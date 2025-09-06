import random
import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from flash_attn.modules.mha import MHA

# from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

# from nn_common import check_nan, check_nan_is_all
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

world_size = 1
rank = 0

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, query, key, value, mask=None):
        attn_output, _ = self.attention(query, key, value, attn_mask=mask)
        return attn_output

class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, cross_attn = False, use_flash_attn = True,return_residual = False):
        super().__init__()
        self.cross_attn = cross_attn
        self.return_residual = return_residual
        self.attention = MHA(embed_dim, num_heads, cross_attn = cross_attn, use_flash_attn = use_flash_attn, return_residual=return_residual)

    def forward(self, x, x_kv=None, mask=None):
        if not self.cross_attn:
            attn_output = self.attention(x, x_kv=None, key_padding_mask=mask)
        else:
            assert x_kv is not None
            attn_output = self.attention(x, x_kv=x_kv,  key_padding_mask=mask)
        return attn_output if not self.return_residual else (attn_output, x)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, inter_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, inter_dim)
        self.fc2 = nn.Linear(inter_dim, embed_dim)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.dropout(x) # remove dropout
        x = self.fc2(x)
        return x

class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Router, self).__init__()
        self.gating = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Compute expert probabilities
        expert_logits = self.gating(x)  # Shape: [batch_size, num_experts]
        expert_probs = F.softmax(expert_logits, dim=-1)
        return expert_probs

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, capacity):
        """
        Mixture of Experts layer.
        :param input_dim: Dimensionality of input features.
        :param output_dim: Dimensionality of output features.
        :param num_experts: Number of experts.
        :param capacity: Number of top experts to use for each input.
        """
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.capacity = capacity
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.router = Router(input_dim, num_experts)

    def forward(self, x):
        """
        Forward pass for Mixture of Experts.
        :param x: Input tensor of shape (batch_size, seq_length, input_dim) or (batch_size, input_dim).
        :return: Output tensor of shape (batch_size, seq_length, output_dim) or (batch_size, output_dim).
        """
        if x.dim() == 2:  # Case: (batch_size, input_dim)
            batch_size, input_dim = x.size()
            seq_length = 1
            x = x.unsqueeze(1)  # Add a dummy sequence length dimension
        elif x.dim() == 3:  # Case: (batch_size, seq_length, input_dim)
            batch_size, seq_length, input_dim = x.size()
        else:
            raise ValueError(f"Unsupported input dimensions: {x.shape}")

        # Flatten sequence and batch dimensions for routing
        x_flat = x.view(batch_size * seq_length, input_dim)

        # Compute expert probabilities
        expert_probs = self.router(x_flat)  # Shape: [batch_size * seq_length, num_experts]
        #debug
        # print("MoE.forward: x_flat shape:", x_flat.shape)
        # print("MoE.forward: expert_probs stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        #     expert_probs.min().item(), expert_probs.max().item(), expert_probs.mean().item()))
        
        # Get top-k experts and their probabilities
        top_k = torch.topk(expert_probs, self.capacity, dim=-1)
        selected_experts = top_k.indices  # Shape: [batch_size * seq_length, capacity]
        selected_probs = top_k.values  # Shape: [batch_size * seq_length, capacity]
        # #debug
        # print("MoE.forward: selected_experts shape:", selected_experts.shape)
        # print("MoE.forward: selected_probs stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        #     selected_probs.min().item(), selected_probs.max().item(), selected_probs.mean().item()))

        # Prepare outputs
        output_dim = self.experts[0].out_features
        outputs = torch.zeros(batch_size * seq_length, output_dim, device=x.device)

        # Process top-k experts
        for i in range(self.capacity):
            expert_index = selected_experts[:, i]  # Indices of the selected expert for each sample
            expert_weight = selected_probs[:, i]  # Probabilities of the selected expert
            
            print(f"MoE.forward: Expert {j} processes {sample_indices.numel()} samples")
            
            # Route inputs to selected experts
            expert_outputs = torch.cat([
                self.experts[j](x_flat[expert_index == j])  # Apply expert `j` to the relevant inputs
                if (expert_index == j).sum() > 0 else torch.zeros(0, output_dim, device=x.device)
                for j in range(self.num_experts)
            ], dim=0)

            # Add weighted outputs
            outputs += expert_outputs * expert_weight.unsqueeze(-1)

        # Reshape outputs back to original dimensions
        outputs = outputs.view(batch_size, seq_length, output_dim)
        if seq_length == 1:  # Remove sequence length dimension if it was added
            outputs = outputs.squeeze(1)

        return outputs

class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, embed_dim, n_routed_experts, n_activated_experts, n_expert_groups, n_limited_groups, score_func="softmax", route_scale=1.0):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = embed_dim
        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, embed_dim))

        # Initialize weights with Xavier/Glorot initialization
        torch.nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.empty(n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        # check_nan(x, f'x in gate')

        scores = F.linear(x, self.weight)

        # check_nan(x, f'scores in gate before softmax')

        if self.score_func == "softmax":
            scores = scores - scores.max(dim=-1, keepdim=True)[0]  # ADDED, Subtract max for numerical stability
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()

        # check_nan(scores, f'scores in gate')

        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)

        # check_nan(scores, f'weights before clamp in gate')

        weights = torch.clamp(weights, min=1e-7)  # ADDED, Avoid very small/zero values
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        # check_nan(scores, f'weights after clamp in gate')

        return weights.type_as(x), indices

class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, embed_dim, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = embed_dim
        # assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // world_size
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(embed_dim, n_routed_experts, n_activated_experts, n_expert_groups=1, n_limited_groups=1, score_func="softmax", route_scale=1.0)
        self.experts = nn.ModuleList([Expert(embed_dim, moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = FeedForward(embed_dim, n_shared_experts * moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        修改后的 MoE.forward：
          1. 将输入展平为 (N, dim)。
          2. 通过 gate 得到每个样本的 topk 专家索引和对应权重（形状均为 [N, topk]）。
          3. 对于每个专家（遍历所有 n_routed_experts），用非零 mask 找出该专家被选择的所有样本，
             并用 index_add_ 聚合专家输出的加权结果。
          4. 加上 shared专家输出后恢复原始形状。
        """
        shape = x.size()
        x_flat = x.view(-1, self.dim)  # [N, dim]
        weights, indices = self.gate(x_flat)  # weights, indices: [N, topk]
        N, topk = weights.shape

        y = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_routed_experts):
            # 找出哪些样本选择了 expert_idx
            mask = (indices == expert_idx)  # [N, topk] boolean
            if mask.sum() == 0:
                continue
            # nonzero 返回形状 [M, 2]，第一列为样本索引，第二列为该样本在 topk 中的位置
            sel = mask.nonzero(as_tuple=False)
            sample_indices = sel[:, 0]  # [M]
            # 对应使用的权重（每个样本可能出现多次则自动累加）
            weight_values = weights[mask].unsqueeze(-1)  # [M, 1]
            # 计算 expert 输出
            expert = self.experts[expert_idx]
            # 注意：如果某些样本可能出现多次，则对相同样本的 expert 输出进行累加
            expert_output = expert(x_flat[sample_indices])  # [M, dim]
            # 累加结果到 y：对于相同 sample index, 将加权输出相加
            y.index_add_(0, sample_indices, expert_output * weight_values)
        # 计算共享专家输出
        z = self.shared_experts(x_flat)
        out_flat = y + z
        return out_flat.view(shape)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts, capacity):
        super(TransformerBlock, self).__init__()
        self.self_attn = Attention(embed_dim, num_heads)
        self.cross_attn = Attention(embed_dim, num_heads)
        # self.ff = FeedForward(embed_dim, ff_dim)
        self.moe = MixtureOfExperts(embed_dim, embed_dim, num_experts, capacity)
        # self.moe = MOE()
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input, mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)

        # Cross-attention
        cross_attn_output = self.cross_attn(x, cross_input, cross_input, mask)
        x = self.norm2(x + cross_attn_output)

        # Feedforward and MoE
        # ff_output = self.ff(x)
        moe_output = self.moe(x)
        # x = self.norm3(x + ff_output + moe_output)
        x = self.norm3(x + moe_output)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=8192):
        """
        Sinusoidal Positional Encoding module.
        :param embed_dim: The dimension of the embeddings.
        :param max_seq_len: The maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        # Create position encoding matrix
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # Shape: (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        encoding = torch.zeros(max_seq_len, embed_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)  # Even indices
        encoding[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        encoding = encoding.unsqueeze(0)  # Add batch dimension: (1, max_seq_len, embed_dim)
        self.register_buffer('positional_encoding', encoding)

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.
        :param x: Input tensor of shape (batch_size, seq_len, embed_dim).
        :return: Tensor with positional encodings added.
        """
        seq_len = x.size(1)
        # return x + self.positional_encoding[:, :seq_len]
        return self.positional_encoding[:, :seq_len]

# implement a transformer by using the transformer block above
class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, num_experts, capacity, seq_length):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(seq_length, embed_dim)
        print(self.embedding)
        self.pos_embedding = PositionalEncoding( embed_dim, max_seq_len=seq_length)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, num_experts, capacity) for _ in range(num_layers)
        ])

    def forward(self,x, cross_input, mask=None):
        x = self.embedding(x) + self.pos_embedding(x)

        # x = self.embedding(x)  # Convert indices to embeddings
        for layer in self.layers:
            x = layer(x,cross_input, mask)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, inter_dim = 10944, layer_id=None):
        super(Block, self).__init__()
        self.self_attn = Attention(embed_dim, num_heads)
        self.n_dense_layers = 3
        n_dense_layers = self.n_dense_layers

        if layer_id is not None:
            self.moe = FeedForward(embed_dim, inter_dim) if layer_id < n_dense_layers else MoE(embed_dim,
                                                                                               n_routed_experts,
                                                                                               n_activated_experts,
                                                                                               n_shared_experts,
                                                                                               moe_inter_dim)
        else:
            self.moe = MoE(embed_dim, n_routed_experts, n_activated_experts,
                           n_shared_experts,
                           moe_inter_dim)

        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input=None, mask=None):
        x_norm = self.norm1(x)
        if cross_input is None:
            # Self-attention
            attn_output = self.self_attn(x_norm, x_norm, x_norm, mask)
            attn_output = x + attn_output
        else:
            # Cross-attention
            cross_input_norm = self.norm1(cross_input)
            attn_output = self.self_attn(x_norm, cross_input_norm, cross_input_norm, mask)
            attn_output = x + attn_output

        # check_nan(attn_output, 'attn_output in self_attn')

        moe_output = self.moe(self.norm3(attn_output))
        moe_output = attn_output + moe_output

        return moe_output

class BlockMoba(nn.Module):
    """
    替换原Block的标准注意力为moba注意力，保留MoE逻辑和同样的init参数
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        n_routed_experts,
        n_activated_experts,
        n_shared_experts,
        moe_inter_dim,
        inter_dim=10944,
        layer_id=None,
        moba_chunk_size=5,
        moba_topk=2
    ):
        super(BlockMoba, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk

        self.n_dense_layers = 9  
        if layer_id is not None and layer_id < self.n_dense_layers:
            self.moe = FeedForward(embed_dim, inter_dim)
        else:
            self.moe = MoE(embed_dim, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input=None, mask=None):
        """
        x: [batch, seq_len, embed_dim]
        cross_input: 如果需要 cross-attention，可以在内部做相应区分
        mask: [batch, seq_len, seq_len], 可选
        """
        x_norm = self.norm1(x)

        if cross_input is None:
            # 自注意力
            attn_output = self._moba_attention(x_norm, x_norm, x_norm, mask)
        else:
            # cross-attention（自行决定是否也用moba）
            cross_input_norm = self.norm1(cross_input)
            attn_output = self._moba_attention(x_norm, cross_input_norm, cross_input_norm, mask)

        # 残差
        out = x + attn_output

        # MoE / FF 处理
        moe_output = self.moe(self.norm3(out))
        out = out + moe_output

        return out
    def _moba_attention(self, q, k, v, mask=None):
        """使用标准注意力替代 moba_attn_varlen"""
        bsz, seqlen, d_model = q.shape
        head_dim = d_model // self.num_heads
        
        # 标准多头注意力实现
        q = q.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return attn_output

    
class TransformerDeepSeek_gaze(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        inter_dim,
        num_heads,
        n_routed_experts,
        n_activated_experts,
        n_shared_experts,
        moe_inter_dim,
        d_model=768,
        out_dim=3,
        dropout_rate=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_rate)  # 新增 Dropout
        self.proj_f1 = nn.Linear(512, d_model)
        self.proj_f2 = nn.Linear(512, d_model)
        self.proj_f3 = nn.Linear(2048, d_model)
        self.proj_patch = nn.Linear(d_model, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList([
            BlockMoba(
                d_model,
                num_heads,
                n_routed_experts,
                n_activated_experts,
                n_shared_experts,
                moe_inter_dim,
                inter_dim=inter_dim,
                layer_id=i,
                moba_chunk_size=5,
                moba_topk=2
            )
            for i in range(num_layers)
        ])
        self.linear_head = nn.Linear(d_model, out_dim)

    def forward(self, raw_inputs, mask=None):
        token_f1 = self.dropout(self.proj_f1(raw_inputs["feature_1"]).unsqueeze(1))
        token_f2 = self.dropout(self.proj_f2(raw_inputs["feature_2"]).unsqueeze(1))
        token_f3 = self.dropout(self.proj_f3(raw_inputs["feature_3"]))
        token_list = [token_f1, token_f2, token_f3]
        if "token_img_patch" in raw_inputs and raw_inputs["token_img_patch"] is not None:
            token_patch = self.dropout(self.proj_patch(raw_inputs["token_img_patch"]))
            token_list.append(token_patch)
        transformer_input = torch.cat(token_list, dim=1)
        batch_size = transformer_input.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat([cls_tokens, transformer_input], dim=1)
        x = transformer_input
        for layer in self.layers:
            x = self.dropout(layer(x, cross_input=None, mask=mask))  # 在每层后加 Dropout
        global_repr = x[:, 0, :]
        out = self.linear_head(global_repr)
        return out
    







