
is_ablation = False  # 消融实验标志，True时不保存checkpoint

import numpy as np
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用性: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch使用的CUDA版本: {torch.version.cuda}")
    print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from gazehub_datasets import (
    DatasetMPIIFaceGazeByGazeHub,
    DatasetEyeDiapByGazeHub,
    DatasetGaze360ByGazeHub,
    DatasetETHXGazeByGazeHub,
)
from config import *
import torch.optim as optim
from utils import leave_one_out, one
from models import GEWithCLIPModel_zhao as GEWithCLIPModel
import torch.nn as nn
from model_zhao_test import TransformerDeepSeek_gaze
import math
import os
from datetime import datetime


class TransformerConfigBase:
    num_layers: int = 12
    embed_dim: int = 512
    inter_dim: int = 2048
    num_heads: int = 8
    # n_routed_experts: int = 8
    # n_activated_experts: int = 4
    n_routed_experts: int = 4
    n_activated_experts: int = 2
    n_shared_experts: int = 2
    moe_inter_dim: int = 1024


config = TransformerConfigBase()


class ZhaoDataset(Dataset):
    def __init__(self, images_path: Path,label_paths: list[Path], ds_name: str):
        super().__init__()
        # Chen-Xianrui
        if ds_name == "MPIIFaceGaze":
            DS = DatasetMPIIFaceGazeByGazeHub
        elif ds_name == "EyeDiap":
            DS = DatasetEyeDiapByGazeHub
        elif ds_name == "Gaze360":
            DS = DatasetGaze360ByGazeHub
        elif ds_name == "ETH-XGaze":
            DS = DatasetETHXGazeByGazeHub
        self.inner_dataset = DS(
            images_path,  # 这里用传入的 images_path
            label_paths,
            CLIP_PREPROCESS,
            CNN_PREPROCESS,
        )

    def __getitem__(self, idx):
        _input, label = self.inner_dataset[idx]
        return _input, label

    def __len__(self):
        return len(self.inner_dataset)


def process_batch(batch, model):
    # 从 DataLoader 中取出的 batch，_input 为输入结构体，label 为 gaze 标签，[B, 3]
    _input, label = batch
    _input.face = _input.face.to(DEVICE)
    _input.other_face = _input.other_face.to(DEVICE)
    label = label.to(DEVICE).float()  # 转换成 half 类型

    # 利用 hook 获取 encoder_i 输出的完整 token 序列（来自 VisionTransformer）
    hook_outputs = {}

    def vt_hook(module, inp, output):
        hook_outputs["full_tokens"] = inp[0].permute(1, 0, 2)  # expect shape [B, L, d_model]

    hook_handle = model.model.visual.transformer.register_forward_hook(vt_hook)
    img_feats = model.encoder_i(_input.face)
    hook_handle.remove()
    if "full_tokens" in hook_outputs:
        full_tokens = hook_outputs["full_tokens"]
        if full_tokens.dim() == 3:
            token_img_patch = full_tokens[:, 1:, :]  # 除去 CLS token，形状 [B, num_patches, d_model]
        else:
            token_img_patch = None
            print("Captured tokens have unexpected dimensions.")
    else:
        token_img_patch = None
        print("Hook did not capture full tokens.")

    # 以下计算全局特征，主要基于 img_feats 及选取的其它信息
    label_feats = model.encoder_t2(model.label_tokens)
    img_norm = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-2)
    label_norm = label_feats / (label_feats.norm(dim=-1, keepdim=True) + 1e-2)
    sim_illum = model.logit_scale.exp() * img_norm @ model.illum_norm.T
    sim_head = model.logit_scale.exp() * img_norm @ model.head_norm.T
    sim_bg = model.logit_scale.exp() * img_norm @ model.bg_norm.T
    sim_label = model.logit_scale.exp() * img_norm @ label_norm.T

    idx_illum = sim_illum.argmax(dim=-1)
    idx_head = sim_head.argmax(dim=-1)
    idx_bg = sim_bg.argmax(dim=-1)
    idx_label = sim_label.argmax(dim=-1)

    selected_illum = model.illum_feats[idx_illum]
    selected_head = model.head_feats[idx_head]
    selected_bg = model.bg_feats[idx_bg]
    selected_label = label_feats[idx_label]

    feature_1 = img_feats + selected_illum + selected_head + selected_bg
    feature_1 = feature_1 / (feature_1.norm(dim=-1, keepdim=True) + 1e-3)
    feature_2 = img_feats + selected_label
    feature_2 = feature_2 / (feature_2.norm(dim=-1, keepdim=True) + 1e-3)

    # 获取 CNN 特征图并生成局部 token
    feature_map = model.main_model(_input.other_face)["features"]  # shape: [B, C, H, W]
    B, C, H, W = feature_map.shape
    tokens_feature3 = feature_map.view(B, C, H * W).transpose(1, 2)  # shape: [B, H*W, C]
    # 这里假设 tokens_feature3 的通道数 C 为512（可根据实际情况调整）

    # 按消融配置控制各特征，若为False则自动补零张量
    features = {}
    features["label"] = label
    # feature_1
    if ABLA_CONFIG.get('use_feature_1', True):
        features["feature_1"] = feature_1
    else:
        features["feature_1"] = torch.zeros_like(feature_1)
    # feature_2
    if ABLA_CONFIG.get('use_feature_2', True):
        features["feature_2"] = feature_2
    else:
        features["feature_2"] = torch.zeros_like(feature_2)
    # feature_3
    if ABLA_CONFIG.get('use_feature_3', True):
        features["feature_3"] = tokens_feature3
    else:
        features["feature_3"] = torch.zeros_like(tokens_feature3)
    # feature_4
    if ABLA_CONFIG.get('use_feature_4', True):
        features["token_img_patch"] = token_img_patch
    else:
        if token_img_patch is not None:
            features["token_img_patch"] = torch.zeros_like(token_img_patch)
        else:
            features["token_img_patch"] = None
    return features

def feature_separation_loss(f1, f2, epsilon=1e-6):

    return ((f1 * f2).sum(dim=-1) / ((f1.norm(dim=-1) * f2.norm(dim=-1)).clamp(min=epsilon).pow(2))).mean()
def angular_loss(pred, target):
    # 归一化
    pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-6)
    target_n = target / (target.norm(dim=-1, keepdim=True) + 1e-6)
    # 计算余弦相似度
    cos_sim = (pred_n * target_n).sum(dim=-1).clamp(-1.0, 1.0)
    # 计算角度（弧度），再取均值
    loss = torch.acos(cos_sim).mean()
    return loss


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    # 构造数据集，这里使用 ZhaoDataset 包裹 DatasetMPIIFaceGazeByGazeHub
    # TODO 多折验证。这里只验证了第一折 Chen-Xianrui
    #for eyediap and mpiiface
    # train_ds = ZhaoDataset(leave_one_out(TRAIN_DATASET_NAME, TRAIN_LABELS_PATH, 0), TRAIN_DATASET_NAME)
    # test_ds = ZhaoDataset(one(TEST_DATASET_NAME, TEST_LABELS_PATH, 0), TEST_DATASET_NAME)
    
    if TRAIN_DATASET_NAME == "ETH-XGaze":
        train_label_path = [TRAIN_LABELS_PATH / "train.label"]
        full_ds = ZhaoDataset(TRAIN_IMAGES_PATH, train_label_path, TRAIN_DATASET_NAME)
        total_len = len(full_ds)
        test_len = int(total_len * 0.15)
        train_len = total_len - test_len
        gen = torch.Generator().manual_seed(SEED)
        train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_len, test_len], generator=gen)
        print(f"{TRAIN_DATASET_NAME}划分: 训练集样本数={train_len}, 测试集样本数={test_len}")
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=gen,
            num_workers=NUM_WORKERS,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
    elif TRAIN_DATASET_NAME == "Gaze360":
        train_label_path = [TRAIN_LABELS_PATH / "train.label"]
        test_label_path = [TRAIN_LABELS_PATH / "test.label"]
        train_ds = ZhaoDataset(TRAIN_IMAGES_PATH, train_label_path, TRAIN_DATASET_NAME)
        test_ds = ZhaoDataset(TEST_IMAGES_PATH, test_label_path, TRAIN_DATASET_NAME)
        print(f"{TRAIN_DATASET_NAME}训练集样本数: {len(train_ds)}, 测试集样本数: {len(test_ds)}")
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
    else:
        train_label_path = leave_one_out(TRAIN_DATASET_NAME, TRAIN_LABELS_PATH, 1)
        val_label_path = one(TEST_DATASET_NAME, TEST_LABELS_PATH, 1)
        train_ds = ZhaoDataset(TRAIN_IMAGES_PATH, train_label_path, TRAIN_DATASET_NAME)
        test_ds = ZhaoDataset(TEST_IMAGES_PATH, val_label_path, TRAIN_DATASET_NAME)
        print("训练集样本数:", len(train_ds))
        print("测试集样本数:", len(test_ds))
        gen = torch.Generator().manual_seed(SEED)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=gen,
            num_workers=NUM_WORKERS,
        )
        print("训练集 batch 数量:", len(train_dl))
        test_dl = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    # 加载主模型（例如 CLIP 模型）到 GPU
    model = GEWithCLIPModel().to(DEVICE)
    for param in model.model.visual.parameters():
        param.requires_grad = False
    for param in model.model.transformer.parameters():
        param.requires_grad = False
    # model.main_model = model.main_model.half()  # 如果 main_model 是 CNN

    # 实例化 TransformerDeepSeek_gaze 模型，注意内部投影层参数维度需跟数据一致：
    config = TransformerConfigBase()
    transformer_model = TransformerDeepSeek_gaze(
        config.num_layers,
        config.embed_dim,
        config.inter_dim,
        config.num_heads,
        config.n_routed_experts,
        config.n_activated_experts,
        config.n_shared_experts,
        config.moe_inter_dim,
        d_model=768,  # 统一投影到 768 维
        out_dim=3  # gaze 输出维度为 3
    ).to(DEVICE)

    criterion = torch.nn.HuberLoss()

    optimizer = torch.optim.AdamW(
    list(transformer_model.parameters()) +
    list(model.fuse_model.parameters()) +
    list(model.main_model.parameters()),
  # 加上这行
    lr=LEARNING_RATE,
    )

    # 添加余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=ETA_MIN
    )


    os.makedirs("log", exist_ok=True)
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"log/{log_time}_{TRAIN_DATASET_NAME}-{TEST_DATASET_NAME}_log.txt"


    def write_log(msg):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


    os.makedirs("checkpoints", exist_ok=True)
    best_angle = float('inf')
    best_model_path = None
    scaler = GradScaler()
    for epoch in range(NUM_EPOCHS):
        # --- 训练 ---
        for i, batch in enumerate(train_dl):
            raw_inputs = process_batch(batch, model=model)
            with autocast():
                output = transformer_model(raw_inputs)
                output_safe = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                label_safe = torch.nan_to_num(raw_inputs["label"], nan=0.0, posinf=1e6, neginf=-1e6)
                loss_gaze = angular_loss(output_safe, label_safe)
                # 新增特征分离损失
                loss_sep = feature_separation_loss(raw_inputs["feature_1"], raw_inputs["feature_2"])
                # 总损失
                lambda_sep = 1.0  # 可调节权重
                loss = loss_gaze + lambda_sep * loss_sep
                pred_n = output_safe / (output_safe.norm(dim=-1, keepdim=True) + 1e-6)
                gt_n = label_safe / (label_safe.norm(dim=-1, keepdim=True) + 1e-6)
                cos_sim = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
                angles = torch.acos(cos_sim) * 180.0 / math.pi
                mean_angle = angles.mean().item()
            if i % 50 == 0:
                log_msg = (
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}——{TRAIN_DATASET_NAME}-{TEST_DATASET_NAME} "
                    f"Epoch {epoch} Step {i}, Loss: {loss.item():.6f}, "
                    f"Angular Loss: {loss_gaze.item():.6f}, Feature Loss: {loss_sep.item():.6f}, "
                    f"Mean Angular Error: {mean_angle:.2f}°"
                )
                print(log_msg)
                write_log(log_msg)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # --- 测试集验证 ---
        transformer_model.eval()
        model.eval()
        test_angles = []
        with torch.no_grad():
            for batch in test_dl:
                raw_inputs = process_batch(batch, model=model)
                for k in raw_inputs:
                    if isinstance(raw_inputs[k], torch.Tensor):
                        raw_inputs[k] = raw_inputs[k].float()
                output = transformer_model(raw_inputs)
                output_safe = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                label_safe = torch.nan_to_num(raw_inputs["label"], nan=0.0, posinf=1e6, neginf=-1e6)
                pred_n = output_safe / (output_safe.norm(dim=-1, keepdim=True) + 1e-6)
                gt_n = label_safe / (label_safe.norm(dim=-1, keepdim=True) + 1e-6)
                cos_sim = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
                angles = torch.acos(cos_sim) * 180.0 / math.pi
                test_angles.append(angles)
        mean_test_angle = torch.cat(test_angles).mean().item() if test_angles else float('nan')
        val_msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}——{TRAIN_DATASET_NAME} Epoch {epoch} Test Mean Angular Error: {mean_test_angle:.2f}°"
        print(val_msg)
        write_log(val_msg)
        # 保存最佳模型（非消融实验时才保存）
        if not is_ablation and mean_test_angle < best_angle:
            best_angle = mean_test_angle
            best_model_path = f"checkpoints/best—separate-added_{TRAIN_DATASET_NAME}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean_test_angle': best_angle
            }, best_model_path)
            print(f"Best model saved to {best_model_path}")
        transformer_model.train()
        model.train()
    scheduler.step()
    # 每轮训练后在训练集上推理并输出2D预测与GT
    if TRAIN_DATASET_NAME == "ETH-XGaze":
        transformer_model.eval()
        model.eval()
        pred_lines = []
        img_idx = 0
        pred_y_list = []
        label_y_list = []
        pred_norm_list = []
        label_norm_list = []
        with torch.no_grad():
            for batch in train_dl:
                raw_inputs = process_batch(batch, model=model)
                for k in raw_inputs:
                    if isinstance(raw_inputs[k], torch.Tensor):
                        raw_inputs[k] = raw_inputs[k].float()
                output = transformer_model(raw_inputs)
                output_safe = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                label_safe = torch.nan_to_num(raw_inputs["label"], nan=0.0, posinf=1e6, neginf=-1e6)
                batch_size = output_safe.shape[0]
                for idx_in_batch in range(batch_size):
                    img_path = train_dl.dataset.inner_dataset.labels[img_idx].Face
                    def ccs_to_pitchyaw(vec):
                        x, y, z = vec
                        y = max(min(y, 1.0), -1.0)
                        pitch = math.asin(y)
                        yaw = math.atan2(x, z)
                        return pitch, yaw
                    pred_3d = output_safe[idx_in_batch].cpu().numpy()
                    label_3d = label_safe[idx_in_batch].cpu().numpy()
                    # 归一化
                    pred_3d_norm = np.linalg.norm(pred_3d) + 1e-6
                    label_3d_norm = np.linalg.norm(label_3d) + 1e-6
                    pred_3d_unit = pred_3d / pred_3d_norm
                    label_3d_unit = label_3d / label_3d_norm
                    pred_y_list.append(pred_3d[1])
                    label_y_list.append(label_3d[1])
                    pred_norm_list.append(pred_3d_norm)
                    label_norm_list.append(label_3d_norm)
                    # 只打印前20个样本的3D向量及y分量（未归一化和归一化后）
                    if img_idx < 20:
                        print(f"[DEBUG] img: {img_path}\npred_3d: {pred_3d}, pred_y: {pred_3d[1]:.6f}, pred_3d_unit: {pred_3d_unit}, pred_y_unit: {pred_3d_unit[1]:.6f}\nlabel_3d: {label_3d}, label_y: {label_3d[1]:.6f}, label_3d_unit: {label_3d_unit}, label_y_unit: {label_3d_unit[1]:.6f}")
                    pred_2d = ccs_to_pitchyaw(pred_3d_unit)
                    label_2d = ccs_to_pitchyaw(label_3d_unit)
                    pred_lines.append(f"{img_path},pred_pitch={pred_2d[0]:.6f},pred_yaw={pred_2d[1]:.6f},label_pitch={label_2d[0]:.6f},label_yaw={label_2d[1]:.6f}")
                    img_idx += 1
        # 统计归一化情况和y分量分布
        pred_y_arr = np.array(pred_y_list)
        label_y_arr = np.array(label_y_list)
        pred_norm_arr = np.array(pred_norm_list)
        label_norm_arr = np.array(label_norm_list)
        # print("[SUMMARY] pred_3d范数: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(pred_norm_arr.mean(), pred_norm_arr.std(), pred_norm_arr.min(), pred_norm_arr.max()))
        # print("[SUMMARY] label_3d范数: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(label_norm_arr.mean(), label_norm_arr.std(), label_norm_arr.min(), label_norm_arr.max()))
        # print("[SUMMARY] pred_y分布: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(pred_y_arr.mean(), pred_y_arr.std(), pred_y_arr.min(), pred_y_arr.max()))
        # print("[SUMMARY] label_y分布: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(label_y_arr.mean(), label_y_arr.std(), label_y_arr.min(), label_y_arr.max()))
        # 检查是否有y分量超出[-1,1]
        pred_y_out = np.sum((pred_y_arr < -1) | (pred_y_arr > 1))
        label_y_out = np.sum((label_y_arr < -1) | (label_y_arr > 1))
        # print(f"[SUMMARY] pred_y超出[-1,1]的数量: {pred_y_out}")
        # print(f"[SUMMARY] label_y超出[-1,1]的数量: {label_y_out}")
        pred_label_path = f"ethxgaze_train_pred_epoch{epoch}_angle{mean_angle:.2f}_with_gt.txt"
        with open(pred_label_path, "w", encoding="utf-8") as f:
            for line in pred_lines:
                f.write(line + "\n")
        print(f"训练集2D gaze及GT已保存到 {pred_label_path}")
    if TRAIN_DATASET_NAME == "ETH-XGaze":
        transformer_model.eval()
        model.eval()
        pred_lines = []
        img_idx = 0
        with torch.no_grad():
            for batch in test_dl:
                raw_inputs = process_batch(batch, model=model)
                for k in raw_inputs:
                    if isinstance(raw_inputs[k], torch.Tensor):
                        raw_inputs[k] = raw_inputs[k].float()
                output = transformer_model(raw_inputs)
                output_safe = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
                label_safe = torch.nan_to_num(raw_inputs["label"], nan=0.0, posinf=1e6, neginf=-1e6)
                batch_size = output_safe.shape[0]
                for idx_in_batch in range(batch_size):
                    pred_3d = output_safe[idx_in_batch].cpu().numpy()
                    label_3d = label_safe[idx_in_batch].cpu().numpy()
                    # 归一化
                    pred_3d_unit = pred_3d / (np.linalg.norm(pred_3d) + 1e-6)
                    label_3d_unit = label_3d / (np.linalg.norm(label_3d) + 1e-6)
                    # 只输出pitch，空格分隔
                    def ccs_to_pitchyaw(vec):
                        x, y, z = vec
                        y = max(min(y, 1.0), -1.0)
                        pitch = math.asin(y)
                        return pitch
                    pred_pitch = ccs_to_pitchyaw(pred_3d_unit)
                    label_pitch = ccs_to_pitchyaw(label_3d_unit)
                    pred_lines.append(f"{pred_pitch:.6f} {label_pitch:.6f}")
                    img_idx += 1
        pred_label_path = f"ethxgaze_pred_pitch_epoch{epoch}_angle{mean_angle:.2f}_with_gt.txt"
        with open(pred_label_path, "w", encoding="utf-8") as f:
            for line in pred_lines:
                f.write(line + "\n")
        print(f"测试集pitch预测及GT已保存到 {pred_label_path}")
   