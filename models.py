# model


import clip
import torch
import torch.nn as nn
from torchvision import models
import copy
import timm
from config import *
from torchvision.models.feature_extraction import create_feature_extractor

class GEWithCLIPModel(nn.Module):
    def __init__(
        self,
        irrelevant_feats_dim=512,
        relevant_feats_dim=512,
    ):
        super().__init__()

        self.illumination_texts = [
            "a face with bright light",
            "a face with low light",
            "a face with shadows",
        ]
        self.headpose_texts = [
            "a frontal face",
            "a profile face",
        ]
        self.background_texts = [
            "a face on bright background",
            "a face on dark background",
        ]
        self.label_texts = [
            "A photo of a face looking left",
            "A photo of a face looking upper left",
            "A photo of a face looking up",
            "A photo of a face looking upper right",
            "A photo of a face looking right",
            "A photo of a face looking lower right",
            "A photo of a face looking down",
            "A photo of a face looking lower left",
        ]
        illum_tokens = clip.tokenize(self.illumination_texts).to(DEVICE)
        headpose_tokens = clip.tokenize(self.headpose_texts).to(DEVICE)
        bg_tokens = clip.tokenize(self.background_texts).to(DEVICE)
        self.label_tokens = clip.tokenize(self.label_texts).to(DEVICE)
        clip_model_1 = copy.deepcopy(CLIP_MODEL)
        clip_model_1.eval()
        with torch.no_grad():
            # encoder_t1
            self.illum_feats = clip_model_1.encode_text(illum_tokens)
            self.head_feats = clip_model_1.encode_text(headpose_tokens)
            self.bg_feats = clip_model_1.encode_text(bg_tokens)

            self.illum_norm = self.illum_feats / self.illum_feats.norm(
                dim=-1, keepdim=True
            )
            self.head_norm = self.head_feats / self.head_feats.norm(
                dim=-1, keepdim=True
            )
            self.bg_norm = self.bg_feats / self.bg_feats.norm(dim=-1, keepdim=True)
        del clip_model_1
        torch.cuda.empty_cache()
        self.model = CLIP_MODEL
        self.encoder_i = CLIP_MODEL.encode_image
        self.encoder_t2 = CLIP_MODEL.encode_text

        if CNN_MODEL == "ResNet-50":
            # ResNet50 CNN for image features
            # 这个模型默认期望输入的数据是一个形状为 (batch_size, 3, 224(height), 224(width)) 的张量（Tensor）。
            main_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            main_model_feats_dim = main_model.fc.in_features
            main_model.fc = nn.Identity()
        elif CNN_MODEL == "ResNet-18":
            main_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            main_model_feats_dim = main_model.fc.in_features
            main_model.fc = nn.Identity()
        elif CNN_MODEL == "EdgeNeXt-Small":
            main_model = timm.create_model("edgenext_small", pretrained=True)
            main_model_feats_dim = main_model.head.fc.in_features
            main_model.head.fc = nn.Identity()
        self.main_model = main_model

        # fusion layers
        fused_dim = irrelevant_feats_dim + relevant_feats_dim + main_model_feats_dim
        self.fuse_model = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, 3)  # 3D gaze output
        )
        self.logit_scale = CLIP_MODEL.logit_scale

    def forward(
        self,
        face,
        other_face,
    ):
        img_feats = self.encoder_i(face)
        label_feats = self.encoder_t2(self.label_tokens)

        # 归一化后计算相似度，选出最高索引
        img_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)
        label_norm = label_feats / label_feats.norm(dim=-1, keepdim=True)

        sim_illum = self.logit_scale.exp() * img_norm @ self.illum_norm.T
        sim_head = self.logit_scale.exp() * img_norm @ self.head_norm.T
        sim_bg = self.logit_scale.exp() * img_norm @ self.bg_norm.T

        scale = self.logit_scale.exp().clamp(max=10)
        sim_label = scale * img_norm @ label_norm.T

        idx_illum = sim_illum.argmax(dim=-1)
        idx_head = sim_head.argmax(dim=-1)
        idx_bg = sim_bg.argmax(dim=-1)
        idx_label = sim_label.argmax(dim=-1)
        selected_illum = self.illum_feats[idx_illum]
        selected_head = self.head_feats[idx_head]
        selected_bg = self.bg_feats[idx_bg]
        selected_label = label_feats[idx_label]

        feature_1 = img_feats + selected_illum + selected_head + selected_bg
        feature_1 = feature_1 / feature_1.norm(dim=-1, keepdim=True)

        feature_2 = img_feats + selected_label
        feature_2 = feature_2 / feature_2.norm(dim=-1, keepdim=True)

        # feature_3 = self.main_model(other_face).view(face.size(0), -1)
        feature_3 = self.main_model(other_face).reshape(face.size(0), -1)

        fused = torch.cat([feature_1, feature_2, feature_3], dim=-1)
        gaze_pred = self.fuse_model(fused)

        return gaze_pred, sim_label, feature_1, feature_2
class GEWithCLIPModel_zhao(nn.Module):
    def __init__(
        self,
        irrelevant_feats_dim=512,
        relevant_feats_dim=512,
    ):
        super().__init__()

        self.illumination_texts = [
            "a face with bright light",
            "a face with low light",
            "a face with shadows",
        ]
        self.headpose_texts = [
            "a frontal face",
            "a profile face",
        ]
        self.background_texts = [
            "a face on bright background",
            "a face on dark background",
        ]
        self.label_texts = [
            "A photo of a face looking left",
            "A photo of a face looking upper left",
            "A photo of a face looking up",
            "A photo of a face looking upper right",
            "A photo of a face looking right",
            "A photo of a face looking lower right",
            "A photo of a face looking down",
            "A photo of a face looking lower left",
        ]
        illum_tokens = clip.tokenize(self.illumination_texts).to(DEVICE)
        headpose_tokens = clip.tokenize(self.headpose_texts).to(DEVICE)
        bg_tokens = clip.tokenize(self.background_texts).to(DEVICE)
        self.label_tokens = clip.tokenize(self.label_texts).to(DEVICE)
        clip_model_1 = copy.deepcopy(CLIP_MODEL)
        clip_model_1.eval()
        with torch.no_grad():
            # encoder_t1
            self.illum_feats = clip_model_1.encode_text(illum_tokens)
            self.head_feats = clip_model_1.encode_text(headpose_tokens)
            self.bg_feats = clip_model_1.encode_text(bg_tokens)

            self.illum_norm = self.illum_feats / self.illum_feats.norm(
                dim=-1, keepdim=True
            )
            self.head_norm = self.head_feats / self.head_feats.norm(
                dim=-1, keepdim=True
            )
            self.bg_norm = self.bg_feats / self.bg_feats.norm(dim=-1, keepdim=True)
        del clip_model_1
        torch.cuda.empty_cache()
        self.model = CLIP_MODEL
        self.encoder_i = CLIP_MODEL.encode_image
        self.encoder_t2 = CLIP_MODEL.encode_text

        if CNN_MODEL == "ResNet-50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # 使用 create_feature_extractor 提取 layer4 输出的特征图
            return_nodes = {"layer4": "features"}  # layer4 输出特征图
            main_model = create_feature_extractor(base_model, return_nodes=return_nodes)
            # main_model.forward(x) 返回一个字典，key 是 "features"，值形状为 [B, 2048, H, W]
            main_model_feats_dim = 2048  # ResNet50 layer4 的输出通道数

        elif CNN_MODEL == "ResNet-18":
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            return_nodes = {"layer4": "features"}
            main_model = create_feature_extractor(base_model, return_nodes=return_nodes)
            main_model_feats_dim = 512  # ResNet18 layer4 的输出通道数
        elif CNN_MODEL == "EdgeNeXt-Small":
            main_model = timm.create_model("edgenext_small", pretrained=True)
            main_model_feats_dim = main_model.head.fc.in_features
            main_model.head.fc = nn.Identity()
        self.main_model = main_model

        # fusion layers
        fused_dim = irrelevant_feats_dim + relevant_feats_dim + main_model_feats_dim
        self.fuse_model = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, 3)  # 3D gaze output
        )
        self.logit_scale = CLIP_MODEL.logit_scale

    def forward(
        self,
        face,
        other_face,
    ):
        img_feats = self.encoder_i(face)
        label_feats = self.encoder_t2(self.label_tokens)

        # 归一化后计算相似度，选出最高索引
        img_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)
        label_norm = label_feats / label_feats.norm(dim=-1, keepdim=True)

        sim_illum = self.logit_scale.exp() * img_norm @ self.illum_norm.T
        sim_head = self.logit_scale.exp() * img_norm @ self.head_norm.T
        sim_bg = self.logit_scale.exp() * img_norm @ self.bg_norm.T

        scale = self.logit_scale.exp().clamp(max=10)
        sim_label = scale * img_norm @ label_norm.T

        idx_illum = sim_illum.argmax(dim=-1)
        idx_head = sim_head.argmax(dim=-1)
        idx_bg = sim_bg.argmax(dim=-1)
        idx_label = sim_label.argmax(dim=-1)
        selected_illum = self.illum_feats[idx_illum]
        selected_head = self.head_feats[idx_head]
        selected_bg = self.bg_feats[idx_bg]
        selected_label = label_feats[idx_label]

        feature_1 = img_feats + selected_illum + selected_head + selected_bg
        feature_1 = feature_1 / feature_1.norm(dim=-1, keepdim=True)

        feature_2 = img_feats + selected_label
        feature_2 = feature_2 / feature_2.norm(dim=-1, keepdim=True)

        # feature_3 = self.main_model(other_face).view(face.size(0), -1)
        feature_3 = self.main_model(other_face).reshape(face.size(0), -1)

        fused = torch.cat([feature_1, feature_2, feature_3], dim=-1)
        gaze_pred = self.fuse_model(fused)

        return gaze_pred, sim_label, feature_1, feature_2
