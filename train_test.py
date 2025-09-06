# imports, GLOBAL_PARAMS, some that can be placed in the front


import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch import nn, optim
import math
from easydict import EasyDict as edict
from tqdm import tqdm
import random
import os
from loggers import get_tensorboard_writer
from utils import gaze_dir_3d_to_class, leave_one_out, one
# from gazehub_datasets import (
#     DatasetMPIIFaceGazeByGazeHub,
#     DatasetEyeDiapByGazeHub,
#     DatasetGaze360ByGazeHub,
#     DatasetETHXGazeByGazeHub,
# )
from models import GEWithCLIPModel
from config import *
from torch.utils.tensorboard import SummaryWriter


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True)
os.environ["PYTHONHASHSEED"] = str(SEED)


cnn_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# train and test

def random_loader(num_batches: int):
    for _ in range(num_batches):
        # 使用 torch.rand 生成 [0,1] 的数据
        faces = torch.rand(BATCH_SIZE, 3, 224, 224, device=DEVICE)
        other_faces = torch.rand(BATCH_SIZE, 3, 224, 224, device=DEVICE)
        # 随机 3D gaze 向量仍可用 randn 或者 rand，根据实际需求调整
        labels = torch.randn(BATCH_SIZE, 3, device=DEVICE)
        yield edict(face=faces, other_face=other_faces), labels
def test(model, test_dl, epoch, log_file):
    total_error = 0.0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(
            test_dl,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}, Testing",
            unit="batch",
        ):
            faces = inputs.face.to(DEVICE)
            other_faces = inputs.other_face.to(DEVICE)
            labels = labels.to(DEVICE)

            gaze_pred, _, _, _ = model(faces, other_faces)
            pred_n = gaze_pred / gaze_pred.norm(dim=-1, keepdim=True)
            gt_n = labels / labels.norm(dim=-1, keepdim=True)
            cos_sim = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
            angles = torch.acos(cos_sim) * 180.0 / math.pi

            total_error += angles.sum().item()
            total_count += angles.numel()

    avg_angular_error = total_error / total_count
    log_file.write(
        f"Epoch {epoch}/{NUM_EPOCHS}, Test Angular Error: {avg_angular_error:.2f}°\n"
    )


def main(
    train_leave: int,
    test_leave: int,
    logger: SummaryWriter,
    isVal: bool = False,
):
    if IS_TRAIN:
        if train_leave is None:
            log_file_path = LOGS_PATH / TRAIN_RUN_NAME / "train.log"
        else:
            log_file_path = LOGS_PATH / TRAIN_RUN_NAME / f"train_{train_leave}.log"
    else:
        if test_leave is None:
            log_file_path = LOGS_PATH / TEST_RUN_NAME / "test.log"
        else:
            log_file_path = LOGS_PATH / TEST_RUN_NAME / f"test_{test_leave}.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_file_path, "w", encoding="utf-8")

    # if TRAIN_DATASET_NAME == "MPIIFaceGaze":
    #     TRAIN_DS = DatasetMPIIFaceGazeByGazeHub
    # elif TRAIN_DATASET_NAME == "EyeDiap":
    #     TRAIN_DS = DatasetEyeDiapByGazeHub
    # elif TRAIN_DATASET_NAME == "Gaze360":
    #     TRAIN_DS = DatasetGaze360ByGazeHub
    # elif TRAIN_DATASET_NAME == "ETH-XGaze":
    #     TRAIN_DS = DatasetETHXGazeByGazeHub
    # if TEST_DATASET_NAME == "MPIIFaceGaze":
    #     TEST_DS = DatasetMPIIFaceGazeByGazeHub
    # elif TEST_DATASET_NAME == "EyeDiap":
    #     TEST_DS = DatasetEyeDiapByGazeHub
    # elif TEST_DATASET_NAME == "Gaze360":
    #     TEST_DS = DatasetGaze360ByGazeHub
    # elif TEST_DATASET_NAME == "ETH-XGaze":
    #     TEST_DS = DatasetETHXGazeByGazeHub

    # train_ds = TRAIN_DS(
    #     TRAIN_IMAGES_PATH,
    #     (
    #         leave_one_out(TRAIN_DATASET_NAME, TRAIN_LABELS_PATH, train_leave)
    #         if train_leave is not None
    #         else [TRAIN_LABELS_PATH / "train.label"]
    #     ),
    #     transform=CLIP_PREPROCESS,
    #     other_transform=cnn_preprocess,
    # )
    # if isVal:
    #     val_ds = TRAIN_DS(
    #         TRAIN_IMAGES_PATH,
    #         [TRAIN_LABELS_PATH / "val.label"],
    #         transform=CLIP_PREPROCESS,
    #         other_transform=cnn_preprocess,
    #     )
    # test_ds = TEST_DS(
    #     TEST_IMAGES_PATH,
    #     (
    #         one(TEST_DATASET_NAME, TEST_LABELS_PATH, test_leave)
    #         if test_leave is not None
    #         else [TEST_LABELS_PATH / "test.label"]
    #     ),
    #     transform=CLIP_PREPROCESS,
    #     other_transform=cnn_preprocess,
    # )
    gen = torch.Generator().manual_seed(SEED)
    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     generator=gen,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=True,
    # )
    num_batches = 5
    train_dl = list(random_loader(num_batches))
    test_dl = list(random_loader(
    ))
    # if isVal:
    #     val_dl = DataLoader(
    #         val_ds,
    #         batch_size=BATCH_SIZE,
    #         shuffle=False,
    #         num_workers=NUM_WORKERS,
    #         pin_memory=True,
    #     )
    # test_dl = DataLoader(
    #     test_ds,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=True,
    # )

    def check_model_nan(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in {name}")
                return True
        return False

# 在模型创建后立即检查
    model = GEWithCLIPModel().to(DEVICE)
    if check_model_nan(model):
        print("Model has NaN after initialization!")

    try:
        # 获取一个示例输入来构建计算图
        sample_batch = next(iter(train_dl))
        sample_faces = sample_batch[0].face.to(DEVICE)
        sample_other_faces = sample_batch[0].other_face.to(DEVICE)

        # 分离梯度以避免将需要梯度的张量作为常量插入
        sample_faces = sample_faces.detach()
        sample_other_faces = sample_other_faces.detach()

        # 确保模型在eval模式下构建图
        model.eval()
        with torch.no_grad():
            # 添加模型计算图到 TensorBoard
            logger.add_graph(model, (sample_faces, sample_other_faces))
    # TODO
    except Exception as e:
        print(f"Warning: Could not add graph to tensorboard: {e}")

    # 定义损失和优化器
    # features_loss = (
    #     lambda f1, f2: (f1 * f2).sum(dim=-1)
    #     / (f1.norm(dim=-1) * f2.norm(dim=-1)).pow(2).mean()
    # )
    epsilon = 1e-6
    features_loss = (
        lambda f1, f2: ((f1 * f2).sum(dim=-1)
                        / ((f1.norm(dim=-1) * f2.norm(dim=-1)).clamp(min=epsilon).pow(2))).mean()
    )
    ce_loss = nn.CrossEntropyLoss()
    gaze_loss = nn.L1Loss()
    # torch.autograd.set_detect_anomaly(True)
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    if IS_TRAIN:
        model.train()
        best_val_sum_loss = float("inf")
        step_count = 0
        for i_epoch in range(NUM_EPOCHS):
            # --- train ---
            model.train()
            train_loss_sum = 0.0
            # for inputs, labels in notebook_tqdm(
            for inputs, labels in tqdm(
                train_dl,
                desc=f"Epoch {i_epoch + 1}/{NUM_EPOCHS}, Training",
                unit="batch",
            ):
                step_count += 1

                faces = inputs.face.to(DEVICE)
                other_faces = inputs.other_face.to(DEVICE)
                labels = labels.to(DEVICE)
                class_indices = torch.tensor(
                    [gaze_dir_3d_to_class(l.cpu()) for l in labels]
                ).to(DEVICE)

                optimizer.zero_grad()
                gaze_pred, sim_label, feature_1, feature_2 = model(
                    faces,
                    other_faces,
                )
                if torch.isnan(faces).any():
                    print(f"NaN detected in faces at step {step_count}")
                if torch.isnan(other_faces).any():
                    print(f"NaN detected in other_faces at step {step_count}")
                if torch.isnan(labels).any():
                    print(f"NaN detected in labels at step {step_count}") 
                features_loss_values = features_loss(feature_1, feature_2)
                ce_loss_values = ce_loss(sim_label, class_indices)
            
                gaze_loss_values = gaze_loss(gaze_pred, labels)
                sum_loss_values = (
                    0*features_loss_values + 0*ce_loss_values + gaze_loss_values
                )

                sum_loss_values.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                # 分别记录loss
                logger.add_scalar(
                    "Loss/train_features_loss", features_loss_values.item(), step_count
                )
                logger.add_scalar(
                    "Loss/train_ce_loss", ce_loss_values.item(), step_count
                )
                logger.add_scalar(
                    "Loss/train_gaze_loss", gaze_loss_values.item(), step_count
                )
                logger.add_scalar(
                    "Loss/train_sum_loss", sum_loss_values.item(), step_count
                )
                logger.flush()

                train_loss_sum += sum_loss_values.item()

            train_avg_sum_loss = train_loss_sum / len(train_dl)
            log_file.write(
                f"Epoch {i_epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_avg_sum_loss:.4f}\n"
            )
            print(f"Epoch {i_epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_avg_sum_loss:.4f}\n")
            # --- validation ---
            if isVal:
                model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    # for inputs, labels in notebook_tqdm(
                    for inputs, labels in tqdm(
                        val_dl,
                        desc=f"Epoch {i_epoch + 1}/{NUM_EPOCHS}, Validating",
                        unit="batch",
                    ):
                        faces = inputs.face.to(DEVICE)
                        other_faces = inputs.other_face.to(DEVICE)
                        labels = labels.to(DEVICE)
                        class_indices = torch.tensor(
                            [gaze_dir_3d_to_class(l.cpu()) for l in labels]
                        ).to(DEVICE)

                        gaze_pred, sim_label, feature_1, feature_2 = model(
                            faces,
                            other_faces,
                        )

                       
                        features_loss_values = features_loss(feature_1, feature_2)
                        ce_loss_values = ce_loss(sim_label, class_indices)
                        gaze_loss_values = gaze_loss(gaze_pred, labels)
                        sum_loss_values = (
                            features_loss_values + ce_loss_values + gaze_loss_values
                        )

                        val_loss_sum += sum_loss_values.item()

                val_avg_sum_loss = val_loss_sum / len(val_dl)
                log_file.write(
                    f"Epoch {i_epoch + 1}/{NUM_EPOCHS}, Validation Loss: {val_avg_sum_loss:.4f}\n"
                )

                if val_avg_sum_loss < best_val_sum_loss:
                    best_val_sum_loss = val_avg_sum_loss
                    log_file.write(
                        f"Epoch {i_epoch + 1}/{NUM_EPOCHS}, Best Validation Loss: {best_val_sum_loss:.4f}\n"
                    )
                    save_path = CHECKPOINTS_PATH / TRAIN_RUN_NAME / f"best.pth"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

            # --- test ---
            if (i_epoch + 1) % TEST_STEP == 0:
                test(model, test_dl, i_epoch + 1, log_file)

            # --- save checkpoint ---
            if (i_epoch + 1) % SAVE_STEP == 0:
                if train_leave is None:
                    save_path = (
                        CHECKPOINTS_PATH / TRAIN_RUN_NAME / f"epoch_{i_epoch + 1}.pth"
                    )
                else:
                    save_path = (
                        CHECKPOINTS_PATH
                        / TRAIN_RUN_NAME
                        / str(train_leave)
                        / f"epoch_{i_epoch + 1}.pth"
                    )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)

    else:
        if train_leave is None:
            save_path = CHECKPOINTS_PATH / TRAIN_RUN_NAME / TEST_CHECKPOINT
        else:
            save_path = (
                CHECKPOINTS_PATH / TRAIN_RUN_NAME / str(train_leave) / TEST_CHECKPOINT
            )
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        test(model, test_dl, TEST_EPOCH, log_file)


# main


# TensorBoard

if TRAIN_DATASET_NAME == "MPIIFaceGaze" and TEST_DATASET_NAME == "MPIIFaceGaze":
    for i in range(3):
        logger, close_logger = get_tensorboard_writer(
            f"{TRAIN_RUN_NAME}_leave_{i}_out"
            if IS_TRAIN
            else f"{TEST_RUN_NAME}_leave_{i}_out"
        )
        print(f"Leave {i} out")
        main(i, i, logger, isVal=False)
        close_logger()

elif TRAIN_DATASET_NAME == "EyeDiap" and TEST_DATASET_NAME == "EyeDiap":
    for i in range(4):
        logger, close_logger = get_tensorboard_writer(
            f"{TRAIN_RUN_NAME}_leave_{i}_out"
            if IS_TRAIN
            else f"{TEST_RUN_NAME}_leave_{i}_out"
        )
        print(f"Leave {i} out")
        main(i, i, logger, isVal=False)
        close_logger()

elif TRAIN_DATASET_NAME == "Gaze360" and TEST_DATASET_NAME == "Gaze360":
    logger, close_logger = get_tensorboard_writer(
        TRAIN_RUN_NAME if IS_TRAIN else TEST_RUN_NAME
    )
    main(None, None, logger, isVal=True)
    close_logger()

elif TRAIN_DATASET_NAME == "ETH-XGaze" and TEST_DATASET_NAME == "ETH-XGaze":
    logger, close_logger = get_tensorboard_writer(
        TRAIN_RUN_NAME if IS_TRAIN else TEST_RUN_NAME
    )
    main(None, None, logger, isVal=False)
    close_logger()

elif TRAIN_DATASET_NAME == "Gaze360" and TEST_DATASET_NAME == "MPIIFaceGaze":
    for i in range(15):
        logger, close_logger = get_tensorboard_writer(
            f"{TRAIN_RUN_NAME}_leave_{i}_out"
            if IS_TRAIN
            else f"{TEST_RUN_NAME}_leave_{i}_out"
        )
        print(f"Leave {i} out")
        main(None, i, logger, isVal=False)
        close_logger()

elif TRAIN_DATASET_NAME == "Gaze360" and TEST_DATASET_NAME == "EyeDiap":
    for i in range(4):
        logger, close_logger = get_tensorboard_writer(
            f"{TRAIN_RUN_NAME}_leave_{i}_out"
            if IS_TRAIN
            else f"{TEST_RUN_NAME}_leave_{i}_out"
        )
        print(f"Leave {i} out")
        main(None, i, logger, isVal=False)
        close_logger()
