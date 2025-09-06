from PIL import Image
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Callable, Optional
import cv2
from easydict import EasyDict as edict
import copy
from config import *
from utils import *

# dataset by GazeHub


class DatasetMPIIFaceGazeByGazeHub(Dataset):
    __transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    __target_transform = lambda label: torch.tensor(label)
    if TRAIN_DATASET_NAME == "Gaze360" and TEST_DATASET_NAME == "MPIIFaceGaze":
        __coefficients = np.array([-1, -1, 1])
    else:
        __coefficients = np.array([1, 1, 1])

    def __init__(
        self,
        images_path: Path,
        label_paths: list[Path],
        transform: Optional[Callable] = None,
        other_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.images_path = images_path
        self.labels = [
            line.split(" ")
            for p in label_paths
            for line in p.read_text(encoding="utf-8").splitlines()[1:]
        ]
        self.labels = [
            edict(
                Face=label[0],
                # 变量名开头不能是数字
                _3DGaze=np.array(label[5].split(",")).astype("float")
                * DatasetMPIIFaceGazeByGazeHub.__coefficients,
            )
            for label in self.labels
        ]
        self.labels = self.labels[:]

        self.transform = (
            transform
            if transform is not None
            else DatasetMPIIFaceGazeByGazeHub.__transform
        )
        self.other_transform = (
            other_transform
            if other_transform is not None
            else DatasetMPIIFaceGazeByGazeHub.__transform
        )
        self.target_transform = (
            target_transform
            if target_transform is not None
            else DatasetMPIIFaceGazeByGazeHub.__target_transform
        )

    def __getitem__(self, idx):
        face_img = cv2.imread(self.images_path / self.labels[idx].Face)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        other_face_img = copy.deepcopy(face_img)
        face_img = self.transform(face_img)
        other_face_img = self.other_transform(other_face_img)
        label = self.target_transform(self.labels[idx]._3DGaze)
        return edict(face=face_img, other_face=other_face_img), label

    def __len__(self):
        return len(self.labels)


class DatasetEyeDiapByGazeHub(Dataset):
    # __transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )
    __transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    __target_transform = lambda label: torch.tensor(label)
    if TRAIN_DATASET_NAME == "Gaze360" and TEST_DATASET_NAME == "EyeDiap":
        __coefficients = np.array([-1, -1, 1])
    else:
        __coefficients = np.array([1, 1, 1])

    def __init__(
        self,
        images_path: Path,
        label_paths: list[Path],
        transform: Optional[Callable] = None,
        other_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.images_path = images_path
        self.labels = [
            line.split(" ")
            for p in label_paths
            for line in p.read_text(encoding="utf-8").splitlines()[1:]
        ]
        self.labels = [
            edict(
                Face=label[0],
                # 变量名开头不能是数字
                _3DGaze=np.array(label[4].split(",")).astype("float")
                * DatasetEyeDiapByGazeHub.__coefficients,
            )
            for label in self.labels
        ]

        self.transform = (
            transform if transform is not None else DatasetEyeDiapByGazeHub.__transform
        )
        self.other_transform = (
            other_transform
            if other_transform is not None
            else DatasetEyeDiapByGazeHub.__transform
        )
        self.target_transform = (
            target_transform
            if target_transform is not None
            else DatasetEyeDiapByGazeHub.__target_transform
        )

    def __getitem__(self, idx):
        face_img = cv2.imread(self.images_path / self.labels[idx].Face)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        other_face_img = copy.deepcopy(face_img)
        face_img = self.transform(face_img)
        other_face_img = self.other_transform(other_face_img)

        label = self.target_transform(self.labels[idx]._3DGaze)

        return edict(face=face_img, other_face=other_face_img), label

    def __len__(self):
        return len(self.labels)


DatasetGaze360ByGazeHub = DatasetEyeDiapByGazeHub


class DatasetETHXGazeByGazeHub(Dataset):
    __transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    __target_transform = lambda label: torch.tensor(label)

    def __init__(
        self,
        images_path: Path,
        label_paths: list[Path],
        transform: Optional[Callable] = None,
        other_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.images_path = images_path
        self.labels = [
            line.split(" ")
            for p in label_paths
            for line in p.read_text(encoding="utf-8").splitlines()[1:]
        ]
        self.labels = [
            edict(
                Face=label[0],
                # 变量名开头不能是数字
                _3DGaze=gaze_pitch_yaw_to_ccs(
                    np.array(label[1].split(",")).astype("float")
                ),
            )
            for label in self.labels
        ]

        self.transform = (
            transform if transform is not None else DatasetETHXGazeByGazeHub.__transform
        )
        self.other_transform = (
            other_transform
            if other_transform is not None
            else DatasetETHXGazeByGazeHub.__transform
        )
        self.target_transform = (
            target_transform
            if target_transform is not None
            else DatasetETHXGazeByGazeHub.__target_transform
        )

    def __getitem__(self, idx):
        face_img = cv2.imread(self.images_path / self.labels[idx].Face)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)
        other_face_img = copy.deepcopy(face_img)
        face_img = self.transform(face_img)
        other_face_img = self.other_transform(other_face_img)

        label = self.target_transform(self.labels[idx]._3DGaze)

        return edict(face=face_img, other_face=other_face_img), label

    def __len__(self):
        return len(self.labels)
