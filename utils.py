import math
import torch
import numpy as np
from pathlib import Path
from typing import List  # 添加这一行


def gaze_dir_3d_to_class(gaze_dir_3d: torch.Tensor) -> int:
    x, y, _ = gaze_dir_3d
    angle = math.degrees(math.atan2(y, x))
    if angle < 0:
        angle += 360
    relative_angle = (angle + 22.5) % 360
    sector = int(relative_angle // 45 % 8)
    return sector


# def leave_one_out(ds_name: str, labels_path: Path, leave: int) -> List[List[str]]:
#     if ds_name == "MPIIFaceGaze":
#         one_filename = f"p{leave:02d}.label"
#     elif ds_name == "EyeDiap":
#         one_filename = f"Cluster{leave}.label"
        

#     return [
#         label_path
#         for label_path in labels_path.glob("*.label")
#         if label_path.name != one_filename
#     ]


# def one(ds_name: str, labels_path: Path, leave: int) -> List[str]:
#     if ds_name == "MPIIFaceGaze":
#         one_filename = f"p{leave:02d}.label"
#     elif ds_name == "EyeDiap":
#         one_filename = f"Cluster{leave}.label"

#     return [labels_path / one_filename]

def leave_one_out(ds_name: str, labels_path: Path, leave: int) -> list:
    if ds_name == "MPIIFaceGaze":
        one_filename = f"p{leave:02d}.label"
    elif ds_name == "EyeDiap":
        one_filename = f"p{leave}.label"
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")

    # 只保留实际存在的文件
    return [
        label_path
        for label_path in labels_path.glob("p*.label")
        if label_path.name != one_filename and label_path.exists()
    ]

def one(ds_name: str, labels_path: Path, leave: int) -> list:
    if ds_name == "MPIIFaceGaze":
        one_filename = f"p{leave:02d}.label"
    elif ds_name == "EyeDiap":
        one_filename = f"p{leave}.label"
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")

    file_path = labels_path / one_filename
    # 只返回实际存在的文件，否则返回空列表
    return [file_path] if file_path.exists() else []
def gaze_pitch_yaw_to_ccs(gaze: np.ndarray, degrees: bool = False):
    """
    将 gaze 的 pitch 和 yaw 转为 CCS 坐标系下的 3D 单位向量。

    参数：
      pitch: 俯仰角（弧度，默认）；
      yaw:   偏航角（弧度，默认）；
      degrees: bool，可选，若为 True 则 pitch/yaw 以角度为单位。

    返回：
      numpy.ndarray, shape=(3,), [x, y, z]，单位向量。
    """
    pitch, yaw = gaze
    if degrees:
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    # 假设：
    #   x 轴向右，
    #   y 轴向上，
    #   z 轴向前（镜头光轴方向）
    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)
    return np.array([x, y, z])
