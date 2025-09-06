# GazeFormer: Context-Aware Gaze Estimation via CLIP and MoE Transformer

> A PyTorch framework for 3D Gaze Estimation by fusing semantic priors from CLIP with visual features through a Mixture-of-Experts Transformer.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

This repository contains the official implementation for **GazeFormer**.
## Key Features

- **Text-Conditioned Feature Fusion:** Dynamically guides gaze-relevant feature extraction using textual prompts for attributes like illumination, head pose, and gaze direction.
- **Multi-Source Token Aggregation:** Fuses semantic embeddings from CLIP, spatial features from a CNN backbone, and raw patch tokens using a Mixture-of-Experts (MoE) transformer architecture.
- **Cross-Dataset Generalization:** Designed and tested for robustness across multiple standard gaze datasets (Gaze360, ETH-XGaze, MPIIFaceGaze, EyeDiap) with minimal changes.
- **Ablation-Ready:** Easily toggle different feature streams (`feature_1` to `feature_4`) via a central `ABLA_CONFIG` for controlled experiments.

## Architecture Overview

GazeFormer processes face images through three parallel streams, which are then tokenized and fed into a transformer for final 3D gaze regression.

1.  **CLIP-Semantic Stream:** A frozen CLIP model encodes the input image and a bank of textual prompts. The most relevant text-based attribute embeddings (e.g., "a face with bright light", "a face looking left") are selected via cosine similarity and fused with the image embedding to create task-aligned and context-compensated features.
2.  **CNN-Visual Stream:** A standard CNN backbone (e.g., ResNet-50) extracts a rich spatial feature map, providing a strong visual-geometric prior.
3.  **Fusion Transformer:** The outputs from the above streams are projected into a common token space. A transformer, enhanced with Mixture-of-Experts (MoE) layers, aggregates these tokens to predict the final 3D gaze vector.

## Setup

### 1. Prerequisites

- Python 3.10+
- PyTorch 1.2.1+
- CUDA 11.3+

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Gazeformer_submission.git
cd Gazeformer_submission
pip install -r requirements.txt
```

A `requirements.txt` should contain:
```
torch
torchvision
timm
easydict
ftfy
regex
opencv-python
numpy
tqdm
wandb
git+https://github.com/openai/CLIP.git
```

### 3. Datasets

Download the required datasets and organize them following the GazeHub convention:

```
datasets/
├── Gaze360/
│   └── GazeHub/
│       ├── Image/
│       └── Label/
├── ETH-XGaze/
│   └── GazeHub/
│       ├── Image/
│       └── Label/
...
```

Update the paths in `config.py` if your structure differs.

## Usage

### Training

The main training script `train.py` handles dataset loading, model initialization, and the training loop.

To start training, run:

```bash
python train.py
```

- **Configuration:** Modify `config.py` to set hyperparameters, select datasets (`TRAIN_DATASET_NAME`, `TEST_DATASET_NAME`), and choose the CNN backbone (`CNN_MODEL`).
- **Ablation Studies:** Enable or disable feature streams by editing the `ABLA_CONFIG` dictionary in `config.py`.
- **Logging:** Training progress and validation results are logged to the `log/` directory and can be monitored with TensorBoard.

### Evaluation

The model is evaluated on the validation set periodically during training. To run a standalone evaluation, you would typically load a checkpoint and run the test loop.


