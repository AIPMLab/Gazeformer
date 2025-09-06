import os
import wandb
from config import *
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Callable


def get_wandb_run(name: str) -> Tuple[wandb.sdk.wandb_init.Run, Callable]:
    os.environ["HTTP_PROXY"] = "http://localhost:17891"
    os.environ["HTTPS_PROXY"] = "http://localhost:17891"
    wandb.login()
    run = wandb.init(
        mode="offline",
        # Set the wandb entity where your project will be logged (generally your team name).
        # entity="my-awesome-team-name",
        # Set the wandb project where this run will be logged.
        project="GE with CLIP",
        name=name,
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": CNN_MODEL,
            "train_dataset": TRAIN_DATASET_NAME,
            "test_dataset": TEST_DATASET_NAME,
            "epochs": NUM_EPOCHS,
        },
    )

    return run, run.finish


def get_tensorboard_writer(name: str) -> Tuple[SummaryWriter, Callable]:
    writer = SummaryWriter(log_dir=RUNS_PATH / name)

    return writer, writer.close
