"""
train.py — experiment entry point.

Run with default config:
    python train.py

Override anything from the command line (Hydra syntax):
    python train.py optimizer=sgd optimizer.lr=0.05 trainer.epochs=30
    python train.py logger.mode=disabled          # quick test, no wandb
    python train.py logger.run_name=resnet18-sgd  # fix a readable run name

Launch a quick local sweep (no wandb agent needed):
    for lr in 1e-3 1e-4; do
        python train.py optimizer.lr=$lr logger.tags="[lr-sweep]"
    done
"""

import torch
import torchvision
import torchvision.transforms as T
import hydra
from omegaconf import DictConfig

from trainer import CustomTrainer
from logger import build_logger


# -----------------------------------------------------------------------
# Model — instantiated here, NOT in YAML.
#
# Why: models often need conditional logic (pretrained weights, frozen
# layers, custom heads), which is awkward to express in a config file.
# The architecture name / hyperparams can still live in config.yaml
# if you want to sweep them; just read them from `cfg` here.
# -----------------------------------------------------------------------

def build_model(num_classes: int = 10) -> torch.nn.Module:
    """
    Edit this function to swap architectures.
    Everything else (trainer, logger, optimizer) stays untouched.
    """
    model = torchvision.models.resnet18(weights=None)
    # Replace the classifier head for the target number of classes.
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


# -----------------------------------------------------------------------
# Data — adjust transforms and dataset to your task.
# -----------------------------------------------------------------------

def build_loaders(cfg: DictConfig):
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


# -----------------------------------------------------------------------
# Metrics — extend this with whatever you need (F1, AUC, mAP, ...).
# Must return a plain dict of {str: float}.
# -----------------------------------------------------------------------

def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> dict:
    preds = outputs.argmax(dim=1)
    acc = (preds == targets).float().mean().item()
    return {"accuracy": acc}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Build components independently — easy to unit-test each one.
    model = build_model(num_classes=10)
    train_loader, val_loader = build_loaders(cfg)

    # Logger: NullLogger when mode=disabled, WandBLogger otherwise.
    # full_cfg is passed so every hyperparameter is stored on the run.
    logger = build_logger(cfg.logger, full_cfg=cfg)

    # Optionally stream gradient histograms to the dashboard.
    if cfg.logger.get("watch_model", False):
        logger.watch(
            model,
            log=cfg.logger.get("watch_log", "gradients"),
            log_freq=cfg.logger.get("watch_log_freq", 100),
        )

    trainer = CustomTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=cfg.trainer,
        logger=logger,
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train()
    finally:
        # Always close the run, even if training crashes midway.
        logger.finish()


if __name__ == "__main__":
    main()