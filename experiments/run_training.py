import argparse
import torch
import yaml
import os

from models import HybridModel
from optimizer import ETDACVO
from training.trainer import train
from utils.seed import set_seed
from utils.logger import log


def load_dataset(config):
    """
    Generic dataset loader.
    Replace with actual dataset loading logic.
    """
    dataset_name = config.get("dataset", "unknown")

    print(f"Loading dataset: {dataset_name}")

    train_loader = None
    val_loader = None

    print("? Implement dataset loader for:", dataset_name)

    return train_loader, val_loader


def main(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_loader, val_loader = load_dataset(config)

    # Model
    num_classes = config.get("num_classes", 2)
    model = HybridModel(num_classes=num_classes)
    model.to(device)

    # ETDACVO controller
    etdacvo = ETDACVO(
        config=config,
        dim=config.get("param_dim", 10)
    )

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        etdacvo=etdacvo,
        config=config,
        device=device
    )

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join(
        "checkpoints",
        f"{config.get('dataset', 'model')}_etdacvo.pth"
    )

    torch.save(model.state_dict(), checkpoint_path)

    log(f"Training completed. Model saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    main(args.config)