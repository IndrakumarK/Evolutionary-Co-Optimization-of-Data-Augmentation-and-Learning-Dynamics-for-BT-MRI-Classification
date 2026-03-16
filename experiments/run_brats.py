import argparse
import torch
from torch.utils.data import DataLoader
import yaml
import os

from models import HybridModel
from optimizer import ETDACVO
from training.trainer import train
from utils.seed import set_seed
from utils.logger import log


def load_brats_dataset(config):
    """
    Placeholder dataset loader.
    Replace with actual BraTS dataset class.
    """
    train_loader = None
    val_loader = None

    print("? Replace load_brats_dataset() with actual BraTS loader.")
    return train_loader, val_loader


def main(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_loader, val_loader = load_brats_dataset(config)

    # Initialize model
    model = HybridModel(num_classes=config.get("num_classes", 4))
    model.to(device)

    # Initialize ETDACVO optimizer controller
    etdacvo = ETDACVO(config=config, dim=config.get("param_dim", 10))

    # Train model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        etdacvo=etdacvo,
        config=config,
        device=device
    )

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/brats_etdacvo.pth")

    log("BraTS experiment completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/brats.yaml")

    args = parser.parse_args()

    main(args.config)