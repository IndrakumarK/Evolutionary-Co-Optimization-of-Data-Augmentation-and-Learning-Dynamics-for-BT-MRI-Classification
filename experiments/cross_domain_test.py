import torch
from torch.utils.data import DataLoader
from models import HybridModel
from training.evaluate import evaluate
from utils.metrics import accuracy
import argparse


def cross_domain_test(model_path,
                      source_loader,
                      target_loader,
                      device="cuda"):
    """
    Evaluate cross-dataset generalization without fine-tuning.
    """

    model = HybridModel(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Evaluating on source dataset...")
    source_acc = evaluate(model, source_loader, device=device)

    print("Evaluating on target dataset (cross-domain)...")
    target_acc = evaluate(model, target_loader, device=device)

    retention = (target_acc / source_acc) * 100

    print("Source Accuracy:", round(source_acc, 4))
    print("Target Accuracy:", round(target_acc, 4))
    print("Cross-Domain Retention (%):", round(retention, 2))

    return {
        "source_accuracy": source_acc,
        "target_accuracy": target_acc,
        "retention_percent": retention
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Placeholder loaders (replace with actual dataset loaders)
    source_loader = None
    target_loader = None

    print("? Please replace dataset loaders before running.")

    # Example call (will not run until loaders are defined)
    # cross_domain_test(args.model_path, source_loader, target_loader, args.device)