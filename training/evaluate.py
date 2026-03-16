import torch
import torch.nn.functional as F
from utils.metrics import accuracy_score, dice_score, f1_score


def evaluate(model,
             dataloader,
             device="cuda",
             task="classification"):
    """
    Evaluate model on validation/test dataset.
    """

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            if task == "classification":
                loss = F.cross_entropy(outputs, targets)

                preds = torch.argmax(outputs, dim=1)

                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

            else:  # segmentation
                loss = F.cross_entropy(outputs, targets)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    results = {"val_loss": avg_loss}

    if task == "classification":
        acc = total_correct / total_samples
        results["accuracy"] = acc
        results["f1"] = f1_score(
            torch.cat(all_preds),
            torch.cat(all_targets)
        )

    else:
        results["dice"] = dice_score(
            torch.cat(all_preds),
            torch.cat(all_targets)
        )

    return results