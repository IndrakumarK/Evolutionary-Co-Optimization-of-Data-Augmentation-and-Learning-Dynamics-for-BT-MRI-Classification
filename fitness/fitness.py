import torch
import torch.nn.functional as F
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim,
    peak_signal_noise_ratio as psnr,
)

try:
    import lpips
    lpips_model = lpips.LPIPS(net="alex")
except Exception:
    lpips_model = None


def compute_fitness(
    original,
    augmented,
    predictions,
    targets,
    task="classification"
):

    # ---- Paper weights ----
    w1 = 0.3  # SSIM
    w2 = 0.2  # PSNR
    w3 = 0.2  # LPIPS
    w4 = 0.3  # Dice

    # ---- ensure correct shape ----
    if original.dim() == 3:
        original = original.unsqueeze(1)
    if augmented.dim() == 3:
        augmented = augmented.unsqueeze(1)

    # ---- cross entropy ----
    ce_loss = F.cross_entropy(predictions, targets)

    # ---- SSIM penalty ----
    ssim_val = ssim(augmented, original)
    ssim_penalty = 1 - ssim_val

    # ---- PSNR penalty ----
    psnr_val = psnr(augmented, original)
    psnr_penalty = torch.clamp(1 / (psnr_val + 1e-6), max=10.0)

    # ---- LPIPS perceptual distance ----
    if lpips_model is not None:

        model = lpips_model.to(original.device)

        # normalize to [-1,1]
        aug_lp = augmented * 2 - 1
        org_lp = original * 2 - 1

        lpips_val = model(aug_lp, org_lp).mean()

    else:

        lpips_val = torch.tensor(0.0, device=original.device)

    # ---- Dice (only segmentation) ----
    if task == "segmentation":

        preds = torch.argmax(predictions, dim=1)

        preds = preds.float()
        targets = targets.float()

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()

        dice = (2 * intersection + 1e-6) / (union + 1e-6)

        dice_penalty = 1 - dice

    else:

        dice_penalty = torch.tensor(0.0, device=original.device)
        w4 = 0.0

    # ---- final fitness ----
    fitness = (
        ce_loss
        + w1 * ssim_penalty
        + w2 * psnr_penalty
        + w3 * lpips_val
        + w4 * dice_penalty
    )

    return fitness