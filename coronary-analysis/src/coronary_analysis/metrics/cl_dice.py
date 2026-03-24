import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 4:
        raise ValueError(f"Expected tensor of shape [B, C, H, W], got {img.shape}")

    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    img = img.clamp(0.0, 1.0)

    skel = F.relu(img - soft_open(img))
    for _ in range(iters):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel = skel + F.relu(delta - skel * delta)

    return skel


class SoftClDiceLoss(nn.Module):
    def __init__(self, iters: int = 10, smooth: float = 1e-6):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        pred_skel = soft_skeletonize(probs, self.iters)
        target_skel = soft_skeletonize(targets, self.iters)

        # topology precision: skeleton of pred inside target
        tprec = (pred_skel * targets).sum(dim=(1, 2, 3)) / (
            pred_skel.sum(dim=(1, 2, 3)) + self.smooth
        )

        # topology sensitivity: skeleton of target inside pred
        tsens = (target_skel * probs).sum(dim=(1, 2, 3)) / (
            target_skel.sum(dim=(1, 2, 3)) + self.smooth
        )

        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice.mean()
