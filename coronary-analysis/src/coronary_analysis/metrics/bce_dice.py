import torch
from torch import nn
from segmentation_models_pytorch import losses as smp_losses

from .cl_dice import SoftClDiceLoss


class BCEDiceCriterion(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp_losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, masks)
        dice_loss = self.dice(logits, masks)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class BCEDiceClDiceCriterion(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        cldice_weight: float = 0.2,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp_losses.DiceLoss(mode="binary", from_logits=True)
        self.cldice = SoftClDiceLoss(iters=10)

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, masks)
        dice_loss = self.dice(logits, masks)
        cldice_loss = self.cldice(logits, masks)

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
            + self.cldice_weight * cldice_loss
        )
