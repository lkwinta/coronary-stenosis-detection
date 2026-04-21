import torch
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score


def compute_dice_iou_metrics(
    logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).long()
    targets = masks.long()

    tp, fp, fn, tn = get_stats(
        preds,
        targets,
        mode="binary",
    )

    iou = iou_score(tp, fp, fn, tn, reduction="micro")
    dice = f1_score(tp, fp, fn, tn, reduction="micro")

    return {"dice": dice.item(), "iou": iou.item()}
