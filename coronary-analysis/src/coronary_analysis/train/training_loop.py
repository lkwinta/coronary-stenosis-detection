import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from coronary_analysis.metrics import compute_dice_iou_metrics


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int | None = None,
    num_epochs: int | None = None,
) -> tuple[float, float, float]:
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    desc = (
        f"Train [{epoch}/{num_epochs}]"
        if epoch is not None and num_epochs is not None
        else "Train"
    )

    with tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    ) as batch_bar:
        for images, masks in batch_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            dice, iou = compute_dice_iou_metrics(logits.detach(), masks)

            total_loss += loss.item() * bs
            total_dice += dice * bs
            total_iou += iou * bs
            total_samples += bs

            batch_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                dice=f"{total_dice / total_samples:.4f}",
                iou=f"{total_iou / total_samples:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    return (
        total_loss / total_samples,
        total_dice / total_samples,
        total_iou / total_samples,
    )


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int | None = None,
    num_epochs: int | None = None,
) -> tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    desc = (
        f"Val [{epoch}/{num_epochs}]"
        if epoch is not None and num_epochs is not None
        else "Val"
    )

    with tqdm(
        loader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    ) as batch_bar:
        for images, masks in batch_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            bs = images.size(0)
            dice, iou = compute_dice_iou_metrics(logits, masks)

            total_loss += loss.item() * bs
            total_dice += dice * bs
            total_iou += iou * bs
            total_samples += bs

            batch_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                dice=f"{total_dice / total_samples:.4f}",
                iou=f"{total_iou / total_samples:.4f}",
            )

    return (
        total_loss / total_samples,
        total_dice / total_samples,
        total_iou / total_samples,
    )


def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 20,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
):
    best_val_loss = float("inf")
    best_model_state = None

    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    with tqdm(range(1, num_epochs + 1), desc="Epochs", dynamic_ncols=True) as epoch_bar:
        for epoch in epoch_bar:
            train_loss, train_dice, train_iou = _train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                num_epochs=num_epochs,
            )
            val_loss, val_dice, val_iou = _validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                num_epochs=num_epochs,
            )

            if scheduler is not None:
                try:
                    scheduler.step(val_dice)
                except TypeError:
                    scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_dice"].append(train_dice)
            history["train_iou"].append(train_iou)
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            history["val_iou"].append(val_iou)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())

            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_dice=f"{train_dice:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_dice=f"{val_dice:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                best="*" if is_best else "",
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history
