import copy
from typing import Callable
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

type MetricFunction = Callable[[torch.Tensor, torch.Tensor], dict[str, float]]


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int | None = None,
    num_epochs: int | None = None,
    metrics_function: MetricFunction | None = None,
) -> tuple[float, dict[str, float]]:
    model.train()

    total_loss = 0.0
    metrics = defaultdict(lambda: 0.0)
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

            if metrics_function is not None:
                batch_metrics = metrics_function(logits, masks)
                for key, value in batch_metrics.items():
                    metrics[key] += value * bs

            total_loss += loss.item() * bs
            total_samples += bs

            batch_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                **{
                    key: f"{value / total_samples:.4f}"
                    for key, value in metrics.items()
                },
            )

    return (
        total_loss / total_samples,
        {key: value / total_samples for key, value in metrics.items()},
    )


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int | None = None,
    num_epochs: int | None = None,
    metrics_function: MetricFunction | None = None,
) -> tuple[float, dict[str, float]]:
    model.eval()

    total_loss = 0.0
    metrics = defaultdict(lambda: 0.0)
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

            if metrics_function is not None:
                batch_metrics = metrics_function(logits, masks)
                for key, value in batch_metrics.items():
                    metrics[key] += value * bs

            total_loss += loss.item() * bs
            total_samples += bs

            batch_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                **{
                    key: f"{value / total_samples:.4f}"
                    for key, value in metrics.items()
                },
            )

    return (
        total_loss / total_samples,
        {key: value / total_samples for key, value in metrics.items()},
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
    metrics_function: MetricFunction | None = None,
    early_stopping_patience: int | None = None,
):
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    with tqdm(range(1, num_epochs + 1), desc="Epochs", dynamic_ncols=True) as epoch_bar:
        for epoch in epoch_bar:
            train_loss, train_metrics = _train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                num_epochs=num_epochs,
                metrics_function=metrics_function,
            )
            val_loss, val_metrics = _validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                num_epochs=num_epochs,
                metrics_function=metrics_function,
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            for key, value in train_metrics.items():
                history.setdefault(f"train_{key}", []).append(value)

            for key, value in val_metrics.items():
                history.setdefault(f"val_{key}", []).append(value)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                best="*" if is_best else "",
            )

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch} due to no improvement for {early_stopping_patience} epochs."
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history
