"""One training loop shared by every experiment (v3 recipe).

Recipe, each piece a known accuracy gain for fine-tuning on small medical sets:
    * freeze_epochs   - first epoch(s) train only head + metadata branch
    * layer-wise LR decay - early backbone layers learn slower than late ones
    * linear warmup then cosine decay of the learning rate, updated every batch
    * mixup (images and metadata mixed with the same lambda)
    * class-weighted cross-entropy with label smoothing
    * EMA - an exponential moving average of the weights; usually 1-2 points
      better than the raw weights and the one that is saved and evaluated
    * early stopping on validation macro-F1; last.pt saved every epoch (resume)
Batches are (images, metadata, targets).
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from .model import set_backbone_trainable


@dataclass
class TrainConfig:
    epochs: int = 30
    freeze_epochs: int = 1        # backbone frozen for this many epochs
    warmup_epochs: float = 2.0    # linear LR warmup after unfreezing
    lr: float = 2e-4              # peak LR of the head; backbone top layer gets lr * layer_decay
    head_lr: float = 1e-3         # LR while the backbone is frozen
    layer_decay: float = 0.75     # each earlier block: LR multiplied by this
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.4      # 0 turns mixup off
    mixup_prob: float = 0.5
    ema_decay: float = 0.999      # 0 turns EMA off
    patience: int = 6
    grad_clip: float = 1.0
    amp: bool = True


def train_config_from(settings: dict) -> TrainConfig:
    known = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in settings.items() if k in known})


# ---------------------------------------------------------------------------
# Optimizer, schedule, EMA
# ---------------------------------------------------------------------------
def param_groups(model, cfg: TrainConfig) -> list[dict]:
    """AdamW groups with an 'lr_scale' each (layer-wise decay for the backbone)."""
    groups = []
    try:
        from timm.optim import param_groups_layer_decay
        groups = param_groups_layer_decay(model.backbone, weight_decay=cfg.weight_decay,
                                          layer_decay=cfg.layer_decay)
        # timm makes one group per block: 12 for ViT-B but ~80 for ConvNeXt/EfficientNet. The decay is
        # rescaled so the earliest layer always gets layer_decay**12 of the top LR, whatever the depth
        # (without this, 40% of ConvNeXt-S would train at <1% of the head LR - found in review).
        depth = len({round(g.get("lr_scale", 1.0), 12) for g in groups}) - 1
        if depth > 12:
            decay = cfg.layer_decay ** (12 / depth)
            groups = param_groups_layer_decay(model.backbone, weight_decay=cfg.weight_decay, layer_decay=decay)
        for group in groups:                     # backbone top layer starts one step below the head
            group["lr_scale"] = group.get("lr_scale", 1.0) * cfg.layer_decay
    except Exception:                            # model without group_matcher: one backbone group
        groups = [{"params": list(model.backbone.parameters()), "lr_scale": cfg.layer_decay,
                   "weight_decay": cfg.weight_decay}]
    groups.append({"params": model.head_parameters(), "lr_scale": 1.0, "weight_decay": cfg.weight_decay})
    return [g for g in groups if g["params"]]


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """Multiplier in [0, 1]: linear rise for warmup_steps, then cosine to 0."""
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


class EMA:
    """Exponential moving average of model weights (buffers copied directly)."""

    def __init__(self, model, decay: float):
        import torch

        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self._torch = torch

    def update(self, model) -> None:
        with self._torch.no_grad():
            for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values(), strict=False):
                if ema_v.dtype.is_floating_point:
                    ema_v.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    ema_v.copy_(v)


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------
def _mixup(images, metas, targets, alpha: float, rng: np.random.Generator):
    import torch

    lam = float(rng.beta(alpha, alpha))
    order = torch.randperm(images.shape[0], device=images.device)
    return (lam * images + (1 - lam) * images[order], lam * metas + (1 - lam) * metas[order],
            targets, targets[order], lam)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device: str, cfg: TrainConfig,
                    lr_for_step: Callable[[int], float] | None, start_step: int, ema: EMA | None,
                    rng: np.random.Generator):
    import torch

    model.train()
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()                   # frozen backbone: keep BatchNorm statistics fixed
    total_loss, correct, seen, step = 0.0, 0, 0, start_step
    use_amp = cfg.amp and device == "cuda"
    for images, metas, targets in loader:
        images, metas, targets = images.to(device), metas.to(device), targets.to(device)
        if lr_for_step is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr_for_step(step) * group.get("lr_scale", 1.0)
        mixed = cfg.mixup_alpha > 0 and rng.random() < cfg.mixup_prob
        if mixed:
            images, metas, y_a, y_b, lam = _mixup(images, metas, targets, cfg.mixup_alpha, rng)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
            logits = model(images, metas)
            loss = (lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)) if mixed \
                else criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * len(targets)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += len(targets)
        step += 1
    return total_loss / max(seen, 1), correct / max(seen, 1), step


def predict_logits(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """(logits, targets) for a whole loader, no learning."""
    import torch

    model.eval()
    outputs, truths = [], []
    with torch.no_grad():
        for images, metas, targets in loader:
            outputs.append(model(images.to(device), metas.to(device)).float().cpu().numpy())
            truths.append(targets.numpy())
    return np.concatenate(outputs), np.concatenate(truths)


def evaluate_epoch(model, loader, criterion, device: str) -> dict:
    import torch

    logits, y_true = predict_logits(model, loader, device)
    loss = criterion(torch.from_numpy(logits).to(device), torch.from_numpy(y_true).to(device)).item()
    y_pred = logits.argmax(1)
    return {"loss": loss, "accuracy": float((y_pred == y_true).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer, epoch: int, best_score: float, history: list,
                    meta: dict, ema_model=None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": (ema_model or model).state_dict(),
                "raw_model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "best_score": best_score, "history": history, **meta}, path)


def fit(model, train_loader, val_loader, class_weights: list[float], cfg: TrainConfig, device: str,
        checkpoint_dir: Path, meta: dict, log_fn: Callable[[dict], None] | None = None,
        resume: bool = True, seed: int = 0) -> pd.DataFrame:
    """Train with early stopping; best.pt holds the best (EMA) weights. Returns per-epoch metrics."""
    import torch

    checkpoint_dir = Path(checkpoint_dir)
    model.to(device)
    rng = np.random.default_rng(seed)
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
    meta = {**meta, "train_config": asdict(cfg)}

    history, best_score, start_epoch, without_gain = [], -1.0, 0, 0
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    saved = None
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["raw_model_state"])
        history, best_score, start_epoch = saved["history"], saved["best_score"], saved["epoch"] + 1
        best_epoch = max(r["epoch"] for r in history if r["val_macro_f1"] >= best_score)
        without_gain = saved["epoch"] - best_epoch
        print(f"Resuming from epoch {start_epoch} (best macro-F1 so far {best_score:.3f})")

    steps_per_epoch = len(train_loader)
    tune_steps = max(cfg.epochs - cfg.freeze_epochs, 1) * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    if ema is not None and saved is not None:
        ema.module.load_state_dict(saved["model_state"])
    optimizer, current_stage = None, None

    for epoch in range(start_epoch, cfg.epochs):
        stage = "frozen" if epoch < cfg.freeze_epochs else "finetune"
        if stage != current_stage:
            set_backbone_trainable(model, trainable=(stage == "finetune"))
            if stage == "frozen":
                optimizer = torch.optim.AdamW(model.head_parameters(), lr=cfg.head_lr,
                                              weight_decay=cfg.weight_decay)
            else:
                optimizer = torch.optim.AdamW(param_groups(model, cfg), lr=cfg.lr)
            if saved is not None and epoch == start_epoch and saved.get("history") \
                    and saved["history"][-1]["stage"] == stage:
                try:                             # resume: keep AdamW's moment estimates
                    optimizer.load_state_dict(saved["optimizer_state"])
                except ValueError:
                    pass
            current_stage = stage
        if stage == "finetune":
            first_step = (epoch - cfg.freeze_epochs) * steps_per_epoch
            schedule = lambda s: cfg.lr * warmup_cosine(s, warmup_steps, tune_steps)  # noqa: E731
        else:
            first_step, schedule = 0, None

        started = time.time()
        train_loss, train_acc, _ = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                                                   cfg, schedule, first_step,
                                                   ema if stage == "finetune" else None, rng)
        if ema is not None and stage == "frozen":
            ema.module.load_state_dict(model.state_dict())   # EMA starts from the warmed-up head
        scored = ema.module if ema is not None else model
        val = evaluate_epoch(scored, val_loader, criterion, device)
        row = {"epoch": epoch, "stage": stage, "lr": max(g["lr"] for g in optimizer.param_groups),
               "train_loss": train_loss, "train_accuracy": train_acc,
               **{f"val_{k}": v for k, v in val.items()}, "seconds": round(time.time() - started, 1)}
        history.append(row)

        improved = val["macro_f1"] > best_score
        if improved:
            best_score, without_gain = val["macro_f1"], 0
            save_checkpoint(best_path, model, optimizer, epoch, best_score, history, meta,
                            ema.module if ema else None)
        else:
            without_gain += 1
        save_checkpoint(last_path, model, optimizer, epoch, best_score, history, meta,
                        ema.module if ema else None)
        print(f"epoch {epoch:>2} [{stage:<8}] train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val acc {val['accuracy']:.3f} macro-F1 {val['macro_f1']:.3f} "
              f"{'*best*' if improved else ''} ({row['seconds']}s)")
        if log_fn is not None:
            log_fn(row)
        if stage == "finetune" and without_gain >= cfg.patience:
            print(f"Early stop: no macro-F1 gain for {cfg.patience} epochs. Best = {best_score:.3f}")
            break
    return pd.DataFrame(history)


def load_checkpoint(path: Path, model, device: str = "cpu") -> dict:
    """Load best (EMA) weights into `model`; return the checkpoint's saved facts."""
    import torch

    saved = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    return {k: v for k, v in saved.items() if k not in ("model_state", "raw_model_state", "optimizer_state")}


def model_from_checkpoint(path: Path, device: str = "cpu", pretrained: bool = False):
    """Rebuild a DermaNet from the facts saved in its checkpoint and load the best weights.

    Returns (model, facts). Needs 'model_name', 'classes', 'use_metadata', 'image_size' in the checkpoint.
    """
    import torch

    from .model import build_model

    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(saved["model_name"], len(saved["classes"]), pretrained=pretrained,
                        use_metadata=saved.get("use_metadata", True), image_size=saved.get("image_size"))
    model.load_state_dict(saved["model_state"])
    facts = {k: v for k, v in saved.items() if k not in ("model_state", "raw_model_state", "optimizer_state")}
    return model.to(device).eval(), facts
