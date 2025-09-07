import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from loss import DetectionLoss
from utils import generate_anchors


def collate_fn(batch):
    """Detection-style collate: returns lists instead of stacked targets."""
    images, targets = zip(*batch)  # tuples of length B
    # Stack images to [B,3,H,W]; keep targets as list of dicts
    images = torch.stack(images, dim=0).contiguous()
    return images, list(targets)


@torch.no_grad()
def validate(model, dataloader, criterion, device, anchors_device):
    """Validate the model."""
    model.eval()
    meter = {"loss_obj": 0.0, "loss_cls": 0.0, "loss_loc": 0.0, "loss_total": 0.0}
    num_batches = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)

        preds = model(images)  # list of 3 tensors
        loss_dict = criterion(preds, targets, anchors_device)

        for k in meter:
            meter[k] += float(loss_dict[k].item())
        num_batches += 1

    if num_batches == 0:
        return {k: 0.0 for k in meter}

    return {k: v / num_batches for k, v in meter.items()}


def train_epoch(model, dataloader, criterion, optimizer, device, anchors_device):
    """Train for one epoch."""
    model.train()
    meter = {"loss_obj": 0.0, "loss_cls": 0.0, "loss_loc": 0.0, "loss_total": 0.0}
    num_batches = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)

        preds = model(images)
        loss_dict = criterion(preds, targets, anchors_device)

        optimizer.zero_grad(set_to_none=True)
        loss_dict["loss_total"].backward()
        optimizer.step()

        for k in meter:
            meter[k] += float(loss_dict[k].item())
        num_batches += 1

    if num_batches == 0:
        return {k: 0.0 for k in meter}

    return {k: v / num_batches for k, v in meter.items()}


def main():
    # ---------------- Configuration ----------------
    batch_size = 16
    learning_rate = 0.001

    num_epochs = 50


    num_workers = 4
    pin_memory = True
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        # else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    # Data paths (modify to your dataset layout)
    ROOT = Path(__file__).resolve().parents[1]          # problems/
    det_root = ROOT / "datasets" / "detection"

    train_image_dir = str(det_root / "train")
    val_image_dir   = str(det_root / "val")
    train_annotation = str(det_root / "train_annotations.json")
    val_annotation   = str(det_root / "val_annotations.json")

    # Results paths
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = results_dir / "best_model.pth"
    log_path = results_dir / "training_log.json"

    # Anchor configuration (per requirement)
    image_size = 224
    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [
        [16, 24, 32],    # Scale 1 (56x56)
        [48, 64, 96],    # Scale 2 (28x28)
        [96, 128, 192],  # Scale 3 (14x14)
    ]

    # ---------------- Data ----------------
    train_set = ShapeDetectionDataset(train_image_dir, train_annotation, transform=None)
    val_set = ShapeDetectionDataset(val_image_dir, val_annotation, transform=None)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---------------- Model / Loss / Opt ----------------
    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    criterion = DetectionLoss(num_classes=3)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Pre-generate anchors once (they don't change with batch)
    anchors_list = generate_anchors(feature_map_sizes, anchor_scales, image_size=image_size)
    anchors_device = [a.to(device) for a in anchors_list]

    # ---------------- Training Loop ----------------
    best_val = float("inf")
    log = []

    for epoch in range(1, num_epochs + 1):
        train_meter = train_epoch(model, train_loader, criterion, optimizer, device, anchors_device)
        val_meter = validate(model, val_loader, criterion, device, anchors_device)

        # Logging
        epoch_log = {
            "epoch": epoch,
            "train": train_meter,
            "val": val_meter,
        }
        log.append(epoch_log)

        # Save log file (update each epoch)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        # Best checkpoint by validation total loss
        val_total = val_meter["loss_total"]
        if val_total < best_val:
            best_val = val_total
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss_total": best_val,
                },
                best_ckpt_path,
            )

        print(
            f"[Epoch {epoch:03d}/{num_epochs}] "
            f"Train total: {train_meter['loss_total']:.4f} | "
            f"Val total: {val_total:.4f} "
            f"(obj {val_meter['loss_obj']:.4f} | cls {val_meter['loss_cls']:.4f} | loc {val_meter['loss_loc']:.4f})"
        )

    print(f"Training complete. Best val loss: {best_val:.6f}. Saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()