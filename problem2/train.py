import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def _eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


def train_heatmap_model(model, train_loader, val_loader, num_epochs=30, device=None, save_path="results/heatmap_model.pth"):
    """
    Train the heatmap-based model with MSE loss on heatmaps.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _eval_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"[Heatmap] Epoch {epoch:02d}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

    # save best
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)
    return history


def train_regression_model(model, train_loader, val_loader, num_epochs=30, device=None, save_path="results/regression_model.pth"):
    """
    Train the direct regression model with MSE loss on normalized coordinates.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _eval_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"[Regress] Epoch {epoch:02d}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

    # save best
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)
    return history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = 32
    epochs = 30

    # ---- Datasets & Loaders ----
    HERE = os.path.dirname(os.path.abspath(__file__))          # .../problems/problem2
    PROBLEMS_ROOT = os.path.dirname(HERE)                      # .../problems
    DATA_ROOT = os.path.join(PROBLEMS_ROOT, "datasets", "keypoints")

    # 数据与标注
    train_img_dir = os.path.join(DATA_ROOT, "train")
    val_img_dir   = os.path.join(DATA_ROOT, "val")
    train_ann = os.path.join(DATA_ROOT, "train_annotations.json")
    val_ann   = os.path.join(DATA_ROOT, "val_annotations.json")

    # Heatmap loaders
    train_ds_hm = KeypointDataset(train_img_dir, train_ann, output_type="heatmap", heatmap_size=64, sigma=2.0)
    val_ds_hm   = KeypointDataset(val_img_dir,   val_ann,   output_type="heatmap", heatmap_size=64, sigma=2.0)

    train_loader_hm = DataLoader(train_ds_hm, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=(device.type=="cuda"))
    val_loader_hm   = DataLoader(val_ds_hm,   batch_size=bs, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))

    # Regression loaders
    train_ds_rg = KeypointDataset(train_img_dir, train_ann, output_type="regression")
    val_ds_rg   = KeypointDataset(val_img_dir,   val_ann,   output_type="regression")

    train_loader_rg = DataLoader(train_ds_rg, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=(device.type=="cuda"))
    val_loader_rg   = DataLoader(val_ds_rg,   batch_size=bs, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))

    # ---- Models ----
    hm_model = HeatmapNet(num_keypoints=5)
    rg_model = RegressionNet(num_keypoints=5)

    # ---- Train ----
    hist_hm = train_heatmap_model(
        hm_model, train_loader_hm, val_loader_hm,
        num_epochs=epochs, device=device, save_path="results/heatmap_model.pth"
    )
    hist_rg = train_regression_model(
        rg_model, train_loader_rg, val_loader_rg,
        num_epochs=epochs, device=device, save_path="results/regression_model.pth"
    )

    # ---- Log ----
    logs = {
        "heatmap":   hist_hm,
        "regression": hist_rg,
        "epochs": epochs,
        "batch_size": bs,
        "optimizer": "Adam",
        "lr": 0.001,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/training_log.json", "w") as f:
        json.dump(logs, f, indent=2)
    print("Saved models and logs to 'results/'")


if __name__ == '__main__':
    main()
