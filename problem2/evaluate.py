import json
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from problem2.dataset import KeypointDataset
from problem2.model import HeatmapNet, RegressionNet


@torch.no_grad()
def extract_keypoints_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Extract (x, y) coordinates from heatmaps by argmax.

    Args:
        heatmaps: Tensor [B, K, H, W]

    Returns:
        coords: Tensor [B, K, 2] with (x, y) in heatmap coordinates (0..W-1, 0..H-1)
    """
    assert heatmaps.dim() == 4, "heatmaps must be [B,K,H,W]"
    B, K, H, W = heatmaps.shape
    # flatten per heatmap and argmax -> idx in [0, H*W)
    flat = heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=-1)                         # [B,K]
    y = (idx // W).to(torch.float32)                  # [B,K]
    x = (idx %  W).to(torch.float32)                  # [B,K]
    coords = torch.stack([x, y], dim=-1)              # [B,K,2]
    return coords


@torch.no_grad()
def compute_pck(
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        thresholds,
        normalize_by: str = "bbox",
) -> dict:
    """
    Compute Percentage of Correct Keypoints (PCK).

    Args:
        predictions: [N, K, 2]  (x,y) in the same coordinate system as GT
        ground_truths: [N, K, 2]
        thresholds: iterable of floats (fraction of the chosen normalization)
        normalize_by: 'bbox' (diagonal) or 'torso'
            - 'bbox': per-sample bounding box diagonal over GT keypoints
            - 'torso': mean of distances between (left_hand,right_hand) and (left_foot,right_foot)
                       若该定义不适用或退化，将回退到 'bbox'

    Returns:
        {thr: accuracy (float in [0,1])}
    """
    assert predictions.shape == ground_truths.shape and predictions.dim() == 3
    N, K, _ = predictions.shape
    preds = predictions.float()
    gts = ground_truths.float()

    # distances per keypoint
    d = torch.linalg.norm(preds - gts, dim=-1)        # [N,K]

    # normalization per sample
    if normalize_by == "torso":
        # 本作业的5点顺序: [head, left_hand, right_hand, left_foot, right_foot]
        # 定义 torso 尺寸为两对肢体（手、脚）间距的均值；若缺失/退化回退 bbox
        lh, rh, lf, rf = 1, 2, 3, 4
        torso = []
        for i in range(N):
            try:
                dh = torch.linalg.norm(gts[i, lh] - gts[i, rh])  # hands
                df = torch.linalg.norm(gts[i, lf] - gts[i, rf])  # feet
                t = (dh + df) / 2.0
                if not torch.isfinite(t) or t < 1e-6:
                    raise ValueError
            except Exception:
                # 回退 bbox
                xmin, _ = gts[i, :, 0].min(dim=0)
                xmax, _ = gts[i, :, 0].max(dim=0)
                ymin, _ = gts[i, :, 1].min(dim=0)
                ymax, _ = gts[i, :, 1].max(dim=0)
                w = (xmax - xmin).clamp(min=1.0)
                h = (ymax - ymin).clamp(min=1.0)
                t = torch.sqrt(w * w + h * h)
            torso.append(t)
        norm = torch.stack(torso, dim=0)              # [N]
    else:
        # bbox diagonal
        xmin, _ = gts[:, :, 0].min(dim=1)
        xmax, _ = gts[:, :, 0].max(dim=1)
        ymin, _ = gts[:, :, 1].min(dim=1)
        ymax, _ = gts[:, :, 1].max(dim=1)
        w = (xmax - xmin).clamp(min=1.0)
        h = (ymax - ymin).clamp(min=1.0)
        norm = torch.sqrt(w * w + h * h)              # [N]

    norm = norm.view(N, 1)                             # [N,1] broadcast to [N,K]
    frac = d / norm                                    # [N,K] normalized distance

    pck_values = {}
    thresholds = list(thresholds)
    for thr in thresholds:
        correct = (frac <= thr).float().mean().item()  # over N*K
        pck_values[thr] = correct
    return pck_values


def plot_pck_curves(pck_heatmap: dict, pck_regression: dict, save_path: str):
    """
    Plot PCK curves comparing both methods.

    Args:
        pck_heatmap: {thr: acc}
        pck_regression: {thr: acc}
    """
    thrs_h = sorted(pck_heatmap.keys())
    thrs_r = sorted(pck_regression.keys())
    xs = sorted(set(thrs_h) | set(thrs_r))

    y_h = [pck_heatmap.get(t, np.nan) for t in xs]
    y_r = [pck_regression.get(t, np.nan) for t in xs]

    plt.figure(figsize=(6, 4.5))
    plt.plot(xs, y_h, marker="o", label="Heatmap")
    plt.plot(xs, y_r, marker="s", label="Regression")
    plt.xlabel("Threshold (fraction of normalization)")
    plt.ylabel("PCK (accuracy)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path):
    """
    Visualize predicted and ground truth keypoints on image.

    Args:
        image:  [1,128,128] tensor or HxW numpy array or PIL 可转 numpy
        pred_keypoints: [K,2]  (x,y) in image coordinates (128x128 if你用的输入大小)
        gt_keypoints:   [K,2]
    """
    # 处理图像到 numpy HxW
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        elif img.ndim == 2:
            pass
        else:
            raise ValueError("image tensor must be [1,H,W] or [H,W]")
    else:
        img = np.array(image)
        if img.ndim == 3:
            img = img[..., 0]  # 取灰度

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)

    # GT (绿色) 与 Pred (红色)
    pk = pred_keypoints.detach().cpu().numpy() if isinstance(pred_keypoints, torch.Tensor) else np.asarray(pred_keypoints)
    gk = gt_keypoints.detach().cpu().numpy() if isinstance(gt_keypoints, torch.Tensor) else np.asarray(gt_keypoints)

    plt.scatter(gk[:, 0], gk[:, 1], s=30, marker="o", facecolors="none", edgecolors="lime", linewidths=2, label="GT")
    plt.scatter(pk[:, 0], pk[:, 1], s=18, marker="x", c="red", linewidths=2, label="Pred")

    # 连线方便观察误差
    for (x1, y1), (x2, y2) in zip(gk, pk):
        plt.plot([x1, x2], [y1, y2], linestyle="--")

    plt.xlim(-0.5, img.shape[1] - 0.5)
    plt.ylim(img.shape[0] - 0.5, -0.5)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()



def main():
    # ---------- Config ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 作业要求的 PCK 阈值
    PCK_THRESHOLDS = [0.05, 0.1, 0.15, 0.2]
    BATCH_SIZE = 32
    IMG_SIZE = 128  # 你输入网络的尺寸

    # 路径（按你的工程：datasets 在 problem2 的上一级）
    ROOT = Path(__file__).resolve().parents[1]              # problems/
    kp_root = ROOT / "datasets" / "keypoints"
    val_image_dir = str(kp_root / "val")
    val_annotation = str(kp_root / "val_annotations.json")

    results_dir = Path(__file__).resolve().parent / "results"
    vis_dir = results_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 已训练好的权重（由 train.py 产出）
    heatmap_pth = results_dir / "heatmap_model.pth"
    regression_pth = results_dir / "regression_model.pth"

    # ---------- Data ----------
    # 用 regression 模式的 dataset 拿到 GT 坐标（[0,1]），评估时统一映射到 128×128
    val_set = KeypointDataset(val_image_dir, val_annotation, output_type="regression")
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    K = val_set.samples[0]["keypoints"].shape[0]  # 关键点个数（你的数据是5）

    # ---------- Models ----------
    hm = HeatmapNet(num_keypoints=K).to(device).eval()
    rg = RegressionNet(num_keypoints=K).to(device).eval()

    if heatmap_pth.exists():
        hm.load_state_dict(torch.load(heatmap_pth, map_location=device))
        print(f"Loaded heatmap model: {heatmap_pth}")
    else:
        print(f"WARNING: {heatmap_pth} not found. Using random init (for demo only).")

    if regression_pth.exists():
        rg.load_state_dict(torch.load(regression_pth, map_location=device))
        print(f"Loaded regression model: {regression_pth}")
    else:
        print(f"WARNING: {regression_pth} not found. Using random init (for demo only).")

    # ---------- Inference & PCK ----------
    preds_hm, preds_rg, gts_all = [], [], []
    sample_cache = []  # 用于后面做可视化：保存少量原图与 GT
    with torch.no_grad():
        for imgs, targets in val_loader:
            # imgs: [B,1,128,128]; targets: [B,2K] in [0,1]
            imgs = imgs.to(device)
            B = imgs.size(0)
            gts = targets.view(B, K, 2).clone()
            # 映射到 128×128 像素坐标
            gts[..., 0] *= IMG_SIZE
            gts[..., 1] *= IMG_SIZE

            # Heatmap -> coords
            hm_out = hm(imgs)  # [B,K,Hm,Wm]
            coords_hm = extract_keypoints_from_heatmaps(hm_out)  # [B,K,2] in Hm×Wm
            Hm, Wm = hm_out.shape[-2:]
            coords_hm[..., 0] *= (IMG_SIZE / Wm)
            coords_hm[..., 1] *= (IMG_SIZE / Hm)

            # Regression -> coords
            rg_out = rg(imgs).view(B, K, 2)  # in [0,1]
            coords_rg = rg_out.clone()
            coords_rg[..., 0] *= IMG_SIZE
            coords_rg[..., 1] *= IMG_SIZE

            preds_hm.append(coords_hm.cpu())
            preds_rg.append(coords_rg.cpu())
            gts_all.append(gts.cpu())

            # 缓存少量样本用于可视化
            if len(sample_cache) < 10:
                for i in range(min(B, 10 - len(sample_cache))):
                    sample_cache.append((
                        imgs[i].detach().cpu(),          # [1,128,128]
                        coords_hm[i].detach().cpu(),     # [K,2]
                        coords_rg[i].detach().cpu(),     # [K,2]
                        gts[i].detach().cpu(),           # [K,2]
                    ))

    preds_hm = torch.cat(preds_hm, dim=0)
    preds_rg = torch.cat(preds_rg, dim=0)
    gts_all = torch.cat(gts_all, dim=0)

    pck_h = compute_pck(preds_hm, gts_all, PCK_THRESHOLDS, normalize_by="bbox")
    pck_r = compute_pck(preds_rg, gts_all, PCK_THRESHOLDS, normalize_by="bbox")

    # 打印并保存指标
    print("PCK (Heatmap):   ", {float(k): float(v) for k, v in pck_h.items()})
    print("PCK (Regression):", {float(k): float(v) for k, v in pck_r.items()})
    with open(results_dir / "keypoints_eval_metrics.json", "w") as f:
        json.dump({
            "thresholds": PCK_THRESHOLDS,
            "PCK_heatmap": {str(k): float(v) for k, v in pck_h.items()},
            "PCK_regression": {str(k): float(v) for k, v in pck_r.items()},
        }, f, indent=2)

    # ---------- Plot PCK curves ----------
    plot_pck_curves(pck_h, pck_r, save_path=str(vis_dir / "pck_comparison.png"))
    print(f"Saved PCK comparison curve to: {vis_dir / 'pck_comparison.png'}")

    # ---------- Visualize predictions (前10张) ----------
    # 分别存两张：heatmap 预测 vs GT、regression 预测 vs GT
    for i, (img, pred_h, pred_r, gt) in enumerate(sample_cache):
        visualize_predictions(
            image=img,
            pred_keypoints=pred_h,
            gt_keypoints=gt,
            save_path=str(vis_dir / f"val_vis_heatmap_{i:03d}.png")
        )
        visualize_predictions(
            image=img,
            pred_keypoints=pred_r,
            gt_keypoints=gt,
            save_path=str(vis_dir / f"val_vis_regression_{i:03d}.png")
        )
    print(f"Saved visualizations to: {vis_dir}")


if __name__ == "__main__":
    main()