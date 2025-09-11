import json
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from evaluate import extract_keypoints_from_heatmaps, visualize_predictions

# --------------------- 轻量训练（用于消融） ---------------------
def _train_simple(model, train_loader, val_loader, epochs=5, device=None):
    """
    仅用于消融的轻量训练循环（不保存 .pth）。
    Heatmap: 目标是热力图 -> MSE；Regression: 目标是坐标 -> MSE。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")

    for _ in range(epochs):
        # train
        model.train()
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(imgs)
            if out.shape[-2:] != targets.shape[-2:]:
                out = nn.functional.interpolate(out, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(out, targets)
            loss.backward()
            opt.step()

        # val（与训练同一目标）
        model.eval()
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                out = model(imgs)
                if out.shape[-2:] != targets.shape[-2:]:
                    out = nn.functional.interpolate(out, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(out, targets)
                loss_sum += loss.item() * imgs.size(0)
                n += imgs.size(0)
        val_loss = loss_sum / max(n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


# --------------------- 模型变体：去掉 skip ---------------------
class HeatmapNetNoSkip(HeatmapNet):
    """
    去掉 skip 的变体：使用一套不依赖 concat 的解码器
    编码器仍复用父类：f1:32@64x64, f2:64@32x32, f3:128@16x16, f4:256@8x8
    解码：256@8x8 -> 128@16x16 -> 64@32x32 -> 32@64x64 -> K@64x64
    """
    def __init__(self, num_keypoints=5):
        super().__init__(num_keypoints)
        # 重新定义一套“无 skip”的解码器（注意通道数不再翻倍）
        self.deconv4_ns = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv3_ns = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv2_ns = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final_ns = nn.Conv2d(32, self.num_keypoints, kernel_size=1)

    def forward(self, x):
        # 复用父类编码器
        f1, f2, f3, f4 = self.encoder(x)   # 我们只用 f4，不再 concat f3/f2
        x = self.deconv4_ns(f4)            # 256 -> 128, 8->16
        x = self.deconv3_ns(x)             # 128 -> 64, 16->32
        x = self.deconv2_ns(x)             # 64  -> 32, 32->64
        heatmaps = self.final_ns(x)        # 32  -> K,  64->64
        return heatmaps

# --------------------- 消融实验 ---------------------
def ablation_study(dataset_hint, epochs=5, batch_size=32):
    """
    消融项目：
      1) 热力图分辨率: 32 / 64 / 128
      2) 高斯 σ: 1.0 / 2.0 / 3.0 / 4.0
      3) 是否使用 skip 连接: with / without
    仅对 HeatmapNet 进行轻量训练与验证（记录 val MSE），保存到 results/ablation_results.json
    并输出三张图：
      - results/ablation_heatmap_resolution.png
      - results/ablation_sigma.png
      - results/ablation_skip.png
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(__file__).resolve().parent / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 从 dataset_hint 推断数据根目录
    HERE = os.path.dirname(os.path.abspath(__file__))          # .../problems/problem2
    PROBLEMS_ROOT = os.path.dirname(HERE)                      # .../problems
    DATA_ROOT = os.path.join(PROBLEMS_ROOT, "datasets", "keypoints")

    # 数据与标注
    train_img = os.path.join(DATA_ROOT, "train")
    val_img   = os.path.join(DATA_ROOT, "val")
    train_ann = os.path.join(DATA_ROOT, "train_annotations.json")
    val_ann   = os.path.join(DATA_ROOT, "val_annotations.json")

    results = {"heatmap_resolution": {}, "sigma": {}, "skip_connections": {}}

    # 统一 DataLoader 构造器
    def make_loaders(hm_size, sigma):
        train_ds = KeypointDataset(str(train_img), str(train_ann),
                                   output_type="heatmap", heatmap_size=hm_size, sigma=sigma)
        val_ds   = KeypointDataset(str(val_img),   str(val_ann),
                                   output_type="heatmap", heatmap_size=hm_size, sigma=sigma)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
        val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))
        return train_ld, val_ld

    # 1) heatmap resolution
    for hm_size in [32, 64, 128]:
        train_ld, val_ld = make_loaders(hm_size, 2.0)
        model = HeatmapNet(num_keypoints=5)
        model = _train_simple(model, train_ld, val_ld, epochs=epochs, device=device)

        # 记录最优 val MSE（重跑一次 val）
        criterion, loss_sum, n = nn.MSELoss(), 0.0, 0
        model.eval()
        with torch.no_grad():
            for imgs, targets in val_ld:
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs)
                if out.shape[-2:] != targets.shape[-2:]:
                    out = nn.functional.interpolate(out, size=targets.shape[-2:], mode="bilinear", align_corners=False)

                loss_sum += criterion(out, targets).item() * imgs.size(0)
                n += imgs.size(0)
        results["heatmap_resolution"][str(hm_size)] = loss_sum / max(n, 1)

    # 2) sigma
    for s in [1.0, 2.0, 3.0, 4.0]:
        train_ld, val_ld = make_loaders(64, s)
        model = HeatmapNet(num_keypoints=5)
        model = _train_simple(model, train_ld, val_ld, epochs=epochs, device=device)

        criterion, loss_sum, n = nn.MSELoss(), 0.0, 0
        model.eval()
        with torch.no_grad():
            for imgs, targets in val_ld:
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs)

                if out.shape[-2:] != targets.shape[-2:]:
                    out = nn.functional.interpolate(out, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                loss_sum += criterion(out, targets).item() * imgs.size(0)
                n += imgs.size(0)
        results["sigma"][str(s)] = loss_sum / max(n, 1)

    # 3) skip connections
    train_ld, val_ld = make_loaders(64, 2.0)
    model_with = HeatmapNet(num_keypoints=5)
    model_with  = _train_simple(model_with,  train_ld, val_ld, epochs=epochs, device=device)
    model_without = HeatmapNetNoSkip(num_keypoints=5)
    model_without = _train_simple(model_without, train_ld, val_ld, epochs=epochs, device=device)

    # 评估 val MSE
    def eval_mse(m, loader):
        crit, s, n = nn.MSELoss(), 0.0, 0
        m.eval()
        with torch.no_grad():
            for imgs, targets in loader:
                imgs, targets = imgs.to(device), targets.to(device)
                out = m(imgs)
                if out.shape[-2:] != targets.shape[-2:]:
                    out = nn.functional.interpolate(out, size=targets.shape[-2:], mode="bilinear",)
                s += crit(out, targets).item() * imgs.size(0)
                n += imgs.size(0)
        return s / max(n, 1)

    results["skip_connections"]["with"] = eval_mse(model_with, val_ld)
    results["skip_connections"]["without"] = eval_mse(model_without, val_ld)

    # 保存 JSON
    with open(save_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Ablation] results saved to {save_dir / 'ablation_results.json'}")

    # 简单可视化（折线/柱状）
    import matplotlib.pyplot as plt

    # (a) resolution
    xs = sorted(results["heatmap_resolution"].keys(), key=lambda z: int(z))
    ys = [results["heatmap_resolution"][k] for k in xs]
    plt.figure(); plt.plot([int(k) for k in xs], ys, marker="o")
    plt.xlabel("Heatmap size"); plt.ylabel("Val MSE (lower better)")
    plt.grid(True, ls="--", alpha=0.4); plt.tight_layout()
    plt.savefig(save_dir / "ablation_heatmap_resolution.png", dpi=180); plt.close()

    # (b) sigma
    xs = sorted(results["sigma"].keys(), key=lambda z: float(z))
    ys = [results["sigma"][k] for k in xs]
    plt.figure(); plt.plot([float(k) for k in xs], ys, marker="o")
    plt.xlabel("Gaussian sigma"); plt.ylabel("Val MSE (lower better)")
    plt.grid(True, ls="--", alpha=0.4); plt.tight_layout()
    plt.savefig(save_dir / "ablation_sigma.png", dpi=180); plt.close()

    # (c) skip
    plt.figure()
    plt.bar(["with", "without"],
            [results["skip_connections"]["with"], results["skip_connections"]["without"]])
    plt.ylabel("Val MSE (lower better)")
    plt.tight_layout(); plt.savefig(save_dir / "ablation_skip.png", dpi=180); plt.close()

    print(f"[Ablation] figures saved to {save_dir}")
    return results


# --------------------- 失败案例分析（用已训练权重） ---------------------
@torch.no_grad()
def analyze_failure_cases(models, test_loader, save_dir="results/failures", threshold=0.05):
    """
    识别并可视化失败样本：
      A: Heatmap 成功 / Regression 失败
      B: Regression 成功 / Heatmap 失败
      C: 两者都失败
    成败标准：各样本 K 点归一化误差（bbox 对角线）平均 <= threshold 视为成功。
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hm = models["heatmap"].eval().to(device)
    rg = models["regression"].eval().to(device)

    cat_A, cat_B, cat_C = [], [], []
    max_vis_per_cat = 12

    for idx, (imgs, targets) in enumerate(test_loader):
        imgs = imgs.to(device)                 # [B,1,128,128]
        B, K2 = targets.shape
        K = K2 // 2
        gts = targets.view(B, K, 2).clone()
        gts[..., 0] *= 128.0                  # 映射到像素坐标
        gts[..., 1] *= 128.0

        # Heatmap -> coords (128 坐标)
        hm_out = hm(imgs)
        coords_hm = extract_keypoints_from_heatmaps(hm_out)
        Hm, Wm = hm_out.shape[-2:]
        coords_hm[..., 0] *= (128.0 / Wm)
        coords_hm[..., 1] *= (128.0 / Hm)

        # Regression -> coords (128 坐标)
        rg_out = rg(imgs).view(B, K, 2)
        coords_rg = rg_out.clone()
        coords_rg[..., 0] *= 128.0
        coords_rg[..., 1] *= 128.0

        # 归一化误差（bbox 对角线）
        d_hm = torch.linalg.norm(coords_hm.cpu() - gts.cpu(), dim=-1)  # [B,K]
        d_rg = torch.linalg.norm(coords_rg.cpu() - gts.cpu(), dim=-1)
        xmin, _ = gts[..., 0].min(dim=1); xmax, _ = gts[..., 0].max(dim=1)
        ymin, _ = gts[..., 1].min(dim=1); ymax, _ = gts[..., 1].max(dim=1)
        diag = torch.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2).clamp(min=1.0)  # [B]
        frac_hm = (d_hm / diag.unsqueeze(1)).mean(dim=1)
        frac_rg = (d_rg / diag.unsqueeze(1)).mean(dim=1)

        succ_hm = frac_hm <= threshold
        succ_rg = frac_rg <= threshold

        imgs_cpu = imgs.cpu()
        for i in range(B):
            bucket = None
            if succ_hm[i] and not succ_rg[i]:
                bucket = ("A", cat_A)
            elif succ_rg[i] and not succ_hm[i]:
                bucket = ("B", cat_B)
            elif (not succ_rg[i]) and (not succ_hm[i]):
                bucket = ("C", cat_C)

            if bucket is not None and len(bucket[1]) < max_vis_per_cat:
                visualize_predictions(
                    image=imgs_cpu[i],
                    pred_keypoints=coords_hm[i].cpu().numpy(),  # 也可以改成回归预测
                    gt_keypoints=gts[i].cpu().numpy(),
                    save_path=str(Path(save_dir) / f"{bucket[0]}_{idx:04d}_{i}.png"),
                )
                bucket[1].append((idx, i))

        if len(cat_A) >= max_vis_per_cat and len(cat_B) >= max_vis_per_cat and len(cat_C) >= max_vis_per_cat:
            break

    summary = {
        "threshold": threshold,
        "heatmap_success_regression_fail": cat_A,
        "regression_success_heatmap_fail": cat_B,
        "both_fail": cat_C,
    }
    with open(Path(save_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Failures] saved images & summary to {save_dir}")
    return summary


# --------------------- 入口：只跑消融 / 失败分析 ---------------------
if __name__ == "__main__":
    RUN_ABLATION  = True   # 开 / 关 消融
    RUN_FAILURES  = True   # 开 / 关 失败案例分析

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径（datasets 在 problem2 的上一级）
    HERE = os.path.dirname(os.path.abspath(__file__))          # .../problems/problem2
    PROBLEMS_ROOT = os.path.dirname(HERE)                      # .../problems
    DATA_ROOT = os.path.join(PROBLEMS_ROOT, "datasets", "keypoints")


    train_img = os.path.join(DATA_ROOT, "train")
    val_img   = os.path.join(DATA_ROOT, "val")
    train_ann = os.path.join(DATA_ROOT, "train_annotations.json")
    val_ann   = os.path.join(DATA_ROOT, "val_annotations.json")

    if RUN_ABLATION:
        # 用一个 heatmap dataset 作为路径提示
        ds_hint = KeypointDataset(str(train_img), str(train_ann),
                                  output_type="heatmap", heatmap_size=64, sigma=2.0)
        ablation_study(ds_hint, epochs=5, batch_size=32)

    if RUN_FAILURES:
        # 用回归标签作为 GT 坐标（[0,1]）
        val_reg = KeypointDataset(str(val_img), str(val_ann), output_type="regression")
        val_loader = DataLoader(val_reg, batch_size=32, shuffle=False,
                                num_workers=2, pin_memory=(device.type == "cuda"))

        # 加载已训练好的权重（由 train.py 产出）
        hm = HeatmapNet(num_keypoints=5)
        rg = RegressionNet(num_keypoints=5)
        hm.load_state_dict(torch.load(os.path.join(HERE, "results", "heatmap_model.pth"), map_location=device))
        rg.load_state_dict(torch.load(os.path.join(HERE, "results", "regression_model.pth"), map_location=device))
        analyze_failure_cases({"heatmap": hm, "regression": rg}, val_loader)
