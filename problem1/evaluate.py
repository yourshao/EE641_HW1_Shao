import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from utils import generate_anchors, compute_iou


# ------------------------ Core metrics ------------------------

def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute Average Precision for a single class.

    Args:
        predictions: List[Dict] with elements:
            {
                "image_id": int,
                "box": Tensor[4] or list[4] in [x1,y1,x2,y2],
                "score": float
            }
        ground_truths: Dict[int, List[Tensor[4]]] mapping image_id -> list of GT boxes (this class only)
        iou_threshold: float, IoU threshold to count as True Positive

    Returns:
        ap: float (VOC-style interpolated AP)
    """
    if len(predictions) == 0:
        # no predictions; AP is 0 unless also no GT (then define 0)
        total_gt = sum(len(v) for v in ground_truths.values())
        return 0.0 if total_gt > 0 else 0.0

    # Sort predictions by descending score
    predictions = sorted(predictions, key=lambda d: d["score"], reverse=True)

    # For each image, track which GT boxes are already matched
    gt_matched = {img_id: torch.zeros(len(gt_boxes), dtype=torch.bool) for img_id, gt_boxes in ground_truths.items()}
    tp = torch.zeros(len(predictions))
    fp = torch.zeros(len(predictions))

    # Count total GT for this class
    total_gt = sum(len(v) for v in ground_truths.values())

    for i, pred in enumerate(predictions):
        img_id = pred["image_id"]
        box_p = torch.as_tensor(pred["box"], dtype=torch.float32).unsqueeze(0)  # [1,4]
        gts = ground_truths.get(img_id, [])

        if len(gts) == 0:
            fp[i] = 1
            continue

        boxes_g = torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in gts], dim=0)  # [G,4]
        ious = compute_iou(box_p, boxes_g).squeeze(0)  # [G]
        best_iou, best_idx = ious.max(dim=0)

        if best_iou.item() >= iou_threshold and not gt_matched[img_id][best_idx]:
            tp[i] = 1
            gt_matched[img_id][best_idx] = True
        else:
            fp[i] = 1

    # Cumulate
    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    denom = (tp_cum + fp_cum).clamp(min=1e-6)
    precision = tp_cum / denom
    recall = tp_cum / max(total_gt, 1)

    # 11-point or continuous AP; here: standard continuous with precision envelope
    # Make precision monotonically decreasing
    precision_envelope = precision.clone()
    for i in range(len(precision_envelope) - 2, -1, -1):
        precision_envelope[i] = max(precision_envelope[i], precision_envelope[i + 1])

    # Integrate w.r.t. recall using trapezoid over unique recall points
    recall_np = recall.detach().cpu().numpy()
    prec_np = precision_envelope.detach().cpu().numpy()
    # Add (0,1) and (1,0) endpoints implicitly if needed
    recall_np = np.concatenate(([0.0], recall_np, [1.0]))
    prec_np = np.concatenate(([prec_np[0] if len(prec_np) > 0 else 0.0], prec_np, [0.0]))
    # Integrate
    ap = 0.0
    for i in range(1, len(recall_np)):
        ap += (recall_np[i] - recall_np[i - 1]) * prec_np[i]
    return float(ap)


# ------------------------ Visualization ------------------------

def visualize_detections(image, predictions, ground_truths, save_path):

    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.copy()
    else:
        raise ValueError("image must be path or PIL.Image")

    draw = ImageDraw.Draw(img, "RGBA")

    # Colors
    # GT: green, Predictions: red (with alpha)
    def draw_box(b, color, width=2, text=None, fill=None):
        x1, y1, x2, y2 = [float(v) for v in b]
        for k in range(width):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)
        if fill is not None:
            draw.rectangle([x1, y1, x2, y2], outline=color, fill=fill)
        if text:
            draw.text((x1 + 2, y1 + 2), text, fill=color)

    # Class names (consistent with your classes)
    class_names = ["circle", "square", "triangle"]

    # Draw GT
    if ground_truths and len(ground_truths.get("boxes", [])) > 0:
        for b, lb in zip(ground_truths["boxes"], ground_truths["labels"]):
            label = int(lb)
            txt = f"GT:{class_names[label]}"
            draw_box(b, color=(0, 200, 0, 255), width=2, text=txt)

    # Draw predictions
    if predictions and len(predictions.get("boxes", [])) > 0:
        for b, sc, lb in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
            label = int(lb)
            score = float(sc)
            txt = f"{class_names[label]} {score:.2f}"
            draw_box(b, color=(255, 50, 50, 255), width=2, text=txt, fill=(255, 0, 0, 30))

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


# ------------------------ Decoding & NMS ------------------------

def _to_cxcywh(boxes: torch.Tensor):
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h


def decode_predictions(predictions: List[torch.Tensor],
                       anchors: List[torch.Tensor],
                       num_classes=3,
                       conf_thresh=0.3):
    """
    Decode raw model outputs into per-image detections BEFORE NMS.
    Returns a list (len=B) of dicts:
      {
        "boxes": Tensor[K,4],
        "scores": Tensor[K],
        "labels": Tensor[K],
        "scale_ids": Tensor[K],  # 0/1/2 indicating which scale head produced the detection
      }
    """
    device = predictions[0].device
    B = predictions[0].shape[0]
    out_per_image = [{"boxes": [], "scores": [], "labels": [], "scale_ids": []} for _ in range(B)]

    for s_id, (pred_s, anch_s) in enumerate(zip(predictions, anchors)):
        B, ch, H, W = pred_s.shape
        A_per = ch // (5 + num_classes)

        # pred_s = pred_s.view(B, A_per, (5 + num_classes), H, W)
        # pred_s = pred_s.permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,A,5+C]
        # pred_s = pred_s.view(B, -1, (5 + num_classes))  # [B, N, 5+C]; N=H*W*A


        pred_s = (
            pred_s
            .reshape(B, A_per, 5 + num_classes, H, W)
            .permute(0, 3, 4, 1, 2)
            .reshape(B, -1, 5 + num_classes)
        )


        # Split
        loc = pred_s[..., 0:4].contiguous()  # [B,N,4] (tx,ty,tw,th)
        obj = pred_s[..., 4].contiguous()  # [B,N]
        cls = pred_s[..., 5:5 + num_classes].contiguous()  # [B,N,C]

        # Anchors
        anc = anch_s.to(device)  # [N,4]
        a_cx, a_cy, a_w, a_h = _to_cxcywh(anc)

        # Decode
        tx, ty, tw, th = loc.unbind(-1)
        p_cx = a_cx.unsqueeze(0) + tx * a_w.unsqueeze(0)
        p_cy = a_cy.unsqueeze(0) + ty * a_h.unsqueeze(0)
        p_w = a_w.unsqueeze(0) * torch.exp(tw)
        p_h = a_h.unsqueeze(0) * torch.exp(th)
        x1 = p_cx - 0.5 * p_w
        y1 = p_cy - 0.5 * p_h
        x2 = p_cx + 0.5 * p_w
        y2 = p_cy + 0.5 * p_h
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B,N,4]

        # Scores: obj * softmax(class)
        obj_sig = torch.sigmoid(obj)  # [B,N]
        cls_prob = F.softmax(cls, dim=-1)  # [B,N,C]
        # For each class, compute combined score and threshold
        comb = obj_sig.unsqueeze(-1) * cls_prob      # [B,N,C]
        best_scores, best_cls = comb.max(dim=-1)     # [B,N], [B,N]

        for b in range(B):
            keep = best_scores[b] >= conf_thresh
            if keep.any():
                out_per_image[b]["boxes"].append(boxes[b][keep])
                out_per_image[b]["scores"].append(best_scores[b][keep].float())
                out_per_image[b]["labels"].append(best_cls[b][keep].to(torch.long))
                out_per_image[b]["scale_ids"].append(
                    torch.full((int(keep.sum()),), s_id, dtype=torch.long, device=device)
                )
    # Concatenate lists into tensors
    for b in range(B):
        if len(out_per_image[b]["boxes"]) == 0:
            # empty tensors
            out_per_image[b] = {
                "boxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
                "scale_ids": torch.zeros((0,), dtype=torch.long, device=device),
            }
        else:
            out_per_image[b] = {
                "boxes": torch.cat(out_per_image[b]["boxes"], dim=0),
                "scores": torch.cat(out_per_image[b]["scores"], dim=0),
                "labels": torch.cat(out_per_image[b]["labels"], dim=0),
                "scale_ids": torch.cat(out_per_image[b]["scale_ids"], dim=0),
            }
    return out_per_image


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float):
    """
    Simple PyTorch NMS (class-agnostic). Returns indices to keep.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        remain = (iou <= iou_thresh).nonzero(as_tuple=False).squeeze(1)
        order = order[1:][remain]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


def apply_nms_per_class(det: Dict[str, torch.Tensor], iou_thresh=0.5, topk_per_class=200):
    """
    Run per-class NMS and optionally cap top-K per class. Returns filtered dict with same keys plus scale_ids.
    """
    boxes, scores, labels, scales = det["boxes"], det["scores"], det["labels"], det["scale_ids"]
    if boxes.numel() == 0:
        return det

    keep_all = []
    for c in labels.unique(sorted=True):
        c = int(c.item())
        idx = (labels == c)
        if idx.sum() == 0:
            continue
        b = boxes[idx]
        s = scores[idx]
        sc = scales[idx]
        # TopK pre-filter
        if b.shape[0] > topk_per_class:
            topk = torch.topk(s, k=topk_per_class).indices
            b, s, sc = b[topk], s[topk], sc[topk]
            idx_indices = idx.nonzero(as_tuple=False).squeeze(1)[topk]
        else:
            idx_indices = idx.nonzero(as_tuple=False).squeeze(1)

        keep_idx_local = nms(b, s, iou_thresh=iou_thresh)
        keep_all.append(idx_indices[keep_idx_local])

    if len(keep_all) == 0:
        kept = torch.zeros((0,), dtype=torch.long, device=boxes.device)
    else:
        kept = torch.cat(keep_all, dim=0)

    return {
        "boxes": boxes[kept],
        "scores": scores[kept],
        "labels": labels[kept],
        "scale_ids": scales[kept],
    }


# ------------------------ Scale analysis ------------------------

def analyze_scale_performance(model, dataloader, anchors):
    """
    Analyze which scales detect which object sizes.
    Generates:
      - results/visualizations/scale_performance.png
      - results/visualizations/scale_stats.json
    Also returns a dict with counts.
    """
    device = next(model.parameters()).device
    save_dir = Path("results/visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Buckets by object side length (sqrt(area)) aligned to your dataset scales
    def size_bucket(box):
        x1, y1, x2, y2 = [float(v) for v in box]
        side = max(1e-6, np.sqrt(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))))
        if side <= 40:  # ~small
            return "small"
        elif side <= 96:  # ~medium
            return "medium"
        else:
            return "large"

    # Counters
    scales = [0, 1, 2]
    buckets = ["small", "medium", "large"]
    stats = {(s, b): 0 for s in scales for b in buckets}

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            raw = model(images)
            decoded = decode_predictions(raw, anchors, num_classes=3, conf_thresh=0.3)
            for b_idx, (det_b, t) in enumerate(zip(decoded, targets)):
                kept = apply_nms_per_class(det_b, iou_thresh=0.5, topk_per_class=200)
                # Match detections to GT to know which are correct
                if len(t["boxes"]) == 0 or kept["boxes"].numel() == 0:
                    continue
                ious = compute_iou(kept["boxes"], t["boxes"].to(device))  # [D, G]
                best_iou, gt_idx = ious.max(dim=1)
                tp_mask = best_iou >= 0.5
                if tp_mask.any():
                    gt_used = set()
                    for i in torch.nonzero(tp_mask, as_tuple=False).squeeze(1).tolist():
                        g = int(gt_idx[i].item())
                        # avoid double-counting same GT multiple detections
                        if g in gt_used:
                            continue
                        gt_used.add(g)
                        s_id = int(kept["scale_ids"][i].item())
                        bucket = size_bucket(t["boxes"][g].tolist())
                        stats[(s_id, bucket)] += 1

    # Save bar chart
    values = np.array([[stats[(s, b)] for b in buckets] for s in scales])  # [3,3]
    x = np.arange(len(buckets))
    width = 0.25
    plt.figure(figsize=(7, 4))
    for i, s in enumerate(scales):
        plt.bar(x + (i - 1) * width, values[i], width=width, label=f"Scale {s + 1}")
    plt.xticks(x, buckets)
    plt.ylabel("True Positives (post-NMS)")
    plt.title("Scale specialization by object size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "scale_performance.png", dpi=150)
    plt.close()

    # Save raw stats
    stats_json = {
        f"scale_{s + 1}": {b: int(stats[(s, b)]) for b in buckets}
        for s in scales
    }
    with open(save_dir / "scale_stats.json", "w") as f:
        json.dump(stats_json, f, indent=2)

    return stats_json


# ------------------------ Anchor coverage visualization ------------------------

def visualize_anchor_coverage(image, gt_boxes, anchors_per_scale, save_dir, iou_thr=0.5):
    """
    For each scale, mark anchor centers that have IoU>=thr with any GT.
    Saves a scatter-like plot overlay on the image for each scale.
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy()
    W, H = img.size
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = anchors_per_scale[0].device if isinstance(anchors_per_scale[0], torch.Tensor) else torch.device("cpu")
    gtb = torch.as_tensor(gt_boxes, dtype=torch.float32, device=device) if len(gt_boxes) > 0 else torch.zeros((0, 4),
                                                                                                              device=device)

    for s_id, anc in enumerate(anchors_per_scale):
        anc_t = anc.to(device)
        if gtb.numel() > 0:
            ious = compute_iou(anc_t, gtb)  # [A, G]
            cover = (ious.max(dim=1).values >= iou_thr)  # [A]
        else:
            cover = torch.zeros((anc_t.shape[0],), dtype=torch.bool, device=device)

        # Draw points for coverage (anchor centers)
        cx, cy, _, _ = _to_cxcywh(anc_t)
        cx = cx.cpu().numpy()
        cy = cy.cpu().numpy()
        cover_np = cover.cpu().numpy()

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.scatter(cx[~cover_np], cy[~cover_np], s=1, alpha=0.2, label="no-match")
        plt.scatter(cx[cover_np], cy[cover_np], s=2, alpha=0.9, label="IoU>=0.5")
        plt.title(f"Anchor coverage (Scale {s_id + 1})")
        plt.legend(markerscale=4)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / f"anchor_coverage_scale{s_id + 1}.png", dpi=180)
        plt.close()


# ------------------------ End-to-end evaluation script ------------------------

def main():
    # ---------- Configuration ----------
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        # else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    image_size = 224
    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]


    anchor_scales = [
        [16, 24, 32],    # Scale 1 (56x56)
        [48, 64, 96],    # Scale 2 (28x28)
        [96, 128, 192],  # Scale 3 (14x14)
    ]

    # anchor_scales = [  #reduce S1 to make small obj more detactable
    #     [8, 12, 16],    # Scale 1 (56x56)
    #     [40, 56, 80],    # Scale 2 (28x28)
    #     [96, 128, 192],  # Scale 3 (14x14)
    # ]
    conf_thresh = 0.5
    nms_iou = 0.6
    iou_ap = 0.5

    # Paths (match your train.py)
    ROOT = Path(__file__).resolve().parents[1]          # problems/
    det_root = ROOT / "datasets" / "detection"
    val_image_dir = str(det_root / "val")
    val_annotation = str(det_root / "val_annotations.json")

    ckpt_path = Path("results/best_model.pth")
    vis_dir = Path("results/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Data / Model / Anchors ----------
    val_set = ShapeDetectionDataset(val_image_dir, val_annotation, transform=None)
    from torch.utils.data import DataLoader
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(targets)

    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("Warning: best_model.pth not found. Evaluating with random-initialized model.")

    model.eval()

    anchors_list = generate_anchors(feature_map_sizes, anchor_scales, image_size=image_size)
    anchors_device = [a.to(device) for a in anchors_list]

    # ---------- Inference & mAP ----------
    class_names = ["circle", "square", "triangle"]
    B = len(val_set)

    # Build GT dict per image_id for each class
    gt_by_class = {c: {} for c in range(3)}  # c -> {img_id: [boxes]}
    for img_id in range(B):
        t = val_set.samples[img_id]
        boxes = torch.as_tensor(t["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(t["labels"], dtype=torch.long)
        for c in range(3):
            gt_by_class[c][img_id] = [boxes[i].tolist() for i in
                                      (labels == c).nonzero(as_tuple=False).squeeze(1).tolist()]

    # Collect predictions
    preds_by_class = {c: [] for c in range(3)}  # list of dict(image_id, box, score)
    img_ptr = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            raw = model(images)
            decoded = decode_predictions(raw, anchors_device, num_classes=3, conf_thresh=conf_thresh)
            for det in decoded:
                kept = apply_nms_per_class(det, iou_thresh=nms_iou, topk_per_class=200)
                # Store by class
                for i in range(kept["boxes"].shape[0]):
                    c = int(kept["labels"][i].item())
                    preds_by_class[c].append({
                        "image_id": img_ptr,
                        "box": kept["boxes"][i].cpu(),
                        "score": float(kept["scores"][i].item())
                    })
                img_ptr += 1

    # Compute AP per class
    ap_per_class = {}
    for c in range(3):
        ap = compute_ap(preds_by_class[c], gt_by_class[c], iou_threshold=iou_ap)
        ap_per_class[class_names[c]] = ap
    mAP = float(np.mean(list(ap_per_class.values()))) if len(ap_per_class) > 0 else 0.0

    print("AP per class:", ap_per_class)
    print("mAP@0.5:", mAP)

    # Save metrics
    with open(Path("results") / "eval_metrics.json", "w") as f:
        json.dump({"AP@0.5": ap_per_class, "mAP@0.5": mAP}, f, indent=2)

    # ---------- Visualize detections for first 10 images ----------
    num_to_vis = min(10, len(val_set))
    # For convenience re-run a single-batch decode to get per-image kept detections
    # (We can iterate one-by-one for simplicity)
    for img_id in range(num_to_vis):
        img_path = val_set.samples[img_id]["path"]
        # Single image forward
        img = Image.open(img_path).convert("RGB")
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(device)
        with torch.no_grad():
            raw = model(x)
            decoded = decode_predictions(raw, anchors_device, num_classes=3, conf_thresh=conf_thresh)[0]
            kept = apply_nms_per_class(decoded, iou_thresh=nms_iou, topk_per_class=200)

        # Build GT for this image
        gt_boxes = torch.as_tensor(val_set.samples[img_id]["boxes"], dtype=torch.float32)
        gt_labels = torch.as_tensor(val_set.samples[img_id]["labels"], dtype=torch.long)

        visualize_detections(
            image=img,
            predictions={"boxes": kept["boxes"].cpu(), "scores": kept["scores"].cpu(), "labels": kept["labels"].cpu()},
            ground_truths={"boxes": gt_boxes, "labels": gt_labels},
            save_path=vis_dir / f"val_det_{img_id:03d}.png"
        )

    # ---------- Anchor coverage visualization (use the first val image) ----------
    if len(val_set) > 0:
        img0_path = val_set.samples[0]["path"]
        gt0 = val_set.samples[0]["boxes"]
        visualize_anchor_coverage(img0_path, gt0, anchors_device, vis_dir, iou_thr=0.5)

    # ---------- Scale specialization analysis ----------
    from torch.utils.data import DataLoader
    val_loader_small = DataLoader(val_set, batch_size=8, shuffle=False,
                                  collate_fn=lambda b: (torch.stack([x for x, _ in b]), [t for _, t in b]))
    stats_json = analyze_scale_performance(model, val_loader_small, anchors_device)
    print("Scale analysis:", stats_json)
    print(f"Visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
