import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import match_anchors_to_targets


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

        # weights
        self.w_obj = 1.0
        self.w_cls = 1.0
        self.w_loc = 2.0

        # per-anchor BCE (objectness)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")
        # per-anchor SmoothL1 (localization)
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, predictions, targets, anchors):
        """
        Args:
            predictions: List[Tensor], one per scale
                shape each: [B, A*(5+C), H, W]
            targets: List[Dict], length B, each has:
                - 'boxes': Tensor [N,4] (x1,y1,x2,y2)
                - 'labels': Tensor [N]    in {0..C-1}
            anchors: List[Tensor], one per scale
                shape each: [H*W*A, 4] (x1,y1,x2,y2)

        Returns:
            Dict with loss_obj, loss_cls, loss_loc, loss_total (all scalars)
        """
        device = predictions[0].device
        B = predictions[0].shape[0]
        C = self.num_classes

        total_obj = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        total_loc = torch.tensor(0.0, device=device)

        total_pos = 0
        total_obj_den = 0  # positives + selected negatives count

        # ---- per-scale processing ----
        for pred_s, anchors_s in zip(predictions, anchors):
            B, ch, H, W = pred_s.shape
            A_per = ch // (5 + C)  # anchors per spatial location

            # # reshape -> [B, H*W*A, 5+C]
            # pred_s = pred_s.view(B, A_per, (5 + C), H, W)
            # pred_s = pred_s.permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,A,5+C]
            # pred_s = pred_s.view(B, -1, (5 + C))  # [B, H*W*A, 5+C]

            pred_s = (
                pred_s
                .reshape(B, A_per, 5 + C, H, W)
                .permute(0, 3, 4, 1, 2)
                .reshape(B, -1, 5 + C)
            )


            # anchors for this scale
            anchors_s = anchors_s.to(device)  # [H*W*A, 4]
            assert anchors_s.shape[0] == pred_s.shape[1], \
                "Anchors count must match predictions at this scale"

            # split predictions
            pred_loc = pred_s[..., 0:4].contiguous()              # [B, N, 4] (tx,ty,tw,th)
            pred_obj = pred_s[..., 4].contiguous()               # [B, N]
            pred_cls = pred_s[..., 5:5 + C].contiguous()             # [B, N, C]

            assert anchors_s.shape[0] == H * W * A_per, \
                f"Anchors {anchors_s.shape[0]} != H*W*A {H}*{W}*{A_per}={H*W*A_per}"

            # ---- per-image matching & loss ----
            for b in range(B):
                t = targets[b]
                gt_boxes = t["boxes"].to(device) if t["boxes"].numel() > 0 else t["boxes"].to(device)
                gt_labels = t["labels"].to(device) if t["labels"].numel() > 0 else t["labels"].to(device)

                # match
                matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                    anchors_s, gt_boxes, gt_labels,
                    pos_threshold=0.5, neg_threshold=0.3
                )
                # matched_labels: [N] with 0=bg, 1..C = classes+1
                # matched_boxes:  [N,4] (x1,y1,x2,y2)

                N_all = pos_mask.shape[0]
                pos_idx = pos_mask
                neg_idx = neg_mask

                num_pos = int(pos_idx.sum().item())

                # ---------- Objectness (BCE with logits) ----------
                obj_target = torch.zeros((N_all,), device=device, dtype=torch.float32)
                obj_target[pos_idx] = 1.0

                obj_loss_all = self.bce_logits(pred_obj[b], obj_target)  # [N_all]

                # Hard Negative Mining (select top-k negatives)
                sel_neg_idx = self.hard_negative_mining(
                    obj_loss_all.detach(), pos_idx, neg_idx, ratio=3
                )

                # objectness loss = positives + selected negatives
                obj_mask = pos_idx | sel_neg_idx
                if obj_mask.any():
                    total_obj += obj_loss_all[obj_mask].sum()
                    total_obj_den += int(obj_mask.sum().item())

                # ---------- Classification (only positives) ----------
                if num_pos > 0:
                    cls_logits_pos = pred_cls[b][pos_idx]            # [P, C]
                    # matched_labels positive are 1..C -> convert to 0..C-1
                    cls_targets = matched_labels[pos_idx] - 1        # [P]
                    cls_loss = F.cross_entropy(cls_logits_pos, cls_targets, reduction="sum")
                    total_cls += cls_loss
                # if no positives: skip (add 0)

                # ---------- Localization (Smooth L1 on encoded deltas) ----------
                if num_pos > 0:
                    # encode gt deltas wrt anchors
                    target_deltas = self.encode_boxes(matched_boxes[pos_idx], anchors_s[pos_idx])  # [P,4]
                    loc_loss = self.smooth_l1(pred_loc[b][pos_idx], target_deltas)  # [P,4]
                    total_loc += loc_loss.sum(dim=-1).sum()

                total_pos += num_pos

        # normalizers
        pos_norm = max(total_pos, 1)  # avoid div-by-zero
        obj_norm = max(total_obj_den, 1)

        loss_obj = total_obj / obj_norm
        loss_cls = total_cls / pos_norm
        loss_loc = total_loc / pos_norm

        loss_total = self.w_obj * loss_obj + self.w_cls * loss_cls + self.w_loc * loss_loc

        return {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_loc": loss_loc,
            "loss_total": loss_total
        }

    @staticmethod
    def encode_boxes(gt_boxes, anc_boxes, eps: float = 1e-6):
        """
        Encode ground-truth boxes relative to anchors (both [x1,y1,x2,y2]).

        Returns:
            Tensor [N,4] as (tx, ty, tw, th)
        """
        # to centers
        def _to_cxcywh(box):
            x1, y1, x2, y2 = box.unbind(dim=-1)
            w = (x2 - x1).clamp(min=eps)
            h = (y2 - y1).clamp(min=eps)
            cx = x1 + 0.5 * w
            cy = y1 + 0.5 * h
            return cx, cy, w, h

        g_cx, g_cy, g_w, g_h = _to_cxcywh(gt_boxes)
        a_cx, a_cy, a_w, a_h = _to_cxcywh(anc_boxes)

        tx = (g_cx - a_cx) / (a_w + eps)
        ty = (g_cy - a_cy) / (a_h + eps)
        tw = torch.log((g_w + eps) / (a_w + eps))
        th = torch.log((g_h + eps) / (a_h + eps))
        return torch.stack([tx, ty, tw, th], dim=-1)

    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        """
        Select hard negative examples using highest objectness loss.

        Args:
            loss: Tensor [N] (per-anchor objectness loss, detached is fine)
            pos_mask: Bool [N]
            neg_mask: Bool [N]
            ratio: negative : positive

        Returns:
            selected_neg_mask: Bool [N]
        """
        with torch.no_grad():
            num_pos = int(pos_mask.sum().item())
            max_negs = int(ratio * max(num_pos, 1))  # when no positives, still keep a few negatives

            neg_losses = loss.clone()
            # mask out non-negs by -inf so they won't be selected
            neg_losses[~neg_mask] = float("-inf")

            # If there are fewer negatives than max_negs, just take all
            k = min(int(neg_mask.sum().item()), max_negs)
            if k == 0:
                return torch.zeros_like(neg_mask, dtype=torch.bool)

            # top-k hardest negatives
            topk_vals, topk_idx = torch.topk(neg_losses, k=k, largest=True)
            selected = torch.zeros_like(neg_mask, dtype=torch.bool)
            selected[topk_idx] = True
            return selected

