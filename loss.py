import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


# Convert [cx, cy, w, h] to [x1, y1, x2, y2]
def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

# Generalized IoU for box pairs
def generalized_box_iou(boxes1, boxes2):
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)                         # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                   # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union

    lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]

    giou = iou - (area_enc - union) / area_enc
    return giou

# Hungarian Matcher
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)  # [B, Q, C]
        out_bbox = outputs["pred_boxes"]  # [B, Q, 4]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]

            # Skip if no boxes
            if tgt_bbox.shape[0] == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64)
                ))
                continue

            if tgt_bbox.ndim == 1:
                tgt_bbox = tgt_bbox.unsqueeze(0)
            if tgt_ids.ndim == 0:
                tgt_ids = tgt_ids.unsqueeze(0)

            cost_class = -out_prob[b].gather(1, tgt_ids.unsqueeze(0).expand(out_prob[b].size(0), -1))
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(out_bbox[b], tgt_bbox)

            total_cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            total_cost = total_cost.detach().cpu()

            if total_cost.shape == (1, 1):
                indices.append((torch.tensor([0], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)))
                continue

            indices_b = linear_sum_assignment(total_cost)
            indices.append((
                torch.as_tensor(indices_b[0], dtype=torch.int64),
                torch.as_tensor(indices_b[1], dtype=torch.int64)
            ))

        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        # Get target classes for matched queries
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,  # 'no-object' class
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # Compute cross-entropy across all queries (including unmatched ones)
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx].clamp(0, 1)
        # src_boxes = outputs['pred_boxes'][idx]

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').mean()
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))).mean()
        return 5.0 * loss_bbox + 2.0 * loss_giou  # weights!

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        loss_cls = self.loss_labels(outputs, targets, indices)
        loss_box = self.loss_boxes(outputs, targets, indices)
        return loss_cls + loss_box