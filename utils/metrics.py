"""Segmentation metrics: IoU, Dice, Pixel Accuracy."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def calculate_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100
) -> torch.Tensor:
    """Calculate Intersection over Union per class.
    
    Args:
        pred: Predicted class indices (B, H, W) or (H, W).
        target: Ground truth class indices (B, H, W) or (H, W).
        num_classes: Total number of classes.
        ignore_index: Index to ignore.
        
    Returns:
        IoU per class tensor of shape (num_classes,).
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Create mask for valid pixels
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    
    iou_per_class = torch.zeros(num_classes, device=pred.device)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou_per_class[cls] = intersection / union
        else:
            iou_per_class[cls] = float('nan')
    
    return iou_per_class


def calculate_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: int = -100
) -> torch.Tensor:
    """Calculate Dice coefficient per class.
    
    Args:
        pred: Predicted class indices.
        target: Ground truth class indices.
        num_classes: Total number of classes.
        smooth: Smoothing factor.
        ignore_index: Index to ignore.
        
    Returns:
        Dice per class tensor of shape (num_classes,).
    """
    pred = pred.flatten()
    target = target.flatten()
    
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    
    dice_per_class = torch.zeros(num_classes, device=pred.device)

    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        total = pred_cls.sum() + target_cls.sum()

        if total > 0:
            dice_per_class[cls] = (2 * intersection) / (total + smooth)
        else:
            # Class absent from BOTH prediction and target: undefined, not perfect.
            # Returning 1.0 here (the old `smooth/smooth` behaviour) silently
            # inflated the mean Dice of rare classes such as RunDown.
            dice_per_class[cls] = float('nan')

    return dice_per_class


def calculate_pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """Calculate pixel-wise accuracy.
    
    Args:
        pred: Predicted class indices.
        target: Ground truth class indices.
        ignore_index: Index to ignore.
        
    Returns:
        Pixel accuracy as float.
    """
    valid = target != ignore_index
    
    if valid.sum() == 0:
        return 0.0
    
    correct = ((pred == target) & valid).sum().float()
    total = valid.sum().float()
    
    return (correct / total).item()


def _legacy_batch_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
    ignore_index: int = -100
) -> torch.Tensor:
    """Reproduce the ORIGINAL per-batch Dice, bug included, for comparison only.

    The original :func:`calculate_dice` returned ``smooth / smooth == 1.0`` for
    any class absent from both prediction and target. That is the behaviour
    that inflated rare-class Dice on the published dashboard. It is preserved
    here verbatim -- and ONLY here -- so a corrected run can be diffed against
    the old results. Never use this for reporting.
    """
    pred = pred.flatten()
    target = target.flatten()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    dice_per_class = torch.zeros(num_classes, device=pred.device)
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        total = pred_cls.sum() + target_cls.sum()
        dice_per_class[cls] = (2 * intersection + smooth) / (total + smooth)
    return dice_per_class


class SegmentationMetrics:
    """Track and compute segmentation metrics over a whole dataset.

    Metrics are accumulated as raw confusion-matrix counts across every batch
    and reduced ONCE in :meth:`compute`. This is the standard dataset-level
    (a.k.a. "global" or "aggregated") protocol used by Cityscapes / ADE20K /
    Pascal VOC.

    Why this matters (and what changed):
      * The previous implementation computed IoU and Dice *per batch* and then
        averaged those per-batch scores. With 57 test images and a rare class
        such as RunDown appearing in only a handful of them, that estimator is
        strongly biased and unstable.
      * Worse, the old per-batch Dice returned ``smooth / smooth == 1.0`` for
        any class absent from both prediction and target. Rare classes were
        scored as "perfect" on every batch that did not contain them, which is
        how the study ended up publishing ``dice_RunDown = 0.90`` alongside
        ``iou_RunDown = 0.376`` -- an impossible pair, since Dice is pinned to
        IoU by ``Dice = 2*IoU / (1 + IoU)`` (0.376 -> 0.546).

    Nothing is dropped: the old per-batch numbers are still computed and
    reported under ``legacy_*`` keys so a corrected run can be diffed directly
    against results already published to the dashboard.

    Example:
        metrics = SegmentationMetrics(num_classes=4, class_names=[...])
        for pred, target in dataloader:
            metrics.update(pred, target)
        results = metrics.compute()
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -100,
        background_index: int = 0
    ) -> None:
        """Initialize metrics tracker.

        Args:
            num_classes: Number of classes.
            class_names: Optional list of class names for reporting.
            ignore_index: Index to ignore in metrics.
            background_index: Class treated as background when reporting the
                defect-only ("foreground") means. Set to None to disable.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        # Dataset-level confusion counts (the primary, corrected statistics).
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)
        self.pixel_correct = 0
        self.pixel_total = 0
        self.batch_count = 0

        # Legacy per-batch accumulators, kept so the corrected numbers can be
        # compared against everything already published.
        self.iou_sum = torch.zeros(self.num_classes)
        self.dice_sum = torch.zeros(self.num_classes)
        self.class_counts = torch.zeros(self.num_classes)
        self.legacy_dice_counts = torch.zeros(self.num_classes)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch of predictions.

        Args:
            pred: Predicted class indices (B, H, W), or logits (B, C, H, W).
            target: Ground truth class indices (B, H, W).
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)

        pred = pred.detach().cpu().flatten()
        target = target.detach().cpu().flatten().long()

        valid = target != self.ignore_index
        pred_v = pred[valid].long()
        target_v = target[valid]

        # Guard against out-of-range labels rather than silently corrupting the
        # confusion matrix (a stray class id in a YOLO label file would
        # otherwise wrap around via bincount).
        in_range = (pred_v >= 0) & (pred_v < self.num_classes) & \
                   (target_v >= 0) & (target_v < self.num_classes)
        if not bool(in_range.all()):
            dropped = int((~in_range).sum())
            logger.warning(
                f"Dropping {dropped} pixel(s) with class ids outside "
                f"[0, {self.num_classes - 1}] while accumulating metrics."
            )
            pred_v = pred_v[in_range]
            target_v = target_v[in_range]

        if target_v.numel() > 0:
            idx = target_v * self.num_classes + pred_v
            counts = torch.bincount(idx, minlength=self.num_classes ** 2)
            self.confusion += counts.reshape(self.num_classes, self.num_classes).to(torch.float64)

        self.pixel_correct += int((pred_v == target_v).sum())
        self.pixel_total += int(target_v.numel())

        # --- legacy per-batch bookkeeping (reported, never used as headline) ---
        iou = calculate_iou(pred, target, self.num_classes, self.ignore_index)
        dice = _legacy_batch_dice(pred, target, self.num_classes, ignore_index=self.ignore_index)

        iou_valid = ~torch.isnan(iou)
        dice_valid = ~torch.isnan(dice)
        self.iou_sum += torch.where(iou_valid, iou, torch.zeros_like(iou))
        self.dice_sum += torch.where(dice_valid, dice, torch.zeros_like(dice))
        self.class_counts += iou_valid.float()
        self.legacy_dice_counts += dice_valid.float()

        self.batch_count += 1

    def compute(self) -> Dict[str, float]:
        """Reduce accumulated counts into a flat metric dictionary.

        Returns:
            Dict of metric name -> float. Every per-class metric is emitted for
            every class; classes absent from both prediction and ground truth
            are reported as NaN and excluded from the means (their support is
            still reported so the absence stays visible).
        """
        conf = self.confusion
        tp = torch.diag(conf)
        pred_sum = conf.sum(dim=0)     # pixels predicted as class c
        target_sum = conf.sum(dim=1)   # pixels labelled class c (support)
        fp = pred_sum - tp
        fn = target_sum - tp
        union = pred_sum + target_sum - tp

        # A class is "present" if it appears in the ground truth or the
        # prediction anywhere in the dataset.
        present = union > 0
        support_present = target_sum > 0

        nan = float('nan')
        iou = torch.where(present, tp / union.clamp(min=1e-12), torch.full_like(tp, nan))
        denom = pred_sum + target_sum
        dice = torch.where(present, (2 * tp) / denom.clamp(min=1e-12), torch.full_like(tp, nan))
        precision = torch.where(pred_sum > 0, tp / pred_sum.clamp(min=1e-12), torch.full_like(tp, nan))
        recall = torch.where(target_sum > 0, tp / target_sum.clamp(min=1e-12), torch.full_like(tp, nan))
        f1 = dice  # for hard labels, per-class F1 and Dice are the same quantity

        def _mean(values: torch.Tensor, mask: torch.Tensor) -> float:
            sel = values[mask & ~torch.isnan(values)]
            return float(sel.mean()) if sel.numel() > 0 else 0.0

        all_mask = present
        if self.background_index is not None and 0 <= self.background_index < self.num_classes:
            fg_mask = present.clone()
            fg_mask[self.background_index] = False
        else:
            fg_mask = present

        pixel_acc = self.pixel_correct / max(self.pixel_total, 1)

        # Frequency-weighted IoU (weights each class by its share of GT pixels).
        total_gt = float(target_sum.sum())
        if total_gt > 0:
            freq = target_sum / total_gt
            fw_terms = torch.where(present & ~torch.isnan(iou), freq * iou, torch.zeros_like(iou))
            fwiou = float(fw_terms.sum())
        else:
            fwiou = 0.0

        results: Dict[str, float] = {
            # --- headline (corrected, dataset-level) ---
            'mean_iou': _mean(iou, all_mask),
            'mean_dice': _mean(dice, all_mask),
            'pixel_accuracy': pixel_acc,
            # --- defect-only means (background excluded) ---
            'mean_iou_defect': _mean(iou, fg_mask),
            'mean_dice_defect': _mean(dice, fg_mask),
            # --- means restricted to classes that actually have ground truth ---
            # A class with zero GT pixels but some predictions scores IoU 0
            # (pure false positives) and drags the means above down. That is
            # the correct penalty, but the GT-present view is reported too so
            # the two readings can be told apart.
            'mean_iou_gt_present': _mean(iou, support_present),
            'mean_dice_gt_present': _mean(dice, support_present),
            'mean_iou_defect_gt_present': _mean(iou, fg_mask & support_present),
            'mean_dice_defect_gt_present': _mean(dice, fg_mask & support_present),
            # --- additional aggregate views ---
            'frequency_weighted_iou': fwiou,
            'mean_accuracy': _mean(recall, support_present),
            'mean_precision': _mean(precision, all_mask),
            'mean_recall': _mean(recall, support_present),
            'mean_f1': _mean(f1, all_mask),
            'mean_precision_defect': _mean(precision, fg_mask),
            'mean_recall_defect': _mean(recall, fg_mask & support_present),
            'mean_f1_defect': _mean(f1, fg_mask),
            # --- provenance ---
            'num_classes_present': int(present.sum()),
            'total_pixels_evaluated': int(self.pixel_total),
            'num_batches': int(self.batch_count),
            'metrics_protocol': 'dataset_level_confusion_v2',
        }

        for i, name in enumerate(self.class_names):
            results[f'iou_{name}'] = float(iou[i])
            results[f'dice_{name}'] = float(dice[i])
            results[f'precision_{name}'] = float(precision[i])
            results[f'recall_{name}'] = float(recall[i])
            results[f'f1_{name}'] = float(f1[i])
            # Support makes "this class is unmeasurable" visible instead of
            # letting a 4-instance class silently drag the mean down.
            results[f'support_px_{name}'] = int(target_sum[i])
            results[f'predicted_px_{name}'] = int(pred_sum[i])
            results[f'tp_{name}'] = int(tp[i])
            results[f'fp_{name}'] = int(fp[i])
            results[f'fn_{name}'] = int(fn[i])

        # --- legacy per-batch numbers, for direct comparison against the
        #     results already published to the dashboard ---
        legacy_iou = self.iou_sum / torch.clamp(self.class_counts, min=1)
        legacy_dice = self.dice_sum / torch.clamp(self.legacy_dice_counts, min=1)
        legacy_valid = self.class_counts > 0
        results['legacy_mean_iou'] = (
            float(legacy_iou[legacy_valid].mean()) if bool(legacy_valid.any()) else 0.0
        )
        results['legacy_mean_dice'] = (
            float(legacy_dice[legacy_valid].mean()) if bool(legacy_valid.any()) else 0.0
        )
        for i, name in enumerate(self.class_names):
            results[f'legacy_iou_{name}'] = float(legacy_iou[i])
            results[f'legacy_dice_{name}'] = float(legacy_dice[i])

        return results

    def get_confusion_matrix(self) -> torch.Tensor:
        """Return the raw accumulated confusion matrix (rows = GT, cols = pred)."""
        return self.confusion.clone()

    def __str__(self) -> str:
        """String representation of current metrics."""
        results = self.compute()
        lines = [
            f"Mean IoU: {results['mean_iou']:.4f}",
            f"Mean IoU (defects only): {results['mean_iou_defect']:.4f}",
            f"Mean Dice: {results['mean_dice']:.4f}",
            f"Pixel Accuracy: {results['pixel_accuracy']:.4f}",
            f"Frequency-weighted IoU: {results['frequency_weighted_iou']:.4f}",
        ]
        for name in self.class_names:
            lines.append(
                f"  {name:<12} IoU={results[f'iou_{name}']:.4f} "
                f"Dice={results[f'dice_{name}']:.4f} "
                f"support={results[f'support_px_{name}']} px"
            )
        return '\n'.join(lines)


def calculate_boundary_tolerant_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    tolerance: int = 3,
    ignore_index: int = -100
) -> torch.Tensor:
    """Calculate IoU with boundary tolerance to handle polygon vs organic shape mismatch.
    
    Dilates both prediction and target masks before computing IoU,
    allowing slack at boundaries where annotations may be imprecise.
    
    Args:
        pred: Predicted class indices (B, H, W) or (H, W).
        target: Ground truth class indices (B, H, W) or (H, W).
        num_classes: Total number of classes.
        tolerance: Dilation kernel size (pixels of boundary slack allowed).
        ignore_index: Index to ignore.
        
    Returns:
        Boundary-tolerant IoU per class tensor of shape (num_classes,).
    """
    import torch.nn.functional as F
    
    pred = pred.flatten()
    target = target.flatten()
    
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    
    # Reshape for morphological operations
    H = W = int(np.sqrt(len(pred)))
    if H * W != len(pred):
        # Fallback to standard IoU if can't reshape
        return calculate_iou(pred.view(-1), target.view(-1), num_classes, ignore_index)
    
    iou_per_class = torch.zeros(num_classes, device=pred.device)
    kernel_size = 2 * tolerance + 1
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float().view(1, 1, H, W)
        target_cls = (target == cls).float().view(1, 1, H, W)
        
        # Dilate both masks using max pooling
        pred_dilated = F.max_pool2d(pred_cls, kernel_size, stride=1, padding=tolerance)
        target_dilated = F.max_pool2d(target_cls, kernel_size, stride=1, padding=tolerance)
        
        # Tolerant intersection: pred overlaps with dilated target OR dilated pred overlaps with target
        intersection = ((pred_cls * target_dilated) + (pred_dilated * target_cls)).clamp(0, 1).sum()
        union = ((pred_cls + target_cls) > 0).float().sum()
        
        if union > 0:
            iou_per_class[cls] = intersection / union
        else:
            iou_per_class[cls] = float('nan')
    
    return iou_per_class


def calculate_instance_detection_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.3,
    ignore_index: int = -100
) -> Dict[str, float]:
    """Calculate instance-level detection metrics (object-level, not pixel-level).
    
    Extracts connected components from both prediction and target,
    then matches them based on IoU threshold. This is more forgiving
    of boundary mismatches since it evaluates "did we find the defect?"
    
    Args:
        pred: Predicted class indices (B, H, W) or (H, W).
        target: Ground truth class indices (B, H, W) or (H, W).
        num_classes: Total number of classes.
        iou_threshold: Minimum IoU to consider a detection matched.
        ignore_index: Index to ignore.
        
    Returns:
        Dictionary with precision, recall, f1 for instance detection.
    """
    from scipy import ndimage
    
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    H = W = int(np.sqrt(len(pred)))
    pred = pred.reshape(H, W)
    target = target.reshape(H, W)
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # Skip background class (class 0)
    for cls in range(1, num_classes):
        pred_mask = (pred == cls).astype(np.uint8)
        target_mask = (target == cls).astype(np.uint8)
        
        # Extract connected components (instances)
        pred_labels, num_pred = ndimage.label(pred_mask)
        target_labels, num_target = ndimage.label(target_mask)
        
        matched_targets = set()
        
        # For each predicted instance, find best matching target
        for pred_id in range(1, num_pred + 1):
            pred_instance = (pred_labels == pred_id)
            best_iou = 0
            best_target = None
            
            for target_id in range(1, num_target + 1):
                target_instance = (target_labels == target_id)
                
                intersection = (pred_instance & target_instance).sum()
                union = (pred_instance | target_instance).sum()
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_target = target_id
            
            if best_iou >= iou_threshold and best_target is not None:
                total_tp += 1
                matched_targets.add(best_target)
            else:
                total_fp += 1  # Predicted instance not matched
        
        # Unmatched target instances are false negatives
        total_fn += num_target - len(matched_targets)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'instance_precision': precision,
        'instance_recall': recall,
        'instance_f1': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


def calculate_boundary_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    tolerance: int = 2,
    ignore_index: int = -100
) -> Dict[str, float]:
    """Calculate boundary-specific metrics using Hausdorff-like distance.
    
    Measures how well prediction boundaries match target boundaries,
    with tolerance for annotation imprecision.
    
    Args:
        pred: Predicted class indices (B, H, W) or (H, W).
        target: Ground truth class indices (B, H, W) or (H, W).
        num_classes: Total number of classes.
        tolerance: Distance threshold in pixels for boundary matching.
        ignore_index: Index to ignore.
        
    Returns:
        Dictionary with boundary precision/recall and average boundary distance.
    """
    from scipy import ndimage
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    if pred.ndim == 3:
        pred = pred[0]
        target = target[0]
    
    H, W = pred.shape
    
    boundary_distances = []
    boundary_recalls = []
    
    for cls in range(1, num_classes):  # Skip background
        pred_mask = (pred == cls).astype(np.uint8)
        target_mask = (target == cls).astype(np.uint8)
        
        # Extract boundaries using erosion
        pred_eroded = ndimage.binary_erosion(pred_mask)
        target_eroded = ndimage.binary_erosion(target_mask)
        
        pred_boundary = pred_mask.astype(bool) ^ pred_eroded
        target_boundary = target_mask.astype(bool) ^ target_eroded
        
        if not target_boundary.any() or not pred_boundary.any():
            continue
        
        # Distance transform from target boundary
        target_dist = ndimage.distance_transform_edt(~target_boundary)
        pred_dist = ndimage.distance_transform_edt(~pred_boundary)
        
        # Average distance from predicted boundary to nearest target boundary
        pred_to_target_dist = target_dist[pred_boundary].mean() if pred_boundary.any() else 0
        target_to_pred_dist = pred_dist[target_boundary].mean() if target_boundary.any() else 0
        
        # Symmetric boundary distance (like Hausdorff but average)
        avg_boundary_dist = (pred_to_target_dist + target_to_pred_dist) / 2
        boundary_distances.append(avg_boundary_dist)
        
        # Boundary recall: % of target boundary within tolerance distance of prediction
        within_tolerance = (pred_dist[target_boundary] <= tolerance).sum()
        total_target_boundary = target_boundary.sum()
        boundary_recall = within_tolerance / total_target_boundary if total_target_boundary > 0 else 0
        boundary_recalls.append(boundary_recall)
    
    return {
        'avg_boundary_distance': np.mean(boundary_distances) if boundary_distances else 0.0,
        'boundary_recall': np.mean(boundary_recalls) if boundary_recalls else 0.0,
        'boundary_recall_per_class': boundary_recalls
    }


if __name__ == "__main__":
    # Self-test: verifies the dataset-level accumulator against hand-computed
    # ground truth, and pins the Dice/IoU identity that the old per-batch
    # implementation violated.
    logging.basicConfig(level=logging.INFO)

    print("=" * 66)
    print("SegmentationMetrics self-test")
    print("=" * 66)

    # --- Test 1: exact values on a tiny, hand-checkable example -------------
    # 4 classes. Class 3 never appears in GT or prediction -> must be NaN,
    # must NOT be counted as a perfect score.
    target = torch.tensor([[[0, 0, 1, 1],
                            [0, 0, 1, 2]]])          # 6x class0? -> counts below
    pred = torch.tensor([[[0, 0, 1, 0],
                          [0, 1, 1, 2]]])

    m = SegmentationMetrics(num_classes=4, class_names=['bg', 'a', 'b', 'c'])
    m.update(pred, target)
    r = m.compute()

    # GT counts: bg=4, a=3, b=1, c=0 | Pred counts: bg=4, a=3, b=1, c=0
    # bg: TP=3 -> IoU 3/(4+4-3)=0.60 ; Dice 6/8 = 0.75
    # a : TP=2 -> IoU 2/(3+3-2)=0.50 ; Dice 4/6
    # b : TP=1 -> IoU 1/(1+1-1)=1.00 ; Dice 1.0
    # c : absent everywhere -> NaN, excluded from means
    assert abs(r['iou_bg'] - 0.60) < 1e-9, r['iou_bg']
    assert abs(r['iou_a'] - 0.50) < 1e-9, r['iou_a']
    assert abs(r['iou_b'] - 1.00) < 1e-9, r['iou_b']
    assert abs(r['dice_bg'] - 0.75) < 1e-9, r['dice_bg']
    assert abs(r['dice_a'] - 4 / 6) < 1e-9, r['dice_a']
    assert np.isnan(r['iou_c']), r['iou_c']
    assert np.isnan(r['dice_c']), r['dice_c']
    assert r['support_px_c'] == 0
    assert r['num_classes_present'] == 3
    assert abs(r['mean_iou'] - (0.60 + 0.50 + 1.00) / 3) < 1e-9, r['mean_iou']
    assert abs(r['mean_iou_defect'] - (0.50 + 1.00) / 2) < 1e-9, r['mean_iou_defect']
    assert abs(r['pixel_accuracy'] - 6 / 8) < 1e-9, r['pixel_accuracy']
    assert r['tp_bg'] == 3 and r['fp_bg'] == 1 and r['fn_bg'] == 1
    print("  [pass] exact per-class IoU/Dice, NaN for absent class, defect-only mean")

    # --- Test 2: the Dice <-> IoU identity that used to be violated ---------
    # For hard labels, Dice must always equal 2*IoU / (1 + IoU).
    torch.manual_seed(0)
    m2 = SegmentationMetrics(num_classes=4, class_names=['bg', 'Dust', 'RunDown', 'Scratch'])
    for _ in range(12):
        # Heavily imbalanced, mimicking the real dataset: RunDown almost never
        # appears, which is exactly the case the old implementation mishandled.
        t = torch.zeros(2, 32, 32, dtype=torch.long)
        t[:, :6, :6] = 1
        if _ == 0:
            t[:, 20:22, 20:22] = 2          # RunDown in a single batch only
        t[:, 10:14, :] = 3
        p = t.clone()
        p[:, :3, :3] = 0                     # miss part of Dust
        p[:, 20:21, 20:22] = 0               # miss half of RunDown
        m2.update(p, t)
    r2 = m2.compute()
    for name in ['bg', 'Dust', 'RunDown', 'Scratch']:
        iou_v, dice_v = r2[f'iou_{name}'], r2[f'dice_{name}']
        if np.isnan(iou_v):
            continue
        expected = 2 * iou_v / (1 + iou_v)
        assert abs(dice_v - expected) < 1e-6, (name, iou_v, dice_v, expected)
    print("  [pass] Dice == 2*IoU/(1+IoU) holds for every class")

    # Demonstrate the bug this replaces: the legacy per-batch Dice for the rare
    # class is inflated far above what its IoU permits.
    legacy_rundown = r2['legacy_dice_RunDown']
    correct_rundown = r2['dice_RunDown']
    print(f"  RunDown  corrected Dice={correct_rundown:.4f}  "
          f"legacy per-batch Dice={legacy_rundown:.4f}  "
          f"(legacy inflated by {legacy_rundown - correct_rundown:+.4f})")
    assert legacy_rundown > correct_rundown, "expected legacy Dice to be inflated"
    print("  [pass] legacy_* keys retained and reproduce the old inflated value")

    # --- Test 3: batch-order invariance (dataset-level accumulation) --------
    a = SegmentationMetrics(num_classes=4, class_names=['bg', 'a', 'b', 'c'])
    b = SegmentationMetrics(num_classes=4, class_names=['bg', 'a', 'b', 'c'])
    torch.manual_seed(1)
    batches = [(torch.randint(0, 3, (2, 16, 16)), torch.randint(0, 3, (2, 16, 16)))
               for _ in range(5)]
    for p_, t_ in batches:
        a.update(p_, t_)
    for p_, t_ in reversed(batches):
        b.update(p_, t_)
    assert abs(a.compute()['mean_iou'] - b.compute()['mean_iou']) < 1e-12
    print("  [pass] result is invariant to batch order")

    # --- Test 4: full key inventory -----------------------------------------
    m3 = SegmentationMetrics(num_classes=4, class_names=['Background', 'Dust', 'RunDown', 'Scratch'])
    m3.update(torch.zeros(1, 8, 8, dtype=torch.long), torch.zeros(1, 8, 8, dtype=torch.long))
    keys = sorted(m3.compute().keys())
    print(f"\n  {len(keys)} metric keys emitted per variant:")
    for k in keys:
        print(f"    - {k}")

    print("\nAll self-tests passed.")
