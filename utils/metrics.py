"""
Evaluation Metrics for Segmentation
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_dice_score(pred, target, num_classes, epsilon=1e-6):
    """
    Compute Dice score for multi-class segmentation.

    Args:
        pred: [B, D, H, W] predicted labels
        target: [B, D, H, W] ground truth labels
        num_classes: Number of classes
        epsilon: Small constant for numerical stability

    Returns:
        mean_dice: Average Dice score across all classes (excluding background)
    """
    dice_scores = []

    for c in range(1, num_classes):  # Skip background (class 0)
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        cardinality = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + epsilon) / (cardinality + epsilon)
        dice_scores.append(dice.item())

    mean_dice = np.mean(dice_scores) if len(dice_scores) > 0 else 0.0

    return mean_dice


def compute_iou(pred, target, num_classes, epsilon=1e-6):
    """
    Compute Intersection over Union (IoU) for multi-class segmentation.

    Args:
        pred: [B, D, H, W] predicted labels
        target: [B, D, H, W] ground truth labels
        num_classes: Number of classes

    Returns:
        mean_iou: Average IoU across all classes (excluding background)
    """
    iou_scores = []

    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection

        iou = (intersection + epsilon) / (union + epsilon)
        iou_scores.append(iou.item())

    mean_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0.0

    return mean_iou


def compute_hausdorff_distance(pred, target, spacing=None, percentile=95):
    """
    Compute Hausdorff Distance between predicted and target masks.

    Args:
        pred: [D, H, W] numpy array, binary mask
        target: [D, H, W] numpy array, binary mask
        spacing: Physical spacing (dz, dy, dx)
        percentile: Percentile for robust HD (default 95 for HD95)

    Returns:
        hd: Hausdorff distance
    """
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    # Get surface points
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)

    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return np.inf

    # Compute distance transforms
    pred_dist = distance_transform_edt(~pred_surface, sampling=spacing)
    target_dist = distance_transform_edt(~target_surface, sampling=spacing)

    # Get distances from surfaces
    pred_to_target = target_dist[pred_surface > 0]
    target_to_pred = pred_dist[target_surface > 0]

    # Compute HD at percentile
    if len(pred_to_target) > 0 and len(target_to_pred) > 0:
        hd = max(
            np.percentile(pred_to_target, percentile),
            np.percentile(target_to_pred, percentile)
        )
    else:
        hd = np.inf

    return hd


def get_surface_points(mask):
    """
    Extract surface points from binary mask using morphological operations.

    Args:
        mask: [D, H, W] binary numpy array

    Returns:
        surface: [D, H, W] binary array with surface points
    """
    from scipy.ndimage import binary_erosion

    # Surface = mask - eroded mask
    eroded = binary_erosion(mask)
    surface = mask & (~eroded)

    return surface


def compute_average_surface_distance(pred, target, spacing=None):
    """
    Compute Average Surface Distance (ASD) between predicted and target masks.

    Args:
        pred: [D, H, W] numpy array, binary mask
        target: [D, H, W] numpy array, binary mask
        spacing: Physical spacing (dz, dy, dx)

    Returns:
        asd: Average surface distance
    """
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    # Get surface points
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)

    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return np.inf

    # Compute distance transforms
    pred_dist = distance_transform_edt(~pred_surface, sampling=spacing)
    target_dist = distance_transform_edt(~target_surface, sampling=spacing)

    # Get distances from surfaces
    pred_to_target = target_dist[pred_surface > 0]
    target_to_pred = pred_dist[target_surface > 0]

    # Compute average
    asd = (pred_to_target.sum() + target_to_pred.sum()) / \
          (len(pred_to_target) + len(target_to_pred))

    return asd


class MetricsCalculator:
    """
    Calculator for all segmentation metrics.
    """

    def __init__(self, num_classes, spacing=None):
        self.num_classes = num_classes
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)

    def compute_all_metrics(self, pred, target):
        """
        Compute all metrics for a single sample.

        Args:
            pred: [D, H, W] predicted labels (numpy or torch)
            target: [D, H, W] ground truth labels (numpy or torch)

        Returns:
            metrics: Dictionary of metric values
        """
        # Convert to numpy if needed
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        metrics = {}

        # Compute metrics for each foreground class
        for c in range(1, self.num_classes):
            pred_c = (pred == c).astype(np.uint8)
            target_c = (target == c).astype(np.uint8)

            # Skip if no ground truth for this class
            if target_c.sum() == 0:
                continue

            # Dice
            intersection = (pred_c * target_c).sum()
            cardinality = pred_c.sum() + target_c.sum()
            dice = 2.0 * intersection / (cardinality + 1e-6)
            metrics[f'dice_class{c}'] = dice

            # IoU
            union = cardinality - intersection
            iou = intersection / (union + 1e-6)
            metrics[f'iou_class{c}'] = iou

            # HD95
            try:
                hd95 = compute_hausdorff_distance(
                    pred_c, target_c, self.spacing, percentile=95
                )
                metrics[f'hd95_class{c}'] = hd95
            except:
                metrics[f'hd95_class{c}'] = np.inf

            # ASD
            try:
                asd = compute_average_surface_distance(
                    pred_c, target_c, self.spacing
                )
                metrics[f'asd_class{c}'] = asd
            except:
                metrics[f'asd_class{c}'] = np.inf

        # Compute mean metrics
        dice_values = [v for k, v in metrics.items() if 'dice' in k]
        if dice_values:
            metrics['mean_dice'] = np.mean(dice_values)

        return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing segmentation metrics")
    print('='*60)

    # Create dummy predictions and targets
    D, H, W = 32, 64, 64
    num_classes = 3

    pred = torch.randint(0, num_classes, (1, D, H, W))
    target = torch.randint(0, num_classes, (1, D, H, W))

    print(f"Pred shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Pred classes: {torch.unique(pred)}")
    print(f"Target classes: {torch.unique(target)}")

    # Test Dice score
    print("\nTesting Dice score...")
    dice = compute_dice_score(pred, target, num_classes)
    print(f"Mean Dice: {dice:.4f}")

    # Test IoU
    print("\nTesting IoU...")
    iou = compute_iou(pred, target, num_classes)
    print(f"Mean IoU: {iou:.4f}")

    # Test full metrics calculator
    print("\nTesting MetricsCalculator...")
    calculator = MetricsCalculator(num_classes=num_classes)
    metrics = calculator.compute_all_metrics(pred[0], target[0])

    print("All metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")