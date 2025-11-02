"""
Loss Functions for Interactive Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    """

    def __init__(self, smooth=1e-5, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, D, H, W] raw network output
            targets: [B, D, H, W] ground truth labels

        Returns:
            dice_loss: scalar loss value
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        # Compute Dice for each class
        dice_scores = []
        for c in range(num_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]

            # Flatten spatial dimensions
            pred_flat = pred_c.reshape(pred_c.shape[0], -1)
            target_flat = target_c.reshape(target_c.shape[0], -1)

            # Compute Dice
            intersection = (pred_flat * target_flat).sum(dim=1)
            cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_scores.append(dice)

        # Average over classes and batch
        dice_scores = torch.stack(dice_scores, dim=1)
        dice_loss = 1.0 - dice_scores.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, D, H, W]
            targets: [B, D, H, W]
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction='none',
            ignore_index=self.ignore_index
        )

        # Get probabilities
        probs = F.softmax(logits, dim=1)

        # Gather probabilities for target class
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        pt = (probs * targets_one_hot).sum(dim=1)

        # Apply focal term
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha = torch.ones(logits.shape[1]) * self.alpha
            else:
                alpha = torch.tensor(self.alpha)

            alpha = alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        # Compute final loss
        focal_loss = (focal_weight * ce_loss).mean()

        return focal_loss


class CombinedLoss(nn.Module):
    """
    Combination of Dice Loss and Cross Entropy Loss.
    """

    def __init__(
        self,
        dice_weight=0.5,
        ce_weight=0.5,
        num_classes=3,
        ignore_index=-100
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes

        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, D, H, W]
            targets: [B, D, H, W]
        """
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)

        total_loss = self.dice_weight * dice + self.ce_weight * ce

        return total_loss


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions")
    print('='*60)

    # Create dummy data
    batch_size = 2
    num_classes = 3
    D, H, W = 16, 32, 32

    logits = torch.randn(batch_size, num_classes, D, H, W)
    targets = torch.randint(0, num_classes, (batch_size, D, H, W))

    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")

    # Test Dice Loss
    print("\nTesting DiceLoss...")
    dice_loss = DiceLoss()
    loss = dice_loss(logits, targets)
    print(f"Dice loss: {loss.item():.4f}")

    # Test Focal Loss
    print("\nTesting FocalLoss...")
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")

    # Test Combined Loss
    print("\nTesting CombinedLoss...")
    combined_loss = CombinedLoss(dice_weight=0.5, ce_weight=0.5, num_classes=3)
    loss = combined_loss(logits, targets)
    print(f"Combined loss: {loss.item():.4f}")

    # Test backward pass
    print("\nTesting backward pass...")
    loss.backward()
    print("Backward pass successful!")