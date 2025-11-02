"""
SAM-Style Prompt Encoder

Converts point coordinates to SAM-style embeddings for FastSAM3D.
This is DIFFERENT from the other prompt encoders which create dense channels.

SAM uses:
- Positional encodings for spatial location
- Learned embeddings for prompt type (positive/negative)
- Sparse representation (not dense heatmaps)
"""

import numpy as np
from typing import List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_encoders.base_encoder import BasePromptEncoder


class SAMPromptEncoder(BasePromptEncoder):
    """
    SAM-style prompt encoder.

    Unlike other encoders (Gaussian, Disk, etc.) which create dense [2, D, H, W] arrays,
    this encoder outputs POINT COORDINATES for SAM's native prompt encoding.

    This is used with FastSAM3D which has its own internal prompt encoder
    (positional embeddings + learned type embeddings).
    """

    def __init__(self):
        super().__init__(name="sam_style")

    def encode(
        self,
        positive_points: List[Tuple[int, int, int]],
        negative_points: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int],
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> dict:
        """
        Encode prompts as point coordinates and labels.

        Args:
            positive_points: List of (z, y, x) coordinates
            negative_points: List of (z, y, x) coordinates
            volume_shape: (D, H, W)
            spacing: Not used for SAM-style

        Returns:
            dict with:
                'points': np.array of shape [N, 3] with (z, y, x) coordinates
                'labels': np.array of shape [N] with 1=positive, 0=negative
        """
        # Validate points
        positive_points = self.validate_points(positive_points, volume_shape)
        negative_points = self.validate_points(negative_points, volume_shape)

        # Combine all points
        all_points = []
        all_labels = []

        # Add positive points (label = 1)
        for point in positive_points:
            all_points.append(point)
            all_labels.append(1)

        # Add negative points (label = 0)
        for point in negative_points:
            all_points.append(point)
            all_labels.append(0)

        if len(all_points) == 0:
            # No points - return empty
            return {
                'points': np.array([]).reshape(0, 3),
                'labels': np.array([])
            }

        return {
            'points': np.array(all_points, dtype=np.float32),  # [N, 3]
            'labels': np.array(all_labels, dtype=np.int64)     # [N]
        }

    def to_dense(
        self,
        points_dict: dict,
        volume_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Convert SAM-style prompts to dense channels for compatibility.

        This is useful if you want to use SAM with the standard training pipeline
        that expects [2, D, H, W] prompt channels.

        Args:
            points_dict: Output from encode()
            volume_shape: (D, H, W)

        Returns:
            prompt_channels: [2, D, H, W] binary marking
        """
        D, H, W = volume_shape

        pos_channel = np.zeros((D, H, W), dtype=np.float32)
        neg_channel = np.zeros((D, H, W), dtype=np.float32)

        points = points_dict['points']
        labels = points_dict['labels']

        for point, label in zip(points, labels):
            z, y, x = int(point[0]), int(point[1]), int(point[2])

            if label == 1:  # Positive
                pos_channel[z, y, x] = 1.0
            else:  # Negative
                neg_channel[z, y, x] = 1.0

        return np.stack([pos_channel, neg_channel], axis=0)


if __name__ == "__main__":
    # Test SAM-style encoder
    print("Testing SAMPromptEncoder")
    print('='*60)

    encoder = SAMPromptEncoder()

    # Test data
    pos_points = [(64, 64, 64), (32, 32, 32)]
    neg_points = [(96, 96, 96)]
    volume_shape = (128, 128, 128)

    # Encode as SAM-style
    print("\nEncoding as SAM-style point coordinates...")
    sam_prompts = encoder.encode(pos_points, neg_points, volume_shape)

    print(f"  Encoder: {encoder}")
    print(f"  Points shape: {sam_prompts['points'].shape}")
    print(f"  Labels shape: {sam_prompts['labels'].shape}")
    print(f"  Points:\n{sam_prompts['points']}")
    print(f"  Labels: {sam_prompts['labels']}")

    # Convert to dense for compatibility
    print("\nConverting to dense channels...")
    dense_prompts = encoder.to_dense(sam_prompts, volume_shape)

    print(f"  Dense shape: {dense_prompts.shape}")
    print(f"  Positive channel non-zeros: {np.count_nonzero(dense_prompts[0])}")
    print(f"  Negative channel non-zeros: {np.count_nonzero(dense_prompts[1])}")

    print("\n✓ SAM-style encoder works!")