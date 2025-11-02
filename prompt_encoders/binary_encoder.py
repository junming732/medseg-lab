"""
Binary Prompt Encoder
Marks clicked points with a value of 1, all other voxels are 0.
This is the simplest approach - single voxel marking.
"""

import numpy as np
from typing import List, Tuple, Optional
from .base_encoder import BasePromptEncoder


class BinaryPromptEncoder(BasePromptEncoder):
    """
    Binary marking: Set clicked voxel to 1, everything else to 0.

    This is your colleague's approach - very simple and sparse.

    Pros:
        - Extremely simple
        - No hyperparameters
        - Exact spatial location

    Cons:
        - Very sparse signal (only 1 voxel per click)
        - Weak gradients for backprop
        - Hard for network to "see" the prompt
    """

    def __init__(self):
        super().__init__(name="binary")

    def encode(
        self,
        positive_points: List[Tuple[int, int, int]],
        negative_points: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int],
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """
        Create binary prompt channels.

        Args:
            positive_points: List of (z, y, x) for positive clicks
            negative_points: List of (z, y, x) for negative clicks
            volume_shape: (D, H, W)
            spacing: Not used in binary encoding

        Returns:
            prompt_channels: [2, D, H, W] array
        """
        D, H, W = volume_shape

        # Initialize channels
        pos_channel = np.zeros((D, H, W), dtype=np.float32)
        neg_channel = np.zeros((D, H, W), dtype=np.float32)

        # Validate points
        positive_points = self.validate_points(positive_points, volume_shape)
        negative_points = self.validate_points(negative_points, volume_shape)

        # Mark positive points
        for z, y, x in positive_points:
            pos_channel[z, y, x] = 1.0

        # Mark negative points
        for z, y, x in negative_points:
            neg_channel[z, y, x] = 1.0

        return np.stack([pos_channel, neg_channel], axis=0)


if __name__ == "__main__":
    # Test the encoder
    encoder = BinaryPromptEncoder()

    # Example usage
    pos_points = [(10, 20, 30), (15, 25, 35)]
    neg_points = [(5, 10, 15)]
    volume_shape = (32, 64, 64)

    prompt_channels = encoder.encode(pos_points, neg_points, volume_shape)

    print(f"Encoder: {encoder}")
    print(f"Output shape: {prompt_channels.shape}")
    print(f"Positive channel non-zero voxels: {np.count_nonzero(prompt_channels[0])}")
    print(f"Negative channel non-zero voxels: {np.count_nonzero(prompt_channels[1])}")
    print(f"Max value in positive channel: {prompt_channels[0].max()}")