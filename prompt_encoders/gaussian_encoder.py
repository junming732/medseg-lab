"""
Gaussian Heatmap Prompt Encoder
Creates smooth 3D Gaussian distributions centered at clicked points.
This is the most common approach in interactive segmentation literature.
"""

import numpy as np
from typing import List, Tuple, Optional
from .base_encoder import BasePromptEncoder


class GaussianPromptEncoder(BasePromptEncoder):
    """
    Gaussian heatmap encoding: Create 3D Gaussian centered at each click.

    This is the standard approach in most interactive segmentation papers.

    Pros:
        - Smooth gradients for optimization
        - Adjustable spatial extent via sigma
        - Widely validated in literature
        - Works well with CNNs

    Cons:
        - Requires tuning sigma hyperparameter
        - Not image-aware (doesn't respect boundaries)
        - Can be computationally expensive for large sigma
    """

    def __init__(self, sigma: float = 3.0):
        """
        Args:
            sigma: Standard deviation of Gaussian in voxels.
                   Typical values: 2-5 voxels
                   Larger sigma = wider influence
        """
        super().__init__(name=f"gaussian_sigma{sigma}")
        self.sigma = sigma

    def encode(
        self,
        positive_points: List[Tuple[int, int, int]],
        negative_points: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int],
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """
        Create Gaussian heatmap prompt channels.

        Args:
            positive_points: List of (z, y, x) for positive clicks
            negative_points: List of (z, y, x) for negative clicks
            volume_shape: (D, H, W)
            spacing: Physical spacing (dz, dy, dx). If provided, sigma is
                    interpreted in physical units. Otherwise in voxels.

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

        # Adjust sigma for physical spacing if provided
        if spacing is not None:
            sigma_z = self.sigma / spacing[0]
            sigma_y = self.sigma / spacing[1]
            sigma_x = self.sigma / spacing[2]
        else:
            sigma_z = sigma_y = sigma_x = self.sigma

        # Add Gaussian for each positive point
        for z_c, y_c, x_c in positive_points:
            gaussian = self._create_gaussian_3d(
                center=(z_c, y_c, x_c),
                volume_shape=volume_shape,
                sigma=(sigma_z, sigma_y, sigma_x)
            )
            pos_channel += gaussian

        # Add Gaussian for each negative point
        for z_c, y_c, x_c in negative_points:
            gaussian = self._create_gaussian_3d(
                center=(z_c, y_c, x_c),
                volume_shape=volume_shape,
                sigma=(sigma_z, sigma_y, sigma_x)
            )
            neg_channel += gaussian

        # Clip to [0, 1] range (in case of overlapping Gaussians)
        pos_channel = np.clip(pos_channel, 0, 1)
        neg_channel = np.clip(neg_channel, 0, 1)

        return np.stack([pos_channel, neg_channel], axis=0)

    def _create_gaussian_3d(
        self,
        center: Tuple[int, int, int],
        volume_shape: Tuple[int, int, int],
        sigma: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Create a 3D Gaussian centered at the given point.

        Args:
            center: (z_c, y_c, x_c) center coordinates
            volume_shape: (D, H, W)
            sigma: (sigma_z, sigma_y, sigma_x) standard deviations

        Returns:
            gaussian: [D, H, W] array with Gaussian distribution
        """
        D, H, W = volume_shape
        z_c, y_c, x_c = center
        sigma_z, sigma_y, sigma_x = sigma

        # Create coordinate grids
        z = np.arange(D, dtype=np.float32)
        y = np.arange(H, dtype=np.float32)
        x = np.arange(W, dtype=np.float32)

        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

        # Compute 3D Gaussian
        # G(x,y,z) = exp(-((x-xc)^2/2σx^2 + (y-yc)^2/2σy^2 + (z-zc)^2/2σz^2))
        gaussian = np.exp(
            -((Z - z_c) ** 2 / (2 * sigma_z ** 2) +
              (Y - y_c) ** 2 / (2 * sigma_y ** 2) +
              (X - x_c) ** 2 / (2 * sigma_x ** 2))
        )

        return gaussian


if __name__ == "__main__":
    # Test the encoder with different sigmas
    for sigma in [2.0, 3.0, 5.0]:
        print(f"\n{'='*60}")
        print(f"Testing GaussianPromptEncoder with sigma={sigma}")
        print('='*60)

        encoder = GaussianPromptEncoder(sigma=sigma)

        # Example usage
        pos_points = [(16, 32, 32)]  # One point in center
        neg_points = []
        volume_shape = (32, 64, 64)

        prompt_channels = encoder.encode(pos_points, neg_points, volume_shape)

        print(f"Encoder: {encoder}")
        print(f"Output shape: {prompt_channels.shape}")
        print(f"Positive channel non-zero voxels (>0.01): {np.sum(prompt_channels[0] > 0.01)}")
        print(f"Max value: {prompt_channels[0].max():.3f}")
        print(f"Value at center: {prompt_channels[0, 16, 32, 32]:.3f}")
        print(f"Value at 1 voxel away: {prompt_channels[0, 16, 32, 33]:.3f}")
        print(f"Value at 3 voxels away: {prompt_channels[0, 16, 32, 35]:.3f}")