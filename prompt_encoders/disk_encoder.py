"""
Disk/Sphere Prompt Encoder
Creates binary spheres with fixed radius around clicked points.
Based on RITM's approach but extended to 3D.
"""

import numpy as np
from typing import List, Tuple, Optional
from .base_encoder import BasePromptEncoder


class DiskPromptEncoder(BasePromptEncoder):
    """
    Disk/Sphere encoding: Create binary sphere with fixed radius at each click.

    Inspired by RITM (Reviving Iterative Training with Mask guidance).
    RITM used 5-pixel radius disks for 2D images.

    Pros:
        - Simple and fast (just geometric computation)
        - Stronger signal than binary marking
        - No need to tune parameters after radius is set
        - Proven effective in RITM

    Cons:
        - Hard boundaries (binary, not smooth)
        - Fixed radius doesn't adapt to image scale
        - Not image-aware
    """

    def __init__(self, radius: int = 3):
        """
        Args:
            radius: Radius of the sphere in voxels.
                   Typical values: 3-5 voxels
                   RITM used radius=5 for 2D natural images
        """
        super().__init__(name=f"disk_r{radius}")
        self.radius = radius

    def encode(
        self,
        positive_points: List[Tuple[int, int, int]],
        negative_points: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int],
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """
        Create disk/sphere prompt channels.

        Args:
            positive_points: List of (z, y, x) for positive clicks
            negative_points: List of (z, y, x) for negative clicks
            volume_shape: (D, H, W)
            spacing: Physical spacing (dz, dy, dx). If provided, radius is
                    adjusted per axis to create physically accurate sphere.

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

        # Adjust radius for physical spacing if provided
        if spacing is not None:
            radius_z = self.radius / spacing[0]
            radius_y = self.radius / spacing[1]
            radius_x = self.radius / spacing[2]
        else:
            radius_z = radius_y = radius_x = self.radius

        # Add sphere for each positive point
        for z_c, y_c, x_c in positive_points:
            sphere = self._create_sphere_3d(
                center=(z_c, y_c, x_c),
                volume_shape=volume_shape,
                radius=(radius_z, radius_y, radius_x)
            )
            pos_channel = np.maximum(pos_channel, sphere)  # Union of spheres

        # Add sphere for each negative point
        for z_c, y_c, x_c in negative_points:
            sphere = self._create_sphere_3d(
                center=(z_c, y_c, x_c),
                volume_shape=volume_shape,
                radius=(radius_z, radius_y, radius_x)
            )
            neg_channel = np.maximum(neg_channel, sphere)

        return np.stack([pos_channel, neg_channel], axis=0)

    def _create_sphere_3d(
        self,
        center: Tuple[int, int, int],
        volume_shape: Tuple[int, int, int],
        radius: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Create a binary 3D sphere centered at the given point.

        Args:
            center: (z_c, y_c, x_c) center coordinates
            volume_shape: (D, H, W)
            radius: (radius_z, radius_y, radius_x)

        Returns:
            sphere: [D, H, W] binary array (0s and 1s)
        """
        D, H, W = volume_shape
        z_c, y_c, x_c = center
        r_z, r_y, r_x = radius

        # Determine bounding box for efficiency
        z_min = max(0, int(z_c - r_z - 1))
        z_max = min(D, int(z_c + r_z + 2))
        y_min = max(0, int(y_c - r_y - 1))
        y_max = min(H, int(y_c + r_y + 2))
        x_min = max(0, int(x_c - r_x - 1))
        x_max = min(W, int(x_c + r_x + 2))

        # Create sphere only in bounding box
        sphere = np.zeros((D, H, W), dtype=np.float32)

        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # Check if point is inside ellipsoid
                    dist_sq = ((z - z_c) / r_z) ** 2 + \
                             ((y - y_c) / r_y) ** 2 + \
                             ((x - x_c) / r_x) ** 2

                    if dist_sq <= 1.0:
                        sphere[z, y, x] = 1.0

        return sphere


if __name__ == "__main__":
    # Test the encoder with different radii
    for radius in [2, 3, 5]:
        print(f"\n{'='*60}")
        print(f"Testing DiskPromptEncoder with radius={radius}")
        print('='*60)

        encoder = DiskPromptEncoder(radius=radius)

        # Example usage
        pos_points = [(16, 32, 32)]  # One point in center
        neg_points = []
        volume_shape = (32, 64, 64)

        prompt_channels = encoder.encode(pos_points, neg_points, volume_shape)

        print(f"Encoder: {encoder}")
        print(f"Output shape: {prompt_channels.shape}")
        print(f"Positive channel non-zero voxels: {np.count_nonzero(prompt_channels[0])}")
        print(f"Max value: {prompt_channels[0].max():.3f}")
        print(f"Theoretical sphere volume: {4/3 * np.pi * radius**3:.1f} voxels")

        # Check a few specific points
        print(f"Value at center (16,32,32): {prompt_channels[0, 16, 32, 32]:.1f}")
        print(f"Value at radius distance (16,32,{32+radius}): {prompt_channels[0, 16, 32, 32+radius]:.1f}")
        if radius > 1:
            print(f"Value just outside (16,32,{32+radius+1}): {prompt_channels[0, 16, 32, 32+radius+1]:.1f}")