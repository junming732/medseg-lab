"""
Geodesic Distance Prompt Encoder
Creates image-aware distance transforms that respect anatomical boundaries.
Based on DeepIGeoS approach.
"""

import numpy as np
from typing import List, Tuple, Optional
from .base_encoder import BasePromptEncoder

try:
    import GeodisTK as geodis
    GEODIS_AVAILABLE = True
except ImportError:
    GEODIS_AVAILABLE = False
    print("Warning: GeodisTK not available. Install with: pip install GeodisTK")


class GeodesicPromptEncoder(BasePromptEncoder):
    """
    Geodesic distance encoding: Create distance maps that respect image intensities.

    Based on DeepIGeoS (Deep Interactive Geodesic Framework).
    Geodesic distance considers image context, unlike Euclidean distance.

    Pros:
        - Image-aware (respects anatomical boundaries)
        - Better guidance along structures
        - Proven effective in DeepIGeoS paper

    Cons:
        - Computationally expensive (especially 3D)
        - Requires GeodisTK library
        - Needs image intensities as input
        - Hyperparameters need tuning (lambda, iterations)
    """

    def __init__(
        self,
        sigma: float = 5.0,
        lamb: float = 1.0,
        iterations: int = 2
    ):
        """
        Args:
            sigma: Controls falloff of distance transform (like Gaussian sigma)
            lamb: Weight for image gradient in geodesic computation
                  Higher = more sensitive to intensity changes
            iterations: Number of iterations for raster scan algorithm
        """
        super().__init__(name=f"geodesic_s{sigma}_l{lamb}")
        self.sigma = sigma
        self.lamb = lamb
        self.iterations = iterations

        if not GEODIS_AVAILABLE:
            raise ImportError(
                "GeodisTK is required for GeodesicPromptEncoder. "
                "Install it with: pip install GeodisTK"
            )

    def encode(
        self,
        positive_points: List[Tuple[int, int, int]],
        negative_points: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int],
        spacing: Optional[Tuple[float, float, float]] = None,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create geodesic distance prompt channels.

        Args:
            positive_points: List of (z, y, x) for positive clicks
            negative_points: List of (z, y, x) for negative clicks
            volume_shape: (D, H, W)
            spacing: Physical spacing (dz, dy, dx)
            image: The actual CT/MRI volume [D, H, W]. REQUIRED for geodesic.
                  If None, falls back to Euclidean distance.

        Returns:
            prompt_channels: [2, D, H, W] array
        """
        if image is None:
            print("Warning: No image provided. Falling back to Euclidean distance.")
            return self._encode_euclidean(
                positive_points, negative_points, volume_shape, spacing
            )

        D, H, W = volume_shape

        # Initialize channels
        pos_channel = np.zeros((D, H, W), dtype=np.float32)
        neg_channel = np.zeros((D, H, W), dtype=np.float32)

        # Validate points
        positive_points = self.validate_points(positive_points, volume_shape)
        negative_points = self.validate_points(negative_points, volume_shape)

        # Normalize image to [0, 255] for GeodisTK
        image_normalized = self._normalize_image(image)

        # Set spacing
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)

        # Process slice-by-slice (true 3D geodesic is very slow)
        for z_slice in range(D):
            # Get slice
            image_slice = image_normalized[z_slice]

            # Find points in this slice
            pos_in_slice = [(y, x) for z, y, x in positive_points if z == z_slice]
            neg_in_slice = [(y, x) for z, y, x in negative_points if z == z_slice]

            # Compute geodesic distance for positive points
            if len(pos_in_slice) > 0:
                geo_dist = self._compute_geodesic_2d(
                    image_slice, pos_in_slice, (H, W), (spacing[1], spacing[2])
                )
                pos_channel[z_slice] = np.exp(-geo_dist / self.sigma)

            # Compute geodesic distance for negative points
            if len(neg_in_slice) > 0:
                geo_dist = self._compute_geodesic_2d(
                    image_slice, neg_in_slice, (H, W), (spacing[1], spacing[2])
                )
                neg_channel[z_slice] = np.exp(-geo_dist / self.sigma)

        # Also add some 3D connectivity (simple Gaussian in z-direction)
        pos_channel = self._add_z_connectivity(pos_channel, positive_points, spacing[0])
        neg_channel = self._add_z_connectivity(neg_channel, negative_points, spacing[0])

        # Clip to [0, 1]
        pos_channel = np.clip(pos_channel, 0, 1)
        neg_channel = np.clip(neg_channel, 0, 1)

        return np.stack([pos_channel, neg_channel], axis=0)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 255] range for GeodisTK."""
        image = image.astype(np.float32)
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min) * 255.0
        return image.astype(np.uint8)

    def _compute_geodesic_2d(
        self,
        image_slice: np.ndarray,
        points: List[Tuple[int, int]],
        shape: Tuple[int, int],
        spacing: Tuple[float, float]
    ) -> np.ndarray:
        """
        Compute 2D geodesic distance from multiple seed points.

        Args:
            image_slice: [H, W] image slice
            points: List of (y, x) coordinates
            shape: (H, W)
            spacing: (dy, dx)

        Returns:
            geo_dist: [H, W] geodesic distance map
        """
        H, W = shape

        # Create seed mask
        seed_mask = np.zeros((H, W), dtype=np.uint8)
        for y, x in points:
            seed_mask[y, x] = 1

        # Compute geodesic distance using raster scan
        geo_dist = geodis.geodesic2d_raster_scan(
            image_slice,
            seed_mask,
            spacing,
            self.lamb,
            self.iterations
        )

        return geo_dist

    def _add_z_connectivity(
        self,
        channel: np.ndarray,
        points: List[Tuple[int, int, int]],
        z_spacing: float
    ) -> np.ndarray:
        """
        Add smooth connectivity in z-direction using Gaussian.
        This helps propagate prompt information across slices.
        """
        D = channel.shape[0]
        sigma_z = 2.0 / z_spacing  # 2 voxels in physical space

        for z_c, y_c, x_c in points:
            # Add Gaussian falloff in z-direction
            for z in range(D):
                dist_z = abs(z - z_c)
                weight = np.exp(-dist_z**2 / (2 * sigma_z**2))
                channel[z, y_c, x_c] = max(channel[z, y_c, x_c], weight)

        return channel

    def _encode_euclidean(
        self,
        positive_points: List[Tuple[int, int, int]],
        negative_points: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int],
        spacing: Optional[Tuple[float, float, float]]
    ) -> np.ndarray:
        """Fallback to Euclidean distance (basically Gaussian) if no image."""
        from .gaussian_encoder import GaussianPromptEncoder

        gaussian_encoder = GaussianPromptEncoder(sigma=self.sigma)
        return gaussian_encoder.encode(
            positive_points, negative_points, volume_shape, spacing
        )


if __name__ == "__main__":
    if GEODIS_AVAILABLE:
        print("Testing GeodesicPromptEncoder with synthetic data")
        print('='*60)

        # Create synthetic image with intensity boundary
        volume_shape = (16, 64, 64)
        image = np.ones(volume_shape, dtype=np.float32) * 100
        # Add a high-intensity region (simulating organ boundary)
        image[:, :, 32:] = 200

        encoder = GeodesicPromptEncoder(sigma=5.0, lamb=1.0)

        # Click on left side of boundary
        pos_points = [(8, 32, 20)]
        neg_points = []

        prompt_channels = encoder.encode(
            pos_points, neg_points, volume_shape, image=image
        )

        print(f"Encoder: {encoder}")
        print(f"Output shape: {prompt_channels.shape}")
        print(f"Max value: {prompt_channels[0].max():.3f}")

        # Compare values on same side vs across boundary
        print(f"\nValues in same intensity region (left of boundary):")
        print(f"  At click (8,32,20): {prompt_channels[0, 8, 32, 20]:.3f}")
        print(f"  5 pixels left (8,32,15): {prompt_channels[0, 8, 32, 15]:.3f}")

        print(f"\nValues across intensity boundary (right side):")
        print(f"  5 pixels right (8,32,25): {prompt_channels[0, 8, 32, 25]:.3f}")
        print(f"  At boundary (8,32,32): {prompt_channels[0, 8, 32, 32]:.3f}")
        print(f"  Beyond boundary (8,32,40): {prompt_channels[0, 8, 32, 40]:.3f}")

        print("\nGeod esic distance should be lower within same region!")
    else:
        print("GeodisTK not available. Skipping test.")
        print("Install with: pip install GeodisTK")