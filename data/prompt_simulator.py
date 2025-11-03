"""
Prompt Simulator

Simulates user clicks during training by sampling points from ground truth masks.
This allows training without actual user interaction.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage


class PromptSimulator:
    """
    Simulates user clicks for training interactive segmentation models.

    Generates positive clicks (on foreground) and negative clicks (on background)
    by sampling from ground truth masks with various strategies.
    """

    def __init__(
        self,
        n_points_range: Tuple[int, int] = (1, 5),
        positive_ratio: float = 0.7,
        error_simulation: bool = True,
        noise_std: float = 2.0,
        sampling_strategy: str = 'random'
    ):
        """
        Args:
            n_points_range: (min, max) number of points to sample
            positive_ratio: Ratio of positive to total points
            error_simulation: Whether to add spatial noise to simulate user error
            noise_std: Standard deviation of Gaussian noise (in voxels)
            sampling_strategy: How to sample points:
                - 'random': Random sampling from foreground/background
                - 'boundary': Sample near object boundaries
                - 'centroid': Sample near object centroids
                - 'mixed': Combination of strategies
        """
        self.n_points_range = n_points_range
        self.positive_ratio = positive_ratio
        self.error_simulation = error_simulation
        self.noise_std = noise_std
        self.sampling_strategy = sampling_strategy

    def simulate(
        self,
        mask: np.ndarray,
        num_classes: int = 2
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """
        Simulate user clicks for a given ground truth mask.

        Args:
            mask: Ground truth mask [D, H, W] with integer labels
                  0 = background, 1+ = foreground classes
            num_classes: Total number of classes (including background)

        Returns:
            positive_points: List of (z, y, x) coordinates for positive clicks
            negative_points: List of (z, y, x) coordinates for negative clicks
        """
        # Determine total number of points
        n_total = np.random.randint(self.n_points_range[0], self.n_points_range[1] + 1)
        n_positive = max(1, int(n_total * self.positive_ratio))
        n_negative = n_total - n_positive

        # Get foreground and background coordinates
        fg_coords = np.where(mask > 0)
        bg_coords = np.where(mask == 0)

        # Sample points based on strategy
        if self.sampling_strategy == 'random':
            positive_points = self._sample_random(fg_coords, n_positive)
            negative_points = self._sample_random(bg_coords, n_negative)

        elif self.sampling_strategy == 'boundary':
            positive_points = self._sample_boundary(mask, n_positive, foreground=True)
            negative_points = self._sample_boundary(mask, n_negative, foreground=False)

        elif self.sampling_strategy == 'centroid':
            positive_points = self._sample_centroid(mask, n_positive)
            negative_points = self._sample_random(bg_coords, n_negative)

        elif self.sampling_strategy == 'mixed':
            # Mix of strategies
            n_pos_random = n_positive // 2
            n_pos_boundary = n_positive - n_pos_random

            pos_random = self._sample_random(fg_coords, n_pos_random)
            pos_boundary = self._sample_boundary(mask, n_pos_boundary, foreground=True)
            positive_points = pos_random + pos_boundary

            negative_points = self._sample_random(bg_coords, n_negative)

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        # Add noise to simulate user error
        if self.error_simulation:
            positive_points = self._add_noise(positive_points, mask.shape)
            negative_points = self._add_noise(negative_points, mask.shape)

        return positive_points, negative_points

    def _sample_random(
        self,
        coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
        n_points: int
    ) -> List[Tuple[int, int, int]]:
        """Randomly sample points from given coordinates."""
        if len(coords[0]) == 0:
            return []

        n_available = len(coords[0])
        n_sample = min(n_points, n_available)

        indices = np.random.choice(n_available, n_sample, replace=False)

        points = [
            (int(coords[0][i]), int(coords[1][i]), int(coords[2][i]))
            for i in indices
        ]

        return points

    def _sample_boundary(
        self,
        mask: np.ndarray,
        n_points: int,
        foreground: bool = True
    ) -> List[Tuple[int, int, int]]:
        """
        Sample points near object boundaries.
        This simulates users clicking to refine boundaries.
        """
        # Compute boundary using morphological operations
        from scipy.ndimage import binary_dilation, binary_erosion

        if foreground:
            # Boundary = dilation - original
            dilated = binary_dilation(mask > 0)
            boundary = dilated & (mask == 0)
        else:
            # Boundary = original - erosion
            eroded = binary_erosion(mask > 0)
            boundary = (mask > 0) & (~eroded)

        # Get boundary coordinates
        boundary_coords = np.where(boundary)

        if len(boundary_coords[0]) == 0:
            # Fallback to random if no boundary found
            if foreground:
                coords = np.where(mask > 0)
            else:
                coords = np.where(mask == 0)
            return self._sample_random(coords, n_points)

        return self._sample_random(boundary_coords, n_points)

    def _sample_centroid(
        self,
        mask: np.ndarray,
        n_points: int
    ) -> List[Tuple[int, int, int]]:
        """
        Sample points near object centroids.
        This simulates users clicking in the middle of objects first.
        """
        # Find connected components
        labeled_mask, n_objects = ndimage.label(mask > 0)

        points = []

        for obj_id in range(1, min(n_objects + 1, n_points + 1)):
            # Find centroid of this object
            obj_coords = np.where(labeled_mask == obj_id)

            if len(obj_coords[0]) > 0:
                centroid_z = int(np.mean(obj_coords[0]))
                centroid_y = int(np.mean(obj_coords[1]))
                centroid_x = int(np.mean(obj_coords[2]))

                points.append((centroid_z, centroid_y, centroid_x))

        # If we need more points, sample randomly
        if len(points) < n_points:
            fg_coords = np.where(mask > 0)
            additional = self._sample_random(fg_coords, n_points - len(points))
            points.extend(additional)

        return points[:n_points]

    def _add_noise(
        self,
        points: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """
        Add Gaussian noise to point coordinates to simulate user error.

        Args:
            points: List of (z, y, x) coordinates
            volume_shape: (D, H, W) to clip coordinates

        Returns:
            noisy_points: Points with added noise
        """
        D, H, W = volume_shape
        noisy_points = []

        for z, y, x in points:
            # Add Gaussian noise
            noise_z = np.random.normal(0, self.noise_std)
            noise_y = np.random.normal(0, self.noise_std)
            noise_x = np.random.normal(0, self.noise_std)

            # Apply noise and clip to volume bounds
            z_new = int(np.clip(z + noise_z, 0, D - 1))
            y_new = int(np.clip(y + noise_y, 0, H - 1))
            x_new = int(np.clip(x + noise_x, 0, W - 1))

            noisy_points.append((z_new, y_new, x_new))

        return noisy_points


if __name__ == "__main__":
    # Test the prompt simulator
    print("Testing PromptSimulator")
    print('='*60)

    # Create synthetic mask
    mask = np.zeros((32, 64, 64), dtype=np.int32)
    # Add a sphere in the center
    for z in range(12, 20):
        for y in range(28, 36):
            for x in range(28, 36):
                if (z-16)**2 + (y-32)**2 + (x-32)**2 < 5**2:
                    mask[z, y, x] = 1

    print(f"Mask shape: {mask.shape}")
    print(f"Foreground voxels: {np.sum(mask > 0)}")

    # Test different sampling strategies
    for strategy in ['random', 'boundary', 'centroid', 'mixed']:
        print(f"\nTesting strategy: {strategy}")
        simulator = PromptSimulator(
            n_points_range=(3, 5),
            positive_ratio=0.7,
            error_simulation=True,
            noise_std=2.0,
            sampling_strategy=strategy
        )

        pos_points, neg_points = simulator.simulate(mask)

        print(f"  Positive points: {len(pos_points)}")
        print(f"    Examples: {pos_points[:2]}")
        print(f"  Negative points: {len(neg_points)}")
        print(f"    Examples: {neg_points[:2]}")