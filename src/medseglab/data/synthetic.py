import torch
import numpy as np
from torch.utils.data import Dataset


class SyntheticVolumetricSeg(Dataset):
    """
    Synthetic 3D volumetric segmentation dataset for testing/debugging.
    Generates random 3D volumes with simple geometric shapes as segmentation targets.
    """

    def __init__(self, num_samples: int, img_size: tuple = (32, 32, 32), seed: int = 42):
        """
        Args:
            num_samples: Number of samples in the dataset
            img_size: Tuple of (D, H, W) for 3D volume dimensions
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Generate a synthetic 3D volume and corresponding segmentation mask.

        Returns:
            image: Tensor of shape (1, D, H, W) - single channel 3D volume
            label: Tensor of shape (1, D, H, W) - binary segmentation mask
        """
        # Use deterministic generation based on idx and seed
        sample_rng = np.random.RandomState(self.seed + idx)

        D, H, W = self.img_size

        # Create base image with random noise
        image = sample_rng.randn(D, H, W).astype(np.float32) * 0.1

        # Create label mask
        label = np.zeros((D, H, W), dtype=np.float32)

        # Add a random sphere/ellipsoid as the segmentation target
        center_d = sample_rng.randint(D // 4, 3 * D // 4)
        center_h = sample_rng.randint(H // 4, 3 * H // 4)
        center_w = sample_rng.randint(W // 4, 3 * W // 4)

        radius_d = sample_rng.randint(D // 8, D // 4)
        radius_h = sample_rng.randint(H // 8, H // 4)
        radius_w = sample_rng.randint(W // 8, W // 4)

        # Create coordinate grids
        d_coords, h_coords, w_coords = np.ogrid[:D, :H, :W]

        # Ellipsoid equation
        mask = (
            ((d_coords - center_d) / radius_d) ** 2 +
            ((h_coords - center_h) / radius_h) ** 2 +
            ((w_coords - center_w) / radius_w) ** 2
        ) <= 1.0

        label[mask] = 1.0

        # Add intensity to image where the object is
        image[mask] += sample_rng.randn() * 0.5 + 1.0

        # Add some Gaussian blur effect for realism (simple convolution)
        image = self._simple_blur(image, sample_rng)

        # Convert to torch tensors and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0)  # (1, D, H, W)
        label = torch.from_numpy(label).unsqueeze(0)  # (1, D, H, W)

        return image, label

    def _simple_blur(self, volume, rng):
        """Apply simple 3D smoothing"""
        from scipy.ndimage import gaussian_filter
        sigma = rng.uniform(0.5, 1.5)
        return gaussian_filter(volume, sigma=sigma)