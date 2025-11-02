"""
LiTS Interactive Dataset

Dataset loader for Liver Tumor Segmentation (LiTS) Challenge
with interactive prompts for training.
"""

import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_encoders import get_encoder
from data.prompt_simulator import PromptSimulator


class LiTSInteractiveDataset(Dataset):
    """
    LiTS dataset with simulated interactive prompts.

    The LiTS challenge contains CT scans of the abdomen with annotations for:
        - Class 0: Background
        - Class 1: Liver
        - Class 2: Liver tumor

    During training, this dataset simulates user clicks on the liver/tumor
    to create interactive prompt channels.
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        encoder_type: str = 'gaussian',
        encoder_kwargs: Optional[Dict] = None,
        simulator_kwargs: Optional[Dict] = None,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        num_classes: int = 3,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data_root: Path to LiTS dataset root
                      Expected structure:
                      data_root/
                          volume-0.nii
                          segmentation-0.nii
                          volume-1.nii
                          segmentation-1.nii
                          ...
            split: 'train' or 'val'
            encoder_type: Type of prompt encoder ('binary', 'gaussian', 'disk', 'geodesic')
            encoder_kwargs: Arguments for prompt encoder
            simulator_kwargs: Arguments for prompt simulator
            patch_size: Size of 3D patches to extract
            num_classes: Number of segmentation classes
            transform: Additional data transformations
        """
        self.data_root = data_root
        self.split = split
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.transform = transform

        # Setup prompt encoder
        if encoder_kwargs is None:
            encoder_kwargs = {'sigma': 3.0} if encoder_type == 'gaussian' else {}
        self.prompt_encoder = get_encoder(encoder_type, **encoder_kwargs)

        # Setup prompt simulator
        if simulator_kwargs is None:
            simulator_kwargs = {
                'n_points_range': (1, 5),
                'positive_ratio': 0.7,
                'sampling_strategy': 'mixed'
            }
        self.prompt_simulator = PromptSimulator(**simulator_kwargs)

        # Get file list
        self.samples = self._get_file_list()

        print(f"Loaded {len(self.samples)} {split} samples")
        print(f"Using prompt encoder: {self.prompt_encoder}")

    def _get_file_list(self):
        """Get list of volume and segmentation file pairs."""
        all_volumes = sorted([
            f for f in os.listdir(self.data_root)
            if f.startswith('volume-') and f.endswith('.nii')
        ])

        samples = []
        for vol_file in all_volumes:
            # Extract ID
            vol_id = vol_file.replace('volume-', '').replace('.nii', '')
            seg_file = f'segmentation-{vol_id}.nii'

            vol_path = os.path.join(self.data_root, vol_file)
            seg_path = os.path.join(self.data_root, seg_file)

            if os.path.exists(seg_path):
                samples.append({
                    'volume': vol_path,
                    'segmentation': seg_path,
                    'id': vol_id
                })

        # Split into train/val (simple 80/20 split)
        n_train = int(0.8 * len(samples))
        if self.split == 'train':
            return samples[:n_train]
        else:
            return samples[n_train:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a training sample with interactive prompts.

        Returns:
            dict with keys:
                - 'image': [1, D, H, W] CT volume patch
                - 'prompts': [2, D, H, W] encoded prompt channels
                - 'mask': [D, H, W] ground truth segmentation
                - 'image_full': Full CT volume (for geodesic encoder)
        """
        sample_info = self.samples[idx]

        # Load volume and segmentation
        volume = self._load_nifti(sample_info['volume'])
        segmentation = self._load_nifti(sample_info['segmentation'])

        # Normalize volume to [0, 1]
        volume = self._normalize_volume(volume)

        # Extract random patch
        patch_data = self._extract_random_patch(volume, segmentation)
        image_patch = patch_data['image']
        mask_patch = patch_data['mask']

        # Simulate user clicks on this patch
        pos_points, neg_points = self.prompt_simulator.simulate(mask_patch)

        # Encode prompts
        # For geodesic encoder, we need the image
        if isinstance(self.prompt_encoder.__class__.__name__, str) and \
           'geodesic' in self.prompt_encoder.__class__.__name__.lower():
            prompt_channels = self.prompt_encoder.encode(
                pos_points, neg_points, mask_patch.shape, image=image_patch[0]
            )
        else:
            prompt_channels = self.prompt_encoder.encode(
                pos_points, neg_points, mask_patch.shape
            )

        # Apply additional transforms if specified
        if self.transform is not None:
            image_patch, prompt_channels, mask_patch = self.transform(
                image_patch, prompt_channels, mask_patch
            )

        return {
            'image': image_patch.astype(np.float32),
            'prompts': prompt_channels.astype(np.float32),
            'mask': mask_patch.astype(np.int64),
            'pos_points': pos_points,
            'neg_points': neg_points,
            'sample_id': sample_info['id']
        }

    def _load_nifti(self, filepath: str) -> np.ndarray:
        """Load NIfTI file and return numpy array."""
        nii = nib.load(filepath)
        data = nii.get_fdata()
        return data

    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize CT volume using window leveling.
        Standard abdomen window: [-150, 250] HU
        """
        volume = np.clip(volume, -150, 250)
        volume = (volume - (-150)) / (250 - (-150))
        return volume

    def _extract_random_patch(
        self,
        volume: np.ndarray,
        segmentation: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract random 3D patch from volume.
        Try to sample patches that contain foreground.
        """
        D, H, W = volume.shape
        pd, ph, pw = self.patch_size

        # Find regions with foreground
        fg_coords = np.where(segmentation > 0)

        if len(fg_coords[0]) > 0 and np.random.rand() < 0.8:
            # Sample patch containing foreground (80% of time)
            idx = np.random.randint(len(fg_coords[0]))
            center_z = fg_coords[0][idx]
            center_y = fg_coords[1][idx]
            center_x = fg_coords[2][idx]

            # Get patch bounds around this center
            start_z = max(0, center_z - pd // 2)
            start_y = max(0, center_y - ph // 2)
            start_x = max(0, center_x - pw // 2)
        else:
            # Random patch (20% of time for variety)
            start_z = np.random.randint(0, max(1, D - pd))
            start_y = np.random.randint(0, max(1, H - ph))
            start_x = np.random.randint(0, max(1, W - pw))

        # Ensure we don't go out of bounds
        start_z = min(start_z, D - pd)
        start_y = min(start_y, H - ph)
        start_x = min(start_x, W - pw)

        # Extract patches
        image_patch = volume[
            start_z:start_z + pd,
            start_y:start_y + ph,
            start_x:start_x + pw
        ]
        mask_patch = segmentation[
            start_z:start_z + pd,
            start_y:start_y + ph,
            start_x:start_x + pw
        ]

        # Add channel dimension to image
        image_patch = image_patch[np.newaxis, ...]  # [1, D, H, W]

        return {
            'image': image_patch,
            'mask': mask_patch
        }


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing LiTSInteractiveDataset")
    print('='*60)

    # You'll need to update this path to your actual LiTS data
    data_root = "/path/to/lits/data"

    if not os.path.exists(data_root):
        print(f"Data root {data_root} does not exist.")
        print("Creating synthetic test...")

        # Create synthetic data for testing
        os.makedirs("./test_data", exist_ok=True)

        # Create a simple test volume and segmentation
        volume = np.random.rand(64, 128, 128) * 400 - 150  # Simulate CT
        segmentation = np.zeros((64, 128, 128))
        # Add liver region
        segmentation[20:40, 50:80, 50:80] = 1
        # Add tumor region
        segmentation[28:32, 60:65, 60:65] = 2

        # Save as NIfTI
        nib.save(nib.Nifti1Image(volume, np.eye(4)), './test_data/volume-0.nii')
        nib.save(nib.Nifti1Image(segmentation, np.eye(4)), './test_data/segmentation-0.nii')

        data_root = "./test_data"

    # Test with different encoders
    for encoder_type in ['binary', 'gaussian', 'disk']:
        print(f"\nTesting with {encoder_type} encoder")
        print('-'*60)

        dataset = LiTSInteractiveDataset(
            data_root=data_root,
            split='train',
            encoder_type=encoder_type,
            patch_size=(32, 64, 64)
        )

        if len(dataset) > 0:
            sample = dataset[0]

            print(f"Image shape: {sample['image'].shape}")
            print(f"Prompts shape: {sample['prompts'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")
            print(f"Positive points: {len(sample['pos_points'])}")
            print(f"Negative points: {len(sample['neg_points'])}")
            print(f"Mask classes: {np.unique(sample['mask'])}")