# Interactive Segmentation with Multiple Prompt Encoders

A modular experimental framework for training 3D interactive segmentation models with different prompt encoding strategies.

## 🎯 Overview

This repository provides a complete experimental setup for comparing different methods of encoding user interactions (clicks/points) for interactive medical image segmentation. It's designed for the LiTS (Liver Tumor Segmentation) challenge but can be easily adapted to other datasets.

## 📁 Project Structure

```
interactive_segmentation/
├── prompt_encoders/          # Different prompt encoding implementations
│   ├── base_encoder.py       # Abstract base class
│   ├── binary_encoder.py     # Binary marking (single voxel = 1)
│   ├── gaussian_encoder.py   # Gaussian heatmaps (standard approach)
│   ├── disk_encoder.py       # Binary spheres (RITM-style)
│   └── geodesic_encoder.py   # Image-aware geodesic distance (DeepIGeoS)
│
├── data/                     # Dataset loaders and prompt simulation
│   ├── lits_dataset.py      # LiTS dataset with interactive prompts
│   └── prompt_simulator.py  # Simulates user clicks during training
│
├── models/                   # Neural network architectures
│   └── unet3d_interactive.py # 3D U-Net with prompt channels
│
├── utils/                    # Utilities
│   ├── losses.py            # Loss functions (Dice, CE, Focal)
│   └── metrics.py           # Evaluation metrics (Dice, HD95, ASD)
│
├── configs/                  # Experiment configurations
│   ├── binary.yaml          # Binary encoder config
│   ├── gaussian_s3.yaml     # Gaussian (sigma=3) config
│   ├── disk_r3.yaml         # Disk (radius=3) config
│   └── ablation_study.yaml  # Multi-experiment ablation study
│
└── tools/                    # Training and evaluation scripts
    ├── train.py             # Main training script
    ├── evaluate.py          # Evaluation script
    └── run_ablation.py      # Run full ablation study
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install nibabel scipy pyyaml tqdm
pip install GeodisTK  # Optional, for geodesic encoder
```

### 2. Prepare Data

Download the LiTS dataset from the challenge website:
https://competitions.codalab.org/competitions/17094

Expected directory structure:
```
/path/to/lits/data/
    volume-0.nii
    segmentation-0.nii
    volume-1.nii
    segmentation-1.nii
    ...
```

Update the `data_root` path in config files.

### 3. Train a Single Model

```bash
# Train with Gaussian encoder (recommended baseline)
python tools/train.py --config configs/gaussian_s3.yaml

# Train with binary encoder (your colleague's approach)
python tools/train.py --config configs/binary.yaml

# Train with disk encoder (RITM-style)
python tools/train.py --config configs/disk_r3.yaml
```

### 4. Run Ablation Study

Compare all encoding methods:

```bash
python tools/run_ablation.py --config configs/ablation_study.yaml
```

## 🧪 Prompt Encoding Methods

### 1. Binary Encoding
**What:** Marks clicked voxel with 1, everything else is 0.

```python
from prompt_encoders import BinaryPromptEncoder

encoder = BinaryPromptEncoder()
prompts = encoder.encode(pos_points, neg_points, volume_shape)
```

**Pros:** Simple, no hyperparameters, exact location
**Cons:** Very sparse, weak gradients, hard for network to see

### 2. Gaussian Heatmaps (Recommended)
**What:** Creates smooth 3D Gaussian distributions centered at clicks.

```python
from prompt_encoders import GaussianPromptEncoder

encoder = GaussianPromptEncoder(sigma=3.0)  # sigma controls spread
prompts = encoder.encode(pos_points, neg_points, volume_shape)
```

**Pros:** Smooth gradients, widely validated, works well with CNNs
**Cons:** Needs sigma tuning

**Hyperparameter:** `sigma` (typical: 2-5 voxels)

### 3. Disk/Sphere Encoding
**What:** Creates binary spheres with fixed radius.

```python
from prompt_encoders import DiskPromptEncoder

encoder = DiskPromptEncoder(radius=3)  # radius in voxels
prompts = encoder.encode(pos_points, neg_points, volume_shape)
```

**Pros:** Simple, stronger signal than binary, proven in RITM
**Cons:** Hard boundaries, not image-aware

**Hyperparameter:** `radius` (typical: 3-5 voxels)

### 4. Geodesic Distance (Advanced)
**What:** Image-aware distance that respects anatomical boundaries.

```python
from prompt_encoders import GeodesicPromptEncoder

encoder = GeodesicPromptEncoder(sigma=5.0, lamb=1.0)
# Requires image intensities
prompts = encoder.encode(pos_points, neg_points, volume_shape, image=ct_volume)
```

**Pros:** Respects boundaries, better guidance
**Cons:** Computationally expensive, needs GeodisTK

**Hyperparameters:** `sigma`, `lamb` (image gradient weight)

## 📊 Customization

### Add Your Own Encoder

1. Create new file in `prompt_encoders/`:

```python
from prompt_encoders.base_encoder import BasePromptEncoder

class MyCustomEncoder(BasePromptEncoder):
    def __init__(self, my_param=1.0):
        super().__init__(name="my_custom")
        self.my_param = my_param

    def encode(self, positive_points, negative_points, volume_shape, spacing=None):
        # Your encoding logic here
        pos_channel = ...  # Shape: [D, H, W]
        neg_channel = ...  # Shape: [D, H, W]
        return np.stack([pos_channel, neg_channel], axis=0)
```

2. Register in `prompt_encoders/__init__.py`:

```python
from .my_custom_encoder import MyCustomEncoder

def get_encoder(encoder_type, **kwargs):
    encoders = {
        'binary': BinaryPromptEncoder,
        'gaussian': GaussianPromptEncoder,
        'disk': DiskPromptEncoder,
        'geodesic': GeodesicPromptEncoder,
        'my_custom': MyCustomEncoder,  # Add this
    }
    ...
```

3. Create config file `configs/my_custom.yaml`:

```yaml
data:
  encoder:
    type: "my_custom"
    my_param: 1.0
```

### Use Different Dataset

Inherit from base dataset or create new:

```python
from data.lits_dataset import LiTSInteractiveDataset

class MyDataset(LiTSInteractiveDataset):
    def _get_file_list(self):
        # Your data loading logic
        pass
```

## 📈 Expected Results

Based on literature and preliminary experiments:

| Method | Typical Dice | Training Speed | Inference Speed |
|--------|--------------|----------------|-----------------|
| Binary | 0.75-0.80 | Fast | Fast |
| Gaussian (σ=3) | **0.82-0.87** | Fast | Fast |
| Disk (r=3) | 0.80-0.85 | Fast | Fast |
| Geodesic | 0.83-0.88 | Slow | Medium |

**Recommendation:** Start with Gaussian (σ=3) as your baseline.

## 🔧 Troubleshooting

### "Module not found" errors
Make sure you're running from the correct directory:
```bash
cd /path/to/interactive_segmentation
python tools/train.py --config configs/gaussian_s3.yaml
```

### Out of memory
Reduce batch size or patch size in config:
```yaml
data:
  batch_size: 1
  patch_size: [64, 64, 64]
```

### GeodisTK not available
Geodesic encoder requires GeodisTK. Install with:
```bash
pip install GeodisTK
```
Or skip geodesic experiments.

## 📚 References

### Papers
- **DeepIGeoS**: Wang et al., "DeepIGeoS: A Deep Interactive Geodesic Framework for Medical Image Segmentation", TPAMI 2019
- **RITM**: Sofiiuk et al., "Reviving Iterative Training with Mask Guidance for Interactive Segmentation", CVPR 2022
- **SAM**: Kirillov et al., "Segment Anything", ICCV 2023
- **LiTS**: Bilic et al., "The Liver Tumor Segmentation Benchmark (LiTS)", Medical Image Analysis 2023

### Datasets
- LiTS Challenge: https://competitions.codalab.org/competitions/17094

