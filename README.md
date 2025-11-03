# Interactive Segmentation with Multiple Prompt Encoders

A modular experimental framework for training 3D interactive segmentation models with different prompt encoding strategies, including FastSAM3D integration.

## Overview

This repository provides a complete experimental setup for comparing different methods of encoding user interactions (clicks/points) for interactive medical image segmentation. Designed for the LiTS (Liver Tumor Segmentation) challenge with support for both CNN-based (U-Net) and transformer-based (FastSAM3D) architectures.

## Features

- 5 Prompt Encoding Methods: Binary, Gaussian, Disk, Geodesic, SAM-style
- 2 Model Architectures: 3D U-Net, FastSAM3D (ViT-based)
- Automated Click Simulation with multiple sampling strategies
- Pretrained Model Support for FastSAM3D
- Complete Training Pipeline with configuration-based experiments
- Ablation study system for comparing methods

## Project Structure

```
medseg-lab/
├── prompt_encoders/          # Prompt encoding implementations
│   ├── base_encoder.py       # Abstract base class
│   ├── binary_encoder.py     # Binary marking (single voxel)
│   ├── gaussian_encoder.py   # Gaussian heatmaps (recommended)
│   ├── disk_encoder.py       # Binary spheres (RITM-style)
│   ├── geodesic_encoder.py   # Image-aware geodesic distance
│   └── sam_encoder.py        # SAM-style point encoding
│
├── data/                     # Dataset loaders and prompt simulation
│   ├── lits_dataset.py       # LiTS dataset with interactive prompts
│   ├── prompt_simulator.py   # Simulates user clicks
│   └── fastsam3d_dataset.py  # FastSAM3D-compatible dataset
│
├── models/                   # Neural network architectures
│   ├── unet3d_interactive.py      # 3D U-Net with prompt channels
│   └── fastsam3d_interactive.py   # FastSAM3D wrapper
│
├── utils/                    # Utilities
│   ├── losses.py             # Loss functions (Dice, CE, Combined)
│   └── metrics.py            # Evaluation metrics (Dice, IoU, HD95)
│
├── configs/                  # Experiment configurations
│   ├── binary.yaml           # Binary encoder config
│   ├── gaussian_s3.yaml      # Gaussian (sigma=3) config
│   ├── disk_r3.yaml          # Disk (radius=3) config
│   ├── fastsam3d.yaml        # FastSAM3D config
│   └── ablation_study.yaml   # Multi-experiment comparison
│
├── tools/                    # Training and evaluation scripts
│   ├── train.py              # Main training script
│   ├── test_components.py    # Component testing
│   └── run_ablation.py       # Run full ablation study
│
├── pretrained_weights/       # Pretrained model checkpoints
│   └── fastsam3d.pth         # FastSAM3D pretrained weights
│
├── pyproject.toml            # Python project configuration
├── Makefile                  # Convenient training commands
└── README.md                 # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Or use make
make setup-venv
source venv/bin/activate
make setup
```

### 2. Prepare Data

Download the LiTS dataset and create a symlink:

```bash
# Create symlink to your data location
mkdir -p data
ln -s /path/to/your/LITS/data data/lits

# Verify
ls data/lits/
# Should show: volume-*.nii, segmentation-*.nii
```

Expected structure:
```
data/lits/
    volume-0.nii
    segmentation-0.nii
    volume-1.nii
    segmentation-1.nii
    ...
```

### 3. Download Pretrained Weights (Optional, for FastSAM3D)

```bash
# Create weights directory
mkdir -p pretrained_weights

# Download FastSAM3D checkpoint
wget https://huggingface.co/techlove/FastSAM3D/resolve/main/FastSAM3D.pth \
  -O pretrained_weights/fastsam3d.pth
```

### 4. Test Setup

```bash
make test
```

### 5. Train Models

Using Makefile:
```bash
# Train with Gaussian encoder (baseline)
make train

# Train with different encoders
make train-binary
make train-disk

# Train FastSAM3D
make train-fastsam3d

# Run all experiments
make train-all
```

Or directly:
```bash
# Gaussian baseline
python tools/train.py --config configs/gaussian_s3.yaml

# FastSAM3D with pretrained weights
python tools/train.py --config configs/fastsam3d.yaml
```

## Prompt Encoding Methods

### 1. Binary Encoding
Marks clicked voxel with 1, everything else is 0.

```python
from prompt_encoders import BinaryPromptEncoder
encoder = BinaryPromptEncoder()
```

Pros: Simple, no hyperparameters
Cons: Very sparse, weak gradients

### 2. Gaussian Heatmaps (Recommended)
Creates smooth 3D Gaussian distributions centered at clicks.

```python
from prompt_encoders import GaussianPromptEncoder
encoder = GaussianPromptEncoder(sigma=3.0)
```

Pros: Smooth gradients, widely validated
Cons: Needs sigma tuning
Hyperparameter: sigma=3.0 (typical: 2-5 voxels)

### 3. Disk/Sphere Encoding
Creates binary spheres with fixed radius.

```python
from prompt_encoders import DiskPromptEncoder
encoder = DiskPromptEncoder(radius=3)
```

Pros: Stronger signal than binary
Cons: Hard boundaries, not image-aware
Hyperparameter: radius=3 (typical: 3-5 voxels)

### 4. Geodesic Distance
Image-aware distance that respects anatomical boundaries.

```python
from prompt_encoders import GeodesicPromptEncoder
encoder = GeodesicPromptEncoder(sigma=5.0, lamb=1.0)
```

Pros: Respects boundaries, better guidance
Cons: Computationally expensive, requires GeodisTK
Installation: `pip install GeodisTK`

### 5. SAM-Style
Point coordinates with positive/negative labels.

```python
from prompt_encoders import SAMPromptEncoder
encoder = SAMPromptEncoder()
```

Pros: Native SAM format, sparse representation
Cons: Requires SAM-compatible model

## Model Architectures

### 3D U-Net (Baseline)
- Architecture: CNN-based encoder-decoder
- Parameters: ~15M
- Input: Image + 2-channel prompts
- Speed: ~100ms per volume

### FastSAM3D
- Architecture: ViT-Tiny (6 layers, distilled)
- Parameters: ~5M
- Input: Image + prompts
- Speed: ~8ms per volume (12.5x faster)
- Pretrained: Available from HuggingFace

FastSAM3D configuration:
```yaml
# configs/fastsam3d.yaml
model:
  type: "fastsam3d_interactive"
  in_channels: 1
  out_channels: 3
  prompt_channels: 2
  image_size: [128, 128, 128]
  embed_dim: 192
  depth: 6
  num_heads: 6
  use_pretrained: true
  pretrained_path: "./pretrained_weights/fastsam3d.pth"
```

## Expected Results

| Method | Architecture | Dice Score | Speed | Parameters |
|--------|-------------|-----------|-------|-----------|
| Binary | U-Net | 0.75-0.80 | 100ms | 15M |
| Gaussian | U-Net | 0.82-0.87 | 100ms | 15M |
| Disk | U-Net | 0.80-0.85 | 100ms | 15M |
| Geodesic | U-Net | 0.83-0.88 | 200ms | 15M |
| FastSAM3D | ViT-Tiny | 0.83-0.88 | 8ms | 5M |

Recommendation: Start with Gaussian (sigma=3) as baseline, then try FastSAM3D.

## Troubleshooting

### Module Import Errors
Check that __init__.py files have all necessary imports, especially in prompt_encoders/.

### Out of Memory
Reduce batch size or patch size:
```yaml
data:
  batch_size: 1
  patch_size: [64, 64, 64]
```

### Symlink Not Working
```bash
# Remove broken symlink
rm -f data/lits

# Recreate with absolute path
ln -s /absolute/path/to/LITS/data data/lits

# Verify
ls -la data/lits
```

### FastSAM3D Checkpoint Loading Error
In models/fastsam3d_interactive.py, ensure torch.load uses:
```python
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
```

### Dataset Finding 0 Samples
Check file naming - must have volume-X.nii paired with segmentation-X.nii:
```bash
ls data/lits/
# Verify symlink: readlink -f data/lits
```

### GeodisTK Not Available
```bash
pip install GeodisTK
```
Or skip geodesic experiments.

## References

### Key Papers
- FastSAM3D: Shen et al., "FastSAM3D: An Efficient Segment Anything Model for 3D Volumetric Medical Images", arXiv 2024
- DeepIGeoS: Wang et al., "DeepIGeoS: A Deep Interactive Geodesic Framework for Medical Image Segmentation", TPAMI 2019
- RITM: Sofiiuk et al., "Reviving Iterative Training with Mask Guidance for Interactive Segmentation", CVPR 2022
- SAM: Kirillov et al., "Segment Anything", ICCV 2023
- LiTS: Bilic et al., "The Liver Tumor Segmentation Benchmark (LiTS)", Medical Image Analysis 2023

### Resources
- LiTS Challenge: https://competitions.codalab.org/competitions/17094
- FastSAM3D Code: https://github.com/arcadelab/FastSAM3D
- FastSAM3D Weights: https://huggingface.co/techlove/FastSAM3D
- FastSAM3D Slicer: https://github.com/arcadelab/FastSAM3D_slicer

