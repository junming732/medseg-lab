# MedSeg Lab — State-of-the-Art Medical Image Segmentation (Starter)

A curated, reproducible playground of classic and modern segmentation methods:
- Baselines: UNet, ResUNet (planned)
- Transformers: **UNETR** (this starter includes a Lightning wrapper)
- Auto-config: nnU-Net v2 (planned via wrapper)
- Foundation/promptable: SAM / MedSAM (planned demos)

This starter ships a **CPU-only synthetic smoke test** so CI can run without GPUs or datasets.  
Replace the synthetic dataset with BTCV/MSD loaders as you progress.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # or: python -m venv .venv; .\.venv\Scripts\activate on Windows
pip install -U pip
pip install -e .[cpu]         # or .[cuda] after you have CUDA-capable PyTorch installed
pre-commit install || true     # optional if you add pre-commit later
pytest -q -k "cpu_smoke"
```

## Train (synthetic example, fast)
```bash
python tools/train.py --config configs/unetr/btcv.yaml --synthetic true --max_epochs 1
```

## Typical real-data (when you add BTCV/MSD loaders)
```bash
python tools/train.py --config configs/unetr/btcv.yaml --data_root /path/to/btcv --max_epochs 300
```

## Results (example table — fill as you run real experiments)
| Model    | Dataset | Dice ↑ | HD95 ↓ | Notes        |
|----------|---------|--------|--------|--------------|
| UNETR    | BTCV    | 0.XX   |  X.X   | transformer  |

## Learning Journal
See [/docs/progress](docs/progress) for dated write-ups tied to PRs.
