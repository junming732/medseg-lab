# 2025-10-05 — UNETR baseline (synthetic smoke)

## Goal
Stand up a Lightning training loop around MONAI's UNETR and validate CI can pass on CPU with synthetic data.

## Setup
- Config: `configs/unetr/btcv.yaml` (synthetic=true)
- Commit: <SHA after PR>

## Results
- Dice (synthetic): ~0.5–0.7 after 1–3 epochs (non-representative)

## What worked
- End-to-end training loop.
- Deterministic seed and small 3D volumes keep runtime minimal.

## What didn’t
- Synthetic data is noisy; real dice will differ.

## Next steps
- Add real BTCV/MSD dataloaders and dataset cards.
- Scale img_size, epochs, and augmentations.
