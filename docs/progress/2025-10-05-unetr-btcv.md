# 2025-10-09 — UNETR baseline (synthetic)
## Goal
Verify end-to-end training on synthetic 3D data using UNETR.
## Config
configs/unetr/btcv.yaml (synthetic=true), epochs=1, commit <SHA>
## Results
val/dice: 0.62 (synthetic, not representative)
## What worked
- CUDA/cuDNN fixed.
- Lightning logs recorded to docs/progress/lightning_logs/.
## What didn’t
- Synthetic metrics noisy.
- Need BTCV dataset loader.
## Next steps
- Add BTCV real data support.
- Add CSV logging for metrics.
