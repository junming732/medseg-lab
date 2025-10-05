from typing import Any, Dict, Tuple
import os, yaml, random, numpy as np, torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from medseglab.models.unetr import UNETRLightning
from medseglab.data.synthetic import SyntheticVolumetricSeg

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataloaders(cfg: Dict[str, Any]):
    dcfg = cfg.get("data", {})
    img_size = tuple(dcfg.get("img_size", [32,32,32]))
    if dcfg.get("synthetic", True):
        train_ds = SyntheticVolumetricSeg(dcfg.get("num_samples_train", 8), img_size, seed=cfg.get("seed", 42))
        val_ds   = SyntheticVolumetricSeg(dcfg.get("num_samples_val", 4), img_size, seed=cfg.get("seed", 123))
    else:
        raise NotImplementedError("Real dataset loaders not yet implemented in starter.")
    bs = int(dcfg.get("batch_size", 1))
    nw = int(dcfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader

def build_model(cfg: Dict[str, Any]) -> UNETRLightning:
    m = cfg.get("model", {})
    return UNETRLightning(
        in_channels=int(m.get("in_channels", 1)),
        out_channels=int(m.get("out_channels", 1)),
        img_size=tuple(m.get("img_size", [32,32,32])),
        feature_size=int(m.get("feature_size", 16)),
        learning_rate=float(m.get("learning_rate", 1e-3)),
    )

def build_trainer(cfg: Dict[str, Any]) -> pl.Trainer:
    t = cfg.get("trainer", {})
    return pl.Trainer(
        max_epochs=int(t.get("max_epochs", 10)),
        precision=t.get("precision", 32),
        log_every_n_steps=int(t.get("log_every_n_steps", 1)),
        enable_checkpointing=bool(t.get("enable_checkpointing", False)),
        enable_progress_bar=bool(t.get("enable_progress_bar", True)),
        accelerator="auto",
        devices=1,
    )

def train_from_config(config_path: str, overrides: Dict[str, Any] | None = None):
    cfg = load_config(config_path)
    if overrides:
        # shallow merge
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    set_seed(int(cfg.get("seed", 42)))
    model = build_model(cfg)
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(model, train_loader, val_loader)
