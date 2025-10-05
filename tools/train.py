import argparse, yaml
from typing import Any, Dict
from medseglab.engine.train import train_from_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--synthetic", type=str, default=None, help="true/false to override data.synthetic")
    ap.add_argument("--max_epochs", type=int, default=None, help="Override trainer.max_epochs")
    args = ap.parse_args()

    overrides: Dict[str, Any] = {}
    if args.synthetic is not None:
        overrides.setdefault("data", {})["synthetic"] = args.synthetic.lower() in {"1","true","yes","y"}
    if args.max_epochs is not None:
        overrides.setdefault("trainer", {})["max_epochs"] = args.max_epochs
    train_from_config(args.config, overrides or None)

if __name__ == "__main__":
    main()
