.PHONY: setup setup-venv activate test train train-all clean

# Setup virtual environment
setup-venv:
	python3 -m venv venv
	@echo "Virtual environment created!"
	@echo "Run 'source venv/bin/activate' to activate it"

# Install dependencies (run AFTER activating venv)
setup:
	python -m pip install -U pip
	pip install -e .
	@echo "Setup complete! Dependencies installed."

# Install with optional dependencies
setup-full:
	python -m pip install -U pip
	pip install -e ".[geodesic,test]"

# Run tests
test:
	python tools/test_components.py

# Train with Gaussian encoder (recommended baseline)
train:
	python tools/train.py --config configs/gaussian_s3.yaml

# Train with different encoders
train-binary:
	python tools/train.py --config configs/binary.yaml

train-gaussian:
	python tools/train.py --config configs/gaussian_s3.yaml

train-disk:
	python tools/train.py --config configs/disk_r3.yaml

train-fastsam3d:
	python tools/train.py --config configs/fastsam3d.yaml

# Run ablation study (compare all methods)
train-all:
	python tools/run_ablation.py --config configs/ablation_study.yaml

# Clean up
clean:
	rm -rf venv
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Help
help:
	@echo "Available commands:"
	@echo "  make setup-venv       - Create virtual environment"
	@echo "  make setup            - Install dependencies (after activating venv)"
	@echo "  make setup-full       - Install with optional dependencies"
	@echo "  make test             - Run component tests"
	@echo "  make train            - Train Gaussian model (default)"
	@echo "  make train-binary     - Train with binary encoder"
	@echo "  make train-gaussian   - Train with Gaussian encoder"
	@echo "  make train-disk       - Train with disk encoder"
	@echo "  make train-fastsam3d  - Train FastSAM3D"
	@echo "  make train-all        - Run ablation study (all methods)"
	@echo "  make clean            - Remove venv and cache files"