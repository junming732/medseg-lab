.PHONY: setup test train

setup:
	python -m pip install -U pip
	pip install -e .[cpu,test]

test:
	pytest -q -k "cpu_smoke"

train:
	python tools/train.py --config configs/unetr/btcv.yaml --synthetic true --max_epochs 1
