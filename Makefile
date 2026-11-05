.PHONY: install test lint format registry smoke clean clean-dry

# src layout: make the packages importable from the repo root.
export PYTHONPATH := $(shell pwd)/src

# Standard engineering entry points.

install:
	pip install -e ".[dev]"

test:
	pytest

test-slow:
	pytest --run-slow

lint:
	ruff check .

format:
	ruff format .

# Print which topology losses and models are registered in this environment.
registry:
	python -c "from core.losses import TOPOLOGY_LOSS_REGISTRY; import models; from core.registry import MODEL_REGISTRY; print('losses:', sorted(TOPOLOGY_LOSS_REGISTRY)); print('models:', sorted(MODEL_REGISTRY))"

# Fast sanity: import everything, build the model, run one tiny forward.
smoke:
	python tests/smoke_model.py

# Remove every __pycache__ dir and stray .pyc/.pyo file (dry-run: make clean-dry)
clean:
	find src scripts tests docs configs -type d -name __pycache__ -not -path '*/baseline_zoo/*' -prune -exec rm -rf {} +
	find src scripts tests docs configs -type f \( -name '*.pyc' -o -name '*.pyo' \) -not -path '*/baseline_zoo/*' -delete

clean-dry:
	find src scripts tests docs configs -type d -name __pycache__ -not -path '*/baseline_zoo/*' -prune -print
	find src scripts tests docs configs -type f \( -name '*.pyc' -o -name '*.pyo' \) -not -path '*/baseline_zoo/*' -print
