.PHONY: install test lint format verify registry smoke clean

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

# Numerical cross-check of our strict ports against the official repos
# (requires the locally pulled repos in ../segment/3rd; CPU only).
verify:
	python scripts/verify_against_official.py

# Print which topology losses and models are registered in this environment.
registry:
	python -c "from core.losses import TOPOLOGY_LOSS_REGISTRY; import models; from core.registry import MODEL_REGISTRY; print('losses:', sorted(TOPOLOGY_LOSS_REGISTRY)); print('models:', sorted(MODEL_REGISTRY))"

# Fast sanity: import everything, build the model, run one tiny forward.
smoke:
	python tests/smoke_model.py

# Remove every __pycache__ dir and stray .pyc/.pyo file (dry-run: make clean-dry)
clean:
	bash scripts/clean_pycache.sh

clean-dry:
	bash scripts/clean_pycache.sh -n
