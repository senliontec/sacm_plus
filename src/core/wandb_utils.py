"""Wandb integration for the self-hosted wandb server (172.16.1.7).

Reference pattern: /home/dev/mywork/project/fencing-algs
(src/train/trainer.py): wandb.init is env-configured (WANDB_BASE_URL +
WANDB_API_KEY), guarded, and EVERY failure falls back to running without
logging — wandb must never take down a training run.
"""

import logging
import os


def wandb_available():
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def setup_wandb(args, name, config):
    """Init a wandb run from CLI args; returns the run or None.

    Non-fatal by design: missing package, unreachable server or bad auth
    all degrade to a warning and training continues without wandb.
    """
    if not getattr(args, 'use_wandb', False):
        return None
    if not wandb_available():
        logging.warning("wandb not installed — continuing without logging")
        return None
    import wandb
    try:
        host = getattr(args, 'wandb_host', None)
        if host:
            os.environ['WANDB_BASE_URL'] = host
        key = getattr(args, 'wandb_api_key', None)
        if key:
            os.environ['WANDB_API_KEY'] = key
        tags = getattr(args, 'wandb_tags', None)
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        run = wandb.init(
            project=getattr(args, 'wandb_project', 'sacm'),
            entity=getattr(args, 'wandb_entity', None) or None,
            name=getattr(args, 'wandb_name', None) or name,
            config=config,
            tags=tags or [],
        )
        logging.info(f"wandb run started: {run.entity}/{run.project}/{run.name}")
        return run
    except Exception as e:
        logging.warning(f"wandb init failed ({e}) — continuing without logging")
        return None


def log_scalars(run, metrics):
    """wandb.log with the same failure-is-non-fatal discipline."""
    if run is None:
        return
    try:
        run.log(metrics)
    except Exception:
        pass


def finish_run(run):
    """Close the run; never raise."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
