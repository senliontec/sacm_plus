"""Wandb integration for the self-hosted wandb server (172.16.1.7).

Reference pattern: /home/dev/mywork/project/fencing-algs
(src/train/trainer.py): wandb.init is env-configured (WANDB_BASE_URL +
WANDB_API_KEY), guarded, and EVERY failure falls back to running without
logging — wandb must never take down a training run.
"""

import logging
import os

# Read the self-hosted wandb key from the environment.  Never commit a
# literal key: this file is tracked and the value would also appear in logs.
WANDB_API_KEY_DEFAULT = os.environ.get('WANDB_API_KEY', '')

_SECRET_MARKERS = ('api_key', 'token', 'secret', 'password')


def redact_secrets(args):
    """Return a log-safe dict of argparse values with credentials masked."""
    safe = {}
    for key, value in vars(args).items():
        if any(marker in key.lower() for marker in _SECRET_MARKERS):
            safe[key] = '***' if value else ''
        else:
            safe[key] = value
    return safe


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
    try:
        import wandb
    except Exception as e:
        logging.warning(f"wandb import failed ({e}) — continuing without logging")
        return None
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
        mode = getattr(args, 'wandb_mode', None) or None
        if mode == 'disabled':
            return None
        wandb_dir = getattr(args, 'wandb_dir', None) or None
        if wandb_dir:
            os.makedirs(wandb_dir, exist_ok=True)
        run = wandb.init(
            project=getattr(args, 'wandb_project', 'sacm'),
            entity=getattr(args, 'wandb_entity', None) or None,
            name=getattr(args, 'wandb_name', None) or name,
            config=config,
            tags=tags or [],
            group=getattr(args, 'wandb_group', None) or None,
            job_type=getattr(args, 'wandb_job_type', None) or None,
            dir=wandb_dir,
            mode=mode,
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


def log_artifact(run, path, name=None, artifact_type='model'):
    """Attach a checkpoint/artifact to the run; never raise."""
    if run is None:
        return
    try:
        import wandb
        artifact = wandb.Artifact(
            name or os.path.splitext(os.path.basename(str(path)))[0],
            type=artifact_type,
        )
        artifact.add_file(str(path))
        run.log_artifact(artifact)
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
