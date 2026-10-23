#!/usr/bin/env bash
# Shared self-hosted W&B environment for sacm_plus.
# Mirrors /home/dev/mywork/project/fencing-algs/_launch_experiments.sh.
# Credentials live in ~/.netrc (created by wandb login); no key is stored here.

export WANDB_BASE_URL="${WANDB_BASE_URL:-http://172.16.1.7:8080}"
export WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"
export WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
export no_proxy="${no_proxy:-},172.16.1.7,172.16.1.0/24"
export NO_PROXY="${NO_PROXY:-},172.16.1.7,172.16.1.0/24"

# Return 0 iff an online W&B run can authenticate (env or ~/.netrc).
wandb_auth_ok() {
  "${PYTHON:-python}" - <<'PY'
import netrc
import os
import sys

if os.environ.get('WANDB_API_KEY'):
    sys.exit(0)
try:
    import wandb
    if getattr(wandb.api, 'api_key', None):
        sys.exit(0)
except Exception:
    pass

host = os.environ.get('WANDB_BASE_URL', 'http://172.16.1.7:8080')
host = host.replace('https://', '').replace('http://', '').split('/')[0]
try:
    auth = netrc.netrc().authenticators(host)
except Exception:
    auth = None
sys.exit(0 if auth else 1)
PY
}
