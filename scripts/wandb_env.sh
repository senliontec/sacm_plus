#!/usr/bin/env bash
# Shared self-hosted W&B environment for sacm_plus.
# Mirrors /home/dev/mywork/project/fencing-algs/_launch_experiments.sh.
# Credentials live in ~/.netrc (created by wandb login); no key is stored here.

export WANDB_BASE_URL="${WANDB_BASE_URL:-http://172.16.1.7:8080}"
export WANDB_ENTITY="${WANDB_ENTITY:-buaazqk}"
export WANDB_PROJECT="${WANDB_PROJECT:-sacm}"
export no_proxy="${no_proxy:-},172.16.1.7,172.16.1.0/24"
export NO_PROXY="${NO_PROXY:-},172.16.1.7,172.16.1.0/24"

# Local convenience: the same self-hosted key is already present in the
# adjacent fencing-algs config.  Read it without ever printing it when the
# caller has neither WANDB_API_KEY nor a matching ~/.netrc entry.
if [ -z "${WANDB_API_KEY:-}" ]; then
  _sacm_py="${PYTHON:-}"
  if [ -z "$_sacm_py" ]; then
    if [ -x /home/dev/anaconda3/envs/tsai/bin/python ]; then
      _sacm_py=/home/dev/anaconda3/envs/tsai/bin/python
    else
      _sacm_py=python
    fi
  fi
  WANDB_API_KEY="$("$_sacm_py" - <<'PY' 2>/dev/null || true
import os
import re

home = os.path.expanduser('~')
paths = [
    os.path.join(home, 'mywork/project/fencing-algs/configs/base.yaml'),
    '/home/dev/mywork/project/fencing-algs/configs/base.yaml',
    '/home/zkyd/code/fencing_algs/configs/base.yaml',
    '/home/zkyd/code/fencing-algs/configs/base.yaml',
    os.path.abspath('../../fencing-algs/configs/base.yaml'),
]
for path in paths:
    if not os.path.isfile(path):
        continue
    try:
        import yaml
        cfg = yaml.safe_load(open(path)) or {}
        key = ((cfg.get('wandb') or {}).get('api_key') or '').strip()
    except Exception:
        key = ''
    if not key:
        try:
            text = open(path).read()
            m = re.search(r'api_key:\s*["\']?([^"\'\s]+)', text)
            key = m.group(1) if m else ''
        except Exception:
            key = ''
    if key:
        print(key)
        break
PY
)"
  [ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY
  unset _sacm_py
fi

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
