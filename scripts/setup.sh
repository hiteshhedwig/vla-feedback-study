#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null
# POLICY_BASE_PYTHON optionally reuses a Python 3.12 environment with GPU Torch
# and LeRobot core dependencies, without changing its installed packages.
if [[ ! -d .venv-policy ]]; then
  if [[ -n "${POLICY_BASE_PYTHON:-}" ]]; then
    uv venv --python "$POLICY_BASE_PYTHON" --system-site-packages .venv-policy
    uv pip install --python .venv-policy/bin/python transformers==5.3.0 num2words==0.5.14 scipy==1.15.3 numpy==2.2.6 packaging==25.0 fsspec==2026.2.0
  else
    uv venv --python 3.12 .venv-policy
    uv pip install --python .venv-policy/bin/python torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv-policy/bin/python -r requirements-policy.txt
  fi
fi
if [[ ! -d .venv-sim ]]; then
  uv venv --python 3.10 .venv-sim
  uv pip install --python .venv-sim/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv-sim/bin/python -r requirements-sim.txt
fi
.venv-policy/bin/python scripts/bootstrap_sources.py
.venv-policy/bin/python scripts/download_model.py
.venv-policy/bin/python scripts/bootstrap_sources.py --verify
