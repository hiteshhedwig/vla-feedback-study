#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -ne 1 ]]; then echo "Usage: bash scripts/run.sh NEW_OUTPUT_DIRECTORY"; exit 2; fi
if [[ -e "$1" ]]; then echo "Refusing to overwrite existing output: $1"; exit 2; fi
mkdir -p artifacts
.venv-policy/bin/python - <<'PORTCHECK'
import socket
with socket.socket() as s:
    if s.connect_ex(("127.0.0.1",8766))==0:
        raise SystemExit("Port 8766 is already active; stop that service before starting a new owned model.")
PORTCHECK
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 .venv-policy/bin/python scripts/policy_server.py > artifacts/server-current.log 2>&1 &
model_pid=$!
trap 'kill "$model_pid" 2>/dev/null || true; wait "$model_pid" 2>/dev/null || true' EXIT
.venv-policy/bin/python - "$model_pid" <<'PY'
import json,sys,time,urllib.request,os
pid=int(sys.argv[1])
for attempt in range(120):
    try:
        os.kill(pid,0)
        with urllib.request.urlopen("http://127.0.0.1:8766/health",timeout=1) as r:meta=json.load(r)
        if meta.get("checkpoint")=="lerobot/smolvla_libero":break
    except Exception:pass
    time.sleep(0.5)
else:raise SystemExit("Policy failed to start; inspect artifacts/server-current.log")
PY
.venv-policy/bin/python scripts/run_study.py --output "$1"
.venv-sim/bin/python scripts/report.py --runs "$1" --output "$1/report"
