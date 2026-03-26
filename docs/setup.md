# Setup and execution

Validated host: Ubuntu 20.04, Ryzen 7 6800HS, RTX 3060 Laptop GPU with 6 GB VRAM,
NVIDIA driver 535.183.01. GPU BF16 execution was tested. The model and all inference
run locally; no cloud account or paid service is required.

## Actual environments

- Policy: Python 3.12, PyTorch 2.10.0+cu128, Transformers 5.3.0 and the pinned
  LeRobot source. Current compatible package versions are recorded in
  configs/policy-versions.json.
- Simulation: a freshly installed project-local Python 3.10 environment, CPU
  PyTorch 2.7.1, MuJoCo 3.2.3, robosuite 1.4.1 and NumPy 1.26.4. See
  configs/simulator-versions.json.
- Checkpoint: lerobot/smolvla_libero at
  a24f1f380461b665158d52acd140152c79e8b36d.
- LeRobot source: a24998f7ba3c77ea445b48c92ad15c14a50e492a.
- LIBERO source/assets: 8f1084e3132a39270c3a13ebe37270a43ece2a01.

Two environments avoid forcing the old robosuite/NumPy simulation stack into the
modern model environment. A small localhost-only HTTP interface transfers RGB bytes
losslessly, plus state and instruction. It is not a remote cloud deployment.

## Install

Require Git, uv, FFmpeg, a compatible NVIDIA driver and EGL rendering support.

~~~bash
bash scripts/setup.sh
~~~

The setup helper obtains sparse pinned source checkouts and only checkpoint/
tokenizer assets. It does not download training datasets. The checkpoint already
contains VLM weights; load_vlm_weights=false avoids an unnecessary second copy.
Both source revisions are checked before use.

The full fresh GPU setup branch follows pinned upstream package requirements, but
was not independently executed as a separate clean-machine installation here.
The tested installation reused existing GPU Torch/LeRobot core packages in a new
system-site-packages virtual environment, installed the policy extras locally,
aligned dependency bounds, and validated all recorded inference inputs.

On a machine with an existing suitable Python 3.12 GPU environment:

~~~bash
POLICY_BASE_PYTHON=/path/to/python bash scripts/setup.sh
~~~

Reuse an existing clean source checkout if desired:

~~~bash
.venv-policy/bin/python scripts/bootstrap_sources.py --reuse-lerobot /path/to/lerobot --reuse-libero /path/to/LIBERO
~~~

Existing source revisions must match the lock and have no tracked changes.
The actual machine reuses source checkouts read-only through .deps symlinks.
Do not edit those upstream checkouts.

The source-download helper, model-download helper and existing-environment setup
path were run and verified. The simulator installation was performed from its pinned
requirements. Model files and environments are excluded by .gitignore.

## Run the milestone

Use a new output directory, with no unrelated service occupying localhost:8766:

~~~bash
bash scripts/run.sh artifacts/my-study
~~~

The launcher starts a local model process, runs the frozen three-task matrix,
generates a report and terminates its owned model process afterward.

For manual debugging, in separate terminals:

~~~bash
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 .venv-policy/bin/python scripts/policy_server.py
.venv-policy/bin/python scripts/run_study.py --output artifacts/my-study
.venv-sim/bin/python scripts/report.py --runs artifacts/my-study --output artifacts/my-report
~~~

Simulation time pauses during policy requests. This is not a real-time deployment.
Each worker has a wall-clock deadline and retains a result/error receipt where
possible; a technical failure stops the batch rather than becoming a robot timeout.

## Inspect recorded evidence without loading the model

Replay a recorded plan using only the CPU simulator:

~~~bash
PYTHONHASHSEED=0 .venv-sim/bin/python scripts/run_trial.py --suite libero_spatial --task 0 --initial 0 --mode open --plan artifacts/study-v2/bowl-state0-reference/actions.jsonl --output artifacts/my-offline-replay
~~~

Open-loop replay loads the source plan's metadata; it does not contact the model
service or require model inference.

~~~bash
PYTHONHASHSEED=0 .venv-sim/bin/python -m pytest tests -q
.venv-sim/bin/python scripts/render_demo.py --runs artifacts/study-v2 --output artifacts/my-new-demo
~~~

Tests audit the shipped primary evidence and execute focused real simulator checks.
Rendering uses the saved videos and adds explanation cards; it does not create new
policy rollouts.
