# VLA Feedback Study

[![Recorded closed-loop and open-loop comparisons across three manipulation tasks](artifacts/demo/preview.gif)](artifacts/demo/complete-walkthrough.mp4)

**[Watch the explained walkthrough](artifacts/demo/complete-walkthrough.mp4)** · [Bowl placement](artifacts/demo/bowl.mp4) · [Drawer opening](artifacts/demo/drawer.mp4) · [Stove switch](artifacts/demo/stove.mp4)

One **pretrained SmolVLA**, one simulated Panda arm, and three tasks: study how fresh model feedback compares with executing a nominal VLA-generated plan.

Two focused experiments use the same pretrained policy and simulator:

| Experiment | Question | Evidence |
| --- | --- | --- |
| [Feedback vs. fixed-plan replay](#what-open-loop-means-here) | Does observing again help after an interruption? | Three tasks, 36 episode records |
| [Action chunk execution](#action-chunking-how-often-should-the-vla-predict-again) | What changes when we execute 1, 10 or 50 predictions before asking the model again? | One bowl scene, three recorded episodes |

There is no policy training or custom grasp trajectory.

## Action chunking: how often should the VLA predict again?

[![Three recorded bowl-placement executions: one, ten and fifty actions per inference](artifacts/milestone2/preview.gif)](artifacts/milestone2/comparison.mp4)

**[Watch the explained three-way comparison](artifacts/milestone2/comparison.mp4)** · [Measured report](artifacts/milestone2/report.md) · [Raw results](artifacts/milestone2/summary.json)

### What actually changes?

SmolVLA predicts **50 future actions in every inference call**. We vary how many of those actions the robot executes before capturing fresh camera images and robot state:

| Mode | What the robot does | When the VLA sees the robot again |
| --- | --- | --- |
| Execute 1 | Use the first prediction; discard the other 49 | After every action |
| Execute 10 | Execute the first 10 predictions in order; discard the remaining 40 | After every 10 actions |
| Execute 50 | Execute the entire predicted chunk in order | After every 50 actions |

Each mode repeats **observe → predict → execute** until LIBERO detects task success or the 150-action budget runs out. All use the same bowl-placement task, predefined initial state, checkpoint and existing Panda controller. There is no injected interruption in this experiment.

The one-action mode approximates execution with feedback after each action. **It is not a separately trained sequential-output model:** the policy still computes all 50 predictions. This experiment studies how we *use* a chunk, rather than comparing two model architectures.

### What happened in this scene?

All three executions completed the bowl-placement task, according to LIBERO's task predicate.

| Actions before fresh feedback | Task outcome | Robot actions executed | Model calls | Total inference time | Inference cost per executed action |
| --- | --- | --- | --- | --- | --- |
| 1 | Success | 78 | 78 | 24.03 s | 0.308 s |
| 10 | Success | 79 | 8 | 2.96 s | 0.037 s |
| 50 | Success | 78 | 2 | 0.58 s | 0.007 s |

**Reusing the chunk avoided repeated model calls while preserving task success in this scene.** Executing 50 predictions per call used 2 calls instead of 78. It reduced total inference work, not the latency of an individual call.

The trade-off is feedback freshness. At the configured 20 Hz command rate, these modes request new observations after up to **0.05 s, 0.5 s or 2.5 s of simulated motion**, respectively. A longer chunk saves inference work but commits to more actions before the next model update. These clean runs do not establish which horizon would respond best to a disturbance.

### How to read the demo

The three panes replay the actual robot executions together at **0.5× speed**. Labels show the execution horizon, final model-call count and measured inference cost. A pane freezes and is labeled after its episode ends.

**The video compares motion, not elapsed wall-clock runtime.** Physics pauses while the synchronous model call runs, so the 24.03 seconds of inference in the one-action mode are not shown as equivalent idle time. The table reports that computation cost separately. No artificial delay was added.

### What this does—and does not—show

This is the deliberately small **Milestone 2** result: one scene, one episode per horizon, separate from the 36-record feedback experiment. All three succeeded; that is not evidence of equal reliability or a universally optimal chunk size.

Run order was 10, 1, 50, and cold-start overhead remains included. These measurements do not demonstrate real-time deadline handling, asynchronous buffering or recovery. Broader task and latency comparisons are left for a later study.

To rerun just this scene into a new output directory:

~~~bash
# Terminal 1
.venv-policy/bin/python scripts/policy_server.py

# Terminal 2
.venv-policy/bin/python scripts/run_chunk_smoke.py --output artifacts/my-chunk-smoke
~~~

## Feedback experiment: closed loop vs. fixed-plan replay

The following 36-record experiment keeps the ten-action execution horizon fixed and changes whether the policy receives fresh feedback.

## What “open loop” means here

| Mode | What generates the commands |
| --- | --- |
| Closed loop | Current agent/wrist images, robot state and task instruction go to SmolVLA; execute ten predicted actions, then observe and predict again. |
| Open loop | Replay a full nominal command sequence previously generated by this same VLA on the same task and initial state. New observations are recorded for evidence but never sent to the model. |

This is **high-level feedback versus nominal trajectory replay**. Both retain robosuite's low-level operational-space feedback controller.

SmolVLA predicts 50 actions, not a whole manipulation task. The replay plan is therefore compiled from a separate nominal closed-loop rollout. It is **not a single-shot full-task VLA prediction**, and its model-inference preparation cost is not free.

## What happens in the videos

Both panes begin from the same simulated state. Commands 30–39 are then withheld from both robots: their Cartesian command becomes zero and the previously executed gripper command is retained. Physics and low-level control continue.

After that half-second interruption:

- The closed-loop policy receives the resulting camera/state feedback at its next scheduled query.
- The open-loop execution continues the fixed nominal command sequence.
- LIBERO independently checks the task goal. A valid neural-network output is not task success.

The clips show the predefined initial state 0 for each task, including the original agent view and a wrist-view inset. Source videos are genuine simulated executions. Playback is 0.5× with labeled reading pauses. A pane held after termination is explicitly labeled; it is not additional robot behavior.

## Measured results

**36 primary episode records:** three tasks × three fixed initial states × two execution modes × two conditions. Zero infrastructure failures.

Under clean conditions, both modes succeed in 9/9 cases. Clean replay is expected to match its nominal reference by construction: this is a reproducibility control, not independent evidence that open-loop VLA planning is generally equivalent.

Under the same ten-command interruption:

| Task | Closed-loop success | Open-loop replay success |
| --- | --- | --- |
| Bowl placement | 3/3 | 0/3 |
| Drawer opening | 3/3 | 1/3 |
| Stove switch | 3/3 | 3/3 |
| **Total** | **9/9** | **4/9** |

**Feedback improved completion here, but did not always reduce time to the goal.**

In the four interrupted pairs where both modes succeeded, replay reached the goal sooner:

- Drawer, state 1: replay 133 commands; closed loop 143.
- Stove, states 0/1/2: replay 88/85/77; closed loop 91/91/82.

That timing comparison excludes five replay failures; it must not be presented as an overall timing victory. Replay also makes zero online model calls, while preparing each reference plan required 30 VLA calls.

These are mixed engineering trade-offs, **not a universal ranking**. Three predefined tasks and three fixed initial states do not establish a broad correlation or reliability guarantee. We did not add tasks until a preferred result appeared.

[Full report and paired outcomes](artifacts/report-v2/report.md) · [Machine-readable metrics](artifacts/report-v2/summary.json) · [Frozen protocol](configs/study.json)

![Success counts by task and condition](artifacts/report-v2/success-by-task.png)

## Genuine model integration

The public [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero) checkpoint contains approximately 450 million parameters and is pinned at `a24f1f380461b665158d52acd140152c79e8b36d`. Its weights are loaded strictly; no parameters are trained or replaced.

| Interface | Actual representation |
| --- | --- |
| Cameras | Agent and wrist RGB, each 256×256; orientation handled as in LeRobot's LIBERO processor |
| State | End-effector world position (3), axis-angle orientation (3), two gripper joint positions (2) |
| Language | The supported LIBERO task instruction |
| Prediction | 50×7 normalized actions |
| Executed action | Checkpoint-denormalized relative translation (3), relative axis-angle rotation (3), gripper (1) |
| Feedback interval | Ten commands in the feedback experiment; 1/10/50 in the chunk-execution comparison |
| Control / physics | 20 Hz commands; existing 2 ms simulation/controller integration |

The checkpoint's existing processors rename cameras, tokenize language and normalize state/actions. The policy internally resizes/pads images to 512×512 and maps pixels to its expected range. We do not substitute Euler angles for axis-angle state or interpret action values as joint positions.

Robosuite's OSC controller scales translation/rotation commands and applies the robot dynamics. Gripper −1 opens and +1 closes. Raw predictions, denormalized values, bounded commands and observed execution are saved separately. Focused physical tests verify translation response and gripper sign.

## Architecture

~~~mermaid
flowchart LR
    A[LIBERO cameras and 8D state] --> B[Lossless localhost transport]
    B --> C[LeRobot checkpoint processors]
    C --> D[Pretrained SmolVLA on local GPU]
    D --> E[Ten-action execution buffer]
    E --> F[Existing Panda OSC controller]
    F --> G[MuJoCo physics and task predicate]
    G --> A
    H[Recorded nominal VLA plan] --> F
~~~

Two local Python environments separate the modern policy dependencies from the older LIBERO/robosuite stack. There is no cloud inference service. The small HTTP process is bound to localhost.

**Simulation pauses during synchronous inference.** The measured policy-pipeline latency in the primary batch is about 0.29 s median and 0.34 s p95. Peak PyTorch GPU allocation was about 1.39 GiB on an RTX 3060 Laptop GPU; driver/context memory is additional. A 20 Hz simulation control rate is not a claim of real-time physical deployment.

Existing software provides the model, preprocessing, physics, cameras, controller, initial states and success predicates. Project code supplies execution-mode selection, the paired protocol, input/action evidence, reporting and explained clips.

## Setup and run

The inspected machine is configured and tested. From this folder, with the model service stopped:

~~~bash
bash scripts/run.sh artifacts/my-new-study
~~~

This starts an owned local policy process, executes the 36-record study in a new directory, generates its report and cleans up that policy process. It refuses an existing output or an already occupied model port.

For setup on another Linux machine, install Git, uv, FFmpeg and an appropriate NVIDIA driver, then:

~~~bash
git clone https://github.com/hiteshhedwig/vla-feedback-study.git
cd vla-feedback-study
bash scripts/setup.sh
~~~

Setup creates separate Python 3.12 policy and Python 3.10 simulator environments, obtains pinned source/model assets and downloads no training dataset. Allow several GB for CUDA dependencies and assets.

On this machine the policy environment reuses the existing GPU Torch/core installation without modifying it; the project-local simulator environment was installed independently. The full fresh GPU dependency-install branch is provided as a recipe, rather than claimed as a separately executed clean-machine installation. [Setup details and versions](docs/setup.md).

To render a new study after it finishes:

~~~bash
.venv-sim/bin/python scripts/render_demo.py --runs artifacts/my-new-study --output artifacts/my-new-demo
~~~

The supplied walkthrough and animation have already been fully decoded and visually checked.

## Evidence and verification

**19 tests pass**, including the original Milestone 1 audits and the new horizon-provenance check. They audit actual recorded inference/execution, not mocked model success:

- Camera/state dimensions and semantic conversions.
- All predicted actions, checkpoint denormalization and execution provenance.
- Identical initial conditions and clean replay physics.
- Matching inputs, model noise and actions before the intervention.
- Exact intervention timing and independent task outcomes.
- Real controller translation/gripper behavior and invalid-command rejection.
- Reexecution of every saved model input on the dependency-aligned runtime.

Each trial contains `scenario.json`, `manifest.json`, raw video, lossless observations, prediction chunks, executed commands, physics state and task result.

The final study is `artifacts/study-v2`. An earlier development batch is retained separately after its reproducibility audit found a pre-intervention divergence. It is excluded from the primary results. Further diagnostics localized a later repeat's drift to two RGB channel values changing by 1/255, with identical robot state and model noise. The task still completed at the same step. We do not promise pixel-perfect simulation repeats.

All **370 primary-study model inputs** were reexecuted after aligning dependency bounds, reproducing their predictions exactly. Collection/runtime versions remain separately recorded. [Reproducibility and diagnostics](docs/reproducibility.md).

[Current test output](artifacts/milestone2/tests.log) · [Runtime revalidation](artifacts/runtime-revalidation.json) · [Verification summary](artifacts/verification-summary.json)

## Scope

Milestone 1 contains the multi-task feedback study. Milestone 2 currently adds only the one-scene execution-horizon smoke check above. Recovery-policy comparisons, broader horizon evaluation and asynchronous inference remain future work. There is no claim of physical-hardware validation, a new controller, a general evaluation platform or a new VLA.

[Protocol and interfaces](docs/protocol.md) · [Reproducibility](docs/reproducibility.md) · [Attribution](THIRD_PARTY_NOTICES.md)
