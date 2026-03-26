# Protocol and interfaces

The experiment asks how fresh high-level VLA feedback compares with executing a
recorded nominal VLA plan. It does not compare two trained policies or two low-level
controllers.

## Frozen task matrix

- LIBERO-Spatial task 0: pick up the black bowl between the plate and ramekin and place it on the plate.
- LIBERO-Goal task 0: open the middle cabinet drawer.
- LIBERO-Goal task 7: turn on the stove.
- Initial-state IDs 0, 1 and 2 for each task.
- Model/checkpoint, supported task text, environment seed 0, explicit model-noise
  schedule, ten-action execution horizon, robot controller and 300-command budget
  are fixed.
- Conditions: clean and a command interruption at zero-based steps 30 through 39.

Task/state selection and interruption were fixed before interrupted-policy outcomes.
The primary artifact design is artifacts/study-v2/design.json.

## Nominal plan preparation

For each of the nine task/initial-state combinations, reset the simulator, perform
ten neutral settling steps and execute SmolVLA under fresh feedback. Record all 300
commands, including after first success if it occurs earlier, so every replay has
a full common-budget nominal sequence. The plan remains entirely VLA-generated.

This same reference supplies the clean closed-loop measurement: use its first
success time and corresponding inference calls. Additional post-success calls are
plan preparation, not needed for the task-completion measurement.

The three further rollouts for the same initial state are clean replay, interrupted
closed-loop execution and interrupted replay. Total: 9 × 4 = 36 episode records.

A reference plan is specialized to that exact initial state. We do not transfer one
initial state's plan to another and attribute the resulting mismatch solely to
online feedback. Clean replay is a reproducibility control, not an independent
generalization experiment.

## Shared intervention and termination

At steps 30–39, withhold the requested Cartesian command and execute six zeros
while retaining the last executed gripper command. The underlying OSC controller
still uses physical state feedback and the simulator still advances. No task
trajectory or recovery action is scripted.

Record both requested and executed commands. At step 40 normal command execution
resumes. Closed-loop inference remains on its ordinary ten-step schedule; no special
recovery-triggered inference is added.

LIBERO's task predicate determines first-attainment success. Evaluation rollouts
terminate on that first success or at 300 commands. A reference continues to 300
solely for full-plan construction. Transient predicate success is not a claim of
sustained goal stability. API success is not task success.

Task time is counted after the ten settling actions: command step 0 begins at task
time zero; the underlying physics state also retains its absolute simulation time.
Inference pauses physics, so video duration omits inference waiting.

## Observation semantics

Raw agentview_image and robot0_eye_in_hand_image are RGB HWC uint8 arrays.
Rotate both dimensions once to match the pinned LeRobot LIBERO convention. Convert
to CHW float pixels in [0,1], and use checkpoint-provided camera renaming:
image -> camera1, image2 -> camera2.

State order:
1. End-effector world x, y, z in metres.
2. End-effector orientation as a three-value axis-angle vector, converted from
   simulator xyzw quaternion.
3. Left/right finger joint positions in the simulator's metre convention.

The eight-value vector is normalized with the checkpoint's supplied MEAN_STD
statistics. It is not eight joint angles or Euler pose. The pretrained policy pads
its internal representations as needed; no fabricated physical state is supplied.

Language is the task's existing natural-language instruction. Its tokenization and
48-token padding come from the supplied processor configuration.

Both raw views and raw state are saved losslessly at every model query, along with
their hashes and the actual model-input tensor shapes.

## Action semantics

The model predicts 50 normalized seven-value actions. Apply the supplied
postprocessor once. Each action contains relative end-effector translation, relative
axis-angle rotation and one gripper command.

The existing OSC_POSE controller accepts normalized Cartesian components bounded
to [-1,1]. Its configured target scale is ±0.05 m translation and ±0.5 radians
axis-angle components; these are controller targets, not instantaneous achieved
displacements. Rotation deltas compose with current orientation through the existing
robosuite controller. The controller handles Jacobians, damping and torque execution.

Gripper −1 opens and +1 closes; no OpenVLA-specific sign conversion is introduced.
Predictions outside controller bounds are clipped and counted, with their original
values retained. Nonfinite or wrong-sized actions are rejected.

Execute the first ten actions per chunk. A new current observation then produces
another chunk. This fixed horizon is not itself an experimental variable.

## Matching and provenance

Each episode starts a fresh simulator/controller instance and uses pinned initial
states. Model noise is an explicit standard-normal CUDA tensor with seed
1000 + control_step // 10, matched across conditions. Torch deterministic algorithms
are enabled. These settings do not guarantee that every rendering context will
produce identical pixels forever.

Verify each paired initial physics state and initial observation. The final batch
also verifies clean replay physics and all pre-intervention control prefixes. Both
camera images, proprioception, model noise and predicted actions agree at the
shared pre-intervention queries.

Manifests retain model/checkpoint/source hashes, upstream commits, package versions,
initial states, precision and timing. CPU simulation and local GPU inference are
isolated in separate environments.

## Interpretation

Per-task success counts are primary. Also retain paired goal steps, online inference
counts/latencies and nominal plan preparation costs. Success-only completion means
are explicitly conditional and cannot replace the full outcome table.

This is a developmental, conditional study with three selected initial states per
task. It cannot establish universal superiority, population reliability or a general
statistical correlation across manipulation. No extra task search was performed to
manufacture a counterexample.
