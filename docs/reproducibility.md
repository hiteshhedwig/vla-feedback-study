# Reproducibility and diagnosis

## Primary evidence

artifacts/study-v2 is the primary 36-record study. artifacts/study-v1, the smoke
checks and setup repeats are development/verification evidence and do not increase
the primary denominator.

The final batch uses explicit model-noise seeds, deterministic Torch algorithms,
fixed Python/NumPy seeds, pinned assets and lossless two-camera recording at every
query. Initial-state, replay and pre-intervention audits pass for the recorded batch.

## Why an earlier batch was retained separately

A first audit found a pre-intervention divergence in one pair despite matching
initial states. Replaying identical saved inputs repeatedly produced identical
model outputs, but that batch did not save every wrist-camera input, preventing a
unique attribution of that original discrepancy.

Instead of silently relaxing the matching check, the next protocol recorded both
views for every query, made the action-sampling noise schedule explicit and enabled
deterministic Torch math. All pre-intervention checks passed in the rerun.
The primary scope/task list and interruption severity were not changed to obtain a
preferred outcome.

## A localized later-repeat discrepancy

The additional aligned-runtime closed-loop check reached the same goal at step 83
as its nominal reference, but did not have an identical entire trajectory.

At query step 50, it had:
- Identical eight-value robot state and matched model noise.
- Identical wrist image.
- Two changed agent-camera RGB channel values, each differing by 1/255.
- A changed next predicted action.

Subsequent robot states and camera observations then diverged. The saved diagnostics
localize this later discrepancy to the observation stage; they do not prove a
specific graphics-driver implementation cause. The original development discrepancy
cannot be retroactively attributed with the missing wrist evidence.

See artifacts/diagnostics/aligned-repeat-input-differences.json. This is not a new
adversarial-input study. It is a concrete limit on claiming bitwise closed-loop
reproducibility.

## Dependency alignment

The initial lightweight policy-extra installation selected NumPy, packaging and
fsspec versions outside some inherited LeRobot/datasets dependency bounds.
The project-local policy environment was aligned to NumPy 2.2.6, packaging 25.0 and
fsspec 2026.2.0. The current dependency check reports no broken requirements.

Collection versions and current runtime versions are stored separately:
configs/policy-collection-versions.json and configs/policy-versions.json.

Every one of the primary study's 370 saved model inputs was then passed through
the current checkpoint/runtime with its original explicit noise seed. All normalized
and denormalized predictions matched exactly. This revalidates the recorded policy
outputs under the aligned runtime; it is not 370 new robot episodes and does not
change recorded latency measurements.

The fresh CPU simulator environment was independently installed and checked with
actual replay and live-model execution. Existing repositories and environments were
not modified.

## What is and is not reproduced

- Clean nominal action replay reproduced the recorded physics in this batch.
- Paired prefix inputs and commands match before the imposed interruption.
- Fixed-input model inference reproduced exactly on the inspected machine.
- A later full feedback rollout can diverge when rendered pixels differ.
- None of this guarantees bitwise equivalence on other GPUs, drivers or library
  builds, or hardware behavior.

A valid rerun should preserve its own inputs, versions, outcome and checks rather
than overwrite these artifacts.
