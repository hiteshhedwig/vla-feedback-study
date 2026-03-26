"""Audit actual learned-policy evidence and test real simulator interface semantics."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUNS=ROOT/"artifacts/study-v2"

def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines()]

@pytest.fixture(scope="module")
def trials():
    batch=read(RUNS/"batch.json")
    assert batch["complete"]
    return [(r,RUNS/r["trial"]) for r in batch["results"]]

def test_complete_predeclared_matrix(trials):
    design=read(RUNS/"design.json")
    assert len(trials)==len(design["tasks"])*len(design["initial_state_ids"])*4==36
    assert all(r["status"] in {"SUCCESS","TASK_TIMEOUT"} for r,_ in trials)
    assert len({(r["task_key"],r["initial_state_id"],r["mode"],r["condition"]) for r,_ in trials})==36

def test_actual_observation_shapes_and_model_identity(trials):
    checkpoints=set()
    sources=set()
    for result,folder in trials:
        meta=read(folder/"manifest.json")
        checkpoints.add(meta["checkpoint_sha256"]);sources.add(meta["trial_source_sha256"])
        obs=np.load(folder/"first-observation.npz")
        assert obs["state"].shape==(8,)
        for key in ["image","image2"]:
            assert obs[key].shape==(256,256,3)
            assert obs[key].dtype==np.uint8
        for pred in lines(folder/"predictions.jsonl"):
            assert pred["input_shapes"]["observation.state"]==[1,8]
            assert pred["input_shapes"]["observation.images.camera1"]==[1,3,256,256]
            assert pred["input_shapes"]["observation.images.camera2"]==[1,3,256,256]
            assert pred["input_shapes"]["observation.language.tokens"]==[1,48]
    assert checkpoints=={"9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"}
    assert len(sources)==1

def test_actual_predictions_denormalize_with_checkpoint_statistics(trials):
    from safetensors.numpy import load_file
    stats=load_file(str(ROOT/".models/smolvla-libero/policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    for result,folder in trials:
        for pred in lines(folder/"predictions.jsonl"):
            normalized=np.array(pred["normalized_actions"])
            action=np.array(pred["actions"])
            assert action.shape==normalized.shape==(50,7)
            expected=normalized*(stats["action.std"]+1e-8)+stats["action.mean"]
            np.testing.assert_allclose(action,expected,atol=2e-6,rtol=2e-6)
            assert np.isfinite(action).all()

def test_executed_actions_really_come_from_policy_or_recorded_plan(trials):
    for result,folder in trials:
        trace=lines(folder/"actions.jsonl")
        predictions=lines(folder/"predictions.jsonl")
        scenario=read(folder/"scenario.json")
        plan=lines(Path(scenario["plan"])) if scenario["plan"] else None
        assert len(trace)==result["steps"]
        assert len(predictions)==result["policy_calls"]
        if plan is not None:
            assert result["policy_calls"]==0
            assert hashlib.sha256(Path(scenario["plan"]).read_bytes()).hexdigest()==result["plan_sha256"]
        for index,row in enumerate(trace):
            expected=plan[index]["command"] if plan is not None else predictions[index//10]["actions"][index%10]
            np.testing.assert_array_equal(row["raw_action"],expected)
            np.testing.assert_array_equal(row["requested_command"],np.clip(expected,-1,1))
            if row["interrupted"]:
                assert 30<=index<40 and result["condition"]=="interrupted"
                np.testing.assert_array_equal(row["command"][:6],[0]*6)
                assert row["command"][6]==trace[index-1]["command"][6]
            else:np.testing.assert_array_equal(row["command"],row["requested_command"])
        if plan is None:
            assert result["policy_calls"]==math.ceil(result["steps"]/10)
        assert result["reference_continues_after_first_success"]==(result["mode"]=="reference")

def test_task_success_is_supported_by_simulator_receipts(trials):
    for result,folder in trials:
        trace=lines(folder/"actions.jsonl")
        successes=[r["step"]+1 for r in trace if r["success"]]
        assert result["success"]==bool(successes)
        assert result["first_success_step"]==(successes[0] if successes else None)
        assert result["status"]==("SUCCESS" if successes else "TASK_TIMEOUT")
        assert result["steps"]<=300
        if result["mode"]=="reference":assert result["steps"]==300

def test_pairs_start_identically_and_clean_replay_matches_physics(trials):
    for result,folder in trials:
        stem=result["task_key"]+"-state"+str(result["initial_state_id"])
        reference=RUNS/(stem+"-reference")
        a=read(reference/"manifest.json");b=read(folder/"manifest.json")
        assert a["initial_simulator_state_hash"]==b["initial_simulator_state_hash"]
        assert a["initial_image_sha256"]==b["initial_image_sha256"]
        expected=lines(reference/"actions.jsonl");actual=lines(folder/"actions.jsonl")
        if result["mode"]=="open" and result["condition"]=="clean":
            np.testing.assert_allclose([x["physics_state"] for x in actual],
                                       [x["physics_state"] for x in expected[:len(actual)]],atol=1e-10,rtol=0)
            assert result["first_success_step"]==read(reference/"result.json")["first_success_step"]
        else:
            count=min(30,len(actual))
            np.testing.assert_allclose([x["physics_state"] for x in actual[:count]],
                                       [x["physics_state"] for x in expected[:count]],atol=1e-10,rtol=0)

def test_all_interrupted_trials_received_the_same_ten_command_intervention(trials):
    for result,folder in trials:
        count=sum(x["interrupted"] for x in lines(folder/"actions.jsonl"))
        assert count==(10 if result["condition"]=="interrupted" else 0)

def test_closed_loop_receives_new_state_after_first_chunk(trials):
    for result,folder in trials:
        if result["mode"]=="open":continue
        predictions=lines(folder/"predictions.jsonl")
        assert len(predictions)>1
        assert np.linalg.norm(np.array(predictions[1]["state"])-np.array(predictions[0]["state"]))>1e-4
        assert predictions[1]["image_sha256"]!=predictions[0]["image_sha256"]

def test_fresh_project_simulator_preserves_the_reference():
    reference=RUNS/"bowl-state0-reference"
    replay=ROOT/"artifacts/fresh-v2-replay-check"
    a,b=lines(reference/"actions.jsonl"),lines(replay/"actions.jsonl")
    np.testing.assert_allclose([x["physics_state"] for x in b],
                               [x["physics_state"] for x in a[:len(b)]],atol=1e-10,rtol=0)
    assert read(replay/"result.json")["first_success_step"]==read(reference/"result.json")["first_success_step"]

def test_fresh_project_closed_loop_preserves_model_execution():
    reference=RUNS/"bowl-state1-reference"
    replay=ROOT/"artifacts/fresh-v2-closed-check"
    a,b=lines(reference/"actions.jsonl"),lines(replay/"actions.jsonl")
    np.testing.assert_allclose([x["physics_state"] for x in b],
                               [x["physics_state"] for x in a[:len(b)]],atol=1e-10,rtol=0)
    assert read(replay/"result.json")["first_success_step"]==read(reference/"result.json")["first_success_step"]

@pytest.fixture()
def simulation():
    from study.simulation import Simulation
    sim=Simulation("libero_spatial",0,0)
    yield sim
    sim.close()

def test_raw_camera_and_state_conversion(simulation):
    from study.simulation import observation
    from robosuite.utils.transform_utils import quat2axisangle
    converted=observation(simulation.obs)
    np.testing.assert_array_equal(converted["image"],simulation.obs["agentview_image"][::-1,::-1])
    np.testing.assert_array_equal(converted["image2"],simulation.obs["robot0_eye_in_hand_image"][::-1,::-1])
    np.testing.assert_allclose(converted["state"][:3],simulation.obs["robot0_eef_pos"],atol=1e-6)
    np.testing.assert_allclose(converted["state"][3:6],quat2axisangle(simulation.obs["robot0_eef_quat"]),atol=1e-5)
    np.testing.assert_allclose(converted["state"][6:],simulation.obs["robot0_gripper_qpos"],atol=1e-6)

@pytest.mark.parametrize("action",[[0]*6,[0]*8,[float("nan")]+[0]*6,[float("inf")]+[0]*6])
def test_invalid_commands_are_rejected_before_physics(simulation,action):
    before=np.asarray(simulation.state())
    with pytest.raises(ValueError):simulation.step(action)
    np.testing.assert_array_equal(simulation.state(),before)

def test_known_controller_translation_and_gripper_sign(simulation):
    before=float(simulation.obs["robot0_eef_pos"][0])
    for _ in range(5):simulation.step([0.1,0,0,0,0,0,-1])
    assert float(simulation.obs["robot0_eef_pos"][0])>before+0.001
    opened=float(np.abs(simulation.obs["robot0_gripper_qpos"]).sum())
    for _ in range(8):simulation.step([0,0,0,0,0,0,1])
    closed=float(np.abs(simulation.obs["robot0_gripper_qpos"]).sum())
    assert closed<opened-0.01


def test_both_cameras_and_noise_match_before_the_intervention(trials):
    for result,folder in trials:
        predictions=lines(folder/"predictions.jsonl")
        for pred in predictions:
            assert pred["noise_seed"]==1000+pred["step"]//10
            obs=np.load(folder/"observations"/("step"+str(pred["step"])+".npz"))
            assert hashlib.sha256(obs["image"].tobytes()).hexdigest()==pred["image_sha256"]
            assert hashlib.sha256(obs["image2"].tobytes()).hexdigest()==pred["image2_sha256"]
        if result["mode"]!="closed" or result["condition"]!="interrupted":continue
        stem=result["task_key"]+"-state"+str(result["initial_state_id"])
        nominal=lines(RUNS/(stem+"-reference")/"predictions.jsonl")
        for a,b in zip(predictions[:4],nominal[:4]):
            assert a["image_sha256"]==b["image_sha256"]
            assert a["image2_sha256"]==b["image2_sha256"]
            assert a["noise_sha256"]==b["noise_sha256"]
            np.testing.assert_array_equal(a["state"],b["state"])
            np.testing.assert_array_equal(a["actions"],b["actions"])

def test_current_dependency_aligned_runtime_reproduces_all_saved_predictions():
    result=read(ROOT/"artifacts/runtime-revalidation.json")
    assert result["passed"]
    assert result["model_inputs_reexecuted"]==370
    assert all(c["maximum_action_difference"]==0 and c["maximum_normalized_difference"]==0
               and c["noise_matches"] for c in result["checks"])
    assert "No broken requirements found" in (ROOT/"artifacts/policy-dependency-check.log").read_text()
    current=read(ROOT/"artifacts/aligned-runtime-closed-check/result.json")
    reference=read(RUNS/"bowl-state1-reference/result.json")
    assert current["status"]==reference["status"]=="SUCCESS"
    assert current["first_success_step"]==reference["first_success_step"]
