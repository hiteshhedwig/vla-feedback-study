"""Audit the three actual one-scene horizon executions."""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/milestone2"
def test_three_horizons_use_the_actual_predicted_actions():
    summary=json.loads((OUT/"summary.json").read_text())
    assert summary["trial_count"]==3
    assert [r["execution_horizon"] for r in summary["conditions"]]==[1,10,50]
    initial_hashes=set()
    for row in summary["conditions"]:
        k=row["execution_horizon"];folder=OUT/("horizon-"+str(k))
        manifest=json.loads((folder/"manifest.json").read_text())
        assert manifest["execution_horizon"]==k
        initial_hashes.add(manifest["initial_simulator_state_hash"])
        trace=[json.loads(x) for x in (folder/"actions.jsonl").read_text().splitlines()]
        predictions=[json.loads(x) for x in (folder/"predictions.jsonl").read_text().splitlines()]
        assert len(predictions)==row["model_calls"]==math.ceil(len(trace)/k)
        for step,event in enumerate(trace):
            np.testing.assert_array_equal(event["raw_action"],predictions[step//k]["actions"][step%k])
        assert row["success"]==any(event["success"] for event in trace)
    assert len(initial_hashes)==1
