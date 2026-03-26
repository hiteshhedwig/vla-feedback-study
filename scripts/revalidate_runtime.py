"""Replay every recorded model input through the current runtime; no new physics."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from study import client

def main():
    runs=ROOT/"artifacts/study-v2"
    batch=json.loads((runs/"batch.json").read_text())
    checks=[]
    for row in batch["results"]:
        folder=runs/row["trial"]
        records=[json.loads(line) for line in (folder/"predictions.jsonl").read_text().splitlines()]
        if not records:continue
        client.reset(1000)
        for record in records:
            obs=np.load(folder/"observations"/("step"+str(record["step"])+".npz"))
            response=client.predict(obs,row["instruction"],record["step"])
            difference=float(np.max(np.abs(np.array(record["actions"])-np.array(response["actions"]))))
            normalized=float(np.max(np.abs(np.array(record["normalized_actions"])-np.array(response["normalized_actions"]))))
            checks.append({"trial":row["trial"],"step":record["step"],
                           "maximum_action_difference":difference,"maximum_normalized_difference":normalized,
                           "noise_matches":record["noise_sha256"]==response["noise_sha256"]})
        print(row["trial"],"checked",len(records),"actual model inputs",flush=True)
    result={"passed":all(c["maximum_action_difference"]==0 and c["maximum_normalized_difference"]==0 and c["noise_matches"] for c in checks),
            "model_inputs_reexecuted":len(checks),"checks":checks,
            "note":"Post dependency-bound alignment, all saved primary-study inputs were reexecuted with the same checkpoint, deterministic math and explicit noise. This does not add robot episodes to the study denominator."}
    (ROOT/"artifacts/runtime-revalidation.json").write_text(json.dumps(result,indent=2)+"\n")
    if not result["passed"]:raise RuntimeError("Current runtime differs from recorded predictions")
    print("All",len(checks),"recorded model predictions reproduced exactly",flush=True)
if __name__=="__main__":main()
