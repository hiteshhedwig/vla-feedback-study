"""One-scene milestone-2 smoke check: execute 1/10/50 predictions per inference."""
import argparse,json,math,os,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/milestone2"
def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUT)
    OUT=parser.parse_args().output.resolve()
    OUT.mkdir(parents=True,exist_ok=False)
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8766/health",timeout=1) as r:json.load(r)
            break
        except Exception:time.sleep(0.5)
    else:raise RuntimeError("Start scripts/policy_server.py first")
    rows=[]
    for k in [10,1,50]:
        out=OUT/("horizon-"+str(k))
        with (OUT/("horizon-"+str(k)+".log")).open("w") as log:
            subprocess.run([str(ROOT/".venv-sim/bin/python"),str(ROOT/"scripts/run_trial.py"),
                "--suite","libero_spatial","--task","0","--initial","0","--mode","closed",
                "--steps","150","--execution-horizon",str(k),"--output",str(out)],
                cwd=ROOT,env={**os.environ,"PYTHONHASHSEED":"0","PYTHONDONTWRITEBYTECODE":"1"},
                stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
        r=json.loads((out/"result.json").read_text())
        predictions=[json.loads(x) for x in (out/"predictions.jsonl").read_text().splitlines()]
        trace=[json.loads(x) for x in (out/"actions.jsonl").read_text().splitlines()]
        assert r["policy_calls"]==math.ceil(r["steps"]/k)==len(predictions)
        for step,state in enumerate(trace):
            assert state["raw_action"]==predictions[step//k]["actions"][step%k]
        row={"execution_horizon":k,"success":r["success"],"status":r["status"],"actions":r["steps"],
             "model_calls":r["policy_calls"],"total_inference_s":sum(r["inference_seconds"]),
             "mean_inference_s":sum(r["inference_seconds"])/r["policy_calls"],
             "amortized_inference_s_per_action":sum(r["inference_seconds"])/r["steps"],
             "wall_seconds":r["wall_seconds"],"provenance_verified":True}
        rows.append(row)
        print(json.dumps(row),flush=True)
        (OUT/"partial-results.json").write_text(json.dumps(rows,indent=2)+"\n")
    rows.sort(key=lambda r:r["execution_horizon"])
    summary={"scene":"LIBERO-Spatial task 0, initial state 0","task":"bowl placement",
             "max_actions":150,"trial_count":3,"conditions":rows,"run_order":[10,1,50],
             "predicted_chunk_size":50,"artificial_delay":False,
             "scope":"One-scene smoke check, not a statistical study or comparison to a natively sequential model.",
             "timing":"Physics pauses during inference; no real-time deadline or action-starvation claim. Wall times include simulator setup; initial cold inference can affect timing."}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# Milestone 2: one-scene action-buffer smoke check\n\n",
       "Same pretrained SmolVLA, same bowl-placement scene and 150-action budget. Every inference predicts 50 actions. Only the execution horizon changes. No artificial delays.\n\n",
       "| Actions used per inference | Outcome | Executed actions | Model calls | Total inference | Inference / executed action |\n| --- | --- | --- | --- | --- | --- |\n"]
    for r in rows:
        lines.append("| {} | {} | {} | {} | {:.2f} s | {:.3f} s |\n".format(r["execution_horizon"],r["status"],r["actions"],r["model_calls"],r["total_inference_s"],r["amortized_inference_s_per_action"]))
    lines+=["\nK=1 discards 49 predictions and replans after each action: it is not a model trained to emit single actions. K=10/50 reuse more of each chunk, amortizing inference but delaying fresh feedback. This checks execution buffering; it does not isolate the model architecture's intrinsic chunk-generation advantage.\n\n",
       "Simulation pauses for inference. These timings show computation cost, not real-time deadline handling. Run order is 10, 1, 50; cold-start and setup overhead are retained. One trial per condition does not establish reliability or a generally optimal horizon. Milestone 1's archived results are unchanged.\n"]
    (OUT/"report.md").write_text("".join(lines))
if __name__=="__main__":main()
