"""Execute the frozen three-task paired feedback study; retain every outcome."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/study.json")
    parser.add_argument("--sim-python",default=str(ROOT/".venv-sim/bin/python"))
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if config["execution_horizon"]!=10 or config["interruption_steps"]!=[30,40]:
        raise ValueError("Milestone 1 freezes horizon and interruption protocol")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"design.json").write_text(json.dumps(config,indent=2)+"\n")
    rows=[]
    reproducibility=[]
    start=time.monotonic()
    for task in config["tasks"]:
        for initial in config["initial_state_ids"]:
            stem=task["id"]+"-state"+str(initial)
            reference=args.output/(stem+"-reference")
            cases=[("reference",False,reference,None),
                   ("open",False,args.output/(stem+"-open-clean"),reference/"actions.jsonl"),
                   ("closed",True,args.output/(stem+"-closed-interrupted"),None),
                   ("open",True,args.output/(stem+"-open-interrupted"),reference/"actions.jsonl")]
            for mode,interrupted,output,plan in cases:
                command=[args.sim_python,str(ROOT/"scripts/run_trial.py"),"--suite",task["suite"],
                         "--task",str(task["task"]),"--initial",str(initial),
                         "--mode",mode,"--steps",str(config["max_actions"]),"--output",str(output)]
                if plan:command+=["--plan",str(plan)]
                if interrupted:command+=["--interruption"]
                with (args.output/(output.name+".log")).open("w") as log:
                    done=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                        env={**os.environ,"MUJOCO_GL":"egl","PYTHONDONTWRITEBYTECODE":"1","PYTHONHASHSEED":"0"},
                                        timeout=120)
                if not (output/"result.json").exists():
                    raise RuntimeError("Worker has no receipt: "+str(output))
                result=json.loads((output/"result.json").read_text())
                result.update(task_key=task["id"],trial=output.name)
                rows.append(result)
                (args.output/"batch.json").write_text(json.dumps({"results":rows,"complete":False},indent=2)+"\n")
                print(output.name,result["status"],"first success",result["first_success_step"],flush=True)
                if done.returncode or result["status"]=="INFRASTRUCTURE_ERROR":
                    raise RuntimeError("Infrastructure error retained: "+str(output))
            reference_trace=[json.loads(x) for x in (reference/"actions.jsonl").read_text().splitlines()]
            for _,_,output,_ in cases[1:]:
                trace=[json.loads(x) for x in (output/"actions.jsonl").read_text().splitlines()]
                count=len(trace) if "open-clean" in output.name else min(30,len(trace))
                delta=float(np.max(np.abs(
                    np.array([x["physics_state"] for x in trace[:count]])-
                    np.array([x["physics_state"] for x in reference_trace[:count]]))))
                entry={"trial":output.name,"compared_steps":count,"maximum_physics_state_difference":delta}
                reproducibility.append(entry)
                (args.output/"prefix-checks.json").write_text(json.dumps(reproducibility,indent=2)+"\n")
                if delta>1e-10:
                    raise RuntimeError("Pre-intervention/clean replay mismatch: "+str(entry))
    summary={"results":rows,"complete":True,"elapsed_wall_seconds":time.monotonic()-start,
             "valid_trials":sum(r["status"] in ["SUCCESS","TASK_TIMEOUT"] for r in rows),
             "infrastructure_failures":sum(r["status"]=="INFRASTRUCTURE_ERROR" for r in rows)}
    (args.output/"batch.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Completed",len(rows),"valid paired episodes",flush=True)
if __name__=="__main__":main()
