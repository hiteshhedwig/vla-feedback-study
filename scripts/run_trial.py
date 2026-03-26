"""A real policy-controlled episode or explicitly labeled nominal action replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import imageio.v2 as imageio
import numpy as np
from study.simulation import Simulation,observation
from study import client

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--suite",required=True)
    parser.add_argument("--task",type=int,required=True)
    parser.add_argument("--initial",type=int,required=True)
    parser.add_argument("--mode",choices=["reference","closed","open"],required=True)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--steps",type=int,default=300)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--interruption",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config={**vars(args),"output":str(args.output),"plan":str(args.plan) if args.plan else None,
            "interruption_steps":[30,40] if args.interruption else None,
            "execution_horizon":10,"model_seed":1000,"environment_seed":0}
    (args.output/"scenario.json").write_text(json.dumps(config,indent=2)+"\n")
    sim=None
    result={"status":"INFRASTRUCTURE_ERROR","success":False,"mode":args.mode,"suite":args.suite,
            "task_id":args.task,"initial_state_id":args.initial,"policy_calls":0,"steps":0,
            "first_success_step":None,"inference_seconds":[],"request_seconds":[],
            "condition":"interrupted" if args.interruption else "clean",
            "clipped_action_elements":0,"scope":"Nominal VLA plan replay vs fresh-feedback VLA; no custom motion controller"}
    started=time.perf_counter()
    try:
        meta=client.reset(1000) if args.mode!="open" else json.loads((args.plan.parent/"manifest.json").read_text())
        sim=Simulation(args.suite,args.task,args.initial)
        obs=observation(sim.obs)
        result["instruction"]=sim.task.language
        initial_hash=hashlib.sha256(np.asarray(sim.state(),dtype=np.float64).tobytes()).hexdigest()
        import subprocess
        code=hashlib.sha256()
        for path in sorted((ROOT/"study").glob("*.py"))+[ROOT/"scripts/run_trial.py"]:
            code.update(path.name.encode());code.update(path.read_bytes())
        meta.update(trial_source_sha256=code.hexdigest(),
                    simulator_versions={"numpy":np.__version__,
                        "mujoco":__import__("mujoco").__version__,
                        "robosuite":__import__("robosuite").__version__,
                        "torch":__import__("torch").__version__},
                    libero_revision=subprocess.check_output(["git","-C",str(ROOT/".deps/LIBERO"),"rev-parse","HEAD"],text=True).strip(),
                    lerobot_revision=subprocess.check_output(["git","-C",str(ROOT/".deps/lerobot"),"rev-parse","HEAD"],text=True).strip(),
                    initial_simulator_state_hash=initial_hash,initial_state=sim.state(),
                    initial_image_sha256=hashlib.sha256(obs["image"].tobytes()).hexdigest(),
                    simulation_paused_during_inference=True,control_hz=20)
        (args.output/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
        imageio.imwrite(args.output/"initial.png",obs["image"])
        np.savez_compressed(args.output/"first-observation.npz",**obs)
        plan=None
        if args.mode=="open":
            if args.plan is None:raise ValueError("Open-loop replay requires a recorded VLA plan")
            plan=[json.loads(line)["command"] for line in args.plan.read_text().splitlines()]
            if len(plan)<args.steps:raise ValueError("Nominal plan shorter than common task budget")
            result["plan_sha256"]=hashlib.sha256(args.plan.read_bytes()).hexdigest()
        (args.output/"observations").mkdir()
        queue=[]
        last_gripper=-1.0
        with imageio.get_writer(args.output/"episode.mp4",fps=20) as video, \
             (args.output/"actions.jsonl").open("w") as log, \
             (args.output/"predictions.jsonl").open("w") as predictions:
            video.append_data(np.concatenate([obs["image"],obs["image2"]],axis=1))
            for step in range(args.steps):
                call=None
                if args.mode=="open":
                    raw=np.asarray(plan[step])
                else:
                    if not queue:
                        response=client.predict(obs,sim.task.language,step)
                        result["policy_calls"]+=1
                        call=result["policy_calls"]-1
                        result["inference_seconds"].append(response["inference_seconds"])
                        result["request_seconds"].append(response["request_seconds"])
                        np.savez_compressed(args.output/"observations"/("step"+str(step)+".npz"),**obs)
                        predictions.write(json.dumps({"step":step,"call":call,**response,
                            "state":obs["state"].tolist(),
                            "image2_sha256":hashlib.sha256(obs["image2"].tobytes()).hexdigest(),
                            "image_sha256":hashlib.sha256(obs["image"].tobytes()).hexdigest()})+"\n")
                        predictions.flush()
                        queue=list(response["actions"][:10])
                        print(args.mode,args.suite,args.task,args.initial,"step",step,
                              "inference",round(response["inference_seconds"],3),flush=True)
                    raw=np.asarray(queue.pop(0),dtype=float)
                if raw.shape!=(7,) or not np.isfinite(raw).all():raise ValueError("Non-finite/wrong-sized action")
                command=np.clip(raw,-1,1)
                result["clipped_action_elements"]+=int(np.count_nonzero(raw!=command))
                before=obs["state"].copy()
                requested=command.copy()
                interrupted=args.interruption and 30<=step<40
                if interrupted:
                    command=np.array([0,0,0,0,0,0,last_gripper],dtype=float)
                success=sim.step(command)
                last_gripper=float(command[6])
                obs=observation(sim.obs)
                result["steps"]=step+1
                if success and result["first_success_step"] is None:
                    result["first_success_step"]=step+1
                    result["first_success_wall_seconds"]=time.perf_counter()-started
                result["success"]=result["first_success_step"] is not None
                log.write(json.dumps({"step":step,"raw_action":raw.tolist(),"command":command.tolist(),
                                      "requested_command":requested.tolist(),"interrupted":interrupted,
                                      "state_before":before.tolist(),"state_after":obs["state"].tolist(),
                                      "physics_state":sim.state(),"success":success,
                                      "policy_call":call,
                                      "source_plan_step":step if args.mode=="open" else None,"simulation_time_s":(step+1)/20})+"\n")
                log.flush()
                video.append_data(np.concatenate([obs["image"],obs["image2"]],axis=1))
                if success and args.mode!="reference":break
        result["status"]="SUCCESS" if result["success"] else "TASK_TIMEOUT"
        result["reference_continues_after_first_success"]=args.mode=="reference"
    except Exception as exc:
        import traceback
        result["error"]=type(exc).__name__+": "+str(exc)
        (args.output/"error.log").write_text(traceback.format_exc())
        raise
    finally:
        result["wall_seconds"]=time.perf_counter()-started
        (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        if sim is not None:sim.close()
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
