"""Replay identical controls and save lossless inputs for a reproducibility diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from study.simulation import Simulation,observation
p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
reference=ROOT/"artifacts/study-v1/stove-state2-reference"
commands=[json.loads(l)["command"] for l in (reference/"actions.jsonl").read_text().splitlines()]
sim=Simulation("libero_goal",7,2)
records=[]
try:
 for step in range(31):
  if step in [0,10,20,30]:
   obs=observation(sim.obs)
   np.savez_compressed(a.output/f"step{step}.npz",**obs)
   records.append({"step":step,"state":sim.state(),"image":hashlib.sha256(obs["image"].tobytes()).hexdigest(),
                   "image2":hashlib.sha256(obs["image2"].tobytes()).hexdigest()})
  if step<30:sim.step(commands[step])
finally:sim.close()
(a.output/"snapshots.json").write_text(json.dumps(records,indent=2)+"\n")
