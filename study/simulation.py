"""Thin access to pinned LIBERO/robosuite; no custom robot controller."""
import os
import random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/".runtime/libero"
RUNTIME.mkdir(parents=True,exist_ok=True)
(RUNTIME/"datasets").mkdir(exist_ok=True)
os.environ["LIBERO_CONFIG_PATH"]=str(RUNTIME)
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".runtime/matplotlib"))
import sys
import yaml
LIBERO=ROOT/".deps/LIBERO"
BASE=LIBERO/"libero/libero"
(RUNTIME/"config.yaml").write_text(yaml.safe_dump({
    "benchmark_root":str(BASE),"bddl_files":str(BASE/"bddl_files"),
    "init_states":str(BASE/"init_files"),"datasets":str(RUNTIME/"datasets"),
    "assets":str(BASE/"assets")}))
sys.path.insert(0,str(LIBERO))
import numpy as np
import torch
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def axis_angle(quaternion):
    q=np.asarray(quaternion,dtype=np.float32)
    w=np.clip(q[3],-1,1)
    denominator=np.sqrt(max(0,1-w*w))
    if denominator<=1e-10:return np.zeros(3,dtype=np.float32)
    return np.asarray(q[:3]/denominator*(2*np.arccos(w)),dtype=np.float32)

def observation(obs):
    cameras={key:np.ascontiguousarray(obs[source][::-1,::-1])
             for key,source in [("image","agentview_image"),("image2","robot0_eye_in_hand_image")]}
    state=np.concatenate([obs["robot0_eef_pos"],axis_angle(obs["robot0_eef_quat"]),
                          obs["robot0_gripper_qpos"]]).astype(np.float32)
    if state.shape!=(8,) or not np.isfinite(state).all():raise ValueError("Invalid simulator state")
    return {**cameras,"state":state}

class Simulation:
    def __init__(self,suite_name,task_id,initial_id,seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        suite=benchmark.get_benchmark_dict()[suite_name]()
        self.task=suite.get_task(task_id)
        base=Path(get_libero_path("bddl_files"))/self.task.problem_folder/self.task.bddl_file
        self.env=OffScreenRenderEnv(bddl_file_name=str(base),camera_heights=256,camera_widths=256,
                                   control_freq=20,ignore_done=True)
        self.env.seed(seed)
        self.env.reset()
        initial_file=Path(get_libero_path("init_states"))/self.task.problem_folder/self.task.init_states_file
        initial_states=torch.load(initial_file,map_location="cpu",weights_only=False)
        self.obs=self.env.set_init_state(initial_states[initial_id])
        for _ in range(10):
            self.obs,_,_,_=self.env.step([0,0,0,0,0,0,-1])
        self.counter=0
    def step(self,action):
        command=np.asarray(action,dtype=float)
        if command.shape!=(7,) or not np.isfinite(command).all():raise ValueError("Invalid action")
        self.obs,_,_,_=self.env.step(command.tolist())
        self.counter+=1
        return bool(self.env.check_success())
    def state(self):
        return self.env.get_sim_state().tolist()
    def close(self):
        self.env.close()
