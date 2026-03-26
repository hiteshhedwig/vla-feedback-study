"""Local-only SmolVLA inference, isolated from the older simulator dependencies."""
import os
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".deps/lerobot/src"))
import base64
from http.server import BaseHTTPRequestHandler,HTTPServer
import hashlib
import json
import time
import traceback
import numpy as np
import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

class Policy:
    def __init__(self):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        path=ROOT/".models/smolvla-libero"
        config=PreTrainedConfig.from_pretrained(str(path))
        config.load_vlm_weights=False
        config.vlm_model_name=str(ROOT/".models/vlm-assets")
        config.device="cuda"
        config.use_amp=True
        config.compile_model=False
        self.policy=SmolVLAPolicy.from_pretrained(str(path),config=config,strict=True).eval().to("cuda")
        self.pre,self.post=make_pre_post_processors(config,pretrained_path=str(path),
            preprocessor_overrides={"tokenizer_processor":{"tokenizer_name":str(ROOT/".models/vlm-assets")}})
        self.metadata={**json.loads((ROOT/"configs/model-lock.json").read_text()),
                       "torch":torch.__version__,"device":torch.cuda.get_device_name(0),
                       "precision":"BF16 autocast; parameter dtypes recorded separately","prediction_horizon":config.chunk_size,
                       "server_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "parameter_count":sum(p.numel() for p in self.policy.parameters()),
                       "parameter_dtypes":sorted({str(p.dtype) for p in self.policy.parameters()}),
                       "noise_schedule":"seed 1000 + query control-step // 10, explicit CUDA generator",
                       "deterministic_algorithms":True,
                       "execution_horizon":10,"state_dim":8,"action_dim":7,
                       "checkpoint_sha256":hashlib.sha256((path/"model.safetensors").read_bytes()).hexdigest()}
        self.reset(0)
    def reset(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.policy.reset()
        return self.metadata
    def infer(self,request):
        state=np.asarray(request["state"],dtype=np.float32)
        if state.shape!=(8,) or not np.isfinite(state).all():raise ValueError("Invalid 8D state")
        batch={"observation.state":torch.from_numpy(state),"task":request["instruction"]}
        for name in ["image","image2"]:
            raw=base64.b64decode(request[name],validate=True)
            image=np.frombuffer(raw,dtype=np.uint8).reshape(256,256,3).copy()
            batch["observation.images."+name]=torch.from_numpy(image).permute(2,0,1).float()/255.0
        torch.cuda.synchronize()
        start=time.perf_counter()
        inputs=self.pre(batch)
        noise_seed=1000+int(request.get("query_step",0))//10
        generator=torch.Generator(device="cuda").manual_seed(noise_seed)
        noise=torch.randn((1,50,32),device="cuda",dtype=torch.float32,generator=generator)
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            normalized=self.policy.predict_action_chunk(inputs,noise=noise)
            actions=self.post(normalized)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        actions=actions.detach().float().cpu().numpy()[0]
        normalized=normalized.detach().float().cpu().numpy()[0]
        if actions.shape!=(50,7) or not np.isfinite(actions).all():raise ValueError("Invalid policy chunk")
        return {"noise_seed":noise_seed,"noise_sha256":hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                "actions":actions.tolist(),"normalized_actions":normalized.tolist(),
                "inference_seconds":seconds,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),
                "input_shapes":{k:list(v.shape) for k,v in inputs.items() if hasattr(v,"shape")}}

def main():
    policy=Policy()
    class Handler(BaseHTTPRequestHandler):
        def send(self,obj,status=200):
            body=json.dumps(obj,allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.send(policy.metadata)
        def do_POST(self):
            try:
                request=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send(policy.reset(request["seed"]) if self.path=="/reset" else policy.infer(request))
            except Exception as exc:
                traceback.print_exc()
                self.send({"error":type(exc).__name__+": "+str(exc)},500)
        def log_message(self,*args):pass
    print("READY",json.dumps(policy.metadata),flush=True)
    HTTPServer(("127.0.0.1",8766),Handler).serve_forever()
if __name__=="__main__":main()
