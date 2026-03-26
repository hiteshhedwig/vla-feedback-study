"""Lossless localhost image/state transport; inference is synchronous."""
import base64
import json
import time
import urllib.request
import numpy as np
URL="http://127.0.0.1:8766"
def request(path,payload=None):
    raw=None if payload is None else json.dumps(payload,allow_nan=False).encode()
    req=urllib.request.Request(URL+path,data=raw,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)
def reset(seed):return request("/reset",{"seed":seed})
def predict(obs,instruction,query_step=0):
    payload={"state":obs["state"].tolist(),"instruction":instruction,"query_step":query_step}
    for name in ["image","image2"]:payload[name]=base64.b64encode(obs[name].tobytes()).decode()
    start=time.perf_counter()
    reply=request("/infer",payload)
    reply["request_seconds"]=time.perf_counter()-start
    actions=np.asarray(reply["actions"],dtype=float)
    if actions.shape!=(50,7) or not np.isfinite(actions).all():raise ValueError("Invalid chunk response")
    return reply
