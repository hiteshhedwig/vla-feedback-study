"""Download only the pinned policy and tokenizer/config assets; no datasets."""
import json
import os
from pathlib import Path
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
MODEL="lerobot/smolvla_libero"
REVISION="a24f1f380461b665158d52acd140152c79e8b36d"
TOKENIZER="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TOKENIZER_REVISION="7b375e1b73b11138ff12fe22c8f2822d8fe03467"
def main():
    checkpoint=ROOT/".models/smolvla-libero"
    snapshot_download(MODEL,revision=REVISION,local_dir=checkpoint,
        allow_patterns=["config.json","model.safetensors","policy_*","README.md","train_config.json"])
    snapshot_download(TOKENIZER,revision=TOKENIZER_REVISION,local_dir=ROOT/".models/vlm-assets",
        allow_patterns=["*.json","*.txt","*.model","*.py","merges*"],ignore_patterns=["*.safetensors","*.bin"])
    metadata={"checkpoint":MODEL,"revision":REVISION,"tokenizer":TOKENIZER,
              "tokenizer_revision":TOKENIZER_REVISION,"no_training_data":True}
    (ROOT/"configs/model-lock.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Pinned model and tokenizer assets downloaded",flush=True)
if __name__=="__main__":main()
