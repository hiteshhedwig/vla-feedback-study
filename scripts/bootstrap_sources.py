"""Fetch/verify pinned upstream sources, optionally reusing clean local checkouts."""
import argparse
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--verify",action="store_true")
    p.add_argument("--reuse-lerobot",type=Path)
    p.add_argument("--reuse-libero",type=Path)
    args=p.parse_args()
    lock=json.loads((ROOT/"configs/sources-lock.json").read_text())
    for name,source in lock.items():
        repo=ROOT/".deps"/name
        reuse=args.reuse_lerobot if name=="lerobot" else args.reuse_libero
        if not repo.exists():
            if args.verify:raise RuntimeError("Missing source: "+name)
            repo.parent.mkdir(exist_ok=True)
            if reuse is not None:repo.symlink_to(reuse.resolve(),target_is_directory=True)
            else:
                subprocess.run(["git","clone","--depth","1","--filter=blob:none","--no-checkout",source["url"],str(repo)],check=True)
                subprocess.run(["git","-C",str(repo),"fetch","--depth","1","origin",source["revision"]],check=True)
                subprocess.run(["git","-C",str(repo),"sparse-checkout","init","--cone"],check=True)
                subprocess.run(["git","-C",str(repo),"sparse-checkout","set",source["sparse"]],check=True)
                subprocess.run(["git","-C",str(repo),"checkout","--detach",source["revision"]],check=True)
        revision=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        changes=subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],text=True)
        if revision!=source["revision"] or changes:raise RuntimeError("Source differs from clean pinned version: "+name)
        print("Verified",name,revision)
if __name__=="__main__":main()
