"""Assemble validated task clips and a compact animated README preview."""
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/demo"
def run(*args):subprocess.run(list(args),check=True)
def main():
    full=OUT/"complete-walkthrough.mp4"
    gif=OUT/"preview.gif"
    if full.exists() or gif.exists():raise FileExistsError("Refusing to overwrite assembled media")
    tasks=["bowl","drawer","stove"]
    durations={}
    for task in tasks:
        file=OUT/(task+".mp4")
        run("ffmpeg","-v","error","-xerror","-i",str(file),"-f","null","-")
        durations[task]=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1",str(file)],text=True))
    playlist=OUT/"playlist.txt"
    playlist.write_text("".join("file '{}.mp4'\n".format(t) for t in tasks))
    run("ffmpeg","-y","-v","error","-f","concat","-safe","0","-i",str(playlist),
        "-c","copy","-movflags","+faststart",str(full))
    clips=[]
    for task,start in [("bowl",20),("drawer",24),("stove",18)]:
        clips += [(task,start,min(start+10,durations[task])),(task,durations[task]-5,durations[task])]
    command=["ffmpeg","-y","-v","error"]
    filters=[]
    for index,(task,start,end) in enumerate(clips):
        command+=["-ss",str(start),"-t",str(end-start),"-i",str(OUT/(task+".mp4"))]
        filters.append(f"[{index}:v]fps=5,scale=960:-1:flags=lanczos,setpts=PTS-STARTPTS[v{index}]")
    filters.append("".join(f"[v{i}]" for i in range(len(clips)))+"concat=n=6:v=1:a=0,split[a][b]")
    filters+=["[a]palettegen=max_colors=64:stats_mode=diff[p]",
              "[b][p]paletteuse=dither=bayer:bayer_scale=5"]
    command+=["-filter_complex",";".join(filters),"-loop","0",str(gif)]
    run(*command)
    for path in [full,gif]:run("ffmpeg","-v","error","-xerror","-i",str(path),"-f","null","-")
    (OUT/"media-validation.json").write_text(json.dumps({"passed":True,
        "fully_decoded":[t+".mp4" for t in tasks]+[full.name,gif.name],
        "task_clip_duration_seconds":durations,"gif_excerpts":clips,
        "note":"All clips use predefined initial state 0; complete per-state outcomes are reported separately."},indent=2)+"\n")
    print("Walkthrough and animation assembled; all five media files fully decoded")
if __name__=="__main__":main()
