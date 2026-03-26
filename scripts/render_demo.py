"""Readable side-by-side clips from recorded simulation videos; no new rollouts."""
import argparse
import json
from pathlib import Path
import subprocess
import textwrap
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W,H,FPS=1440,1080,20
COLORS={"bg":"#101820","fg":"#eef2f5","muted":"#bac4cc","green":"#59d5bb","red":"#ffb166"}
def read(p):return json.loads(p.read_text())
def font(n,bold=False):return ImageFont.truetype(BOLD if bold else FONT,n)
def words(draw,x,y,text,width=55,color=None,size=21):
    for line in textwrap.wrap(text,width):
        draw.text((x,y),line,font=font(size),fill=color or COLORS["fg"]);y+=size+7
def frames(path):
    with imageio.get_reader(path) as r:return [frame for frame in r]

def render(runs,out,task):
    left=runs/(task+"-state0-closed-interrupted")
    right=runs/(task+"-state0-open-interrupted")
    results=[read(p/"result.json") for p in [left,right]]
    videos=[frames(p/"episode.mp4") for p in [left,right]]
    maximum=max(r["steps"] for r in results)
    title={"bowl":"Bowl placement","drawer":"Drawer opening","stove":"Stove switch"}[task]
    output=out/(task+".mp4")
    encoder=subprocess.Popen(["ffmpeg","-y","-v","error","-f","rawvideo","-pixel_format","rgb24",
        "-video_size",f"{W}x{H}","-framerate",str(FPS),"-i","-","-an","-c:v","libx264",
        "-threads","2","-preset","fast","-crf","23","-pix_fmt","yuv420p","-movflags","+faststart",str(output)],
        stdin=subprocess.PIPE)
    timeline=[]
    def picture(source,card=None):
        canvas=Image.new("RGB",(W,H),COLORS["bg"])
        d=ImageDraw.Draw(canvas)
        d.text((24,20),"VLA Feedback Study | "+title,font=font(32,True),fill=COLORS["fg"])
        d.text((24,72),"Same initial state and action budget. Low-level robot feedback remains enabled in both.",
               font=font(20),fill=COLORS["muted"])
        for i,(x,label,result,clip) in enumerate(zip([24,756],
                ["CLOSED LOOP | fresh VLA observations","OPEN LOOP | nominal VLA plan replay"],results,videos)):
            d.text((x,125),label,font=font(22,True),fill=COLORS["green"] if i==0 else COLORS["red"])
            index=min(source,len(clip)-1)
            raw=clip[index]
            main=Image.fromarray(raw[:,:256]).resize((660,660),Image.Resampling.LANCZOS)
            canvas.paste(main,(x,170))
            wrist=Image.fromarray(raw[:,256:]).resize((142,142),Image.Resampling.LANCZOS)
            canvas.paste(wrist,(x+506,182))
            d=ImageDraw.Draw(canvas)
            d.rectangle((x+503,179,x+651,327),outline=COLORS["fg"],width=2)
            d.rectangle((x+506,302,x+648,324),fill=COLORS["bg"])
            d.text((x+523,303),"Wrist input",font=font(15),fill=COLORS["fg"])
            complete=result["first_success_step"] is not None and source>=result["first_success_step"]
            status=("SUCCESS at {:.2f} simulated s".format(result["first_success_step"]/20) if complete else
                    "TASK TIMEOUT at 15.00 simulated s" if source>=result["steps"] and not result["success"] else
                    "Task in progress")
            d.text((x,846),status,font=font(23,True),fill=COLORS["green"] if complete else COLORS["red"])
            if source>result["steps"]:
                description="This pane holds its final recorded frame after termination. No additional robot motion is implied."
            elif card=="intro":
                description=("This policy receives both cameras, robot state and instruction; it replans every ten commands." if i==0 else
                    "Every action came from this VLA's nominal reference rollout on the same initial state. No new observations are used here.")
            elif card in ("on","off") or 31<=source<=50:
                description=("Ten commands are withheld from both robots. Fresh feedback lets this policy observe the resulting execution lag." if i==0 else
                    "Ten commands are withheld here too. Replay then continues its fixed nominal sequence without observing the lag.")
            elif card=="outcome":
                description=("Outcome comes from LIBERO's task predicate. Valid model predictions alone do not establish task completion." if i==0 else
                    "Replay has zero online inference calls, but producing its nominal plan required thirty VLA calls.")
            else:
                description=("Current camera/state observations drive new predicted actions. No custom recovery or scripted grasp is added." if i==0 else
                    "The recorded plan is executed as controller commands. This is not a single-shot full-task prediction.")
            words(d,x,890,description,width=55,size=21)
        phase="READING PAUSE: recorded frames held" if card else "RECORDED EXECUTION AT 0.5x SPEED"
        d.text((24,1037),f"{phase} | Source step {source}/{maximum} | {source/20:.2f} simulated s",
               font=font(21,True),fill=COLORS["green"])
        if 31<=source<=40:
            d.rectangle((465,770,975,817),fill=COLORS["bg"])
            d.text((479,779),"10-COMMAND INTERRUPTION ACTIVE",font=font(22,True),fill=COLORS["red"])
        return canvas.tobytes()
    output_count=0
    try:
        for source in range(maximum+1):
            hold=6 if source==0 else 5 if source in (31,41) else 8 if source==maximum else 0
            card="intro" if source==0 else "on" if source==31 else "off" if source==41 else "outcome"
            if hold:
                data=picture(source,card)
                encoder.stdin.write(data*(hold*FPS))
                timeline.append({"source_step":source,"hold_seconds":hold,"label":card})
                output_count+=hold*FPS
            encoder.stdin.write(picture(source)*2)
            output_count+=2
    finally:
        encoder.stdin.close();encoder.wait(timeout=60)
    if encoder.returncode:raise RuntimeError("FFmpeg encode failed")
    subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(output),"-f","null","-"],check=True)
    metadata={"file":output.name,"source_trials":[str(left.resolve().relative_to(ROOT)),str(right.resolve().relative_to(ROOT))],
              "source_outcomes":[r["status"] for r in results],"initial_state_id":0,
              "duration_seconds":output_count/FPS,"reading_holds":timeline,"fully_decoded":True,
              "note":"Recorded simulation videos. 0.5x playback and labeled editorial holds; no new physics/model rollout."}
    (out/(task+".json")).write_text(json.dumps(metadata,indent=2)+"\n")
    print("Rendered and decoded",output,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--task",choices=["bowl","drawer","stove","all"],default="all")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    tasks=["bowl","drawer","stove"] if args.task=="all" else [args.task]
    for task in tasks:
        if (args.output/(task+".mp4")).exists():raise FileExistsError("Refusing to overwrite demo")
        render(args.runs,args.output,task)
    if args.task=="all":
        playlist=args.output/"playlist.txt"
        playlist.write_text("".join("file '{}.mp4'\n".format(t) for t in tasks))
        full=args.output/"complete-walkthrough.mp4"
        subprocess.run(["ffmpeg","-y","-v","error","-f","concat","-safe","0","-i",str(playlist),
                        "-c","copy","-movflags","+faststart",str(full)],check=True)
        subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(full),"-f","null","-"],check=True)
        (args.output/"media-validation.json").write_text(json.dumps({"passed":True,"fully_decoded":[t+".mp4" for t in tasks]+[full.name]},indent=2)+"\n")
if __name__=="__main__":main()
