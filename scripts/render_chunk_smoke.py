"""Compact three-pane replay of the one-scene execution-horizon check."""
import json,subprocess
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/milestone2"
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
def f(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
def main():
    summary=json.loads((OUT/"summary.json").read_text())
    rows=summary["conditions"]
    videos=[]
    for row in rows:
        with imageio.get_reader(OUT/("horizon-"+str(row["execution_horizon"]))/"episode.mp4") as reader:
            videos.append([frame for frame in reader])
    maximum=max(r["actions"] for r in rows)
    output=OUT/"comparison.mp4"
    encoder=subprocess.Popen(["ffmpeg","-y","-v","error","-f","rawvideo","-pixel_format","rgb24",
       "-video_size","1440x900","-framerate","20","-i","-","-an","-c:v","libx264",
       "-threads","2","-preset","fast","-crf","23","-pix_fmt","yuv420p","-movflags","+faststart",str(output)],
       stdin=subprocess.PIPE)
    def picture(step,paused):
        im=Image.new("RGB",(1440,900),"#101820");d=ImageDraw.Draw(im)
        d.text((20,20),"Milestone 2 | One bowl scene: action execution horizons",font=f(30,True),fill="white")
        d.text((20,70),"Same SmolVLA predicts 50 actions. Execute 1, 10 or 50 before observing and inferring again.",font=f(20),fill="#bac4cc")
        for i,(row,clip) in enumerate(zip(rows,videos)):
            x=20+i*475;k=row["execution_horizon"]
            d.text((x,115),"Execute "+str(k)+" action"+("s" if k!=1 else "")+" / inference",font=f(22,True),fill="#59d5bb")
            frame=clip[min(step,len(clip)-1)]
            im.paste(Image.fromarray(frame[:,:256]).resize((450,450),Image.Resampling.LANCZOS),(x,155))
            d=ImageDraw.Draw(im)
            finished=step>=row["actions"]
            status=row["status"] if finished else "Executing recorded actions"
            d.text((x,625),status,font=f(23,True),fill="#59d5bb" if finished and row["success"] else "#ffb166")
            for line,text in enumerate([
                "Final model calls: "+str(row["model_calls"]),
                "Total inference: {:.2f} s".format(row["total_inference_s"]),
                "Inference/action: {:.3f} s".format(row["amortized_inference_s_per_action"]),
                "Final action count: "+str(row["actions"]),
                "Pane held after termination" if finished else "Feedback every {:g} simulated s".format(k/20)]):
                d.text((x,670+line*30),text,font=f(20),fill="#eef2f5")
        d.text((20,845),("READING PAUSE" if paused else "RECORDED SIMULATION AT 0.5x")+" | Source step "+str(step)+
               " | Physics pauses during inference; timings above are wall-clock inference costs.",font=f(19),fill="#59d5bb")
        return im.tobytes()
    try:
        encoder.stdin.write(picture(0,True)*80)
        for step in range(maximum+1):encoder.stdin.write(picture(step,False)*2)
        encoder.stdin.write(picture(maximum,True)*120)
    finally:
        encoder.stdin.close();encoder.wait(timeout=60)
    if encoder.returncode:raise RuntimeError("Video encoding failed")
    subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(output),"-f","null","-"],check=True)
    (OUT/"media-validation.json").write_text(json.dumps({"passed":True,"fully_decoded":True,
        "file":output.name,"note":"Recorded executions, 0.5x playback plus labeled reading holds. No synthetic robot motion."},indent=2)+"\n")
if __name__=="__main__":main()
