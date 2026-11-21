#!/usr/bin/env python3
# 通用瞬态音效捞回(固化账本A6+A7):从原片抠出"尖锐瞬态"(键盘/click/敲击),压制持续人声咝音。
# 原理:高通(SFX带)后,用快包络vs慢包络之差做mask——瞬态(快攻击)mask高,持续音(咝音/元音)mask低。
# 不需要知道是什么声音,任何瞬态SFX通用。输出可直接按目标峰值混入成片。
# 用法: transient_recover.py <原片音视频> <out.wav> [--band 2500] [--peak -6]
import sys,subprocess,argparse,wave,numpy as np
ap=argparse.ArgumentParser();ap.add_argument("inp");ap.add_argument("out")
ap.add_argument("--band",type=int,default=2500);ap.add_argument("--peak",type=float,default=-6.0)
ap.add_argument("--sr",type=int,default=48000)
a=ap.parse_args()
tmp="/tmp/_tr_hp.wav"
subprocess.run(["ffmpeg","-y","-loglevel","error","-i",a.inp,"-vn","-ac","1","-ar",str(a.sr),"-af",f"highpass=f={a.band}",tmp],check=True)
w=wave.open(tmp);n=w.getnframes();sr=w.getframerate()
x=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float32)/32768;w.close()
ax=np.abs(x)
def mov(sig,ms):  # O(n)滑动平均(cumsum),补齐到原长
    k=max(1,int(sr*ms/1000));c=np.cumsum(np.insert(sig,0,0));m=(c[k:]-c[:-k])/k
    pad=len(sig)-len(m);return np.concatenate([np.full(pad//2,m[0] if len(m) else 0),m,np.full(pad-pad//2,m[-1] if len(m) else 0)])
env_f=mov(ax,3);env_s=mov(ax,70)          # 快3ms / 慢70ms
mask=np.clip((env_f-1.3*env_s)/(env_f+1e-4),0,1)  # 攻击处高、持续处低
mask=mov(mask,2)                           # 轻微平滑防颗粒
trans=x*mask
pk=float(np.max(np.abs(trans)))+1e-9
trans=trans/pk*(10**(a.peak/20))           # 归到目标峰值
o=np.clip(trans*32767,-32768,32767).astype(np.int16)
ww=wave.open(a.out,'w');ww.setnchannels(1);ww.setsampwidth(2);ww.setframerate(sr);ww.writeframes(o.tobytes());ww.close()
# 报瞬态密度自检
W=int(sr*0.01);env=[np.max(np.abs(trans[i:i+W])) for i in range(0,len(trans)-W,W)]
on=sum(1 for i in range(1,len(env)) if env[i-1]>1e-4 and env[i]>1e-9 and 20*np.log10(env[i]/env[i-1])>6)
print(f'{{"out":"{a.out}","peakDb":{a.peak},"transients_per_sec":{round(on/(len(env)*0.01),1)}}}')
