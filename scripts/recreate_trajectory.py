import argparse
import csv
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def font(n,bold=False):
    windows=Path('C:/Windows/Fonts')/('msyhbd.ttc' if bold else 'msyh.ttc')
    candidates=[os.environ.get('CHART_FONT_BOLD' if bold else 'CHART_FONT'),
                os.environ.get('CHART_FONT'),windows,
                Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
                Path('/System/Library/Fonts/PingFang.ttc')]
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(str(path),n)
    raise SystemExit('Set CHART_FONT to the path of a Chinese-capable font.')

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def trajectory(cache, predictions):
    import math
    sid,end,width,gamma='8686948',27030,1200,1.24
    labels={int(t):int(stage) for t,stage in
            (line.split() for line in (cache/'labels'/f'{sid}_labeled_sleep.txt').read_text().splitlines())}
    with (cache/'heart_rate'/f'{sid}_heartrate.txt').open() as f:
        hr=[(float(t),float(v)) for t,v in csv.reader(f)
            if end-1800<=float(t)<=end and 25<=float(v)<=220]
    records={int(float(r['timestamp_sec'])):r for r in rows(predictions)
             if r['sid']==sid and r['classifier']=='HistGradientBoosting'}
    # Frozen bounds from the original 8686948 baseline.
    lo,hi=.5421305514730945,.7238625890929589
    im=Image.new('RGB',(1800,1055),'white')
    d=ImageDraw.Draw(im)
    L,R,T,B=151,1616,230,864
    x=lambda t: L+(t-(end-1800))*(R-L)/1800
    d.rectangle((L,145,x(end-width),B),fill='#F1F3F6')
    stages={-1:'#DDE2E8',0:'#E9C964',1:'#90D5D0',2:'#597D9A',3:'#574766',5:'#C778A8'}
    for t in range(end-1800,end,30):
        d.rectangle((x(t),104,x(t+30),146),fill=stages.get(labels.get(t,-1),'#DDE2E8'))
    d.text((L,57),'PSG 分期',font=font(36,True),fill='#28364C')
    for pos,label,stage in [(470,'N1',1),(635,'N2',2),(800,'N3',3),(970,'REM',5),(1190,'清醒',0)]:
        d.rectangle((pos,63,pos+40,90),fill=stages[stage])
        d.text((pos+50,77),label,font=font(28),anchor='lm',fill='#28364C')
    scores=[]; thresholds=[]; trigger=None; filtered=None
    for t in range(end-1800,end+1,30):
        r=records.get(t)
        if t==end-width:
            filtered=None
        threshold=lo+(hi-lo)*((end-t)/width)**gamma if t>=end-width else None
        if threshold is not None:
            thresholds.append((t,threshold))
        if not r or r['is_unknown']=='1' or not r['P_Wake']:
            filtered=None
            scores.append((t,None))
            continue
        value=sum(w*float(r[k] or .5) for w,k in
                  [(1,'P_Wake'),(.6,'P_Light'),(-.4,'P_REM'),(-1,'P_Deep'),(.2,'arousal_index')])
        alpha=.40 if filtered is None or value>=filtered else .85
        filtered=value if filtered is None else alpha*value+(1-alpha)*filtered
        scores.append((t,filtered))
        if trigger is None and threshold is not None and filtered>=threshold:
            trigger=t
    assert (trigger,labels[trigger])==(26190,1)
    hv=[v for _,v in hr]
    lower=math.floor(min(hv)/10)*10
    upper=math.ceil(max(hv)/10)*10
    y=lambda v: B-(v+.55)*(B-T)/1.45
    yh=lambda v: B-(v-lower)*(B-T)/(upper-lower)
    for v in (-.4,-.2,0,.2,.4,.6,.8):
        d.line((L,y(v),R,y(v)),fill='#E1E7ED',width=3)
        d.text((L-12,y(v)),f'{v:.1f}',font=font(34),anchor='rm',fill='#173F5F')
    for v in range(lower,upper+1,10):
        d.text((R+15,yh(v)),str(v),font=font(33),anchor='lm',fill='#A75442')
    d.text((L,182),'适宜度 / 阈值',font=font(34,True),fill='#173F5F')
    d.text((R,182),'心率（次/分）',font=font(34,True),anchor='ra',fill='#A75442')
    d.line([(x(t),yh(v)) for t,v in hr],fill='#E99980',width=4)
    for a,b in zip(scores,scores[1:]):
        if a[1] is not None and b[1] is not None:
            d.line((x(a[0]),y(a[1]),x(b[0]),y(b[1])),fill='#173F5F',width=7)
    # Interpolate threshold analytically at a finer resolution for a smooth dashed path.
    for t in range(end-width,end,36):
        ts=range(t,min(t+21,end)+1,3)
        d.line([(x(u),y(lo+(hi-lo)*((end-u)/width)**gamma)) for u in ts],fill='#D1495B',width=6)
    for sy in range(T,B,29):
        d.line((x(end-width),sy,x(end-width),min(sy+17,B)),fill='#687488',width=6)
    d.line((x(trigger),T,x(trigger),B),fill='#20826D',width=6)
    d.text((x(end-width)-13,T+29),'基线监测',font=font(31,True),anchor='ra',fill='#53627A')
    d.text((x(end-width)+15,T+29),'← 窗口开启（−20 分钟）',font=font(31,True),fill='#53627A')
    d.text((x(trigger)+15,T+79),'−14.0 分钟 · N1 触发',font=font(32,True),fill='#14775E')
    d.line((L,T,L,B,R,B),fill='#53627A',width=3)
    for t in range(-30,1,5):
        d.text((x(end+t*60),B+14),str(t),font=font(33),anchor='mt',fill='#28364C')
    d.text(((L+R)//2,951),'距最晚闹铃的时间（分钟）',font=font(36),anchor='mm',fill='#28364C')
    for pos,label,c in [(160,'平滑得分','#173F5F'),(490,'动态阈值','#D1495B'),(827,'PPG 心率','#E99980'),(1170,'N1 触发','#20826D')]:
        d.line((pos,1018,pos+51,1018),fill=c,width=7)
        d.text((pos+65,1018),label,font=font(34),anchor='lm',fill='#28364C')
    return im

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    trajectory(args.data_dir,args.predictions).save(args.output,dpi=(300,300))
