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

def timeline():
    img=Image.new('RGB',(1680,920),'#202039')
    d=ImageDraw.Draw(img)
    def x(m):
        return round(105+m*21.1)
    d.text((100,58),'醒前 69 分钟分期',font=font(53,True),fill='white')
    d.text((100,134),'华为手环记录 · 全夜清醒 0 分钟',font=font(38),fill='#DBD3EB')
    d.rectangle((x(22),210,x(47),490),fill='#343451')
    for a,b,stage in [(0,15,'深睡'),(15,28,'浅睡'),(28,35,'REM'),(35,42,'浅睡'),(42,49,'REM'),(49,69,'浅睡')]:
        fill={'深睡':'#7562AF','浅睡':'#AE91E4','REM':'#F09ABB'}[stage]
        d.rectangle((x(a),320,x(b)-3,456),fill=fill)
        d.text(((x(a)+x(b))/2,365),stage,font=font(32,True),anchor='mm',fill='#202039')
        d.text(((x(a)+x(b))/2,413),f'{b-a} 分钟',font=font(29),anchor='mm',fill='#202039')
    for m,t in [(0,'07:38'),(22,'08:00'),(47,'08:25'),(69,'08:47')]:
        d.text((x(m),510),t,font=font(34),anchor='mm',fill='white')
    for m,y,label,c in [(23,600,'08:01  激进早醒（见浅睡即响）','#FFC38A'),(37,694,'08:15  动态策略','#9CE6BA'),(47,787,'08:25  定点 / 只等清醒','#FF9CA4')]:
        xx=x(m)
        d.line((xx,459,xx,y-6),fill=c,width=6)
        d.polygon([(xx,455),(xx-12,478),(xx+12,478)],fill=c)
        d.text((xx+16,y),label,font=font(37,True),anchor='lm',fill=c)
    return img

def frontier():
    im=Image.new('RGB',(2040,685),'white')
    d=ImageDraw.Draw(im)
    d.text((40,14),'唤醒风险—早醒代价折中 · 不利阶段唤醒率 (%)',font=font(40,True),fill='#28364C')
    L,R,T,B=152,1910,137,535
    x=lambda v:L+v*(R-L)/26
    y=lambda v:B-v*(B-T)/50
    for v in (0,10,20,30,40,50):
        d.line((L,y(v),R,y(v)),fill='#E2E8F0',width=3)
        d.text((L-17,y(v)),str(v),font=font(36),anchor='rm',fill='#53627A')
    for v in (0,5,10,15,20,25):
        d.text((x(v),B+14),str(v),font=font(35),anchor='mt',fill='#53627A')
    d.line((L,T,L,B,R,B),fill='#53627A',width=4)
    for groups,color_ in [(HOME_COHORT,'#7B4B94'),(PUBLIC_COHORT,'#247F77')]:
        pts=sorted((t,risk) for _,risk,t in groups)
        edge=[]
        best=float('inf')
        for t,risk in pts:
            if risk<best:
                edge.append((x(t),y(risk)))
                best=risk
        d.line(edge,fill=color_,width=8)
        # V2 is dominated by V3 in the student cohort. Show it as a comparison branch.
        if groups is HOME_COHORT:
            a,b=[(x(groups[i][2]),y(groups[i][1])) for i in (3,2)]
            for j in range(0,20,2):
                points=[(a[0]+(b[0]-a[0])*k/20,a[1]+(b[1]-a[1])*k/20) for k in (j,j+1)]
                d.line(points,fill=color_,width=5)
        for i,(_,risk,t) in enumerate(groups):
            xx,yy=x(t),y(risk)
            d.ellipse((xx-12,yy-12,xx+12,yy+12),fill=color_)
            label=('V0 定点','V1 只等清醒','V2 激进早醒','V3 动态 + 非对称')[i]
            if i in (0,1):
                positions=((.24,49),(1.70,43.8)) if groups is HOME_COHORT else ((.30,30.1),(8.35,37.6))
                tx,ty=positions[i]
                d.text((x(tx),y(ty)),label,font=font(32,True),anchor='lm',fill=color_)
            elif i==2 and groups is PUBLIC_COHORT:
                d.text((xx-16,yy-18),label,font=font(32,True),anchor='rs',fill=color_)
            else:
                d.text((xx+17,yy-12),label,font=font(32,True),anchor='ls',fill=color_)
    d.text((850,77),'本组大学生 · 20 分钟窗',font=font(34,True),fill='#7B4B94')
    d.text((1430,77),'公开库 · 30 分钟窗',font=font(34,True),fill='#247F77')
    d.text(((L+R)/2,639),'相对定点闹钟的平均早醒时间 ΔT（分钟）',font=font(43),anchor='mm',fill='#28364C')
    return im

def cohort_data():
    with (Path(__file__).resolve().parents[1]/'data'/'cohort_comparison.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.reader(f))[1:]
    return ([tuple(map(float,r[1:4])) for r in rows],[tuple(map(float,r[4:7])) for r in rows])

HOME_COHORT,PUBLIC_COHORT=cohort_data()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path('results'))
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    timeline().save(args.output_dir/'huawei_example.png',dpi=(300,300))
    frontier().save(args.output_dir/'pareto_frontier.png',dpi=(300,300))
