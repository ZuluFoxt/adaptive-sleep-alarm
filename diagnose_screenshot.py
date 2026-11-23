"""Summarize a wearable's final 60 minutes and optionally replay ordered labels."""
import argparse
import math
from collections import Counter
from pathlib import Path

STAGES = 'WLDR'
RISK = dict(W=0.0, L=0.0, D=1.0, R=0.65)
WINDOWS = (0, 5, 10, 15, 20, 25, 30)

def phenotype(minutes):
    if minutes['W'] >= 9:
        return '早醒或觉醒较多'
    if minutes['D'] >= 6:
        return '清晨深睡残留'
    if minutes['R'] >= 18:
        return 'REM 偏多'
    return '浅睡为主或混合型'

def replay(sequence, cost):
    """Replay historical virtual deadlines; each decision uses only labels seen by then."""
    results=[]
    for width in WINDOWS:
        wakes=[]
        for deadline in range(max(WINDOWS),len(sequence)):
            start=deadline-width
            # Prefer a safe label toward the end as the required score relaxes.
            def accepted(t):
                remaining=(deadline-t)/width
                score=1.0 if sequence[t]=='W' else .6 if sequence[t]=='L' else -1.0
                return score >= .20+.70*remaining**1.24
            t=next((t for t in range(start,deadline) if accepted(t)),deadline)
            wakes.append((RISK[sequence[t]],deadline-t))
        mean_risk=sum(r for r,_ in wakes)/len(wakes)
        mean_lead=sum(t for _,t in wakes)/len(wakes)
        results.append((width,mean_risk,mean_lead,mean_risk+cost*mean_lead))
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--minutes',nargs=4,type=float,metavar=('WAKE','LIGHT','N3','REM'))
    source.add_argument('--sequence',help='60 个逐分钟阶段字符，按时间顺序使用 W/L/D/R')
    source.add_argument('--sequence-file',type=Path,help='同一格式的文本文件')
    parser.add_argument('--cost',type=float,default=.02,help='每提前一分钟的损失权重，默认 0.02')
    args=parser.parse_args()
    if not math.isfinite(args.cost) or args.cost<0:
        parser.error('--cost 必须是有限非负数')
    if args.sequence is not None or args.sequence_file:
        raw=args.sequence if args.sequence is not None else args.sequence_file.read_text(encoding='utf-8-sig')
        sequence=''.join(raw.upper().split())
        if len(sequence)!=60 or set(sequence)-set(STAGES):
            parser.error('序列需恰好 60 个 W/L/D/R 字符')
        minutes=Counter(sequence)
    else:
        sequence=None
        if any(not math.isfinite(n) or n<0 for n in args.minutes) or not math.isclose(sum(args.minutes),60,abs_tol=.01):
            parser.error('四阶段分钟数必须非负且合计 60')
        minutes=dict(zip(STAGES,args.minutes))
    print(f"N3 占比 {minutes['D']/60:.1%}，REM 占比 {minutes['R']/60:.1%}，合计 {(minutes['D']+minutes['R'])/60:.1%}")
    print(f'启发式表型：{phenotype(minutes)}')
    if sequence is None:
        print('窗口回测需要阶段顺序。补充 --sequence 或 --sequence-file 后可计算 W*。')
    else:
        results=replay(sequence,args.cost)
        best=min(results,key=lambda row:(row[3],row[2],row[0]))
        print('窗口(分钟)  平均阶段损失  平均提前(分钟)  综合损失')
        for width,risk,lead,loss in results:
            print(f'{width:>6} {risk:>14.3f} {lead:>16.2f} {loss:>12.3f}')
        print(f'本段序列的回测候选 W* = {best[0]} 分钟（0 表示定点）。')
    print('结果描述这一段设备记录；表型与窗口均为探索性估计。')

if __name__=='__main__':
    main()
