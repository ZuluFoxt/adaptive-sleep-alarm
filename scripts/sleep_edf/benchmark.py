import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import csv

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# Step 1: 自动下载并解析 Sleep-EDF Expanded 金标准 PSG 分期数据
# ==============================================================================
def extract_pre_awakening_sequences(n_subjects=35):
    print(f"[1/4] 正在从 PhysioNet 拉取 Sleep-EDF Expanded 前 {n_subjects} 名受试者的 PSG 专家分期标签...")
    subject_ids = list(range(n_subjects))
    # fetch_data 会自动跳过缺失编号并下载 [psg.edf, hypnogram.edf]
    work = Path(__file__).resolve().parent
    with (work / 'age_records.csv').open(newline='') as manifest:
        files = [(None, work / 'labels' / row['fname']) for row in csv.DictReader(manifest)
                 if int(row['subject']) in subject_ids and row['record type'] == 'Hypnogram']
    
    stage_map = {
        'Sleep stage W': 'W',
        'Sleep stage 1': 'L',  # N1 归为浅睡
        'Sleep stage 2': 'L',  # N2 归为浅睡
        'Sleep stage 3': 'D',  # N3 慢波深睡
        'Sleep stage 4': 'D',  # N4 慢波深睡
        'Sleep stage R': 'R',  # REM 期
    }
    
    sequences = []
    meta_records = []
    
    for idx, (psg_file, hyp_file) in enumerate(files):
        annot = mne.read_annotations(hyp_file)
        # 将整夜转化为 30 秒为单位的序列
        max_time = int(annot.onset[-1] + annot.duration[-1])
        n_epochs = max_time // 30
        epochs = ['W'] * n_epochs
        
        for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
            if desc in stage_map:
                st = stage_map[desc]
                s_ep = int(onset // 30)
                e_ep = int((onset + dur) // 30)
                for ep in range(max(0, s_ep), min(n_epochs, e_ep)):
                    epochs[ep] = st
        
        # 寻找夜间主睡眠块及其清晨结束点 T_wake
        # 规则: 找到持续入睡 (非连续 W >= 30分钟) 的主区间终点
        is_sleep = np.array([1 if s != 'W' else 0 for s in epochs])
        if is_sleep.sum() < 240:  # 睡眠少于 2 小时跳过
            continue
            
        # 用滑动窗口找到睡眠最密集的 6~9 小时核心区，并定位其清晨唤醒边界
        sleep_indices = np.where(is_sleep == 1)[0]
        # 过滤掉白天零星点，定位主睡眠结束点: 后面连续 60 个 epoch (30分钟) 均为 W
        t_wake = None
        for ep in range(sleep_indices[len(sleep_indices)//2], len(epochs) - 60):
            if is_sleep[ep] == 1 and is_sleep[ep+1:ep+61].sum() == 0:
                t_wake = ep + 1
                break
        if t_wake is None or t_wake < 180:
            continue
            
        pre_90m = epochs[t_wake - 180 : t_wake] # 醒前最后 90 分钟 (180 epochs)
        pre_60m = epochs[t_wake - 120 : t_wake] # 醒前最后 60 分钟 (120 epochs)
        
        # 计算最后 60 分钟各期占比并自动归类表型
        p_D = pre_60m.count('D') / 120.0
        p_R = pre_60m.count('R') / 120.0
        p_L = pre_60m.count('L') / 120.0
        p_W = pre_60m.count('W') / 120.0
        
        # 表型自动分类规则
        if p_W >= 0.15:
            phenotype = '表型IV: 清晨早醒/碎片化型 (Wake≥15%)'
        elif p_D >= 0.10:
            phenotype = '表型III: 慢波(N3)后移残留型 (N3≥10%)'
        elif p_R >= 0.35:
            phenotype = '表型I: 清晨长REM主导型 (REM≥35%)'
        else:
            phenotype = '表型II: 浅睡-REM高频均衡交替型'
            
        sequences.append(pre_90m)
        meta_records.append({
            'record_id': hyp_file.name[:8],
            'p_Deep': p_D, 'p_REM': p_R, 'p_Light': p_L, 'p_Wake': p_W,
            'phenotype': phenotype
        })
        
    df_meta = pd.DataFrame(meta_records)
    print(f"成功提取 {len(sequences)} 晚有效临床 PSG 晨间唤醒序列！表型分布如下：")
    print(df_meta['phenotype'].value_counts())
    return sequences, df_meta

# ==============================================================================
# Step 2: 唤醒窗口决策算法蒙特卡洛仿真引擎
# ==============================================================================
def run_monte_carlo_benchmark(sequences, window_min=20, n_seeds=200):
    print(f"\n[2/4] 正在 {len(sequences)} 晚 PSG 数据上运行全时段截止点遍历 x {n_seeds} 随机种子仿真...")
    W_ep = window_min * 2  # 20分钟 = 40 个 30秒 epoch
    
    # 模拟基于 DREAMT 文献校准的单通道 PPG 预测得分分布 (均值与标准差)
    score_params = {
        'W': (0.85, 0.22),
        'L': (0.55, 0.25),
        'R': (-0.40, 0.28),
        'D': (-0.90, 0.20)
    }
    
    results = {name: {'n3': [], 'rem': [], 'loss': []} for name in [
        'Fixed', 'WakeOnly', 'Greedy_k3', 'Dynamic_Sym_k3', 'Dynamic_Asym_Proposed'
    ]}
    
    rng = np.random.default_rng(2026)
    
    for seq in sequences:
        # seq 长度为 180 (90分钟), 遍历最后 60 分钟内的所有截止点 T_end in [60, 180)
        for t_end in range(60, 180, 2): # 每 1 分钟步长遍历一次 T_end
            true_window = seq[t_end - W_ep : t_end + 1] # 长度 W_ep + 1
            
            # 1. Fixed Alarm (直接在 t_end 唤醒)
            st_end = true_window[-1]
            results['Fixed']['n3'].append(1 if st_end == 'D' else 0)
            results['Fixed']['rem'].append(1 if st_end == 'R' else 0)
            results['Fixed']['loss'].append(0.0)
            
            # 蒙特卡洛生成 n_seeds 组含噪 PPG 连续得分轨迹 S(t)
            means = np.array([score_params[s][0] for s in true_window])
            stds  = np.array([score_params[s][1] for s in true_window])
            S_noisy = np.clip(rng.normal(means, stds, size=(n_seeds, len(true_window))), -1.0, 1.0)
            
            # 预计算对称滤波 (k=3) 与非对称滤波 (alpha_up=0.4, alpha_down=0.85)
            S_sym3 = np.zeros_like(S_noisy)
            S_asym = np.zeros_like(S_noisy)
            S_sym3[:, 0] = S_noisy[:, 0]
            S_asym[:, 0] = S_noisy[:, 0]
            for t in range(1, len(true_window)):
                s_idx = max(0, t - 2)
                S_sym3[:, t] = S_noisy[:, s_idx:t+1].mean(axis=1)
                alpha = np.where(S_noisy[:, t] >= S_asym[:, t-1], 0.40, 0.85)
                S_asym[:, t] = alpha * S_noisy[:, t] + (1 - alpha) * S_asym[:, t-1]
                
            # 动态递减阈值 Theta(t)
            t_steps = np.arange(len(true_window))
            tau = (W_ep - t_steps) / float(W_ep)
            theta = 0.22 + 0.63 * (tau ** 1.15)
            theta[-1] = -999.0 # 截止点强制兜底
            
            # 评估各含噪策略
            for name, cond_matrix in [
                ('WakeOnly', S_sym3 >= 0.78),
                ('Greedy_k3', S_sym3 >= 0.35),
                ('Dynamic_Sym_k3', S_sym3 >= theta[None, :]),
                ('Dynamic_Asym_Proposed', S_asym >= theta[None, :])
            ]:
                cond_matrix[:, -1] = True # 最晚时刻兜底
                trig_idx = np.argmax(cond_matrix, axis=1) # 每个 seed 第一个触发的时刻
                trig_stages = [true_window[i] for i in trig_idx]
                results[name]['n3'].append(trig_stages.count('D') / n_seeds)
                results[name]['rem'].append(trig_stages.count('R') / n_seeds)
                results[name]['loss'].append(np.mean((W_ep - trig_idx) * 0.5))
                
    summary = []
    for name, d in results.items():
        n3_m = np.mean(d['n3']) * 100
        rem_m = np.mean(d['rem']) * 100
        loss_m = np.mean(d['loss'])
        summary.append({
            'Strategy': name,
            'Deep_N3_Rate(%)': round(n3_m, 2),
            'REM_Rate(%)': round(rem_m, 2),
            'Total_Adverse_Rate(%)': round(n3_m + rem_m, 2),
            'Mean_Sleep_Loss(min)': round(loss_m, 2)
        })
    df_sum = pd.DataFrame(summary)
    print("\n[3/4] 公开 PSG 数据集仿真测试结果汇总：")
    print(df_sum.to_string(index=False))
    df_sum.to_csv('sleep_edf_alarm_benchmark.csv', index=False)
    return df_sum

if __name__ == '__main__':
    seqs, df_meta = extract_pre_awakening_sequences(n_subjects=35)
    df_sum = run_monte_carlo_benchmark(seqs, window_min=20, n_seeds=1000)

    df_meta.to_csv('sleep_edf_phenotypes.csv', index=False, encoding='utf-8-sig')
    fig, (rates, loss) = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    labels = ['Fixed', 'WakeOnly', 'Greedy k3', 'Dynamic Sym k3', 'Dynamic Asym']
    y = np.arange(len(labels))
    deep = df_sum['Deep_N3_Rate(%)']
    rem = df_sum['REM_Rate(%)']
    rates.barh(y, deep, label='N3/N4', color='#335e87')
    rates.barh(y, rem, left=deep, label='REM', color='#d18a54')
    rates.set_yticks(y, labels)
    rates.invert_yaxis()
    rates.set_xlabel('Wake-up in N3/N4 or REM (%)')
    rates.legend()
    for i, value in enumerate(df_sum['Total_Adverse_Rate(%)']):
        rates.annotate(f'{value:.2f}%', (deep.iloc[i] + rem.iloc[i], i),
                       xytext=(5, 0), textcoords='offset points', va='center')
    rates.margins(x=0.22)
    bars = loss.barh(y, df_sum['Mean_Sleep_Loss(min)'], color='#397968')
    loss.bar_label(bars, fmt='%.2f', padding=5)
    loss.set_yticks(y, labels)
    loss.invert_yaxis()
    loss.set_xlabel('Mean minutes before deadline')
    loss.margins(x=0.20)
    fig.suptitle(f'Sleep-EDF Expanded | {len(seqs)} nights | 1,000 simulations per deadline')
    fig.savefig('public_psg_benchmark.png', dpi=300)
