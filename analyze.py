import matplotlib.pylab as plt
import pandas as pd, numpy as np 
import os, glob, json
import matplotlib.pyplot as plt
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
exp_dir = 'experiments/intra_req_dp_e2e/'

def read_exp_data(exp_dir_name):
    epd_dp_dir = os.path.join(exp_dir, exp_dir_name)
    exp_df = []
    for exp_file in glob.glob(f'{epd_dp_dir}/*.exp'):
        reqs_num, reqs_rate = os.path.basename(exp_file).split('.e')[0].split('-')[1:]
        exp_data = json.load(open(exp_file, 'rb'))
        exp_df.append({
            'reqs_num': reqs_num,
            'reqs_rate': reqs_rate,
            'server_latency': np.mean([datum['server_latency'] for datum in exp_data]),
            'ttft': np.mean([datum['ttft'] for datum in exp_data]),
            'min_ttft': np.min([datum['ttft'] for datum in exp_data]),
            'max_ttft': np.max([datum['ttft'] for datum in exp_data]),
            'tpot': np.mean([datum['tpot'] for datum in exp_data]),
            'min_tpot': np.min([datum['tpot'] for datum in exp_data]),
            'max_tpot': np.max([datum['tpot'] for datum in exp_data]),
            'ttft_raw': [datum['ttft'] for datum in exp_data],
            'tpot_raw': [datum['tpot'] for datum in exp_data]
        })
    exp_df = pd.DataFrame(exp_df)
    exp_df = exp_df.sort_values(by=['reqs_num', 'reqs_rate'])
    return exp_df


baselines = ['epd', 'pd', 'd']
base_dir = 'imgs2'
baseline_dfs = {}
for baseline in baselines:
    baseline_df = read_exp_data(os.path.join(base_dir, baseline))
    baseline_dfs[baseline] = baseline_df

# PLOT SLOs
ttft_thresh = 1.4
tpot_thresh = 0.04
def calculate_percentage(df_ttft, df_tpot, ttft_th, tpot_th):
    percentages = []
    for req_rate in df_ttft.index:
        ttft_values = df_ttft.loc[req_rate]
        tpot_values = df_tpot.loc[req_rate]
        assert len(ttft_values) == len(tpot_values), "The lengths of ttft_raw and tpot_raw should be the same"
        condition_arr = [1 if ttft < ttft_th and tpot < tpot_th else 0 for ttft, tpot in zip(ttft_values, tpot_values)]
        percentage = np.mean(condition_arr)
        percentages.append({'reqs_rate': req_rate,
                            'SLO': percentage})
    
    return pd.DataFrame(percentages).set_index('reqs_rate')

plot_df = []
for baseline_name, baseline_df in baseline_dfs.items():
    df_ttft = baseline_df.set_index("reqs_rate")['ttft_raw']
    df_tpot = baseline_df.set_index("reqs_rate")['tpot_raw']
    result = calculate_percentage(df_ttft, df_tpot, ttft_thresh, tpot_thresh)
    result.columns = [baseline_name]
    plot_df.append(result)

plot_df = pd.concat(plot_df, axis=1)

baselines_map = {'epd': 'EPD', 'pd': 'DistServe', 'd': 'vLLM'}
plot_df_filtered = plot_df[baselines_map.keys()]
plot_df_filtered = plot_df_filtered*100
plot_df_filtered.index = plot_df_filtered.index.astype(float)
plot_df_filtered.rename(columns=baselines_map, inplace=True)
plot_df_filtered.plot(marker='.')
plt.axhline(y=90, color='k', linestyle='--', linewidth=1) 
plt.grid()
plt.xlabel('Request Rate (req/s)', fontsize=15)
plt.ylabel('SLO Attainment(%)', fontsize=15)
plt.title(f'TTFT<{ttft_thresh}s and TPOT<{tpot_thresh}s | #Images/Req=2', fontsize=15)
plt.savefig('SLO.png')