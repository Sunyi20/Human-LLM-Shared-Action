import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import os

base_path = "data/pycortex"

# Voxel encoding
LLM_deepseek_voxel = pd.read_csv(os.path.join(base_path, "voxel_encoding_results_csv/LLM_deepseek_voxel_encoding_results.csv"))
LLM_qwen7B_voxel = pd.read_csv(os.path.join(base_path, "voxel_encoding_results_csv/LLM_qwen_7B_voxel_encoding_results.csv"))
MLLM_qwen7B_voxel = pd.read_csv(os.path.join(base_path, "voxel_encoding_results_csv/MLLM_qwen_7B_voxel_encoding_results.csv"))
MLLM_qwen72B_voxel = pd.read_csv(os.path.join(base_path, "voxel_encoding_results_csv/MLLM_qwen_72B_voxel_encoding_results.csv"))

def reshape_model_data(model_data, model_name, method_name):
    id_col = model_data.columns[0]
    melted = model_data.melt(id_vars=[id_col], var_name='region', value_name='value')
    melted['model_type'] = model_name
    melted['method'] = method_name
    return melted[['model_type', 'value', 'region', 'method']]

voxel_data = pd.concat([
    reshape_model_data(LLM_deepseek_voxel, "Deepseek-R1", "Voxel_Encoding"),
    reshape_model_data(LLM_qwen7B_voxel, "Qwen2.5-7B", "Voxel_Encoding"),
    reshape_model_data(MLLM_qwen7B_voxel, "Qwen2.5-VL-7B", "Voxel_Encoding"),
    reshape_model_data(MLLM_qwen72B_voxel, "Qwen2.5-VL-72B", "Voxel_Encoding")
])

model_levels = ["Qwen2.5-7B", "Deepseek-R1", "Qwen2.5-VL-7B", "Qwen2.5-VL-72B"]
voxel_data['model_type'] = pd.Categorical(voxel_data['model_type'], categories=model_levels, ordered=True)
comparisons = [
    ("Qwen2.5-7B", "Deepseek-R1"),
    ("Qwen2.5-VL-7B", "Qwen2.5-VL-72B"),
    ("Qwen2.5-7B", "Qwen2.5-VL-7B"),
    ("Deepseek-R1", "Qwen2.5-VL-72B")
]

model_colors = {
    "Deepseek-R1":    "#334D66",
    "Qwen2.5-7B":     "#3F8F68",
    "Qwen2.5-VL-7B":  "#D89A2B",
    "Qwen2.5-VL-72B": "#8E4A6D"
}

def draw_comparison_plot(data, ylabel="Decoding Accuracy"):
    regions = data['region'].unique()
    n_models = len(model_levels)
    n_regions = len(regions)

    dodge_width = 0.82
    offsets = np.linspace(-dodge_width/2 + dodge_width/(2*n_models),
                          dodge_width/2 - dodge_width/(2*n_models),
                          n_models)
    model_offset = {model: offsets[i] for i, model in enumerate(model_levels)}

    global_min = data['value'].min()
    global_max = data['value'].max()
    value_span = global_max - global_min
    if np.isclose(value_span, 0):
        value_span = max(abs(global_max), 0.01)

    all_test_results = [] 
    for i, region in enumerate(regions):
        region_data = data[data['region'] == region]
        
        for pair in comparisons:
            m1, m2 = pair
            vals1 = region_data[region_data['model_type'] == m1]['value']
            vals2 = region_data[region_data['model_type'] == m2]['value']

            if len(vals1) < 2 or len(vals2) < 2:
                continue

            t_stat, p_value = ttest_ind(vals1, vals2, equal_var=False)  # Welch's t-test
            
            if not np.isnan(p_value):
                all_test_results.append({
                    'region': region,
                    'region_idx': i,
                    'pair': pair,
                    'p_value': p_value,
                    't_stat': t_stat,
                    'm1': m1,
                    'm2': m2
                })

    if len(all_test_results) > 0:
        p_values = [r['p_value'] for r in all_test_results]
        rejected, corrected_pvalues, _, _ = multipletests(
            p_values, alpha=0.05, method='fdr_bh'
        )
        
        for i, result in enumerate(all_test_results):
            result['corrected_p'] = corrected_pvalues[i]
            result['significant'] = rejected[i]
    else:
        for result in all_test_results:
            result['corrected_p'] = 1.0
            result['significant'] = False
    sig_annotations = []
    for result in all_test_results:
        if not result['significant']:
            label = "n.s."
        else:
            corrected_p = result['corrected_p']
            if corrected_p < 0.001:
                label = "***"
            elif corrected_p < 0.01:
                label = "**"
            elif corrected_p < 0.05:
                label = "*"
            else:
                label = "n.s."
        
        region = result['region']
        region_idx = result['region_idx']
        m1, m2 = result['pair']
        
        x_left = region_idx + model_offset[m1]
        x_right = region_idx + model_offset[m2]
        x_mid = (x_left + x_right) / 2
        
        pair_idx = comparisons.index(result['pair'])
        y = global_max + value_span * (0.08 + 0.07 * pair_idx)
        
        sig_annotations.append({
            "x_left": x_left,
            "x_right": x_right,
            "x_mid": x_mid,
            "y": y,
            "label": label,
            "pair": result['pair']
        })

    if len(sig_annotations) > 0:
        y_top = max(a["y"] for a in sig_annotations) + value_span * 0.12
    else:
        y_top = global_max + value_span * 0.10

    y_bottom = global_min - value_span * 0.05

    fig, ax = plt.subplots(figsize=(25, 7))

    ax.axhline(y=0, linestyle='--', color='gray', linewidth=0.28)

    for model in model_levels:
        offset = model_offset[model]
        color = model_colors[model]

        positions = np.arange(n_regions) + offset
        data_by_region = [data[(data['region'] == reg) & (data['model_type'] == model)]['value'].dropna().values
                          for reg in regions]
        vp = ax.violinplot(data_by_region, positions=positions, showmeans=False,
                           showmedians=False, showextrema=False,
                           widths=0.2, bw_method='scott')
        for pc in vp['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.42)
            pc.set_edgecolor('none')
            pc.set_linewidth(0.2)

        for reg_idx, reg_values in enumerate(data_by_region):
            if len(reg_values) == 0:
                continue
            x_center = reg_idx + offset
            jitter = np.random.uniform(-0.08, 0.08, size=len(reg_values))  # jitter.width = 0.08
            ax.scatter(np.full_like(reg_values, x_center) + jitter, reg_values,
                       color=color, alpha=0.42, s=25, linewidth=0)

        bp = ax.boxplot(data_by_region, positions=positions, widths=0.09,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=color, alpha=0.78, linewidth=0.28, color='0.25'),
                        whiskerprops=dict(linewidth=0.28, color='0.25'),
                        capprops=dict(linewidth=0.28, color='0.25'),
                        medianprops=dict(linewidth=0.28, color='0.25'))
        
        medians = [np.median(vals) for vals in data_by_region if len(vals) > 0]
        ax.scatter(positions[:len(medians)], medians, marker='o',
                   facecolor='white', edgecolor='0.15', s=20, linewidth=0.25, zorder=10)

    bracket_height = value_span * 0.025
    for ann in sig_annotations:
        x_left, x_right, x_mid, y, label = ann['x_left'], ann['x_right'], ann['x_mid'], ann['y'], ann['label']
        ax.plot([x_left, x_right], [y, y], color='black', linewidth=0.35)
        ax.plot([x_left, x_left], [y - bracket_height, y], color='black', linewidth=0.35)
        ax.plot([x_right, x_right], [y - bracket_height, y], color='black', linewidth=0.35)
        
        if label == "n.s.":
            ax.text(x_mid, y + value_span * 0.005, label, ha='center', va='bottom',
                    fontsize=13, fontweight='normal', fontstyle='italic')
        else:
            ax.text(x_mid, y + value_span * 0.005, label, ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_xlabel("Brain Region", fontsize=24)
    ax.set_xticks(np.arange(n_regions))
    ax.set_xticklabels(regions, rotation=0, ha='center', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', which='both', length=3, width=0.3, color='black')
    ax.set_ylim(y_bottom, y_top)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.4)
    ax.spines['bottom'].set_linewidth(0.4)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(width=0.3, color='black')

    handles = [plt.Rectangle((0,0),1,1, color=model_colors[m]) for m in model_levels]
    ax.legend(handles, model_levels, loc='lower center', 
              bbox_to_anchor=(0.5, 0.97), ncol=4,
              frameon=False, fontsize=22, columnspacing=0.8)

    plt.subplots_adjust(top=0.88)
    
    return fig
voxel_fig = draw_comparison_plot(voxel_data, ylabel="Encoding Performance ($r$)")

os.makedirs("figures", exist_ok=True)
voxel_fig.savefig(os.path.join(os.getcwd(), "figures/voxel_encoding_comparison.pdf"), bbox_inches='tight')

plt.show()
