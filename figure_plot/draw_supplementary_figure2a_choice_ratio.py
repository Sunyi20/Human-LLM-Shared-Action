import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sys
import os

def plot_models_selection(models_info, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, (info, ax) in enumerate(zip(models_info, axes)):
        filepath = info['filepath']
        title = info['title']
        color = info['color']
        edgecolor = info['edgecolor']
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = [int(line.strip()) for line in f if line.strip() in ('1', '2', '3')]
            
            counter = Counter(data)
            total = sum(counter.values())
        else:
            print(f"Warning: File not found {filepath}")
            counter = Counter()
            total = 0

        ids = [1, 2, 3]
        counts = [counter.get(i, 0) for i in ids]
        percentages = [c / total * 100 if total > 0 else 0 for c in counts]

        bars = ax.bar(ids, counts, color=color, edgecolor=edgecolor, width=0.6)

        for bar, pct in zip(bars, percentages):
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(counts)*0.01),
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xlabel('The selected video ID', fontsize=11)
        if i == 0:
            ax.set_ylabel('Trials', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(ids)
        ax.set_xticklabels(['1', '2', '3'])

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    base_dir = "/nfs/diskstation/DataStation/ChangdeDu/NHB2026_action_Latex/code/figure_plot/data/choice_ratio"
    models_info = [
        {
            'title': 'Qwen2.5-7B',
            'filepath': os.path.join(base_dir, 'LLM_qwen_7B_choice.txt'),
            'color': (0.2, 0.6, 0.4),  # customColor1
            'edgecolor': (0.2, 0.6, 0.4)
        },
        {
            'title': 'DeepSeek-R1',
            'filepath': os.path.join(base_dir, 'LLM_deepseek_choice.txt'),
            'color': (0.2, 0.3, 0.4),  # customColor5
            'edgecolor': (0.2, 0.3, 0.4)
        },
        {
            'title': 'Qwen2.5-VL-7B',
            'filepath': os.path.join(base_dir, 'MLLM_qwen_7B_choice.txt'),
            'color': (0.9, 0.6, 0.1),  # customColor4
            'edgecolor': (0.9, 0.6, 0.1)
        },
        {
            'title': 'Qwen2.5-VL-72B',
            'filepath': os.path.join(base_dir, 'MLLM_qwen_72B_choice.txt'),
            'color': (0.6, 0.2, 0.4),  # customColor2
            'edgecolor': (0.6, 0.2, 0.4)
        }
    ]
    
    output_pdf = os.path.join('4_models_choice_ratio.pdf')
    plot_models_selection(models_info, output_pdf)