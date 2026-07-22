import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import datamapplot
import colorcet
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
datamapplot.rendering_helpers.DEFAULT_FONT_FAMILY = "sans-serif"

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.0
plt.rcParams["figure.autolayout"] = False

full_label_mapping = {
    -1: "Unknown",
    0: "defense",
    1: "ingestion",
    2: "locomotion",
    3: "manipulation",
    4: "self-directed",
    5: "social gesture",
    6: "social interaction",
    7: "social symbolic",
}

label_order = [
    "defense", "ingestion", "locomotion", "manipulation",
    "self-directed", "social gesture", "social interaction",
    "social symbolic", "Unknown"
]

label_color_map = {
    "defense":            "#CE4E09",
    "ingestion":          "#FBFF00",
    "locomotion":         "#0376C4",
    "manipulation":       "#0C8603",
    "self-directed":      "#FB9A99",
    "social gesture":     "#00CED1",
    "social interaction": "#5F05C0",
    "social symbolic":    "#E31A1C",
    "Unknown":            "#BFBFBF",
}


plot_configs = [
    {
        "name": "LLM_deepseek",
        "mat_file": "data/tsne/spose_embedding_sorted_merge_tsne_LLM_deepseek.mat",
        "npy_file": "data/tsne/video_category_index_mit_355.npy",
        "output": "figures/tsne_LLM_deepseek.pdf"
    },
    {
        "name": "LLM_qwen_7B",
        "mat_file": "data/tsne/spose_embedding_sorted_merge_tsne_LLM_qwen_7B.mat",
        "npy_file": "data/tsne/video_category_index_mit_355.npy",
        "output": "figures/tsne_LLM_qwen_7B.pdf"
    },
    {
        "name": "MLLM_qwen_7B",
        "mat_file": "data/tsne/spose_embedding_sorted_merge_tsne_MLLM_qwen_7B.mat",
        "npy_file": "data/tsne/video_category_index_mit_355.npy",
        "output": "figures/tsne_MLLM_qwen_7B.pdf"
    },
    {
        "name": "MLLM_qwen_72B",
        "mat_file": "data/tsne/spose_embedding_sorted_merge_tsne_MLLM_qwen_72B.mat",
        "npy_file": "data/tsne/video_category_index_mit_355.npy",
        "output": "figures/tsne_MLLM_qwen_72B.pdf"
    },
    {
        "name": "human_odd_one_out",
        "mat_file": "data/tsne/spose_embedding_sorted_merge_tsne_human_odd_one_out.mat",
        "npy_file": "data/tsne/video_category_index_human_256.npy",
        "output": "figures/tsne_Human.pdf"
    },
]

for cfg in plot_configs:
    data = loadmat(cfg['mat_file'])
    X = data["Ytsne"]
    category_index = np.load(cfg['npy_file'])

    point_labels = np.array(
        [full_label_mapping.get(int(i), "Unknown") for i in category_index], 
        dtype=object
    )
    
    mask = np.isin(point_labels, label_order)
    X_plot = X[mask]
    labels_plot = point_labels[mask]
    X_plot = X_plot * 0.05
    
    fig, ax = datamapplot.create_plot(
        X_plot,
        labels_plot,
        title=None,
        label_font_size=0,
        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0,
            "ec": "none",
        },
        glow_keywords={"n_levels": 10, "kernel": "exponential", "kernel_bandwidth": 0.25},
        marker_size_array=np.full(len(X_plot), 5),
        label_color_map=label_color_map,
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.autoscale_view(tight=True)

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    fig.savefig(
        cfg['output'],
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    
    plt.close(fig)

legend_labels = [lbl for lbl in label_order if lbl != "Unknown"]

fig_legend, ax_legend = plt.subplots(figsize=(14, 0.5))
ax_legend.set_axis_off()

legend_handles = []
for label in legend_labels:
    color = label_color_map[label]
    handle = Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=color,
        markersize=12,
        markeredgewidth=0.8,
        markeredgecolor='#333333',
        label=label
    )
    legend_handles.append(handle)

legend = ax_legend.legend(
    handles=legend_handles,
    loc='center',
    ncol=len(legend_labels),
    frameon=False,
    framealpha=0.95,
    fancybox=False,
    edgecolor='#AAAAAA',
    fontsize=14,
    columnspacing=1.0,
    handletextpad=0.5,
    borderpad=0.4,
    borderaxespad=0.1,
)

fig_legend.savefig(
    "figures/tsne_label.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
    bbox_extra_artists=[legend],
)
plt.close(fig_legend)