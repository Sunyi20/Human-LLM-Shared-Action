import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import to_rgba
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import glob

def create_circular_tree(similarity_matrix, labels, categories, scores,
                        figsize=(16, 16), bg_color='white',
                        label_spacing=1.0, min_radius=.2, max_radius=1.0):

    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix[distance_matrix < 0] = 0
    from scipy.spatial.distance import squareform
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, 'ward')

    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax = fig.add_subplot(111, polar=True)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ddata = dendrogram(Z, no_plot=True)

    icoord = np.array(ddata['icoord'])
    dcoord = np.array(ddata['dcoord'])
    color_list = ddata['color_list']

    max_d = np.max(dcoord)
    min_d = np.min(dcoord)

    def depth_to_radius(depth, max_depth, min_depth):
        normalized_depth = (max_depth - depth) / (max_depth - min_depth)
        transformed = normalized_depth ** 1.1 
        radius = min_radius + transformed * (max_radius - min_radius)
        return radius

    def to_polar(x, y, max_x):
        theta = (x / max_x) * 2 * np.pi
        r = depth_to_radius(y, max_d, min_d)
        return theta, r

    line_alpha = 0.8
    max_icoord = np.max(icoord)

    for xs, ys, c in zip(icoord, dcoord, color_list):
        rgba_color = to_rgba(c, alpha=line_alpha)

        for i in range(0, len(xs) - 1):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[i + 1], ys[i + 1]

            theta1, r1 = to_polar(x1, y1, max_icoord)
            theta2, r2 = to_polar(x2, y2, max_icoord)

            if i in [0, 2]:
                ax.plot([theta1, theta1], [r1, r2], color=rgba_color, linewidth=1.0)
            else:
                if r1 < min_radius + 1e-7:
                    center_point = 0
                    ax.plot([theta1, theta1], [center_point, r1], color=rgba_color, linewidth=1.0)
                    ax.plot([theta2, theta2], [center_point, r1], color=rgba_color, linewidth=1.0)
                else:
                    arc = np.linspace(theta1, theta2, 50)
                    r_arc = np.ones_like(arc) * r1
                    points = np.column_stack((arc, r_arc))
                    path = Path(points, [Path.MOVETO] + [Path.LINETO] * (len(points) - 1))
                    patch = patches.PathPatch(path, edgecolor=rgba_color, facecolor='none', linewidth=1.0)
                    ax.add_patch(patch)

    leaf_positions = ddata['leaves']
    leaf_count = len(leaf_positions)
    base_angle_step = 2 * np.pi / leaf_count
    leaf_angles = np.linspace(0, 2 * np.pi, leaf_count, endpoint=False)
    unique_categories = list(set(categories))

    fixed_colors = {
        'defense': '#d62728',           
        'ingestion': '#ff7f0e',         
        'locomotion': '#1f77b4',        
        'manipulation': '#9467bd',      
        'self-directed': '#e377c2',     
        'social gesture': '#17becf',    
        'social interaction': '#2ca02c',
        'social symbolic': '#bcbd22',   
        'Unknown': 'gray'
    }
    cmap = plt.get_cmap('tab20')
    category_colors = {}
    
    for i, cat in enumerate(unique_categories):
        if cat in fixed_colors:
            category_colors[cat] = fixed_colors[cat]
        else:
            category_colors[cat] = cmap(i % 20)
            
    min_size, max_size = 10, 100
    if max(scores) != min(scores):
        norm_scores = (np.array(scores) - min(scores)) / (max(scores) - min(scores))
    else:
        norm_scores = np.ones_like(scores) * 0.5

    for i, leaf_idx in enumerate(leaf_positions):
        angle = leaf_angles[i]
        name = labels[leaf_idx]
        category = categories[leaf_idx]
        score = norm_scores[leaf_idx]
        
        color = category_colors.get(category, 'gray')
        
        node_size = min_size + score * (max_size - min_size)
        
        ax.scatter(angle, max_radius + 0.02, s=node_size, 
                   color=color, alpha=0.8, zorder=10)
        text_angle = np.degrees(angle)
        if 90 < text_angle < 270:
            text_angle += 180
            ha = 'right'
            rotation_mode = 'anchor'
            text_r_offset = 0.04
        else:
            ha = 'left'
            rotation_mode = 'anchor'
            text_r_offset = 0.04
        ax.text(
            angle,
            max_radius + text_r_offset,
            name,
            rotation=text_angle,
            rotation_mode=rotation_mode,
            horizontalalignment=ha,
            verticalalignment='center',
            fontsize=6,
            color='black',
            alpha=0.8
        )
    return fig, ax

np.random.seed(42)


def process_and_plot_action_tree(config):
    print(f"Processing: {config['name']}...")
    
    features = np.loadtxt(config['feature_path'])
    num_actions = features.shape[0]
    
    action_labels = []
    with open(config['label_path'], 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                action_labels.append(parts[1])
    
    categories_dir = "data/category_split/categories/"
    action_to_category = {}
    for cat_file in glob.glob(os.path.join(categories_dir, "*.txt")):
        category_name = os.path.splitext(os.path.basename(cat_file))[0]
        with open(cat_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    action_to_category[parts[1]] = category_name

    action_categories = [action_to_category.get(label, 'Unknown') for label in action_labels]
    action_scores = np.ones(num_actions)
    similarity_matrix = cosine_similarity(features)

    fig, ax = create_circular_tree(
        similarity_matrix=similarity_matrix,
        labels=action_labels,
        categories=action_categories,
        scores=action_scores,
        figsize=(15, 15),
        min_radius=0.3,
        max_radius=1.0
    )
    
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    plt.savefig(config['save_path'], bbox_inches='tight')
    plt.close(fig)

configs = [
    {
        "name": "Human",
        "feature_path": "data/Human/human_odd_one_out_spose_embedding_256_averaged.txt",
        "label_path": "data/folder_list/folder_list_human_odd_one_out_overlap_0_based.txt",
        "save_path": "figures/human_action_tree.pdf"
    },
    {
        "name": "LLM Qwen 7B",
        "feature_path": "data/LLM_qwen_7B/qwen_7B_spose_embedding_sorted_final.txt",
        "label_path": "data/folder_list/folder_list.txt",
        "save_path": "figures/LLM_qwen_7B_action_tree.pdf"
    },
    {
        "name": "LLM DeepSeek",
        "feature_path": "data/LLM_deepseek/deepseek_spose_embedding_sorted_final.txt",
        "label_path": "data/folder_list/folder_list.txt",
        "save_path": "figures/LLM_deepseek_action_tree.pdf"
    },
    {
        "name": "MLLM Qwen 7B",
        "feature_path": "data/MLLM_qwen_7B/qwen_7B_VL_spose_embedding_sorted_final.txt",
        "label_path": "data/folder_list/folder_list.txt",
        "save_path": "figures/MLLM_qwen_7B_action_tree.pdf"
    },
    {
        "name": "MLLM Qwen 72B",
        "feature_path": "data/MLLM_qwen_72B/qwen_72B_VL_spose_embedding_sorted_final.txt",
        "label_path": "data/folder_list/folder_list.txt",
        "save_path": "figures/MLLM_qwen_72B_action_tree.pdf"
    }
]

if __name__ == "__main__":
    for cfg in configs:
        process_and_plot_action_tree(cfg)
