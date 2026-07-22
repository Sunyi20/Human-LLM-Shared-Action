import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from community import community_louvain
from matplotlib.patches import Ellipse


DATASETS = {
    "human": {
        "title": "Human",
        "embedding": "data/Human/human_odd_one_out_spose_embedding_sorted_final.txt",
        "seed": 1,
        "threshold": 0.19,
        "output": "human_odd_one_out_cluster_nature",
        "colors": [
            "#0B7F7A",
            "#E86A1C",
            "#D6D900",
            "#18C7BD",
            "#2B9ED8",
            "#59A14F",
            "#C77AAE",
            "#7A7A7A",
        ],
    },
    "mllm_qwen_72b": {
        "title": "Qwen-72B-VL",
        "embedding": "data/MLLM_qwen_72B/qwen_72B_VL_spose_embedding_sorted_final.txt",
        "seed": 41,
        "threshold": 0.19,
        "output": "qwen_VL_72B_cluster_nature",
        "colors": [
            "#2B9ED8",
            "#E86A1C",
            "#18C7BD",
            "#C77AAE",
            "#18C7BD",
            "#7A7A7A",
        ],
    },
    "mllm_qwen_7b": {
        "title": "Qwen-7B-VL",
        "embedding": "data/MLLM_qwen_7B/qwen_7B_VL_spose_embedding_sorted_final.txt",
        "seed": 0,
        "threshold": 0.19,
        "output": "qwen_VL_7B_cluster_nature",
        "colors": [
            "#59A14F",
            "#8E63B0",
            "#E86A1C",
            "#2B9ED8",
            "#18C7BD",
            "#C77AAE",
            "#7A7A7A",
        ],
    },
    "llm_qwen_7b": {
        "title": "Qwen-7B",
        "embedding": "data/LLM_qwen_7B/qwen_7B_spose_embedding_sorted_final.txt",
        "seed": 41,
        "threshold": 0.19,
        "output": "qwen_7B_cluster_nature",
        "colors": [
            "#C77AAE",
            "#0B7F7A",
            "#2B9ED8",
            "#18C7BD",
            "#59A14F",
            "#E86A1C",
            "#7A7A7A",
        ],
    },
    "llm_deepseek": {
        "title": "DeepSeek",
        "embedding": "data/LLM_deepseek/deepseek_spose_embedding_sorted_final.txt",
        "seed": 1,
        "threshold": 0.19,
        "output": "deepseek_cluster_nature",
        "colors": [
            "#E86A1C",
            "#2B9ED8",
            "#D6D900",
            "#18C7BD",
            "#C77AAE",
            "#59A14F",
            "#7A7A7A",
        ],
    },
}


def apply_publication_style():
    """Nature-style defaults: editable text, restrained typography, white background."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })


def rescale(values, out_min, out_max):
    values = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        return np.full(values.shape, (out_min + out_max) / 2.0)
    return out_min + (out_max - out_min) * (values - vmin) / (vmax - vmin)


def lighten_color(hex_color, amount=0.72):
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255
    return rgb + (1 - rgb) * amount


def build_graph_like_original(embedding_path, threshold):
    """Reproduce the original scripts' graph construction and retained-node reindexing."""
    embeddings = np.loadtxt(embedding_path)
    corr_matrix = np.corrcoef(embeddings.T)
    n_original = len(corr_matrix)

    G = nx.from_numpy_array(np.matrix(corr_matrix))
    F = G.copy()
    F.remove_edges_from([
        (n1, n2)
        for n1, n2, w in F.edges(data="weight")
        if w < threshold or n1 == n2
    ])

    iso_list = list(nx.isolates(F))
    F.remove_nodes_from(iso_list)
    idx_update = [i + 1 for i in range(n_original) if i not in iso_list]

    # Match the original re-indexing step exactly:
    # H node ids are 0..N-1; displayed labels are idx_update[i].
    F_adj = nx.adjacency_matrix(F)
    A_adj = F_adj.todense()
    H = nx.from_numpy_array(A_adj)

    return H, corr_matrix, iso_list, idx_update


def collect_background_edges(corr_matrix, idx_update, threshold, background_min):
    """Non-backbone pairwise correlations among retained nodes."""
    background_edges = []
    background_weights = []
    n_nodes = len(idx_update)

    for i in range(n_nodes):
        original_i = idx_update[i] - 1
        for j in range(i + 1, n_nodes):
            original_j = idx_update[j] - 1
            corr = float(corr_matrix[original_i, original_j])
            if background_min <= corr < threshold:
                background_edges.append((i, j))
                background_weights.append(corr)

    return background_edges, np.array(background_weights, dtype=float)


def draw_community_ellipses(ax, H, pos, partition, colors):
    """Soft community blobs, without text labels."""
    for community in sorted(set(partition.values())):
        nodes = [node for node in H.nodes() if partition[node] == community]
        if not nodes:
            continue

        xy = np.array([pos[node] for node in nodes], dtype=float)
        center = xy.mean(axis=0)
        color = colors[community % len(colors)]

        if len(nodes) == 1:
            width = height = 0.22
            angle = 0
        else:
            cov = np.cov(xy.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width = 2.35 * np.sqrt(max(eigvals[0], 1e-5)) + 0.22
            height = 2.35 * np.sqrt(max(eigvals[1], 1e-5)) + 0.22

        ax.add_patch(Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            facecolor=lighten_color(color),
            edgecolor="none",
            alpha=0.70,
            zorder=0,
        ))


def save_figure(fig, output_prefix, formats):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt in {"tif", "tiff", "png"}:
            save_kwargs["dpi"] = 600
        fig.savefig(f"{output_prefix}.{fmt}", **save_kwargs)


def draw_network(dataset_name, config, output_dir, threshold=None, background_min=0.0,
                 no_blobs=False, no_title=False, formats=("svg", "pdf", "tiff")):
    threshold = config["threshold"] if threshold is None else threshold
    seed = config["seed"]
    embedding_path = config["embedding"]
    colors = config["colors"]

    H, corr_matrix, iso_list, idx_update = build_graph_like_original(
        embedding_path,
        threshold,
    )
    if H.number_of_nodes() == 0:
        raise ValueError(f"{dataset_name}: no nodes remain after thresholding.")

    # Preserve each original script's random seed and layout call.
    np.random.seed(seed)
    partition = community_louvain.best_partition(H)
    layout_k = 1 / np.sqrt(H.number_of_nodes())
    pos = nx.spring_layout(
        H,
        seed=seed,
        # k=layout_k,
        # iterations=800,
    )
    pagerank = nx.pagerank(H, alpha=0.85)

    backbone_weights = np.array([H[u][v]["weight"] for u, v in H.edges()], dtype=float)
    backbone_widths = (
        rescale(backbone_weights, 1.05, 2.35)
        if len(backbone_weights) else []
    )
    backbone_alphas = (
        rescale(backbone_weights, 0.48, 0.82)
        if len(backbone_weights) else []
    )

    background_edges, background_weights = collect_background_edges(
        corr_matrix,
        idx_update,
        threshold,
        background_min,
    )
    background_widths = (
        rescale(background_weights, 0.24, 0.78)
        if len(background_weights) else []
    )
    background_alphas = (
        rescale(background_weights, 0.09, 0.26)
        if len(background_weights) else []
    )

    pr_values = np.array([pagerank[node] for node in H.nodes()], dtype=float)
    node_sizes = 230 + 1150 * rescale(pr_values, 0, 1)
    node_colors = [colors[partition[node] % len(colors)] for node in H.nodes()]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_axis_off()
    ax.set_aspect("equal")

    if not no_blobs:
        draw_community_ellipses(ax, H, pos, partition, colors)

    # Weak, sub-threshold relationships: visible but subordinate.
    for (u, v), width, alpha in zip(background_edges, background_widths, background_alphas):
        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=[(u, v)],
            ax=ax,
            width=float(width),
            alpha=float(alpha),
            edge_color="#7F7F7F",
        )

    # Backbone edges: thresholded network from the original code.
    for (u, v), width, alpha in zip(H.edges(), backbone_widths, backbone_alphas):
        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=[(u, v)],
            ax=ax,
            width=float(width),
            alpha=float(alpha),
            edge_color="#1F1F1F",
        )

    nx.draw_networkx_nodes(
        H,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="white",
        linewidths=0.9,
        alpha=0.98,
    )

    labels = {i: idx_update[i] for i in range(len(H.nodes()))}
    label_artists = nx.draw_networkx_labels(
        H,
        pos,
        labels=labels,
        ax=ax,
        font_family="Arial",
        font_size=12,
        font_weight="normal",
        font_color="black",
    )
    for text in label_artists.values():
        text.set_path_effects([
            pe.Stroke(linewidth=2.0, foreground="white", alpha=0.75),
            pe.Normal(),
        ])

    modularity = community_louvain.modularity(partition, H)

    output_prefix = Path(output_dir) / config["output"]
    save_figure(fig, output_prefix, formats)
    plt.close(fig)

    print(f"\n[{dataset_name}] {config['title']}")
    print(f"  embedding: {embedding_path}")
    print(f"  seed={seed}, threshold={threshold:.3f}, background_min={background_min:.3f}")
    print(f"  removed isolates: {iso_list}")
    print(f"  retained labels: {idx_update}")
    print(f"  network: {H.number_of_nodes()} nodes, {H.number_of_edges()} backbone edges")
    print(f"  weak edges: {len(background_edges)}")
    print(f"  modularity Q: {modularity:.4f}")
    for fmt in formats:
        print(f"  saved: {output_prefix}.{fmt.lower().lstrip('.')}")


def parse_dataset_selection(value):
    if value == "all":
        return list(DATASETS.keys())

    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in DATASETS]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. Available: all, "
            f"{', '.join(DATASETS.keys())}"
        )
    return selected


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Draw Nature-style Figure 5 community networks for Human and four LLM/MLLM models. "
            "The original backbone construction and per-script seeds are preserved."
        )
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help=(
            "Dataset id or comma-separated ids. Use all, human, mllm_qwen_72b, "
            "mllm_qwen_7b, llm_qwen_7b, llm_deepseek."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override backbone threshold for all selected datasets. Default uses each original script value.",
    )
    parser.add_argument(
        "--background_min",
        type=float,
        default=0.0,
        help="Minimum correlation for weak background edges. Use -1 to show all sub-threshold pairs.",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for SVG/PDF/TIFF outputs.",
    )
    parser.add_argument(
        "--formats",
        default="svg,pdf",
        help="Comma-separated output formats, e.g. svg,pdf,tiff or pdf.",
    )
    parser.add_argument("--no_blobs", action="store_true")
    parser.add_argument("--no_title", action="store_true")
    args = parser.parse_args()

    apply_publication_style()
    selected = parse_dataset_selection(args.dataset)
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())

    for dataset_name in selected:
        draw_network(
            dataset_name=dataset_name,
            config=DATASETS[dataset_name],
            output_dir=args.output_dir,
            threshold=args.threshold,
            background_min=args.background_min,
            no_blobs=args.no_blobs,
            no_title=args.no_title,
            formats=formats,
        )


if __name__ == "__main__":
    main()
