import numpy as np
import os
import argparse
import math
import textwrap
from tqdm import tqdm
from os.path import join as pjoin
from PIL import Image
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = pjoin(SCRIPT_DIR, "data")
DEFAULT_OUTPUT_ROOT = pjoin(DEFAULT_DATA_ROOT, "pycortex", "topvideos_roi")
DEFAULT_MODELS = ["LLM_deepseek", "LLM_qwen_7B", "MLLM_qwen_7B", "MLLM_qwen_72B"]
DEFAULT_ROSE_COMBINE_ORDER = ["MT", "MST", "FST", "FFC", "TPOJ3"]

MODEL_EMBEDDING_FILES = {
    "LLM_deepseek": "deepseek_spose_embedding_sorted_final.txt",
    "LLM_qwen_7B": "qwen_7B_spose_embedding_sorted_final.txt",
    "MLLM_qwen_7B": "qwen_7B_VL_spose_embedding_sorted_final.txt",
    "MLLM_qwen_72B": "qwen_72B_VL_spose_embedding_sorted_final.txt",
}

ROSE_BASE_COLOR = "#D9D9D9"
ROSE_HIGHLIGHT_COLORS = ["#D55E00", "#0072B2", "#009E73"]
DIMENSION_LABEL_WRAP_WIDTH = 16


def get_wrapped_dimension_label_lines(label: str, width: int = DIMENSION_LABEL_WRAP_WIDTH):
    wrapped_lines = textwrap.wrap(
        label.strip(),
        width=width,
        break_long_words=False,
        break_on_hyphens=True,
    )
    return wrapped_lines if wrapped_lines else [label]


def wrap_dimension_label(label: str, width: int = DIMENSION_LABEL_WRAP_WIDTH):
    return "\n".join(get_wrapped_dimension_label_lines(label, width=width))


def draw_rose_diagram(
    ax,
    values: np.ndarray,
    dimension_labels: list = None,
    top_n_labels: int = 3,
    radial_padding: float = 1.45,
):
    fig = ax.figure
    n_dims = values.shape[0]
    dims = np.arange(n_dims)
    
    processed_values = np.maximum(values, 0)
    top_n_labels = min(top_n_labels, n_dims)
    top_indices = np.argsort(values)[::-1][:top_n_labels]
    top_index_to_rank = {dim_index: rank for rank, dim_index in enumerate(top_indices)}
    colors = np.full(n_dims, ROSE_BASE_COLOR, dtype=object)
    for rank, dim_index in enumerate(top_indices):
        colors[dim_index] = ROSE_HIGHLIGHT_COLORS[rank % len(ROSE_HIGHLIGHT_COLORS)]

    theta = 2 * np.pi * dims / n_dims
    width = 2 * np.pi / n_dims
    max_r_val = processed_values.max() if processed_values.any() else 1

    bars = ax.bar(
        theta,
        processed_values,
        width=width,
        bottom=0,
        color=colors,
        edgecolor="black",
        linewidth=0.35,
        align="edge",
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, max_r_val * radial_padding)
    ax.set_axis_off()

    for dim_index in top_indices:
        rank = top_index_to_rank[dim_index]
        label = f"Dimension {dim_index + 1}"
        if dimension_labels and dim_index < len(dimension_labels):
            label = dimension_labels[dim_index]
        display_label_lines = get_wrapped_dimension_label_lines(label)
        display_label = "\n".join(display_label_lines)

        angle = theta[dim_index] + width / 2
        angle_deg = np.degrees(angle) % 360
        rotation = angle_deg
        ha = "center"
        if 90 < angle_deg < 270:
            rotation = angle_deg + 180

        petal_height = processed_values[dim_index]
        if petal_height <= 0:
            continue

        label_radius = petal_height * 0.65
        fig_width, fig_height = fig.get_size_inches()
        ax_bbox = ax.get_position()
        radius_inches = min(fig_width * ax_bbox.width, fig_height * ax_bbox.height) * 0.42
        radial_room_inches = (petal_height / (max_r_val * radial_padding)) * radius_inches * 0.58
        text_radius_inches = (label_radius / (max_r_val * radial_padding)) * radius_inches
        tangential_room_inches = max(text_radius_inches * width * 0.42, 0.04)
        wrapped_label_length = max(len(line) for line in display_label_lines)
        line_count = len(display_label_lines)
        fontsize_by_length = radial_room_inches * 72 / max(wrapped_label_length * 0.5, 1)
        fontsize_by_height = tangential_room_inches * 120 / max(line_count* 0.8, 1)
        # fontsize = max(4.7, min(9.5, fontsize_by_length, fontsize_by_height))
        fontsize = 18

        ax.text(
            angle,
            label_radius,
            display_label,
            color="black",
            fontsize=fontsize,
            fontweight="bold",
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            linespacing=0.9,
            clip_path=bars[dim_index],
            clip_on=True,
        )


def plot_rose_diagram(
    values: np.ndarray,
    output_path: str = "rose_plot.png",
    dimension_labels: list = None,
    top_n_labels: int = 3,
):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"projection": "polar"})
    draw_rose_diagram(
        ax=ax,
        values=values,
        dimension_labels=dimension_labels,
        top_n_labels=top_n_labels,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bbox = ax.get_window_extent(renderer=renderer)
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())

    pad = 0.05
    bbox_inches = bbox_inches.expanded(1.0 + pad, 1.0 + pad)
    
    fig.savefig(
        output_path, 
        dpi=200, 
        bbox_inches=bbox_inches,
        pad_inches=0,
        facecolor='white'
    )
    plt.close(fig)


def plot_combined_rose_diagrams(
    roi_betas_dict,
    roi_order,
    output_path,
    dimension_labels=None,
    top_n_labels: int = 3,
):
    fig, axs = plt.subplots(
        1,
        len(roi_order),
        figsize=(2.25 * len(roi_order), 2.35),
        subplot_kw={"projection": "polar"},
    )
    fig.subplots_adjust(left=0.002, right=0.998, bottom=0.13, top=0.995, wspace=-0.18)

    for ax, roi_name in zip(np.asarray(axs).flatten(), roi_order):
        if roi_name in roi_betas_dict:
            draw_rose_diagram(
                ax=ax,
                values=roi_betas_dict[roi_name],
                dimension_labels=dimension_labels,
                top_n_labels=top_n_labels,
                radial_padding=1.18,
            )
        else:
            ax.set_axis_off()
        ax.set_title(roi_name, fontsize=11, fontweight="bold", y=-0.08)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    bboxes = []
    for ax in axs:
        if hasattr(ax, 'get_window_extent'):
            bbox = ax.get_window_extent(renderer=renderer)
            bboxes.append(bbox)

    if bboxes:
        from matplotlib.transforms import Bbox
        union_bbox = Bbox.union(bboxes)
        bbox_inches = union_bbox.transformed(fig.dpi_scale_trans.inverted())
        pad = 0.1
        bbox_inches = bbox_inches.expanded(1.0 + pad, 1.0 + pad)
        
        fig.savefig(
            output_path, 
            dpi=200, 
            bbox_inches=bbox_inches,
            pad_inches=0,
            facecolor='white'
        )
    else:
        fig.savefig(
            output_path, 
            dpi=200, 
            bbox_inches='tight', 
            pad_inches=0.1, 
            facecolor='white'
        )
    
    plt.close(fig)

def load_dimension_names(naming_file: str):
    names = {}
    with open(naming_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            prefix, name = line.split(":", 1)
            parts = prefix.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                names[int(parts[-1]) - 1] = name.strip()

    if not names:
        return []

    max_index = max(names)
    return [names.get(dim_index, f"Dimension {dim_index + 1}") for dim_index in range(max_index + 1)]


def save_top_dimensions(values: np.ndarray, output_path: str, dimension_labels=None, top_n: int = 3):
    top_n = min(top_n, values.shape[0])
    top_indices = np.argsort(values)[::-1][:top_n]

    with open(output_path, "w") as f:
        f.write("Rank\tDimension_1based\tDimension_Name\tActivation_Value\n")
        for rank, dim_index in enumerate(top_indices, start=1):
            dim_name = ""
            if dimension_labels and dim_index < len(dimension_labels):
                dim_name = dimension_labels[dim_index]
            f.write(f"{rank}\t{dim_index + 1}\t{dim_name}\t{values[dim_index]:.8g}\n")


def get_safe_roi_name(roi_name):
    return roi_name.replace('/', '_').replace('\\', '_')


def resolve_input_npy(output_root, model, input_npy=None):
    if input_npy:
        return input_npy

    candidates = [
        pjoin(output_root, model, f"{model}_all_roi_betas.npy"),
        pjoin(output_root, f"{model}_all_roi_betas.npy"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def extract_specific_frames(gif_path, frame_indices, target_size=(64, 64)):
    frames = []
    try:
        with Image.open(gif_path) as gif:
            total_frames = gif.n_frames
            for idx in frame_indices:
                if idx < 1 or idx > total_frames:
                    continue
                gif.seek(idx - 1)
                frame = np.array(gif.convert('RGB').resize(target_size))
                frames.append(frame)
    except Exception as e:
        print(f"Error {gif_path}: {e}")
    return frames

def add_filmstrip_border(image, border_height=None, hole_width=None, hole_height=None, hole_spacing=None,
                         border_color=(0, 0, 0), hole_color=(255, 255, 255)):
    height, width, _ = image.shape

    if border_height is None:
        border_height = max(4, int(height * 0.14))  
    if hole_width is None:
        hole_width = max(3, int(width * 0.06))     
    if hole_height is None:
        hole_height = max(2, int(border_height * 0.6))
    if hole_spacing is None:
        hole_spacing = max(6, int(width * 0.08))    
    
    border = np.full((border_height, width, 3), border_color, dtype=np.uint8)
    for x in range(hole_spacing // 2, width, hole_spacing):
        x_start = max(x - hole_width // 2, 0)
        x_end = min(x + hole_width // 2, width)
        border[border_height // 2 - hole_height // 2: border_height // 2 + hole_height // 2, x_start:x_end] = hole_color
    return np.vstack([border, image, border])

def create_filmstrip_image(frames_list, gap=2):
    if not frames_list:
        return None
    gap_array = np.full((frames_list[0].shape[0], gap, 3), (255, 255, 255), dtype=np.uint8)
    concatenated_image = frames_list[0]
    for frame in frames_list[1:]:
        concatenated_image = np.hstack((concatenated_image, gap_array, frame))
    return add_filmstrip_border(concatenated_image)

def process_gif_to_image_array(input_gif_path, frame_indices):
    try:
        frames = extract_specific_frames(input_gif_path, frame_indices)
        if frames:
            return create_filmstrip_image(frames)
    except Exception as e:
        print(f"Error {input_gif_path}: {str(e)}")
    return None

def vcorrcoef(X, y):
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
    r = r_num / r_den
    return r

class TopObjectsProfile:
    def __init__(
        self,
        roiname,
        roibetas,
        img_emb,
        img_fnames,
        outdir,
        gif_map=None,
        frame_indices=None,
        similarity_metric="rectcosine",
        use_fisher=True,
        plot_k=8,
        plot_dpi=300,
        plot_fileformat="pdf",
    ):
        self.roiname = roiname
        self.roibetas = roibetas
        self.img_emb = img_emb
        self.img_fnames = img_fnames
        self.outdir = outdir
        self.gif_map = gif_map if gif_map is not None else {}
        self.frame_indices = frame_indices if frame_indices is not None else []
        self.similarity_metric = similarity_metric
        self.use_fisher = use_fisher
        self.plot_k = plot_k
        self.plot_dpi = plot_dpi
        self.plot_fileformat = plot_fileformat
        self._check_inputs()

    def _check_inputs(self):
        assert self.similarity_metric in ("correlation", "rectcosine", "cosine")
        assert self.img_emb.shape[-1] == self.roibetas.shape[0]
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def find_top_objects(self):
        if self.similarity_metric == "rectcosine":
            betas = self.roibetas.copy()
            betas[betas < 0] = 0
            s = cosine_similarity(self.img_emb, betas.reshape(1, -1)).flatten()
        elif self.similarity_metric == "cosine":
            s = cosine_similarity(self.img_emb, self.roibetas.reshape(1, -1)).flatten()
        elif self.similarity_metric == "correlation":
            s = vcorrcoef(self.img_emb, self.roibetas)
        if self.use_fisher and self.similarity_metric == "correlation":
            s = np.arctanh(s)
        sort_inds = np.argsort(s)
        fnames_sorted = self.img_fnames[sort_inds]
        s_sorted = s[sort_inds]
        return fnames_sorted, s_sorted, sort_inds

    def make_fig_v2(self, figwidth=20):
        fnames_sorted, s_sorted, sort_inds = self.find_top_objects()
        
        top_k_indices = sort_inds[-self.plot_k:]
        
        print(f"\n--- ROI: {self.roiname} ---")
        
        filmstrip_images = []

        for i, video_idx in enumerate(reversed(top_k_indices)):
            rank = i + 1
            video_name = self.img_fnames[video_idx]
            sim_score = s_sorted[-(i + 1)]
            
            print(f"  Rank {rank}: '{video_name}' (Sim: {sim_score:.4f})")

            if (video_idx + 1) in self.gif_map:
                input_gif_path = self.gif_map[video_idx + 1]
                img_array = process_gif_to_image_array(input_gif_path, self.frame_indices)
                if img_array is not None:
                    filmstrip_images.append(img_array)


        if not filmstrip_images:
            return

        sample_image = filmstrip_images[0]
        h, w, _ = sample_image.shape
        image_aspect_ratio = w / h

        n_cols = min(2, len(filmstrip_images))
        n_rows = math.ceil(len(filmstrip_images) / n_cols)
        figheight = figwidth * n_rows / (n_cols * image_aspect_ratio)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(figwidth, figheight))
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        for i, ax in enumerate(np.asarray(axs).flatten()):
            if i < len(filmstrip_images):
                ax.imshow(filmstrip_images[i])
            ax.axis('off')

        outfile = pjoin(self.outdir, f"Top{self.plot_k}_{get_safe_roi_name(self.roiname)}.{self.plot_fileformat}")
        fig.savefig(outfile, dpi=self.plot_dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print("-" * (20 + len(self.roiname)))

def load_filenames_from_txt(filepath):
    filenames = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    filenames.append(parts[1])
    return np.array(filenames)

def get_model_embedding_file(data_root, model, embedding_file=None):
    if embedding_file:
        return embedding_file
    if model not in MODEL_EMBEDDING_FILES:
        raise ValueError(f"Unknown model '{model}'. Please specify --embedding_file.")
    return pjoin(data_root, model, MODEL_EMBEDDING_FILES[model])


def get_model_naming_file(data_root, model, naming_file=None):
    if naming_file:
        return naming_file
    return pjoin(data_root, model, f"{model}_dimension_naming.txt")


def get_model_output_dir(args, model, n_models):
    if args.output_dir:
        return pjoin(args.output_dir, model) if n_models > 1 else args.output_dir
    return pjoin(args.output_root, model)


def process_model(args, model, n_models):
    input_npy = resolve_input_npy(args.output_root, model, args.input_npy)
    output_dir = get_model_output_dir(args, model, n_models)
    rose_output_dir = pjoin(output_dir, "rose_plots") if args.rose_output_dir is None else args.rose_output_dir
    top_videos_output_dir = pjoin(output_dir, "top_videos") if args.top_videos_output_dir is None else args.top_videos_output_dir
    naming_file = get_model_naming_file(args.data_root, model, args.naming_file)

    roi_betas_dict = np.load(input_npy, allow_pickle=True).item()
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_rose_plots:
        os.makedirs(rose_output_dir, exist_ok=True)
        dimension_labels = load_dimension_names(naming_file)
    else:
        dimension_labels = None

    if not args.skip_top_videos:
        embedding_file = get_model_embedding_file(args.data_root, model, args.embedding_file)
        image_embeddings = np.loadtxt(embedding_file)
        image_filenames = load_filenames_from_txt(args.filenames_file)

        gif_map = {}
        for f in os.listdir(args.gif_dir):
            if f.endswith('.gif') and '_' in f:
                idx = int(f.split('_')[0])
                gif_map[idx] = pjoin(args.gif_dir, f)
    else:
        image_embeddings = None
        image_filenames = None
        gif_map = {}

    print(f"\n=== Model: {model} ===")

    if args.combined_rose_only:
        combined_output_path = pjoin(
            rose_output_dir,
            f"{model}_combined_rose_{'_'.join(args.rose_combine_order)}.pdf",
        )
        plot_combined_rose_diagrams(
            roi_betas_dict=roi_betas_dict,
            roi_order=args.rose_combine_order,
            output_path=combined_output_path,
            dimension_labels=dimension_labels,
            top_n_labels=args.top_n_dimensions,
        )
        return

    rois_to_process = {k: v for k, v in roi_betas_dict.items() if k in args.rois} if args.rois else roi_betas_dict
    for roi_name, beta_values in tqdm(rois_to_process.items()):
        safe_roi_name = get_safe_roi_name(roi_name)

        if not args.skip_rose_plots:
            rose_output_path = pjoin(rose_output_dir, f"{safe_roi_name}_rose_plot.pdf")
            top_dims_output_path = pjoin(rose_output_dir, f"{safe_roi_name}_top{args.top_n_dimensions}_dimensions.txt")
            plot_rose_diagram(
                values=beta_values,
                output_path=rose_output_path,
                dimension_labels=dimension_labels,
                top_n_labels=args.top_n_dimensions,
            )
            save_top_dimensions(
                values=beta_values,
                output_path=top_dims_output_path,
                dimension_labels=dimension_labels,
                top_n=args.top_n_dimensions,
            )

        if args.skip_top_videos:
            continue

        roi_top_objects_outdir = pjoin(top_videos_output_dir, safe_roi_name)
        
        top_objects_analyzer = TopObjectsProfile(
            roiname=roi_name, 
            roibetas=beta_values, 
            img_emb=image_embeddings, 
            img_fnames=image_filenames, 
            outdir=roi_top_objects_outdir,
            gif_map=gif_map,
            frame_indices=args.frame_indices,
            plot_k=args.plot_k
        )
        top_objects_analyzer.make_fig_v2()

    if not args.skip_rose_plots and not args.skip_combined_rose:
        combined_output_path = pjoin(
            rose_output_dir,
            f"{model}_combined_rose_{'_'.join(args.rose_combine_order)}.pdf",
        )
        plot_combined_rose_diagrams(
            roi_betas_dict=roi_betas_dict,
            roi_order=args.rose_combine_order,
            output_path=combined_output_path,
            dimension_labels=dimension_labels,
            top_n_labels=args.top_n_dimensions,)
        


def main(args):
    models = args.models if args.models else ([args.model] if args.model else DEFAULT_MODELS)
    args.filenames_file = args.filenames_file or pjoin(args.data_root, "folder_list", "folder_list.txt")
    args.gif_dir = args.gif_dir or pjoin(args.data_root, "gifs_renamed")

    for model in models:
        process_model(args, model, len(models))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--models", nargs='+', type=str, default=None)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input_npy", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--rose_output_dir", type=str, default=None)
    parser.add_argument("--top_videos_output_dir", type=str, default=None)
    parser.add_argument("--embedding_file", type=str, default=None,)
    parser.add_argument("--naming_file", type=str, default=None)
    parser.add_argument("--filenames_file", type=str, default=None)
    parser.add_argument("--rois", nargs='+', type=str, default=['MT', 'MST', 'FST'])
    parser.add_argument("--plot_k", type=int, default=4)
    parser.add_argument("--gif_dir", type=str, default=None)
    parser.add_argument("--frame_indices", nargs='+', type=int, default=[1, 8, 15])
    parser.add_argument("--top_n_dimensions", type=int, default=3,)
    parser.add_argument("--rose_combine_order", nargs='+', type=str, default=DEFAULT_ROSE_COMBINE_ORDER,)
    parser.add_argument("--combined_rose_only", action="store_true",)
    parser.add_argument("--skip_rose_plots", action="store_true")
    parser.add_argument("--skip_combined_rose", action="store_true")
    parser.add_argument("--skip_top_videos", action="store_true")
    
    args = parser.parse_args()
    main(args)
