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

DEFAULT_MODELS = [
    "LLM_deepseek",
    "LLM_qwen_7B",
    "MLLM_qwen_7B",
    "MLLM_qwen_72B",
]

DEFAULT_ROSE_COMBINE_ORDER = [
    "MT",
    "MST",
    "FST",
    "FFC",
    "TPOJ3",
]

MODEL_EMBEDDING_FILES = {
    "LLM_deepseek": "deepseek_spose_embedding_sorted_final.txt",
    "LLM_qwen_7B": "qwen_7B_spose_embedding_sorted_final.txt",
    "MLLM_qwen_7B": "qwen_7B_VL_spose_embedding_sorted_final.txt",
    "MLLM_qwen_72B": "qwen_72B_VL_spose_embedding_sorted_final.txt",
}


DONUT_OTHERS_COLOR = "#D0D0D0"

DONUT_HIGHLIGHT_COLORS = [
    "#E76F51",
    "#56B4E9",
    "#6CCF9F",
]

DIMENSION_LABEL_WRAP_WIDTH = 18


def get_wrapped_dimension_label_lines(
    label: str,
    width: int = DIMENSION_LABEL_WRAP_WIDTH,
):
    wrapped_lines = textwrap.wrap(
        label.strip(),
        width=width,
        break_long_words=False,
        break_on_hyphens=True,
    )

    return wrapped_lines if wrapped_lines else [label]


def wrap_dimension_label(
    label: str,
    width: int = DIMENSION_LABEL_WRAP_WIDTH,
):
    return "\n".join(
        get_wrapped_dimension_label_lines(
            label,
            width=width,
        )
    )


def _prepare_donut_segments(
    values: np.ndarray,
    dimension_labels=None,
    top_n_labels: int = 3,
):
    values = np.asarray(values, dtype=float).reshape(-1)
    processed_values = np.maximum(values, 0)

    n_dims = processed_values.size
    top_n_labels = min(
        max(int(top_n_labels), 0),
        n_dims,
    )

    top_indices = np.argsort(
        -processed_values,
        kind="stable",
    )[:top_n_labels]

    top_values = processed_values[top_indices]

    others_value = max(
        float(processed_values.sum() - top_values.sum()),
        0.0,
    )

    segment_labels = []

    for dim_index in top_indices:
        if dimension_labels and dim_index < len(dimension_labels):
            segment_labels.append(dimension_labels[dim_index])
        else:
            segment_labels.append(
                f"Dimension {dim_index + 1}"
            )

    segment_values = list(top_values)

    segment_colors = [
        DONUT_HIGHLIGHT_COLORS[
            rank % len(DONUT_HIGHLIGHT_COLORS)
        ]
        for rank in range(len(top_indices))
    ]

    segment_values.append(others_value)
    segment_labels.append("Others")
    segment_colors.append(DONUT_OTHERS_COLOR)

    if np.sum(segment_values) <= 0:
        segment_values = (
            [0.0] * len(top_indices)
            + [1.0]
        )

    return (
        np.asarray(segment_values),
        segment_labels,
        segment_colors,
        top_indices,
    )


def draw_donut_diagram(
    ax,
    values: np.ndarray,
    dimension_labels: list = None,
    top_n_labels: int = 3,
    donut_width: float = 0.38,
    startangle: float = 90,
    center_label: str = None,
):
    (
        segment_values,
        segment_labels,
        segment_colors,
        _,
    ) = _prepare_donut_segments(
        values=values,
        dimension_labels=dimension_labels,
        top_n_labels=top_n_labels,
    )

    wedges, _ = ax.pie(
        segment_values,
        colors=segment_colors,
        startangle=startangle,
        counterclock=False,
        labels=None,
        normalize=True,
        wedgeprops={
            "width": donut_width,
            "edgecolor": "white",
            "linewidth": 1.1,
        },
    )

    total = float(segment_values.sum())

    inner_radius = 1.0 - donut_width
    label_radius = (
        inner_radius
        + donut_width * 0.65
    )

    for wedge, label, value in zip(
        wedges,
        segment_labels,
        segment_values,
    ):
        if value <= 0 or total <= 0:
            continue

        middle_angle = (
            wedge.theta1 + wedge.theta2
        ) / 2.0

        angle = np.deg2rad(middle_angle)
        fraction = float(value / total)
        display_label = wrap_dimension_label(
            label,
            width=13,
        )

        if fraction >= 0.055:
            radius = label_radius
        else:
            radius = 1.11

        if fraction >= 0.12:
            fontsize = 9.8
        elif fraction >= 0.075:
            fontsize = 9.2
        elif fraction >= 0.055:
            fontsize = 8.7
        else:
            fontsize = 8.2

        if label == "Others":
            fontweight = "normal"
        else:
            fontweight = "bold"

        ax.text(
            radius * np.cos(angle),
            radius * np.sin(angle),
            display_label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            linespacing=0.88,
            color="black",
            clip_on=False,
        )
    if center_label:
        ax.text(
            0,
            0,
            center_label,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="black",
            linespacing=0.9,
            zorder=10,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_donut_diagram(
    values: np.ndarray,
    output_path: str = "donut_plot.png",
    dimension_labels: list = None,
    top_n_labels: int = 3,
):
    fig, ax = plt.subplots(
        figsize=(9.2, 9.2)
    )

    draw_donut_diagram(
        ax=ax,
        values=values,
        dimension_labels=dimension_labels,
        top_n_labels=top_n_labels,
        donut_width=0.38,
        center_label=None,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )

    plt.close(fig)


def plot_combined_donut_diagrams(
    roi_betas_dict,
    roi_order,
    output_path,
    dimension_labels=None,
    top_n_labels: int = 3,
):
    n_rois = len(roi_order)

    fig, axs = plt.subplots(
        1,
        n_rois,
        figsize=(
            2.7 * n_rois,
            2.75,
        ),
    )

    axs = np.atleast_1d(axs)

    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.02,
        top=0.99,
        wspace=0.06,
    )

    for ax, roi_name in zip(
        axs.flatten(),
        roi_order,
    ):
        if roi_name in roi_betas_dict:
            draw_donut_diagram(
                ax=ax,
                values=roi_betas_dict[roi_name],
                dimension_labels=dimension_labels,
                top_n_labels=top_n_labels,
                donut_width=0.40,
                center_label=roi_name,
            )
        else:
            ax.set_axis_off()

        ax.set_title("")

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )

    plt.close(fig)


draw_rose_diagram = draw_donut_diagram
plot_rose_diagram = plot_donut_diagram
plot_combined_rose_diagrams = plot_combined_donut_diagrams


def load_dimension_names(naming_file: str):
    names = {}

    with open(
        naming_file,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            prefix, name = line.split(":", 1)
            parts = prefix.split()

            if (
                len(parts) >= 2
                and parts[-1].isdigit()
            ):
                dim_index = int(parts[-1]) - 1
                names[dim_index] = name.strip()

    if not names:
        return []

    max_index = max(names)

    return [
        names.get(
            dim_index,
            f"Dimension {dim_index + 1}",
        )
        for dim_index in range(max_index + 1)
    ]


def save_top_dimensions(
    values: np.ndarray,
    output_path: str,
    dimension_labels=None,
    top_n: int = 3,
):
    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    processed_values = np.maximum(
        values,
        0,
    )

    top_n = min(
        top_n,
        processed_values.shape[0],
    )

    top_indices = np.argsort(
        -processed_values,
        kind="stable",
    )[:top_n]

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "Rank\t"
            "Dimension_1based\t"
            "Dimension_Name\t"
            "Activation_Value\n"
        )

        for rank, dim_index in enumerate(
            top_indices,
            start=1,
        ):
            dim_name = ""

            if (
                dimension_labels
                and dim_index < len(dimension_labels)
            ):
                dim_name = dimension_labels[dim_index]

            f.write(
                f"{rank}\t"
                f"{dim_index + 1}\t"
                f"{dim_name}\t"
                f"{values[dim_index]:.8g}\n"
            )


def get_safe_roi_name(roi_name):
    return (
        roi_name
        .replace("/", "_")
        .replace("\\", "_")
    )


def resolve_input_npy(
    output_root,
    model,
    input_npy=None,
):
    if input_npy:
        return input_npy

    candidates = [
        pjoin(
            output_root,
            model,
            f"{model}_all_roi_betas.npy",
        ),
        pjoin(
            output_root,
            f"{model}_all_roi_betas.npy",
        ),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def extract_specific_frames(
    gif_path,
    frame_indices,
    target_size=(64, 64),
):
    frames = []

    try:
        with Image.open(gif_path) as gif:
            total_frames = gif.n_frames

            for idx in frame_indices:
                if idx < 1 or idx > total_frames:
                    continue

                gif.seek(idx - 1)

                frame = np.array(
                    gif.convert("RGB").resize(
                        target_size
                    )
                )

                frames.append(frame)

    except Exception as e:
        print(
            f"Error GIF:"
            f"{gif_path}: {e}"
        )

    return frames


def add_filmstrip_border(
    image,
    border_height=None,
    hole_width=None,
    hole_height=None,
    hole_spacing=None,
    border_color=(0, 0, 0),
    hole_color=(255, 255, 255),
):
    height, width, _ = image.shape

    if border_height is None:
        border_height = max(
            4,
            int(height * 0.14),
        )

    if hole_width is None:
        hole_width = max(
            3,
            int(width * 0.06),
        )

    if hole_height is None:
        hole_height = max(
            2,
            int(border_height * 0.6),
        )

    if hole_spacing is None:
        hole_spacing = max(
            6,
            int(width * 0.08),
        )

    border = np.full(
        (
            border_height,
            width,
            3,
        ),
        border_color,
        dtype=np.uint8,
    )

    for x in range(
        hole_spacing // 2,
        width,
        hole_spacing,
    ):
        x_start = max(
            x - hole_width // 2,
            0,
        )

        x_end = min(
            x + hole_width // 2,
            width,
        )

        y_start = (
            border_height // 2
            - hole_height // 2
        )

        y_end = (
            border_height // 2
            + hole_height // 2
        )

        border[
            y_start:y_end,
            x_start:x_end,
        ] = hole_color

    return np.vstack(
        [
            border,
            image,
            border,
        ]
    )


def create_filmstrip_image(
    frames_list,
    gap=2,
):
    if not frames_list:
        return None

    gap_array = np.full(
        (
            frames_list[0].shape[0],
            gap,
            3,
        ),
        (255, 255, 255),
        dtype=np.uint8,
    )

    concatenated_image = frames_list[0]

    for frame in frames_list[1:]:
        concatenated_image = np.hstack(
            (
                concatenated_image,
                gap_array,
                frame,
            )
        )

    return add_filmstrip_border(
        concatenated_image
    )


def process_gif_to_image_array(
    input_gif_path,
    frame_indices,
):
    try:
        frames = extract_specific_frames(
            input_gif_path,
            frame_indices,
        )

        if frames:
            return create_filmstrip_image(
                frames
            )

    except Exception as e:
        print(
            f"Error {input_gif_path}: "
            f"{str(e)}"
        )

    return None


def vcorrcoef(X, y):
    Xm = np.reshape(
        np.mean(X, axis=1),
        (X.shape[0], 1),
    )

    ym = np.mean(y)

    r_num = np.sum(
        (X - Xm) * (y - ym),
        axis=1,
    )

    r_den = np.sqrt(
        np.sum(
            (X - Xm) ** 2,
            axis=1,
        )
        * np.sum(
            (y - ym) ** 2
        )
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        r = r_num / r_den

    return np.nan_to_num(
        r,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


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

        self.gif_map = (
            gif_map
            if gif_map is not None
            else {}
        )

        self.frame_indices = (
            frame_indices
            if frame_indices is not None
            else []
        )

        self.similarity_metric = similarity_metric
        self.use_fisher = use_fisher
        self.plot_k = plot_k
        self.plot_dpi = plot_dpi
        self.plot_fileformat = plot_fileformat

        self._check_inputs()

    def _check_inputs(self):
        if self.similarity_metric not in (
            "correlation",
            "rectcosine",
            "cosine",
        ):
            raise ValueError(
                "similarity_metric must be "
                "'correlation'、'rectcosine' "
                "or 'cosine'。"
            )

        if (
            self.img_emb.shape[-1]
            != self.roibetas.shape[0]
        ):
            raise ValueError(
                "Embedding dimension and ROI beta not same"
            )

        os.makedirs(
            self.outdir,
            exist_ok=True,
        )

    def find_top_objects(self):
        if self.similarity_metric == "rectcosine":
            betas = self.roibetas.copy()
            betas[betas < 0] = 0

            s = cosine_similarity(
                self.img_emb,
                betas.reshape(1, -1),
            ).flatten()

        elif self.similarity_metric == "cosine":
            s = cosine_similarity(
                self.img_emb,
                self.roibetas.reshape(1, -1),
            ).flatten()

        else:
            s = vcorrcoef(
                self.img_emb,
                self.roibetas,
            )

        if (
            self.use_fisher
            and self.similarity_metric == "correlation"
        ):
            s = np.clip(
                s,
                -0.999999,
                0.999999,
            )
            s = np.arctanh(s)

        sort_inds = np.argsort(s)
        fnames_sorted = self.img_fnames[sort_inds]
        s_sorted = s[sort_inds]

        return (
            fnames_sorted,
            s_sorted,
            sort_inds,
        )

    def make_fig_v2(
        self,
        figwidth=20,
    ):
        (
            fnames_sorted,
            s_sorted,
            sort_inds,
        ) = self.find_top_objects()

        del fnames_sorted

        top_k_indices = sort_inds[
            -self.plot_k:
        ]

        print(
            f"\n--- ROI: {self.roiname} ---"
        )

        filmstrip_images = []

        for i, video_idx in enumerate(
            reversed(top_k_indices)
        ):
            rank = i + 1
            video_name = self.img_fnames[video_idx]
            sim_score = s_sorted[-(i + 1)]

            print(
                f"  Rank {rank}: "
                f"'{video_name}' "
                f"(Sim: {sim_score:.4f})"
            )

            gif_index = video_idx + 1

            if gif_index in self.gif_map:
                input_gif_path = self.gif_map[
                    gif_index
                ]

                img_array = (
                    process_gif_to_image_array(
                        input_gif_path,
                        self.frame_indices,
                    )
                )

                if img_array is not None:
                    filmstrip_images.append(
                        img_array
                    )

            else:
                print(
                    "    -> Warning: No GIF"
                    f"Index {gif_index}。"
                )

        if not filmstrip_images:
            print(
                f"    -> Warning: No GIF: ROI {self.roiname} "
            )
            return

        sample_image = filmstrip_images[0]
        height, width, _ = sample_image.shape
        image_aspect_ratio = width / height

        n_cols = min(
            2,
            len(filmstrip_images),
        )

        n_rows = math.ceil(
            len(filmstrip_images) / n_cols
        )

        figheight = (
            figwidth
            * n_rows
            / (
                n_cols
                * image_aspect_ratio
            )
        )

        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                figwidth,
                figheight,
            ),
        )

        fig.subplots_adjust(
            wspace=0,
            hspace=0,
            left=0,
            right=1,
            bottom=0,
            top=1,
        )

        for i, ax in enumerate(
            np.asarray(axs).reshape(-1)
        ):
            if i < len(filmstrip_images):
                ax.imshow(
                    filmstrip_images[i]
                )

            ax.axis("off")

        outfile = pjoin(
            self.outdir,
            (
                f"Top{self.plot_k}_"
                f"{get_safe_roi_name(self.roiname)}."
                f"{self.plot_fileformat}"
            ),
        )

        fig.savefig(
            outfile,
            dpi=self.plot_dpi,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.close(fig)

        print(
            "-" * (
                20
                + len(self.roiname)
            )
        )


def load_filenames_from_txt(filepath):
    filenames = []

    with open(
        filepath,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(" ", 1)

            if len(parts) == 2:
                filenames.append(parts[1])

    return np.asarray(filenames)


def get_model_embedding_file(
    data_root,
    model,
    embedding_file=None,
):
    if embedding_file:
        return embedding_file

    if model not in MODEL_EMBEDDING_FILES:
        raise ValueError(
            f"Unknown model '{model}'. "
            "Please specify --embedding_file."
        )

    return pjoin(
        data_root,
        model,
        MODEL_EMBEDDING_FILES[model],
    )


def get_model_naming_file(
    data_root,
    model,
    naming_file=None,
):
    if naming_file:
        return naming_file

    return pjoin(
        data_root,
        model,
        f"{model}_dimension_naming.txt",
    )


def get_model_output_dir(
    args,
    model,
    n_models,
):
    if args.output_dir:
        if n_models > 1:
            return pjoin(
                args.output_dir,
                model,
            )

        return args.output_dir

    return pjoin(
        args.output_root,
        model,
    )


def process_model(
    args,
    model,
    n_models,
):
    input_npy = resolve_input_npy(
        args.output_root,
        model,
        args.input_npy,
    )

    output_dir = get_model_output_dir(
        args,
        model,
        n_models,
    )

    if args.rose_output_dir is None:
        donut_output_dir = pjoin(
            output_dir,
            "donut_plots",
        )
    else:
        donut_output_dir = args.rose_output_dir

    if args.top_videos_output_dir is None:
        top_videos_output_dir = pjoin(
            output_dir,
            "top_videos",
        )
    else:
        top_videos_output_dir = (
            args.top_videos_output_dir
        )

    naming_file = get_model_naming_file(
        args.data_root,
        model,
        args.naming_file,
    )

    roi_betas_dict = np.load(
        input_npy,
        allow_pickle=True,
    ).item()

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    if not args.skip_rose_plots:
        os.makedirs(
            donut_output_dir,
            exist_ok=True,
        )

        dimension_labels = (
            load_dimension_names(
                naming_file
            )
        )
    else:
        dimension_labels = None

    if not args.skip_top_videos:
        embedding_file = (
            get_model_embedding_file(
                args.data_root,
                model,
                args.embedding_file,
            )
        )

        image_embeddings = np.loadtxt(
            embedding_file
        )

        image_filenames = (
            load_filenames_from_txt(
                args.filenames_file
            )
        )

        gif_map = {}

        for filename in os.listdir(
            args.gif_dir
        ):
            if (
                filename.endswith(".gif")
                and "_" in filename
            ):
                prefix = filename.split(
                    "_",
                    1,
                )[0]

                try:
                    idx = int(prefix)
                except ValueError:
                    continue

                gif_map[idx] = pjoin(
                    args.gif_dir,
                    filename,
                )

    else:
        image_embeddings = None
        image_filenames = None
        gif_map = {}

    print(
        f"\n=== Model: {model} ==="
    )
    print(
        f"Reading ROI beta: {input_npy}"
    )
    print(
        f"Outputdir: {output_dir}"
    )

    if not args.skip_rose_plots:
        print(
            f"Dimension Naming: {naming_file}"
        )

    if args.combined_rose_only:
        combined_output_path = pjoin(
            donut_output_dir,
            (
                f"{model}_combined_donut_"
                f"{'_'.join(args.rose_combine_order)}"
                ".pdf"
            ),
        )

        plot_combined_donut_diagrams(
            roi_betas_dict=roi_betas_dict,
            roi_order=args.rose_combine_order,
            output_path=combined_output_path,
            dimension_labels=dimension_labels,
            top_n_labels=args.top_n_dimensions,
        )

        print(
            f"{combined_output_path}"
        )

        return

    if args.rois:
        rois_to_process = {
            roi_name: beta_values
            for roi_name, beta_values
            in roi_betas_dict.items()
            if roi_name in args.rois
        }
    else:
        rois_to_process = roi_betas_dict

    for roi_name, beta_values in tqdm(
        rois_to_process.items()
    ):
        safe_roi_name = get_safe_roi_name(
            roi_name
        )

        if not args.skip_rose_plots:
            donut_output_path = pjoin(
                donut_output_dir,
                (
                    f"{safe_roi_name}_"
                    "donut_plot.pdf"
                ),
            )

            top_dims_output_path = pjoin(
                donut_output_dir,
                (
                    f"{safe_roi_name}_top"
                    f"{args.top_n_dimensions}_"
                    "dimensions.txt"
                ),
            )

            plot_donut_diagram(
                values=beta_values,
                output_path=donut_output_path,
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

        roi_top_objects_outdir = pjoin(
            top_videos_output_dir,
            safe_roi_name,
        )

        top_objects_analyzer = (
            TopObjectsProfile(
                roiname=roi_name,
                roibetas=beta_values,
                img_emb=image_embeddings,
                img_fnames=image_filenames,
                outdir=roi_top_objects_outdir,
                gif_map=gif_map,
                frame_indices=args.frame_indices,
                plot_k=args.plot_k,
            )
        )

        top_objects_analyzer.make_fig_v2()

    if (
        not args.skip_rose_plots
        and not args.skip_combined_rose
    ):
        combined_output_path = pjoin(
            donut_output_dir,
            (
                f"{model}_combined_donut_"
                f"{'_'.join(args.rose_combine_order)}"
                ".pdf"
            ),
        )

        plot_combined_donut_diagrams(
            roi_betas_dict=roi_betas_dict,
            roi_order=args.rose_combine_order,
            output_path=combined_output_path,
            dimension_labels=dimension_labels,
            top_n_labels=args.top_n_dimensions,
        )

        print(
            f"Save {combined_output_path}"
        )


def main(args):
    if args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        models = DEFAULT_MODELS

    if args.filenames_file is None:
        args.filenames_file = pjoin(
            args.data_root,
            "folder_list",
            "folder_list.txt",
        )

    if args.gif_dir is None:
        args.gif_dir = pjoin(
            args.data_root,
            "gifs_renamed",
        )

    for model in models:
        process_model(
            args,
            model,
            len(models),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--input_npy",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--rose_output_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--top_videos_output_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--embedding_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--naming_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--filenames_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--rois",
        nargs="+",
        type=str,
        default=[
            "MT",
            "MST",
            "FST",
            "TPOJ3",
            "FFC",
        ],
    )

    parser.add_argument(
        "--plot_k",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gif_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--frame_indices",
        nargs="+",
        type=int,
        default=[1, 8, 15],
    )

    parser.add_argument(
        "--top_n_dimensions",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--rose_combine_order",
        nargs="+",
        type=str,
        default=DEFAULT_ROSE_COMBINE_ORDER,
    )

    parser.add_argument(
        "--combined_rose_only",
        action="store_true",
    )

    parser.add_argument(
        "--skip_rose_plots",
        action="store_true",
    )

    parser.add_argument(
        "--skip_combined_rose",
        action="store_true",
    )

    parser.add_argument(
        "--skip_top_videos",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)