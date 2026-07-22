
import os
import glob
from matplotlib.ticker import FormatStrFormatter
import nibabel as nib
import numpy as np
import cortex
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import argparse


DATA_DIRS = [
    'data/pycortex/data/voxel_encoding_average_weights/qwen_7B/',
    'data/pycortex/data/voxel_encoding_average_weights/deepseek/',
    'data/pycortex/data/voxel_encoding_average_weights/qwen_7B_VL/',
    'data/pycortex/data/voxel_encoding_average_weights/qwen_72B_VL/',
]


def cifti_to_pycortex(cifti_data, cifti_template_path, subject_id, vmin, vmax):
    cifti_template = nib.load(cifti_template_path)
    brain_model_axis = cifti_template.header.get_axis(1)

    n_verts_left  = cortex.db.get_surf(subject_id, 'flat', 'left')[0].shape[0]
    n_verts_right = cortex.db.get_surf(subject_id, 'flat', 'right')[0].shape[0]

    pycortex_data_left  = np.full(n_verts_left,  np.nan)
    pycortex_data_right = np.full(n_verts_right, np.nan)

    for name, data_slice, model in brain_model_axis.iter_structures():
        data_for_struct = cifti_data[data_slice]
        vertex_indices  = model.vertex
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            pycortex_data_left[vertex_indices]  = data_for_struct
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            pycortex_data_right[vertex_indices] = data_for_struct

    full_pycortex_data = np.abs(np.hstack([pycortex_data_left, pycortex_data_right]))

    vtx_data = cortex.Vertex(full_pycortex_data, subject_id,
                             cmap="J4s", vmin=vmin, vmax=vmax)
    return vtx_data


def find_npy(data_dir, subject_id):
    pattern = os.path.join(data_dir, subject_id, '*_avg_pearson_r.npy')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    if len(files) > 1:
        print(f"  [警告] {data_dir}/{subject_id} 下找到多个文件，取第一个: {files[0]}")
    return files[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifti_template',
        default='/nfs/diskstation/DataStation/public_dataset/HAD_human_action_fMRI_dataset/derivatives/ciftify/sub-01/results/ses-action01_task-action_run-1_Atlas.dtseries.nii')
    parser.add_argument('--output_dir',
        default='data/pycortex/voxel_results/')
    parser.add_argument('--vmin', type=float, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subject_ids = sorted([
        d for d in os.listdir(DATA_DIRS[0])
        if os.path.isdir(os.path.join(DATA_DIRS[0], d)) and d.startswith('sub-')
    ])
    if not subject_ids:
        raise FileNotFoundError(f"{DATA_DIRS[0]} no dir")

    for subject_id in subject_ids:
        npy_paths = [find_npy(d, subject_id) for d in DATA_DIRS]

        local_vmax = 0.0
        for d, p in zip(DATA_DIRS, npy_paths):
            if p is None:
                continue
            local_max = np.max(np.abs(np.load(p)))
            print(f"  {os.path.basename(d)}: max = {local_max:.6f}")
            if local_max > local_vmax:
                local_vmax = local_max
        print(f"  → vmax = {local_vmax:.6f}")

        fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

        for ax, p in zip(axes, npy_paths):
            ax.axis('off')
            if p is None:
                continue

            vtx_data = cifti_to_pycortex(
                np.load(p), args.cifti_template,
                'test', args.vmin, local_vmax
            )

            tmp_fig = cortex.quickshow(
                vtx_data,
                with_colorbar=False,
                with_curvature=True,
                recache=True,
                with_rois=False,
                with_labels=False
            )
            tmp_fig.canvas.draw()
            img = np.frombuffer(tmp_fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(tmp_fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(tmp_fig)

            ax.imshow(img)

        output_path = os.path.join(args.output_dir, f"{subject_id}_four_models.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
