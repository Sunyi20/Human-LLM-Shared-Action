import os
import argparse
from matplotlib.ticker import FormatStrFormatter
import nibabel as nib
import numpy as np
import cortex
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

DATA_DIRS = [
    'data/pycortex/data/searchlight/LLM_qwen_7B',
    'data/pycortex/data/searchlight/LLM_deepseek',
    'data/pycortex/data/searchlight/MLLM_qwen_7B',
    'data/pycortex/data/searchlight/MLLM_qwen_72B',
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

    full_pycortex_data = np.hstack([pycortex_data_left, pycortex_data_right])
    full_pycortex_data[full_pycortex_data == 0] = np.nan

    vtx_data = cortex.Vertex(full_pycortex_data, subject_id,
                             cmap="nipy_spectral", vmin=vmin, vmax=vmax)
    return vtx_data


def find_npy(data_dir, subject_id):
    path = os.path.join(data_dir, subject_id, f'{subject_id}_searchlight_r_values.npy')
    return path if os.path.exists(path) else None

def load_data(path):
    return np.load(path)



def render_brain_image(vtx_data):
    tmp_fig = cortex.quickshow(
        vtx_data,
        with_colorbar=False,
        with_curvature=True,
        recache=True,
        with_rois=False,
        with_labels=False,
    )
    tmp_fig.canvas.draw()
    img = np.frombuffer(tmp_fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(tmp_fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(tmp_fig)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifti_template',
        default='/nfs/diskstation/DataStation/public_dataset/HAD_human_action_fMRI_dataset/derivatives/ciftify/sub-01/results/ses-action01_task-action_run-1_Atlas.dtseries.nii')
    parser.add_argument('--output_dir',
        default='data/pycortex/searchlight_results')
    parser.add_argument('--vmin', type=float, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subject_ids = sorted([
        d for d in os.listdir(DATA_DIRS[0])
        if os.path.isdir(os.path.join(DATA_DIRS[0], d)) and d.startswith('sub-')
    ])


    for subject_id in subject_ids:

        npy_paths = [find_npy(d, subject_id) for d in DATA_DIRS]

        local_vmax = 0.0
        for d, p in zip(DATA_DIRS, npy_paths):
            if p is None:
                continue
            local_max = np.max(load_data(p))
            print(f"  {os.path.basename(d)}: max = {local_max:.6f}")
            if local_max > local_vmax:
                local_vmax = local_max

        fig_cb, ax_cb = plt.subplots(figsize=(8, 1))
        cmap = plt.get_cmap('nipy_spectral')
        norm = mcolors.Normalize(vmin=args.vmin, vmax=local_vmax)

        cb = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax_cb, 
            orientation='horizontal',
            ticks=[args.vmin, 0.10, 0.20, local_vmax]
        )
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cb.ax.tick_params(labelsize=22) 

        cb_output_path = os.path.join(args.output_dir, f"{subject_id}_colorbar.pdf")
        fig_cb.savefig(cb_output_path, dpi=300, bbox_inches='tight')
        plt.close(fig_cb)
