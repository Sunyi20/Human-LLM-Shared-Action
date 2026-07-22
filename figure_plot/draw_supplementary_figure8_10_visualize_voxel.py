import os
import glob
import argparse
import nibabel as nib
import numpy as np
import cortex
import matplotlib.pyplot as plt

def cifti_to_pycortex(cifti_data, cifti_template_path, subject_id, vmin, vmax):

    cifti_template = nib.load(cifti_template_path)
    brain_model_axis = cifti_template.header.get_axis(1)
    
    n_verts_left = cortex.db.get_surf(subject_id, 'flat', 'left')[0].shape[0]
    n_verts_right = cortex.db.get_surf(subject_id, 'flat', 'right')[0].shape[0]
    
    pycortex_data_left = np.full(n_verts_left, np.nan)
    pycortex_data_right = np.full(n_verts_right, np.nan)
    
    for name, data_slice, model in brain_model_axis.iter_structures():
        data_for_struct = cifti_data[data_slice]
        vertex_indices = model.vertex
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            pycortex_data_left[vertex_indices] = data_for_struct
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            pycortex_data_right[vertex_indices] = data_for_struct

    # full_pycortex_data = np.hstack([pycortex_data_left, pycortex_data_right])
    full_pycortex_data = np.abs(np.hstack([pycortex_data_left, pycortex_data_right]))
    full_pycortex_data[full_pycortex_data == 0] = np.nan
    valid_data = full_pycortex_data[~np.isnan(full_pycortex_data)]
    
    # data_min = 0
    data_max = np.max(valid_data) if len(valid_data) > 0 else 1
    print(data_max)

    vtx_data = cortex.Vertex(full_pycortex_data, subject_id, cmap="J4s",vmin=vmin, vmax=vmax)
    # vtx_data = cortex.Vertex(full_pycortex_data, subject_id, cmap="nipy_spectral",vmin=vmin, vmax=vmax)

    return vtx_data

if __name__ == '__main__':
    subject_id = 'NHB2026_main'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', 
        default='data/pycortex/data',
    )
    parser.add_argument(
        '--cifti_template', 
        default='/nfs/diskstation/DataStation/public_dataset/HAD_human_action_fMRI_dataset/derivatives/ciftify/sub-01/results/ses-action01_task-action_run-1_Atlas.dtseries.nii'
    )
    parser.add_argument(
        '--output_dir', 
        default='data/pycortex/results_pdf'
    )
    parser.add_argument('--vmin', type=float, default=0)
    parser.add_argument('--vmax', type=float, default=0.19)
    parser.add_argument('--cmap', default='J4s')
    args = parser.parse_args()

    MODELS = [
        "qwen_7B",
        "deepseek",
        "qwen_7B_VL_v2",
        "qwen_72B_VL"
    ]
    
    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in MODELS:
        npy_path = os.path.join(args.data_dir, f"{model_name}_mean_pearson_r_fdr_masked.npy")
        
        if not os.path.exists(npy_path):
            continue
            
        data_to_visualize = np.load(npy_path)
        pycortex_data = cifti_to_pycortex(data_to_visualize, args.cifti_template, subject_id, args.vmin, args.vmax)

        output_image_path = os.path.join(args.output_dir, f"voxel_{model_name}_main.pdf")
        fig = cortex.quickshow(
            pycortex_data, 
            with_colorbar=False, 
            with_curvature=True, 
            recache=True, 
            linewidth=10,
            with_rois=True, 
            with_labels=False
        )

        fig.savefig(output_image_path, dpi=300)
        plt.close(fig)
