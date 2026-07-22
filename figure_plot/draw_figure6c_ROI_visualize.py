import cortex
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib

    
def cifti_to_pycortex(cifti_data, cifti_template_path, subject_id, roi_name):
    cifti_template = nib.load(cifti_template_path)
    brain_model_axis = cifti_template.header.get_axis(1)
    
    n_verts_left = cortex.db.get_surf(subject_id, 'flat', 'left')[0].shape[0]
    n_verts_right = cortex.db.get_surf(subject_id, 'flat', 'right')[0].shape[0]

    pycortex_data_left = np.full(n_verts_left, np.nan)
    pycortex_data_right = np.full(n_verts_right, np.nan)
    
    for name, data_slice, model in brain_model_axis.iter_structures():
        if name not in ('CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT'):
            continue
        data_for_struct = cifti_data[data_slice]
        vertex_indices = model.vertex
        
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            pycortex_data_left[vertex_indices] = data_for_struct
            
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            pycortex_data_right[vertex_indices] = data_for_struct

    full_pycortex_data = np.hstack([pycortex_data_left, pycortex_data_right])
    cortex.utils.add_roi((full_pycortex_data, subject_id), name=roi_name)
    return


if __name__ == '__main__':
    subject_id = 'NHB2026_main'
    
    cifti_template_file = 'support_files/ses-action01_task-action_run-1_Atlas.dtseries.nii'

    rois_to_draw = {
        # '1': './roi_numpy_masks/one_mask.npy',
        # '2': './roi_numpy_masks/two_mask.npy',
        # '3a': './roi_numpy_masks/three_a_mask.npy',
        '3b': 'data/pycortex/data/roi_numpy_masks/three_b_mask.npy',
        # '4': './roi_numpy_masks/four_mask.npy'
        # 'FFC': './roi_numpy_masks/FFC_mask.npy',
        # 'FST': './roi_numpy_masks/FST_mask.npy',
        # 'MST': './roi_numpy_masks/MST_mask.npy',
        # 'MT': './roi_numpy_masks/MT_mask.npy',
        # 'TPOJ': './roi_numpy_masks/TPOJ_mask.npy',
        # 'V1': './roi_numpy_masks/V1_mask.npy',
        # 'V2': './roi_numpy_masks/V2_mask.npy',
        # 'V3': './roi_numpy_masks/V3_mask.npy',
        # 'V4': './roi_numpy_masks/V4_mask.npy',
        # 'V4t': './roi_numpy_masks/V4t_mask.npy',
        # 'LO3': './roi_numpy_masks/LO3_mask.npy',
        
    }

    for roi_name, npy_data_path in rois_to_draw.items():
        roi_mask_data = np.load(npy_data_path)
        cifti_to_pycortex(roi_mask_data, cifti_template_file, subject_id, roi_name)
