import argparse
import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
import scipy.io as sio
import nibabel as nib
from tqdm import tqdm
import cortex


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate TopObjects profiles for predefined MMP ROIs.")
    parser.add_argument(
        "--beta_dir",
        type=str,
        default="data/pycortex/beta_file_encoding/",
    )
    parser.add_argument(
        "--model", 
        type=str,
        default="LLM_deepseek"
    )
    parser.add_argument(
        "--ndims",
        type=int,
        default=30
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/pycortex/topvideos_roi",
    )
    parser.add_argument(
        "--support_path",
        type=str,
        help="Path to the directory with ROI support files.",
        default="support_files/"
    )
    args = parser.parse_args()
    return args

def roi_mask(subject, roi_name, roi_all_names, roi_index):
    """
    Parameters:
    ----------
    roi_name : list or str
    roi_all_names: list
        all of the roi names in roilbl_mmp.csv
    roi_index: ndarray
        the correponding label of each ROI in 32k space
    """
    select_index = []
    left_roi_key = f'L_{roi_name}_ROI'
    index = cortex.utils.get_roi_verts(subject)
    if left_roi_key in roi_all_names.iloc[:, 0].values:
        roi_tmp_index = roi_all_names[roi_all_names.iloc[:, 0] == left_roi_key].index[0] + 1
        select_index.extend([roi_tmp_index, roi_tmp_index + 180])
    else:
        print(roi_name)
        select_index.extend(index[roi_name])
    
    mask = np.asarray([True if x in select_index else False for x in roi_index[0]])
    return mask

def main(args):
    roi_all_names = pd.read_csv(pjoin(args.support_path, 'roilbl_mmp.csv'))
    roi_index_mat = sio.loadmat(pjoin(args.support_path, 'MMP_mpmLR32k.mat'))
    roi_index = roi_index_mat['glasser_MMP']  # Shape: (1, 59412)
    n_voxels = roi_index.shape[1]

    all_roi_labels = roi_all_names.iloc[:, 0].str.replace(r'^L_', '', regex=True).str.replace(r'_ROI$', '', regex=True)
    rois_to_analyze = {roi_label: roi_label for roi_label in all_roi_labels}
    
    custom_rois = ['3b_section1', '3b_section2', '3b_section3', '3b_section4', '3b_section5', '3b_section6', '3b_section7']
    for roi in custom_rois:
        rois_to_analyze[roi] = roi

    print(rois_to_analyze)


    subject_list = [f"{sub_id:02d}" for sub_id in range(1, 31)]
    all_subs_roi_betas = {}

    for sub_id in tqdm(subject_list):
        subject = f"sub-{sub_id}"
        beta_dir = pjoin(args.beta_dir, args.model)
        beta_subject_dir = pjoin(args.beta_dir, args.model, subject)
        
        all_betas_sub = np.zeros((n_voxels, args.ndims))
        for dim_i in range(args.ndims):
            beta_filename = pjoin(beta_subject_dir, f'{subject}_betas_dim-{dim_i+1}.dtseries.nii')
            beta_cifti = nib.load(beta_filename)
            all_betas_sub[:, dim_i] = beta_cifti.get_fdata()[0, :n_voxels]
        np.save(pjoin(beta_dir, f'{subject}_encoding_betas.npy'), all_betas_sub)   
        all_betas_sub = np.abs(all_betas_sub)
        current_sub_betas = {}
        for roi_key, roi_name in rois_to_analyze.items():
            mask = roi_mask('test', roi_name, roi_all_names, roi_index)
            if mask.sum() > 0:
                roi_mean_beta = all_betas_sub[mask, :].mean(axis=0)
                current_sub_betas[roi_key] = roi_mean_beta
        
        if current_sub_betas:
            all_subs_roi_betas[subject] = current_sub_betas


    avg_roibetas = {}
    valid_rois = list(rois_to_analyze.keys())
    
    for roi_key in valid_rois:
        betas_for_this_roi = [
            all_subs_roi_betas[sub][roi_key] 
            for sub in all_subs_roi_betas 
            if roi_key in all_subs_roi_betas[sub]
        ]
        if betas_for_this_roi:
            avg_roibetas[roi_key] = np.mean(betas_for_this_roi, axis=0)

    results = []
    for roi_key, betas in avg_roibetas.items():
        if betas.size > 0:
            sorted_dim_indices = np.argsort(betas)[::-1]
            total_beta_value = np.sum(betas)
            
            top1_dim = sorted_dim_indices[0] + 1
            top1_beta = betas[sorted_dim_indices[0]]
            
            top2_dim = sorted_dim_indices[1] + 1 if len(sorted_dim_indices) > 1 else 'N/A'
            top2_beta = betas[sorted_dim_indices[1]] if len(sorted_dim_indices) > 1 else 'N/A'
            
            top3_dim = sorted_dim_indices[2] + 1 if len(sorted_dim_indices) > 2 else 'N/A'
            top3_beta = betas[sorted_dim_indices[2]] if len(sorted_dim_indices) > 2 else 'N/A'

            results.append({
                'ROI': roi_key,
                'Total_Beta_Value': total_beta_value,
                'Top1_Dimension': top1_dim,
                'Top1_Beta_Value': top1_beta,
                'Top2_Dimension': top2_dim,
                'Top2_Beta_Value': top2_beta,
                'Top3_Dimension': top3_dim,
                'Top3_Beta_Value': top3_beta,
            })

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Total_Beta_Value', ascending=False)
        
        model_outdir = pjoin(args.outdir, args.model)
        os.makedirs(model_outdir, exist_ok=True)
        output_filename_csv = pjoin(model_outdir, f'{args.model}_all_roi_dimensions.csv')
        results_df.to_csv(output_filename_csv, index=False, float_format='%.4f')
        print("\n--- Top 5 ROIs ---")
        print(results_df.head(5).to_string(index=False))

        if avg_roibetas:
            output_filename_npy = pjoin(args.outdir, f'{args.model}_all_roi_betas.npy')
            np.save(output_filename_npy, avg_roibetas)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
