import nibabel as nib
import numpy as np
from scipy.spatial import distance
import pandas as pd
from tqdm import tqdm
import os
from collections import deque
import csv
from scipy import stats
from statsmodels.stats.multitest import multipletests 

def load_cifti_and_surfaces(cifti_file, left_surf_file, right_surf_file):
    cifti_img = nib.load(cifti_file)
    cifti_data = cifti_img.get_fdata()

    bmaps = cifti_img.header.get_index_map(1).brain_models

    left_surf = nib.load(left_surf_file)
    right_surf = nib.load(right_surf_file)

    left_coords = left_surf.darrays[0].data  # shape: (~32k, 3)
    right_coords = right_surf.darrays[0].data  # shape: (~32k, 3)

    left_triangles = left_surf.darrays[1].data
    right_triangles = right_surf.darrays[1].data

    all_coords = []
    cifti_indices = []
    hemisphere_labels = []
    vertex_indices = []
    
    for i, bm in enumerate(bmaps):
        indices = bm._vertex_indices 
        
        if 'LEFT' in bm.brain_structure:
            coords = left_coords[indices]
            hemisphere = 'left'
        elif 'RIGHT' in bm.brain_structure:
            coords = right_coords[indices]
            hemisphere = 'right'
        else:
            continue 

        start_idx = bm.index_offset
        end_idx = start_idx + bm.index_count
  
        all_coords.append(coords)
        cifti_indices.extend(range(start_idx, end_idx))
        hemisphere_labels.extend([hemisphere] * len(indices))
        vertex_indices.extend(indices)

    coords_array = np.vstack(all_coords)  # shape: (~59k, 3)
    
    return cifti_data, coords_array, cifti_indices, hemisphere_labels, vertex_indices, left_triangles, right_triangles


def build_surface_neighborhoods(coords, vertex_indices, hemisphere_labels, radius=10):
    neighborhoods = {}

    coords = np.array(coords)
    vertex_indices = np.array(vertex_indices)
    hemisphere_labels = np.array(hemisphere_labels)

    for hemi in ['left', 'right']:
        hemi_mask = hemisphere_labels == hemi
        hemi_coords = coords[hemi_mask]
        hemi_vertex_indices = vertex_indices[hemi_mask]
        hemi_cifti_indices = np.where(hemi_mask)[0]

        for idx in range(len(hemi_vertex_indices)):
            dists = distance.cdist([hemi_coords[idx]], hemi_coords)[0]
            within_radius = np.where(dists <= radius)[0]

            cifti_neighbors = [hemi_cifti_indices[n] for n in within_radius]
            neighborhoods[hemi_cifti_indices[idx]] = cifti_neighbors

    return neighborhoods


def save_searchlight_results_to_cifti(results, template_cifti, output_file):
    template = nib.load(template_cifti)

    header = template.header.copy()

    data = np.zeros((1, len(results)))
    data[0, :] = results

    new_img = nib.Cifti2Image(data, header)

    new_img.to_filename(output_file)
    
    return output_file

def run_surface_searchlight_analysis(subject, radius, model):
    base_dir = "data/pycortex/data/HAD_per_subject_features"
    base_fmri_dir = "/nfs/diskstation/DataStation/public_dataset/HAD_human_action_fMRI_dataset/derivatives/ciftify/"
    cifti_file = os.path.join(base_fmri_dir, f"{subject}/results/ses-action01_task-action_cycle-1_beta.dscalar.nii")
    left_surf_file = os.path.join(base_fmri_dir, f"{subject}/standard_fsLR_surface/{subject}.L.midthickness.32k_fs_LR.surf.gii")
    right_surf_file = os.path.join(base_fmri_dir, f"{subject}/standard_fsLR_surface/{subject}.R.midthickness.32k_fs_LR.surf.gii")
    output_dir = f"data/pycortex/data/searchlight/{model}/"
    os.makedirs(output_dir, exist_ok=True)
    output_radius = os.path.join(output_dir, f"{subject}")
    os.makedirs(output_radius, exist_ok=True)

    cifti_data, coords, cifti_indices, hemisphere_labels, vertex_indices, left_triangles, right_triangles = load_cifti_and_surfaces(
        cifti_file, left_surf_file, right_surf_file
    )

    neighborhoods = build_surface_neighborhoods(coords, vertex_indices, hemisphere_labels, radius)

    searchlight_p_values = np.ones(59412)
    searchlight_results = np.zeros(59412)
    
    feature_file = os.path.join(base_dir, model, f"{subject}_predicted_embedding.txt")
    fmri_file = os.path.join("data/pycortex/data/fMRI_searchlight_all_brain_per_subject", subject, "whole_brain_responses.npy")
    
    data_fmri = np.load(fmri_file) # (720, 59412)
    data_human = np.loadtxt(feature_file)

    rsm_human = np.corrcoef(data_human)
    neighborhood_items = list(neighborhoods.items())

    for idx, (vertex_idx, neighbor_indices) in enumerate(tqdm(neighborhood_items)):
        center_voxels = data_fmri[:][:, neighbor_indices]  # shape: (n, num_neighbors)
        rsm_center = np.corrcoef(center_voxels)
        r, p_value = stats.pearsonr(rsm_human.flatten(), rsm_center.flatten())

        if np.isnan(r):
            raise ValueError(f"Correlation result is NaN at vertex index {vertex_idx}")
        searchlight_results[vertex_idx] = r
        searchlight_p_values[vertex_idx] = p_value
    reject, fdr_p_values_corrected, _, _ = multipletests(searchlight_p_values, method='fdr_bh', alpha=0.05)

    output_r_file = os.path.join(output_radius, f"{subject}_selected_all_searchlight_r_values.npy")
    output_p_file = os.path.join(output_radius, f"{subject}_selected_all_searchlight_p_values.npy")
    output_fdr_p_file = os.path.join(output_radius, f"{subject}_selected_all_searchlight_fdr_p_values.npy") 
    np.save(output_r_file, searchlight_results)
    np.save(output_p_file, searchlight_p_values)
    np.save(output_fdr_p_file, fdr_p_values_corrected) 

    output_csv = os.path.join(output_radius, f"{subject}_selected_all_searchlight_results.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vertex_index', 'correlation', 'p_value', 'fdr_p_value'])
        for i, r_val in enumerate(searchlight_results):
            writer.writerow([i, r_val, searchlight_p_values[i], fdr_p_values_corrected[i]])
    return searchlight_results


    
if __name__ == "__main__":
    subject_list = [f'sub-{i:02d}' for i in range(1,31)]
    radius = [6]
    model = ['LLM_qwen_7B']
    for subject in tqdm(subject_list):
        for r in tqdm(radius):
            for m in model:
                run_surface_searchlight_analysis(subject, r, m)
