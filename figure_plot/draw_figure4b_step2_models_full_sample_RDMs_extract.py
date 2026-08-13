import os
import numpy as np
import scipy.io as sio

def extract_rdm_subset_batch(control_file, source_mat_file, output_mat_file):
    indices_to_extract = []
    with open(control_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            try:
                idx = int(parts[0])
                if idx != -1:
                    indices_to_extract.append(idx)
            except ValueError:
                pass

    mat_contents = sio.loadmat(source_mat_file)
    processed_data = {}
    variable_names = [k for k in mat_contents.keys() if not k.startswith('__')]

    for var_name in variable_names:
        data_matrix = mat_contents[var_name]

        if not isinstance(data_matrix, np.ndarray) or data_matrix.ndim != 2:
            continue
        
        rows, cols = data_matrix.shape
        max_index = rows - 1
        valid_indices = [i for i in indices_to_extract if 0 <= i <= max_index]
        
        subset_matrix = data_matrix[np.ix_(valid_indices, valid_indices)]
        processed_data[var_name] = subset_matrix

    sio.savemat(output_mat_file, processed_data)

if __name__ == "__main__":
    control_txt_path = 'data/folder_list/folder_list_MiT_human_odd_one_out_overlap.txt'
    source_rdm_mat_path = 'data/consistency_plot/RDM_feature_models.mat' 
    output_mat_path = 'data/consistency_plot/RDM_human_overlap_255_feature_models.mat'
    extract_rdm_subset_batch(control_txt_path, source_rdm_mat_path, output_mat_path)
