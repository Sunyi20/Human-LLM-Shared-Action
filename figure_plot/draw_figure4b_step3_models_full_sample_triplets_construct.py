import numpy as np
import scipy.io as sio
import os

def perform_odd_one_out_mapped(rdm_mat_path, triplets_file_path, output_dir):
    mat_contents = sio.loadmat(rdm_mat_path)
    rdm_keys = [k for k in mat_contents.keys() if not k.startswith('__')]
    valid_triplets = []

    with open(triplets_file_path, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) == 3:
                valid_triplets.append(parts)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model_name in rdm_keys:
        rdm = mat_contents[model_name]
        if not isinstance(rdm, np.ndarray):
            continue 
        reordered_triplets = []
        out_of_bounds_count = 0
        max_rdm_index = rdm.shape[0] - 1

        for triplet in valid_triplets:
            raw_a, raw_b, raw_c = triplet

            if raw_a > 248:
                a = raw_a - 3
            else:
                a = raw_a
            if raw_b > 248:
                b = raw_b - 3
            else:
                b = raw_b
            if raw_c > 248: 
                c = raw_c - 3
            else:
                c = raw_c
        
            idx_a = (a - 1) // 3
            idx_b = (b - 1) // 3
            idx_c = (c - 1) // 3

            if (idx_a > max_rdm_index or 
                idx_b > max_rdm_index or 
                idx_c > max_rdm_index):
                print(raw_a, raw_b, raw_c)
                out_of_bounds_count += 1
                continue

            dist_ab = rdm[idx_a, idx_b]
            dist_ac = rdm[idx_a, idx_c]
            dist_bc = rdm[idx_b, idx_c]
            
            score_a = dist_ab + dist_ac
            score_b = dist_ab + dist_bc
            score_c = dist_ac + dist_bc
            
            scores = [score_a, score_b, score_c]
            max_idx = np.argmax(scores) 

            current_triplet = [raw_a, raw_b, raw_c]
            odd_one = current_triplet[max_idx]
            others = [x for i, x in enumerate(current_triplet) if i != max_idx]
            
            new_order =  others + [odd_one]
            reordered_triplets.append(new_order)

        output_filename = f"triplets_sorted_{model_name}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f_out:
            for t in reordered_triplets:
                f_out.write(f"{t[0]} {t[1]} {t[2]}\n")

if __name__ == "__main__":
    rdm_mat_path = 'data/consistency_plot/RDM_human_overlap_255_feature_models.mat'
    triplets_file_path = 'data/consistency_plot/human_triplets_full_sample_54455_filtered.txt'
    output_dir = 'data/consistency_plot/human_all_models_triplets_41/feature_calculate/original'
    perform_odd_one_out_mapped(rdm_mat_path, triplets_file_path, output_dir)