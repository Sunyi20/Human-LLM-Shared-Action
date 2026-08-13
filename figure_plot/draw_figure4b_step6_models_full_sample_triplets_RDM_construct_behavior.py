import numpy as np
import scipy.io as sio
import os
from scipy.stats import pearsonr

model_sets = [
    'LLM_behavior', 
    'MLLM_behavior', 
    'chatglm4_6v', 
    'gpt_oss', 
    'qwen2_5_72b_v',
    'qwen3_235b'
]

num_objects = 41
# Base directory for the new triplets
base_dir = f'data/consistency_plot/human_all_models_triplets_{num_objects}/behavior_calculate/reshape/'
all_rdms = {}
all_splithalf_rdms = {}


for model in model_sets:
    print("model name:", model)
    filename = f'triplets_sorted_{model}_reshape.txt'
    file_path = os.path.join(base_dir, filename)
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        continue

    mid_index = len(lines) // 2
    lines_first_half = lines[:mid_index]
    lines_second_half = lines[mid_index:]

    # ##############################################
    # ## full data
    # ##############################################
    print("full data")
    trial_count_full = np.zeros((num_objects, num_objects), dtype=int)
    combination_count_full = np.zeros((num_objects, num_objects), dtype=int)

    # First pass to fill trial_count and combination_count
    for line in lines:
        img1, img2, img3 = map(int, line.strip().split())
        if img1 == -1:
            combination_count_full[img2, img3] += 1
            combination_count_full[img3, img2] += 1
        elif img2 == -1:
            combination_count_full[img1, img3] += 1
            combination_count_full[img3, img1] += 1
        elif img3 == -1:
            trial_count_full[img1, img2] += 1
            trial_count_full[img2, img1] += 1
            combination_count_full[img1, img2] += 1
            combination_count_full[img2, img1] += 1
            
        else:
            trial_count_full[img1, img2] += 1
            trial_count_full[img2, img1] += 1
            combination_count_full[img1, img2] += 1
            combination_count_full[img2, img1] += 1
            combination_count_full[img1, img3] += 1
            combination_count_full[img3, img1] += 1
            combination_count_full[img2, img3] += 1
            combination_count_full[img3, img2] += 1

    similarity_matrix_full = np.zeros((num_objects, num_objects), dtype=float)
    # Compute similarity matrix
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            if combination_count_full[i, j] > 0:
                numerator = trial_count_full[i, j]
                similarity_score = numerator / combination_count_full[i, j]
                similarity_matrix_full[i, j] = similarity_score
                similarity_matrix_full[j, i] = similarity_score

    np.fill_diagonal(similarity_matrix_full, 1)

    ##############################################
    ## first_half data
    ##############################################
    print("first_half data")
    trial_count_first_half = np.zeros((num_objects, num_objects), dtype=int)
    combination_count_first_half = np.zeros((num_objects, num_objects), dtype=int)

    # First pass to fill trial_count and combination_count
    for line in lines_first_half:
        img1, img2, img3 = map(int, line.strip().split())
        if img1 == -1:
            combination_count_first_half[img2, img3] += 1
            combination_count_first_half[img3, img2] += 1
        elif img2 == -1:
            combination_count_first_half[img1, img3] += 1
            combination_count_first_half[img3, img1] += 1
        elif img3 == -1:
            trial_count_first_half[img1, img2] += 1
            trial_count_first_half[img2, img1] += 1
            combination_count_first_half[img1, img2] += 1
            combination_count_first_half[img2, img1] += 1
            
        else:
            trial_count_first_half[img1, img2] += 1
            trial_count_first_half[img2, img1] += 1
            combination_count_first_half[img1, img2] += 1
            combination_count_first_half[img2, img1] += 1
            combination_count_first_half[img1, img3] += 1
            combination_count_first_half[img3, img1] += 1
            combination_count_first_half[img2, img3] += 1
            combination_count_first_half[img3, img2] += 1

    similarity_matrix_first_half = np.zeros((num_objects, num_objects), dtype=float)
    # Compute similarity matrix
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            if combination_count_first_half[i, j] > 0:
                numerator = trial_count_first_half[i, j]
                similarity_score = numerator / combination_count_first_half[i, j]
                similarity_matrix_first_half[i, j] = similarity_score
                similarity_matrix_first_half[j, i] = similarity_score

    np.fill_diagonal(similarity_matrix_first_half, 1)

    ##############################################
    ## second_half data
    ##############################################
    print("second_half data")
    trial_count_second_half = np.zeros((num_objects, num_objects), dtype=int)
    combination_count_second_half = np.zeros((num_objects, num_objects), dtype=int)

    # First pass to fill trial_count and combination_count
    for line in lines_second_half:
        img1, img2, img3 = map(int, line.strip().split())
        if img1 == -1:
            combination_count_second_half[img2, img3] += 1
            combination_count_second_half[img3, img2] += 1
        elif img2 == -1:
            combination_count_second_half[img1, img3] += 1
            combination_count_second_half[img3, img1] += 1
        elif img3 == -1:
            trial_count_second_half[img1, img2] += 1
            trial_count_second_half[img2, img1] += 1
            combination_count_second_half[img1, img2] += 1
            combination_count_second_half[img2, img1] += 1
            
        else:
            trial_count_second_half[img1, img2] += 1
            trial_count_second_half[img2, img1] += 1
            combination_count_second_half[img1, img2] += 1
            combination_count_second_half[img2, img1] += 1
            combination_count_second_half[img1, img3] += 1
            combination_count_second_half[img3, img1] += 1
            combination_count_second_half[img2, img3] += 1
            combination_count_second_half[img3, img2] += 1

    similarity_matrix_second_half = np.zeros((num_objects, num_objects), dtype=float)
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            if combination_count_second_half[i, j] > 0:
                numerator = trial_count_second_half[i, j]
                similarity_score = numerator / combination_count_second_half[i, j]
                similarity_matrix_second_half[i, j] = similarity_score
                similarity_matrix_second_half[j, i] = similarity_score

    np.fill_diagonal(similarity_matrix_second_half, 1)

    upper_tri_indices = np.triu_indices(num_objects, k=1)
    vec_split1 = similarity_matrix_first_half[upper_tri_indices]
    vec_split2 = similarity_matrix_second_half[upper_tri_indices]
    corr, _ = pearsonr(vec_split1, vec_split2)
    print(f"{model} Split-Half Correlation: {corr:.4f}")

    all_rdms[model] = similarity_matrix_full
    all_splithalf_rdms[model + '_split1'] = similarity_matrix_first_half
    all_splithalf_rdms[model + '_split2'] = similarity_matrix_second_half
    os.makedirs('data/consistency_plot/human_all_models_RSMs', exist_ok=True)
    sio.savemat('data/consistency_plot/human_all_models_RSMs/RSM41_behavior_models.mat', all_rdms)
    sio.savemat('data/consistency_plot/human_all_models_RSMs/RSM41_behavior_models_splithalf.mat', all_splithalf_rdms)