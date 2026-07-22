import numpy as np
import scipy.io as sio
import os
from scipy.stats import pearsonr
# Configuration for human data
# Removing 3 objects (244, 245, 246) from 768 -> 765 objects
num_objects = 41
triplets_file_path = 'data/consistency_plot/human_triplets_full_sample_54455_filtered_reshape.txt'
output_dir = 'data/consistency_plot/human_all_models_RSMs'

print(f"Processing human triplets from: {triplets_file_path}")

try:
    with open(triplets_file_path, 'r') as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"File not found: {triplets_file_path}")
    exit()

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
    parts = list(map(int, line.strip().split()))
    # Handle potential different triplet formats (assuming img1, img2, img3 are first 3)
    img1, img2, img3 = parts[0], parts[1], parts[2]
    
    # img1 = map_index(raw_img1)
    # img2 = map_index(raw_img2)
    # img3 = map_index(raw_img3)
    
    # Skip triplets containing excluded objects
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
# ## first_half data
##############################################
print("first_half data")
trial_count_first_half = np.zeros((num_objects, num_objects), dtype=int)
combination_count_first_half = np.zeros((num_objects, num_objects), dtype=int)

# First pass to fill trial_count and combination_count
for line in lines_first_half:
    parts = list(map(int, line.strip().split()))
    img1, img2, img3 = parts[0], parts[1], parts[2]
    
    # img1 = map_index(raw_img1)
    # img2 = map_index(raw_img2)
    # img3 = map_index(raw_img3)
    
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
# ## second_half data
##############################################
print("second_half data")
trial_count_second_half = np.zeros((num_objects, num_objects), dtype=int)
combination_count_second_half = np.zeros((num_objects, num_objects), dtype=int)

# First pass to fill trial_count and combination_count
for line in lines_second_half:
    parts = list(map(int, line.strip().split()))
    img1, img2, img3 = parts[0], parts[1], parts[2]
    
    # img1 = map_index(raw_img1)
    # img2 = map_index(raw_img2)
    # img3 = map_index(raw_img3)
    
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
# Compute similarity matrix
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
print(f"Split-Half Correlation: {corr:.4f}")

# Save results
os.makedirs(output_dir, exist_ok=True)

full_mat_path = os.path.join(output_dir, f'RSM{num_objects}_triplet.mat')
split_mat_path = os.path.join(output_dir, f'RSM{num_objects}_triplet_splithalf.mat')

print(f"Saving full RSM to {full_mat_path}")
sio.savemat(full_mat_path, {f'RSM{num_objects}_triplet': similarity_matrix_full})

print(f"Saving split-half RSMs to {split_mat_path}")
sio.savemat(split_mat_path, {
    f'RSM{num_objects}_triplet_split1': similarity_matrix_first_half,
f'RSM{num_objects}_triplet_split2': similarity_matrix_second_half
})