import numpy as np
import scipy.io as sio

num_objects = 41

index_files = [
    'data/Human/wordposition_1.txt',
    'data/Human/wordposition_2.txt',
    'data/Human/wordposition_3.txt'
]

id_map = {}  
index_data = []
for fpath in index_files:
    with open(fpath, 'r') as f:
        ids = [int(line.strip()) for line in f.readlines()[:num_objects]]
        index_data.append(ids)

for action_idx in range(num_objects):
    for file_idx in range(len(index_files)):
        original_id = index_data[file_idx][action_idx]
        id_map[original_id] = action_idx

with open('data/raw_triplets/human_triplets_779467.txt', 'r') as file:
    lines = file.readlines()

trial_count_full = np.zeros((num_objects, num_objects), dtype=int)
combination_count_full = np.zeros((num_objects, num_objects), dtype=int)

for line in lines:
    raw_img1, raw_img2, raw_img3 = map(int, line.strip().split())

    mapped = []
    raw_ids = [raw_img1, raw_img2, raw_img3]
    for i, raw_id in enumerate(raw_ids):
        if raw_id in id_map:
            mapped.append((i, id_map[raw_id]))
    if len(mapped) < 2:
        continue
    
    action_ids = [m[1] for m in mapped]
    positions = [m[0] for m in mapped]

    unique_actions = list(set(action_ids))
    if len(unique_actions) < 2:
        continue
    
    if len(mapped) == 3:
        img1, img2, img3 = action_ids
        trial_count_full[img1, img2] += 1
        trial_count_full[img2, img1] += 1
        
        combination_count_full[img1, img2] += 1
        combination_count_full[img2, img1] += 1
        combination_count_full[img1, img3] += 1
        combination_count_full[img3, img1] += 1
        combination_count_full[img2, img3] += 1
        combination_count_full[img3, img2] += 1
    
    elif len(mapped) == 2:
        a1, a2 = action_ids
        p1, p2 = positions
        
        if a1 != a2:
            if p1 == 0 and p2 == 1:
                trial_count_full[a1, a2] += 1
                trial_count_full[a2, a1] += 1
            elif p1 == 1 and p2 == 0:
                trial_count_full[a1, a2] += 1
                trial_count_full[a2, a1] += 1
            
            combination_count_full[a1, a2] += 1
            combination_count_full[a2, a1] += 1

similarity_matrix_full = np.zeros((num_objects, num_objects), dtype=float)
for i in range(num_objects):
    for j in range(i + 1, num_objects):
        if combination_count_full[i, j] > 0:
            numerator = trial_count_full[i, j]
            similarity_score = numerator / combination_count_full[i, j]
            similarity_matrix_full[i, j] = similarity_score
            similarity_matrix_full[j, i] = similarity_score

np.fill_diagonal(similarity_matrix_full, 1)

sio.savemat('data/Human/RSM_human.mat', {'RSM_triplet': similarity_matrix_full})
sio.savemat('data/Human/RDM_human.mat', {'RDM_triplet': 1 - similarity_matrix_full})