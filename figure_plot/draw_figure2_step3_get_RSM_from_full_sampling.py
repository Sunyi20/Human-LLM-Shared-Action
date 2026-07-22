import numpy as np
import scipy.io as sio
import os


num_objects = 48
# Input paths
triplet_data_path = 'data/MLLM_qwen_7B/qwen_7B_VL_full_sample_triplets.txt'
full_features_path = "data/MLLM_qwen_7B/qwen_7B_VL_spose_embedding_sorted_final.txt"
all_words_path = "data/folder_list/folder_list.txt"
selected_words_path = "data/folder_list/folder_list_MiT_human_odd_one_out_full_sample.txt"

# Output paths
rsm_out = 'data/MLLM_qwen_7B/RSM_48_MLLM_qwen_7B.mat'
rdm_out = 'data/MLLM_qwen_7B/RDM_48_MLLM_qwen_7B.mat'
feature_out = "data/MLLM_qwen_7B/qwen_7B_spose_embedding_sorted_full_sample.txt"
# --- Part 1: Similarity Matrix Calculation (RSM/RDM) ---
print("Processing triplet data for Similarity Matrices...")

with open(triplet_data_path, 'r') as file:
    lines = file.readlines()
trial_count = np.zeros((num_objects, num_objects), dtype=int)
combination_count = np.zeros((num_objects, num_objects), dtype=int)
for line in lines:
    parts = line.strip().split()
    if not parts: continue
    
    # img1 is the "odd one out", img2 and img3 are the "similar pair"
    img1, img2, img3 = map(int, parts)
    
    # Update trial count for the pair chosen as similar
    trial_count[img2, img3] += 1
    trial_count[img3, img2] += 1
    
    # trial_count[img1, img2] += 1
    # trial_count[img2, img1] += 1
    
    # Update combination count for all possible pairs in the triplet
    pairs = [(img1, img2), (img1, img3), (img2, img3)]
    for i, j in pairs:
        combination_count[i, j] += 1
        combination_count[j, i] += 1
# Calculate Similarity Score: P(chosen as similar | appeared together)
similarity_matrix = np.eye(num_objects, dtype=float)
for i in range(num_objects):
    for j in range(i + 1, num_objects):
        if combination_count[i, j] > 0:
            score = trial_count[i, j] / combination_count[i, j]
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score
# Save MAT files
sio.savemat(rsm_out, {'RSM_triplet': similarity_matrix})
sio.savemat(rdm_out, {'RDM_triplet': 1 - similarity_matrix})
print(f"Saved RSM and RDM to {os.path.dirname(rsm_out)}")
# --- Part 2: Feature Extraction & Sorting ---
print("Extracting features for selected objects...")

full_features = np.loadtxt(full_features_path)

def get_word_list(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        # Split by first space to remove index/prefix if exists
        return [line.split(' ', 1)[1] if ' ' in line else line for line in lines]
all_words = get_word_list(all_words_path)
selected_words = get_word_list(selected_words_path)
# Map selected words to their indices in the original embedding file
try:
    word_positions = [all_words.index(word) for word in selected_words]
    print(f"Matched {len(word_positions)} objects for feature extraction.")
    
    selected_features = full_features[word_positions, :]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(feature_out), exist_ok=True)
    np.savetxt(feature_out, selected_features)
    print(f"Saved sorted features to {feature_out}")
    
except ValueError as e:
    print(f"Error: One of the selected words was not found in the master list. {e}")
