import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import savemat


def calculate_pairwise_rdm_from_triplets(triplets, num_categories, odd_position='last'):
    dissimilarity_counts = np.zeros((num_categories, num_categories), dtype=np.float32)
    pair_counts = np.zeros((num_categories, num_categories), dtype=np.float32)

    for triplet in triplets:
        if len(triplet) != 3:
            continue
        
        if odd_position == 'first':
            odd, sim1, sim2 = triplet
        elif odd_position == 'last':
            sim1, sim2, odd = triplet

        if not all(0 <= x < num_categories for x in [odd, sim1, sim2]):
            continue

        pairs = [(odd, sim1), (odd, sim2), (sim1, sim2)]
        for i, j in pairs:
            pair_counts[i, j] += 1
            pair_counts[j, i] += 1

        dissimilar_pairs = [(odd, sim1), (odd, sim2)]
        for i, j in dissimilar_pairs:
            dissimilarity_counts[i, j] += 1
            dissimilarity_counts[j, i] += 1

    rdm = np.divide(dissimilarity_counts, pair_counts, 
                    out=np.zeros_like(dissimilarity_counts), 
                    where=pair_counts!=0)

    rdm_normalized = rdm.copy()
    for i in range(num_categories):
        row = rdm_normalized[i, :]
        mask = np.ones_like(row, dtype=bool)
        mask[i] = False
        non_diagonal_elements = row[mask]
        
        if non_diagonal_elements.size > 0:
            min_val = np.min(non_diagonal_elements)
            max_val = np.max(non_diagonal_elements)
            range_val = max_val - min_val

            if range_val > 0:
                rdm_normalized[i, mask] = (non_diagonal_elements - min_val) / range_val
            else:
                rdm_normalized[i, mask] = 0

    np.fill_diagonal(rdm_normalized, 0)
    rdm_normalized = (rdm_normalized + rdm_normalized.T) / 2.0
    return rdm_normalized

def plot_rdm(rdm, labels=None):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(rdm, cmap='viridis', square=True, xticklabels=labels or False, yticklabels=labels or False)
    ax.set_title('Pairwise Representational Dissimilarity Matrix (RDM)')
    ax.set_xlabel('Action Category Index')
    ax.set_ylabel('Action Category Index')
    plt.show()

def load_triplets_from_file(filepath):
    triplets = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parts = line.replace(',', ' ').split()
                    if len(parts) == 3:
                        triplet = tuple(map(int, parts))
                        triplets.append(triplet)
                except ValueError:
                    print(f"Warning: {line_num} {line}")
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    return triplets

if __name__ == '__main__':
    NUM_ACTION_CATEGORIES = 355
    models_config = {
        'LLM_qwen_7B': {
            'filepath': 'data/raw_triplets/Qwen2.5_7B_536457.txt',
            'odd_position': 'last'
        },
        'LLM_deepseek': {
            'filepath': 'data/raw_triplets/deepseek_r1_542359.txt',
            'odd_position': 'first'
        },
        'MLLM_qwen_7B': {
            'filepath': 'data/raw_triplets/Qwen2.5_7B_VL_521912.txt',
            'odd_position': 'first'
        },

        'MLLM_qwen_72B': {
            'filepath': 'data/raw_triplets//Qwen2.5_72B_VL_547335.txt',
            'odd_position': 'last'
        }
    }

    all_rdms_to_save = {}

    for model_name, config in models_config.items():
        TRIPLET_DATA_FILE = config['filepath']
        odd_pos = config['odd_position']
        loaded_triplets = load_triplets_from_file(TRIPLET_DATA_FILE)
        
        if loaded_triplets:
            pairwise_rdm = calculate_pairwise_rdm_from_triplets(loaded_triplets, NUM_ACTION_CATEGORIES, odd_position=odd_pos)
            plot_rdm(pairwise_rdm)
            all_rdms_to_save[model_name] = pairwise_rdm

    if all_rdms_to_save:
        output_mat_path = 'data/varpartition/all_LLM_rdm.mat'
        savemat(output_mat_path, all_rdms_to_save)
