import numpy as np
import os

def extract_features_for_selected_objects():
    full_features_path = "data/Human/human_odd_one_out_spose_embedding_sorted_delete.txt"
    all_words_path = "data/folder_list/folder_list_human_odd_one_out_expanded.txt"  
    selected_words_path = "data/folder_list/folder_list_full_sample.txt"  
    output_path = "data/Human/spose_embedding_sorted_human_full_sample_delete.txt"

    full_features = np.loadtxt(full_features_path)
    with open(all_words_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        all_words = [line.split(' ', 1)[1] if ' ' in line else line for line in lines]

        
    with open(selected_words_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        selected_words = [line.split(' ', 1)[1] if ' ' in line else line for line in lines]
    print(f"Loading {len(selected_words)} selected words from {selected_words_path}")


    word_positions_1 = []
    word_positions_2 = []
    word_positions_3 = []
    
    for word in selected_words:
        indices = [i for i, x in enumerate(all_words) if x == word]
        if len(indices) >= 3:
            word_positions_1.append(indices[0])
            word_positions_2.append(indices[1])
            word_positions_3.append(indices[2])
        else:
            print(f"Warning: '{word}' has only {len(indices)} matches, at least 3 required")

    print(f"Found {len(word_positions_1)} matching index groups")
    
    for i, positions in enumerate([word_positions_1, word_positions_2, word_positions_3], 1):
        selected_features = full_features[positions, :]
        base, ext = os.path.splitext(output_path)
        current_output_path = f"{base}_{i}{ext}"
        np.savetxt(current_output_path, selected_features)
        print(f"Saved extracted features with shape {selected_features.shape} to {current_output_path}")
        pos_filename = f"wordposition_{i}.txt"
        np.savetxt(pos_filename, np.array(positions, dtype=int), fmt='%d')
        print(f"Saved word positions to {pos_filename}")

if __name__ == "__main__":
    extract_features_for_selected_objects()