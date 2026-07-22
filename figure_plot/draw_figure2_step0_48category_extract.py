import numpy as np

def extract_features_for_selected_objects():
    full_features_path = "data/LLM_deepseek/deepseek_spose_embedding_sorted_final.txt" 
    all_words_path = "data/folder_list/folder_list.txt" 
    selected_words_path = "data/folder_list/folder_list_full_sample_41.txt"  
    output_path = "data/LLM_deepseek/deepseek_spose_embedding_sorted_full_sample.txt"  

    try:
        full_features = np.loadtxt(full_features_path)
    except Exception as e:
        return
    
    try:
        with open(all_words_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            all_words = [line.split(' ', 1)[1] if ' ' in line else line for line in lines]
    except Exception as e:
        return
        
    try:
        with open(selected_words_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            selected_words = [line.split(' ', 1)[1] if ' ' in line else line for line in lines]
    except Exception as e:
        return
    
    word_positions = []
    for word in selected_words:
        try:
            index = all_words.index(word)
            word_positions.append(index)
        except ValueError:
            print(f"Warning: No '{word}'")
    
    
    selected_features = full_features[word_positions, :]
    np.savetxt(output_path, selected_features)
    
    np.savetxt("wordposition48.txt", np.array(word_positions, dtype=int), fmt='%d')

if __name__ == "__main__":
    extract_features_for_selected_objects()