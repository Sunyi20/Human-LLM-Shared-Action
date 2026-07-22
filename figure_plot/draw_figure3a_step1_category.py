import os
import numpy as np
import re
import glob

def generate_category_index_by_subaction():
    activities_dir = 'data/category_split/categories'
    folder_list_file = 'data/folder_list/folder_list.txt'
    output_file = 'data/tsne/video_category_index_mit_355.npy'
    category_files = glob.glob(os.path.join(activities_dir, "*.txt"))
    category_files.sort()
    
    main_categories = [os.path.splitext(os.path.basename(f))[0] for f in category_files]
    print(f"Detect categories {main_categories}")
    
    category_to_index = {cat: idx for idx, cat in enumerate(main_categories)}
    
    subaction_names = []
    if not os.path.exists(folder_list_file):
        print(f"Error: File not found {folder_list_file}")
        return

    with open(folder_list_file, 'r') as f:
        for line in f:
            match = re.search(r'^\d+\s+(.*)', line.strip())
            if match:
                subaction_names.append(match.group(1))
    subaction_to_category = {}
    
    for category_name in main_categories:
        txt_file = os.path.join(activities_dir, f"{category_name}.txt")
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        subaction = parts[1].strip()
                        subaction_to_category[subaction] = category_to_index[category_name]
                    elif len(parts) == 1: 
                        parts = line.strip().split()
                        if len(parts) >= 2:
                             subaction = " ".join(parts[1:])
                             subaction_to_category[subaction] = category_to_index[category_name]
                             
    category_index = []
    for subaction in subaction_names:
        if subaction in subaction_to_category:
            category_index.append(subaction_to_category[subaction])
        else:
            print(f"Warning: Subaction '{subaction}' not found in any main category")
            category_index.append(-1)
    
    category_index = np.array(category_index, dtype=np.int32)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, category_index)
    print(f"Save to {output_file}")

if __name__ == '__main__':
    generate_category_index_by_subaction()