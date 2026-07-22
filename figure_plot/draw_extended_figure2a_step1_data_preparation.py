import pandas as pd
import os

FOLDER_LIST_PATH = 'data/folder_list/folder_list.txt'
DIR_TARGET = 'data/category_split/target'
DIR_CATEGORY = 'data/category_split/categories'
DIR_ACTIVITY = 'data/category_split/activities'
OUTPUT_FILE = 'data_action.xlsx'


def parse_folder_list(file_path):
    data = []
    id_to_index = {}   
    name_to_index = {} 
    full_str_to_index = {} 

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
        
            parts = line.split(' ', 1)
            action_id = parts[0]
            action_name = parts[1] if len(parts) > 1 else ""
            
            data.append({
                'id': action_id,
                'name': action_name,
                'root': 0,
                'target': None,
                'category': None,
                'activity': None
            })
            
            id_to_index[action_id] = i
            name_to_index[action_name] = i
            full_str_to_index[line] = i

    return pd.DataFrame(data), id_to_index, name_to_index, full_str_to_index

def process_classification_dir(dir_path, df, column_name, lookup_maps):
    id_map, name_map, full_map = lookup_maps
    for filename in os.listdir(dir_path):
        if not filename.endswith('.txt'):
            continue
        
        class_label = os.path.splitext(filename)[0]
        file_path = os.path.join(dir_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                content = line.strip()
                if not content:
                    continue
                target_idx = None
                
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                content = line.strip()
                if not content:
                    continue
                
                target_idx = None
                parts = content.split()

                if len(parts) > 0 and parts[0] in id_map:
                    target_idx = id_map[parts[0]]
                
                elif content in name_map:
                    target_idx = name_map[content]
                
                elif content in full_map:
                    target_idx = full_map[content]
                
                if target_idx is not None:
                    df.at[target_idx, column_name] = class_label

if __name__ == '__main__':
    df, *maps = parse_folder_list(FOLDER_LIST_PATH)
    lookup_maps = maps # (id_map, name_map, full_map)
    process_classification_dir(DIR_TARGET, df, 'target', lookup_maps)
    process_classification_dir(DIR_CATEGORY, df, 'category', lookup_maps)
    process_classification_dir(DIR_ACTIVITY, df, 'activity', lookup_maps)
    final_df = df[['root', 'target', 'category', 'activity', 'name']]
    final_df = final_df.fillna('Unknown')
    final_df.to_excel(OUTPUT_FILE, index=False)