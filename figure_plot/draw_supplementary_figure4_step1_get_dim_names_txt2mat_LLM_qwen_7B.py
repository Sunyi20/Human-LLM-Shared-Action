import numpy as np
import scipy.io as sio


DIM_MAPPING = {
    1: 1, 
    2: 10,
    3: 4, 
    4: 7, 
    5: 2, 
    6: 25,
    7: 6, 
    8: 3,  
    9: 15,
    10: 26, 
    11: 30,
    12: 14, 
    13: 11,
    14: 16,
    15: 21,
    16: 18,
    17: 13,
    18: 22,
    19: 8,
    20: 12,
    21: 9,
    22: 19,
    23: 20,
    24: 28,
    25: 5,
    26: 23,
    27: 29,
    28: 17,
    29: 24,
    30: 27

}

def load_data(num_subjects=9):
    mat_data = []
    for i in range(num_subjects):
        txt_filename = 'data/dim_visualization/raw_LLM_qwen_7B/subj%s.txt' % (i+1)
        with open(txt_filename, 'r') as file:
            lines = file.readlines()
            row_data = []
            for line in lines:
                row_data.append(line)
            mat_data.append(row_data)
    
    return np.array(mat_data, dtype=object)

def remap_dimensions(mat_data, dim_mapping):
    num_subjects, num_dims = mat_data.shape
    remapped_data = np.empty((num_subjects, num_dims), dtype=object)
    
    for old_dim, new_dim in dim_mapping.items():
        old_idx = old_dim - 1
        new_idx = new_dim - 1
        remapped_data[:, new_idx] = mat_data[:, old_idx]
    
    return remapped_data

if __name__ == "__main__":
    mat_data = load_data()
    mat_data_remapped = remap_dimensions(mat_data, DIM_MAPPING)
    sio.savemat('data/dim_visualization/dimlabel_answers_LLM_qwen_7B.mat', 
                {'dimlabel_answers_LLM_qwen_7B': mat_data_remapped}, 
                format='5', appendmat=False)