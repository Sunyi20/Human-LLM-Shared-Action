import numpy as np
import scipy.io as sio

mat_data = []
for i in range(8):
    txt_filename = 'data/dim_visualization/raw_MLLM_qwen_72B/Qwen_subj%s.txt' %(i+1)
    with open(txt_filename, 'r') as file:
        lines = file.readlines()
        row_data = []
        for line in lines:
            row_data.append(line)
        mat_data.append(row_data)

mat_data = np.array(mat_data, dtype=object)
sio.savemat('data/dim_visualization/dimlabel_answers_MLLM_qwen_72B.mat', {'dimlabel_answers_qwen': mat_data}, format='5', appendmat=False)