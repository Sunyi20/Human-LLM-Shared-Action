import numpy as np
import scipy.io as sio

mat_data = []
for i in range(9):
    txt_filename = 'data/dim_visualization/raw_LLM_deepseek/Deepseek_subj%s.txt' %(i+1)
    with open(txt_filename, 'r') as file:
        lines = file.readlines()
        row_data = []
        for line in lines:
            row_data.append(line)
        mat_data.append(row_data)

mat_data = np.array(mat_data, dtype=object)
sio.savemat('data/dim_visualization/dimlabel_answers_LLM_deepseek.mat', {'dimlabel_answers_deepseek': mat_data}, format='5', appendmat=False)