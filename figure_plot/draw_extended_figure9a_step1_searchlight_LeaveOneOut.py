import os
import numpy as np
import scipy.io as sio
from himalaya.ridge import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


subject_list = [f'sub-{i:02d}' for i in range(1,31)]
base_dir = "data/pycortex/data/HAD_per_subject_features/MLLM_qwen_72B"

for subject in tqdm(subject_list, desc="Processing Subject"):
    feature_file = os.path.join(base_dir, f"{subject}_predicted_embedding.txt")
    fmri_file = os.path.join("data/pycortex/data/fMRI_searchlight_all_brain_per_subject", subject, "whole_brain_responses.npy")
    
    voxels = np.load(fmri_file) # (720, 59412)
    data_human = np.loadtxt(feature_file)

    print(voxels.shape, data_human.shape)
    tol = 3
    alphas = np.logspace(-tol, tol, 100)
    

    loo = LeaveOneOut()
    score_list = []
    i = 0


    for train_index, test_index in loo.split(data_human):
        print(f"Processing subject {subject}, fold {test_index[0]}")
        model = RidgeCV(alphas=alphas, fit_intercept=True, solver='svd', solver_params=None, cv=5,)
        model.fit(data_human[train_index], voxels[train_index])
        pred = model.predict(data_human[test_index])
        score = mean_squared_error(pred.reshape(-1), voxels[test_index].reshape(-1))
        score_list.append(score)

    output_dir = "data/pycortex/data/searchlight/LeaveOneOut/MLLM_qwen_72B/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, subject), exist_ok=True)

    score_file = os.path.join(output_dir, f"{subject}/score_{subject}.npy")
    indices_file = os.path.join(output_dir, f"{subject}/indices_{subject}.npy")
    
    np.save(score_file, score_list)
    np.save(indices_file, indices)
