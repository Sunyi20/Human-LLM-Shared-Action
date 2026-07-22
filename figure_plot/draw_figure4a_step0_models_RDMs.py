import os
import numpy as np
from scipy.io import savemat
from glob import glob
import re

def main():
    data_dir = "data/consistency_plot/models_RDM"
    output_file = os.path.join(data_dir, "RDM_feature_models.mat")
    npy_files = glob(os.path.join(data_dir, "*.npy"))
    data_dict = {} 
    for npy_file in npy_files:
        file_name = os.path.basename(npy_file).replace('.npy', '')
        var_name = re.sub(r'[^a-zA-Z0-9_]', '_', file_name)
        if var_name[0].isdigit():
            var_name = f"rdm_{var_name}"
        data = np.load(npy_file)
        data_dict[var_name] = data
    savemat(output_file, data_dict)

if __name__ == "__main__":
    main()