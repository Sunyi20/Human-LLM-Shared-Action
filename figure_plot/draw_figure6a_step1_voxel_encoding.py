import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from scipy.stats import pearsonr
from os.path import join as pjoin
from sklearn.model_selection import KFold
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from himalaya.ridge import RidgeCV

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run cross-validated linear regression")
    parser.add_argument("--sub", type=str, help="subject id", default="01")
    parser.add_argument("--fmri_dir", type=str, default="data/pycortex/data/fMRI_searchlight_all_brain_per_subject")
    parser.add_argument("--embedding_dir", type=str, default="data/pycortex/data/HAD_per_subject_features")
    parser.add_argument("--model", type=str, default="MLLM_qwen_7B")
    parser.add_argument("--outdir", type=str, default="data/pycortex/data/voxel_encoding_average_weights")
    return parser.parse_args()

def load_data(sub_id, fmri_dir, embedding_dir, model):
    # Load fMRI data
    fmri_file = os.path.join(fmri_dir, f"sub-{sub_id}", "whole_brain_responses.npy")
    data_fmri = np.load(fmri_file)
    
    # Load embedding data
    embedding_file = pjoin(embedding_dir, model, f"sub-{sub_id}_predicted_embedding.txt")
    embedding_data = np.loadtxt(embedding_file)
    return data_fmri, embedding_data

def r2_score(Real, Pred):
    """Calculate R-squared score."""
    SSres = np.mean((Real - Pred) ** 2, 0)
    SStot = np.var(Real, 0, ddof=0)
    return 1 - SSres / SStot

def process_subject(sub_id, args):
    print(f"\nProcessing subject: {sub_id}")
    
    # Setup paths
    sub_outdir = pjoin(args.outdir, args.model, f"sub-{sub_id}")
    os.makedirs(sub_outdir, exist_ok=True)

    # Load data
    Y, X = load_data(sub_id, args.fmri_dir, args.embedding_dir, args.model)
    # X: design matrix (trials x features), Y: response data (trials x voxels)
    
    print(f"Data shapes - X: {X.shape}, Y: {Y.shape}")
    
    # Cross-validation setup
    n_splits = 6
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 3, 100)

    all_Y_test = []
    all_Y_predict = []
    all_weights = np.zeros((X.shape[1], Y.shape[1]))

    # Cross-validation loop
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model = RidgeCV(alphas=alphas, fit_intercept=True, solver='svd', cv=5)
        model.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)

        all_Y_test.append(Y_test)
        all_Y_predict.append(Y_predict)
        all_weights += model.coef_

    # Aggregate results
    all_Y_test = np.concatenate(all_Y_test, axis=0)
    all_Y_predict = np.concatenate(all_Y_predict, axis=0)
    avg_weights = all_weights / n_splits

    # Calculate metrics
    pearson_r = np.array([pearsonr(all_Y_test[:, i], all_Y_predict[:, i])[0] 
                          for i in range(all_Y_test.shape[1])])
    
    r2_scores = np.array([r2_score(all_Y_test[:, i], all_Y_predict[:, i]) 
                          for i in range(all_Y_test.shape[1])])

    # Save results
    np.save(pjoin(sub_outdir, f"sub-{sub_id}_avg_weights.npy"), avg_weights)
    np.save(pjoin(sub_outdir, f"sub-{sub_id}_avg_pearson_r.npy"), pearson_r)
    np.save(pjoin(sub_outdir, f"sub-{sub_id}_avg_r_sqs.npy"), r2_scores)
    
    print(f"Completed sub-{sub_id}")

def main():
    args = parse_arguments()
    subject_list = [f"{sub_id:02d}" for sub_id in range(1, 31)]
    print(f"Will process {len(subject_list)} subjects: {subject_list}")
    
    for sub_id in tqdm(subject_list, desc="Processing subjects"):
        process_subject(sub_id, args)

if __name__ == "__main__":
    main()