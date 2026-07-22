import argparse
import os
import numpy as np
from os.path import join as pjoin
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Decode video features from fMRI voxel responses.")
    parser.add_argument(
        "--fmri_dir",
        type=str,
        help="path to fMRI data directory",
        default="data/pycortex/data/fMRI_searchlight_all_brain_per_subject",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        help="path to video embedding directory",
        default="data/pycortex/data/HAD_per_subject_features",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="path to output directory",
        default="data/pycortex/beta_file_decoding/",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model name",
        default="LLM_qwen_7B",
    )
    args = parser.parse_args()
    return args


def load_subject_fmri_data(sub_id, fmri_dir):
    subject = f"sub-{sub_id}"
    fmri_file = os.path.join(fmri_dir, subject, "whole_brain_responses.npy")
    if not os.path.exists(fmri_file):
        raise FileNotFoundError(f"fMRI file not found: {fmri_file}")
    data_fmri = np.load(fmri_file)
    return data_fmri


def load_subject_video_embedding(sub_id, embedding_dir, model):
    embedding_file = pjoin(embedding_dir, model, f"sub-{sub_id}_predicted_embedding.txt")
    embedding_data = np.loadtxt(embedding_file)
    return embedding_data


def safe_zscore(data, axis=0):
    data_z = zscore(data, axis=axis)
    data_z = np.nan_to_num(data_z, nan=0.0)
    return data_z


def zscore_by_train(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1 
    X_train_z = (X_train - mean) / std
    X_test_z = (X_test - mean) / std
    return X_train_z, X_test_z


def correlation_score(y_true, y_pred):
    corrs = np.zeros(y_true.shape[1])
    for i in range(y_true.shape[1]):
        if np.std(y_true[:, i]) == 0 or np.std(y_pred[:, i]) == 0:
            corrs[i] = 0
        else:
            corrs[i], _ = pearsonr(y_true[:, i], y_pred[:, i])
    return corrs


def main(args):
    subject_list = [f"{sub_id:02d}" for sub_id in range(1, 31)]

    os.makedirs(args.outdir, exist_ok=True)

    for sub_id in tqdm(subject_list, desc="Processing Subjects"):
        print(f"\n--- Decoding for sub-{sub_id} ---")
        try:
            fmri_data = load_subject_fmri_data(sub_id, args.fmri_dir)       # (n_samples, n_voxels)
            embedding_data = load_subject_video_embedding(sub_id, args.embedding_dir, args.model)  # (n_samples, n_features)
        except Exception as e:
            print(f"Skipping sub-{sub_id}: {e}")
            continue
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=False)
        y_pred_all = np.zeros_like(embedding_data, dtype=np.float64)
        y_true_all = np.zeros_like(embedding_data, dtype=np.float64)

        for fold_idx, (train_index, test_index) in enumerate(kf.split(fmri_data), start=1):
            X_train_raw, X_test_raw = fmri_data[train_index], fmri_data[test_index]
            y_train_raw, y_test_raw = embedding_data[train_index], embedding_data[test_index]

            X_train, X_test = zscore_by_train(X_train_raw, X_test_raw)
            y_train, y_test = zscore_by_train(y_train_raw, y_test_raw)

            decoder = RidgeCV(alphas=[1, 10, 100, 1000, 10000, 100000])
            decoder.fit(X_train, y_train)
            y_pred = decoder.predict(X_test)

            y_pred_all[test_index] = y_pred
            y_true_all[test_index] = y_test

        prediction_accuracy = correlation_score(y_true_all, y_pred_all)
        mean_accuracy = np.mean(prediction_accuracy)

        print(f"  Mean Correlation Accuracy: {mean_accuracy:.4f}")

        X_full = safe_zscore(fmri_data, axis=0)
        y_full = safe_zscore(embedding_data, axis=0)

        decoder_final = RidgeCV(alphas=[1, 10, 100, 1000, 10000, 100000])
        decoder_final.fit(X_full, y_full)
        decoding_betas = decoder_final.coef_  # Shape: (n_features, n_voxels)
        decoding_betas = safe_zscore(decoding_betas, axis=1)

        sub_outdir = pjoin(args.outdir, args.model)
        os.makedirs(sub_outdir, exist_ok=True)
        np.save(pjoin(sub_outdir, f"sub-{sub_id}_predicted_embeddings.npy"), y_pred_all)
        np.save(pjoin(sub_outdir, f"sub-{sub_id}_decoding_accuracy.npy"), prediction_accuracy)
        np.save(pjoin(sub_outdir, f"sub-{sub_id}_decoding_betas.npy"), decoding_betas)

        print(f"  Results saved to {sub_outdir}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)