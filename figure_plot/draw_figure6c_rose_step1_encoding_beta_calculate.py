from os.path import join as pjoin
import os
import argparse
import numpy as np
from scipy.stats import zscore
from helper_functions.glm import FracRidgeVoxelwise
import nibabel as nib

def parse_arguments():
    parser = argparse.ArgumentParser(
    )
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
        default="data/pycortex/data/HAD_per_subject_features"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="path to output directory",
        default="data/pycortex/beta_file/",
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="model name", 
        default="LLM_deepseek"
    )
    parser.add_argument(
        "--zscore_X",
        action="store_true",
        default=True,
        help="zscore regressors",
    )
    parser.add_argument(
        "--zscore_y",
        action="store_true",
        default=True,
        help="zscore responses",
    )
    args = parser.parse_args()
    return args

def load_subject_fmri_data(sub_id, fmri_dir):
    subject = f"sub-{sub_id}"
    fmri_file = os.path.join(fmri_dir, subject, "whole_brain_responses.npy")

    if not os.path.exists(fmri_file):
        raise FileNotFoundError(f"fMRI file not found: {fmri_file}")
    try:
        data_fmri = np.load(fmri_file)
        print(f"Successfully loaded fMRI data for sub-{sub_id}")
        print(f"fMRI data shape: {data_fmri.shape}")
        return data_fmri
    except Exception as e:
        print(f"Error loading fMRI file {fmri_file}: {e}")
        raise


def load_subject_video_embedding(sub_id, embedding_dir, model="qwen"):
    embedding_file = pjoin(embedding_dir, model, f"sub-{sub_id}_predicted_embedding.txt")
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
    try:
        embedding_data = np.loadtxt(embedding_file)
        print(f"Successfully loaded embedding data for sub-{sub_id}")
        print(f"Embedding shape: {embedding_data.shape}")
        return embedding_data
    except Exception as e:
        print(f"Error loading embedding file {embedding_file}: {e}")
        raise


def prepare_data(sub_id, fmri_dir, embedding_dir, model):
    data_fmri = load_subject_fmri_data(sub_id, fmri_dir)
    embedding_data = load_subject_video_embedding(sub_id, embedding_dir, model)
    return data_fmri, embedding_data


def main(args):
    subject_list = [f"{sub_id:02d}" for sub_id in range(1, 31)]
    for sub_id in subject_list:
        sub_outdir = pjoin(args.outdir, args.model, f"sub-{sub_id}")
        os.makedirs(sub_outdir, exist_ok=True)
        y, X_dims = prepare_data(sub_id, args.fmri_dir, args.embedding_dir, args.model)
        print(y.shape)
        if args.zscore_X:
            print("zscoring X")
            X_dims = zscore(X_dims, axis=0)
        
        print("loading responses")
        if args.zscore_y:
            print("zscoring y")
            y = zscore(y, axis=0)

        print("Start ridge regression")
        fr = FracRidgeVoxelwise(
            n_splits=7,
            test_size=0.0, 
            fracs=np.arange(0.01, 1.01, 0.01),
            run_pcorr=False,
        )
        betas, _, _, best_fracs, _ = fr.tune_and_eval(X_dims, y)
        result_dir = sub_outdir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        fracs_filename = pjoin(result_dir, f'sub-{sub_id}_best_fracs.npy')
        np.save(fracs_filename, best_fracs)

        beta_filename = pjoin(result_dir, f'sub-{sub_id}_betas.npy')
        np.save(beta_filename, betas)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)