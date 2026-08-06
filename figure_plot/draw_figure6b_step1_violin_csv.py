import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORT_DIR = SCRIPT_DIR / "support_files"
PYCORTEX_DIR = SCRIPT_DIR / "data" / "pycortex"

MODELS = ["LLM_qwen_7B", "MLLM_qwen_7B", "MLLM_qwen_72B", "LLM_deepseek"]
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 31)]

SEARCHLIGHT_RADII = [6]
SEARCHLIGHT_ROOT = Path(
    os.environ.get(
        "SEARCHLIGHT_ROOT",
        "data/pycortex/data/searchlight",
    )
)

VOXEL_OUTPUT_DIR = PYCORTEX_DIR / "voxel_encoding_results_csv"
SEARCHLIGHT_OUTPUT_DIR = PYCORTEX_DIR / "searchlight_results_csv"

VOXEL_REGIONS = [
    "MT",
    "MST",
    "FST",
    "FFC",
    "TPOJ3",
    "LO1",
    "LO2",
    "IPS0",
    "IPS1",
    "IPS2",
    "M1",
    "PM",
    "S1",
    "S2",
]
SEARCHLIGHT_REGIONS = [
    "MT",
    "MST",
    "FST",
    "FFC",
    "TPOJ3",
    "LO1",
    "LO2",
    "IPS0",
    "IPS1",
    "IPS2",
    "M1",
    "PM",
    "S1",
    "S2",
]

REGION_ROIS = {
    "MT": ["MT"],
    "MST": ["MST"],
    "FST": ["FST"],
    "FFC": ["FFC"],
    "TPOJ3": ["TPOJ3"],
    "LO1": ["LO1"],
    "LO2": ["LO2"],
    "IPS0": ["V7"],  # retinotopic IPS0 ~= HCP V7
    "IPS1": ["IP0"],  # retinotopic IPS1 ~= HCP IP0
    "IPS2": ["IPS1", "MIP"],
    "V3a": ["V3A"],
    "V3b": ["V3B"],
    "V4": ["V4"],
    "M1": ["4"],
    "PM": ["6a", "6d", "FEF", "55b", "PEF", "6r", "IFJp"],
    "S1": ["1", "2", "3a", "3b"],
    "S2": ["OP4", "PFop", "PFcm"],
}


def roi_mask(roi_names, roi_all_names, roi_index):
    """Return a bilateral HCP-MMP ROI mask in 32k cortical space."""
    if isinstance(roi_names, str):
        roi_names = [roi_names]

    selected_indices = []
    for roi_name in roi_names:
        left_roi = f"L_{roi_name}_ROI"
        matches = roi_all_names.loc[roi_all_names.isin([left_roi]).any(axis=1)].index
        if matches.empty:
            raise ValueError(f"ROI {left_roi} not found in roilbl_mmp.csv")

        left_index = matches[0] + 1
        selected_indices.extend([left_index, left_index + 180])

    return np.isin(roi_index[0], selected_indices)


def build_region_masks(regions):
    roi_all_names = pd.read_csv(SUPPORT_DIR / "roilbl_mmp.csv")
    roi_index = sio.loadmat(SUPPORT_DIR / "MMP_mpmLR32k.mat")["glasser_MMP"]

    masks = {}
    for region in regions:
        masks[region] = roi_mask(REGION_ROIS[region], roi_all_names, roi_index)
        print(f"Select {masks[region].sum()} voxels in {region}")
    return masks


def summarize_by_region(data, region_masks, regions):
    data = np.asarray(data).ravel()[:59412]
    return [np.nanmean(data[region_masks[region]]) for region in regions]


def save_region_table(results, output_file, regions):
    df_regions = pd.DataFrame(results, index=SUBJECTS, columns=regions)
    df_regions.to_csv(output_file)
    print(f"Saved {output_file}")
    print("Mean across subjects:")
    print(df_regions.mean())
    return df_regions


def export_voxel_encoding_csv(region_masks):
    VOXEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        base_data_path = PYCORTEX_DIR / "data" / "voxel_encoding_average_weights" / model
        if not base_data_path.exists():
            continue

        results = np.full((len(SUBJECTS), len(VOXEL_REGIONS)), np.nan)

        for sub_idx, subject in enumerate(tqdm(SUBJECTS,)):
            encoding_file = base_data_path / subject / f"{subject}_avg_pearson_r.npy"
            if not encoding_file.exists():
                continue

            encoding_data = np.load(encoding_file)
            results[sub_idx, :] = summarize_by_region(
                encoding_data, region_masks, VOXEL_REGIONS
            )

        if np.isnan(results).all():
            continue

        output_file = VOXEL_OUTPUT_DIR / f"{model}_voxel_encoding_results.csv"
        print(f"\n=== Model: {model} | Voxel encoding ===")
        save_region_table(results, output_file, VOXEL_REGIONS)
        print("========================================\n")


def export_searchlight_csv(region_masks):
    SEARCHLIGHT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        for radius in SEARCHLIGHT_RADII:
            base_data_path = (
                SEARCHLIGHT_ROOT
                / model
            )
            if not base_data_path.exists():
                continue

            results = np.full((len(SUBJECTS), len(SEARCHLIGHT_REGIONS)), np.nan)

            for sub_idx, subject in enumerate(
                tqdm(SUBJECTS)
            ):
                searchlight_file = (
                    base_data_path
                    / subject
                    / f"{subject}_searchlight_r_values.npy"
                )
                if not searchlight_file.exists():
                    continue

                searchlight_data = np.load(searchlight_file)
                results[sub_idx, :] = summarize_by_region(
                    searchlight_data, region_masks, SEARCHLIGHT_REGIONS
                )

            if np.isnan(results).all():
                continue

            output_file = SEARCHLIGHT_OUTPUT_DIR / f"{model}_searchlight_results.csv"
            print(f"\n=== Model: {model} | Searchlight radius: {radius} ===")
            save_region_table(results, output_file, SEARCHLIGHT_REGIONS)
            print("========================================\n")


def main():
    regions = list(dict.fromkeys(VOXEL_REGIONS + SEARCHLIGHT_REGIONS))
    region_masks = build_region_masks(regions)
    export_voxel_encoding_csv(region_masks)
    export_searchlight_csv(region_masks)


if __name__ == "__main__":
    main()
