# Human–LLM Shared Action

Research code for the paper **“The computational geometry of action: convergence across artificial intelligence, human behaviour and cortex”**

This repository contains the data-collection pipeline, sparse positive
similarity embedding (SPoSE) analyses, and figure-generation scripts used to
compare action representations in humans, language models, vision-language
models, and the human brain.

- **Data:** [OSF project 5t23v](https://osf.io/5t23v/)
- **Code:** [github.com/Sunyi20/Human-LLM-Shared-Action](https://github.com/Sunyi20/Human-LLM-Shared-Action)

## Overview

The repository supports three main stages of the analysis:

1. Generate action triplets, collect odd-one-out judgments from text-only and
   vision-language models, and align them with human behavioral data.
2. Learn interpretable action dimensions with SPoSE.
3. Reproduce behavioral, representational-similarity, feature-attribution, and
   fMRI encoding/searchlight analyses and their corresponding figures.

## Repository structure

```text
Human-LLM-Shared-Action/
├── Data_Collection/
│   ├── step1_data_preprocessing/   # Generate train, test, and full-sample triplets
│   ├── step2_data_collecting/      # Query hosted or local language/vision models
│   └── step3_data_cleaning/        # Merge, deduplicate, and prune collected results
├── Action_SPoSE_Learning/
│   ├── models/                     # SPoSE model definitions
│   ├── tripletize.py               # Convert feature matrices into triplet datasets
│   ├── train.py                    # Train sparse positive embeddings
│   ├── sampling.py                 # Sample synthetic judgments from trained models
│   └── plotting.py                 # SPoSE diagnostics and visualizations
└── figure_plot/
    ├── draw_figure*.{py,m,r}       # Main-figure analysis and plotting scripts
    ├── draw_extended_figure*       # Extended-data figure scripts
    ├── draw_supplementary_figure*  # Supplementary figure scripts
    ├── helper_functions/           # Shared Python and MATLAB utilities
    └── support_files/              # Atlas, label, font, and template files
```

Figure scripts are named by manuscript figure, panel, and processing order. For
example, files beginning with `draw_figure6c_rose_step1_` should be run before
the corresponding `step2` and `step3` scripts.

## Data availability

Large input and derived datasets are hosted in the public
[OSF project](https://osf.io/5t23v/) rather than this GitHub repository. They
include:

- `fMRI_searchlight_all_brain_per_subject`: whole-brain responses for each
  participant. Each `whole_brain_response.npy` array has shape
  `(720, 59412)`.
- `HAD_per_subject_features`: prediction-error features and SPoSE
  representations for the videos viewed by each participant.
- `MiT355_features`: prediction-error features for each of the 355 Moments in
  Time action categories.

Download only the datasets required by the analysis you intend to run. Most
figure scripts expect data below a local `data/` directory, while the data
collection scripts use the `HUMAN_ACTION_ROOT` environment variable described
below.

## Requirements

The repository combines Python, MATLAB, and R. A full snapshot of the Python
research environment is provided in [`requirements.txt`](requirements.txt).
Most packages are version-pinned; packages originally exported as local Conda
build paths were normalized to portable package names and remain unpinned.

### Python

Python 3.10 is a conservative choice for the legacy and current scripts in this
repository. A minimal scientific Python environment can be created with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "numpy<1.24" scipy pandas matplotlib seaborn scikit-learn statsmodels tqdm \
  pillow opencv-python torch "openai>=1" requests numba scikit-image
```

Additional scripts require one or more of the following packages:

- **Local vision-language inference:** `transformers`, `qwen-vl-utils`
- **Neuroimaging:** `nibabel`, `nilearn`, `pycortex`, `himalaya`,
  `fracridge`, `fastl2lir`
- **Feature extraction:** OpenAI CLIP, `scikit-image`
- **Visualization and utilities:** `networkx`, `python-louvain`, `d3blocks`,
  `pyppeteer`, `datamapplot`, `colorcet`, `moviepy`, `pypdf`

Install the optional packages required by the scripts you plan to run. GPU
workflows also require a CUDA-compatible PyTorch installation. The NumPy upper
bound above preserves compatibility with legacy SPoSE code that uses deprecated
NumPy scalar aliases.

### R

The R figure scripts use:

```r
install.packages(c(
  "tidyverse", "ggraph", "tidygraph", "readxl", "glue", "shadowtext",
  "ggplot2", "dplyr", "ggpubr"
))
```

### MATLAB

A recent MATLAB installation is required for the `.m` figure scripts. Shared
MATLAB utilities and several third-party plotting functions are bundled under
`figure_plot/helper_functions/`. Some analyses may additionally require the
relevant Statistics, Machine Learning, Image Processing, or Text Analytics
toolboxes.

## Installation

```bash
git clone https://github.com/Sunyi20/Human-LLM-Shared-Action.git
cd Human-LLM-Shared-Action

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Then install the dependencies for the component you intend to use.

To reproduce the full Linux/CUDA 12.4 research environment:

```bash
python -m pip install -r requirements.txt
```

For CPU-only systems or a different CUDA version, install the matching PyTorch
build first and then install the remaining packages as needed.

## Usage

### 1. Prepare triplets

The preprocessing scripts generate category triplets for training, testing,
noise-ceiling estimation, or full sampling:

```bash
python Data_Collection/step1_data_preprocessing/full_sample_list_generate.py
python Data_Collection/step1_data_preprocessing/train_list_generate.py
```

Review the output filenames and category counts in each script before running a
new experiment.

### 2. Collect model judgments

Set a project-data root containing the expected `data/` and `list/`
subdirectories:

```text
HUMAN_ACTION_ROOT/
├── data/
│   ├── folder_list.txt
│   ├── transcriptions_selected/
│   └── videos_selected/
├── list/
└── run_experiment/        # Created by the API-based collection scripts
```

Export the data root and the credential required by the selected provider:

```bash
export HUMAN_ACTION_ROOT="/path/to/human_action_data"
export CSTCLOUD_API_KEY="your-api-key"

python Data_Collection/step2_data_collecting/run_deepseek_v4_284B.py 1
```

The final positional argument selects a triplet split. Other hosted-model
scripts use the same pattern. Use `SILICONFLOW_API_KEY` for the SiliconFlow
runner.

For local Qwen2.5-VL inference:

```bash
export HUMAN_ACTION_ROOT="/path/to/human_action_data"
export QWEN25_VL_MODEL_PATH="/path/to/Qwen-2.5-7B"

python Data_Collection/step2_data_collecting/run_qwen2.5_VL_7B_GPU.py 1
```

Never commit API keys or other credentials.

### 3. Learn SPoSE action dimensions

`tripletize.py` converts a `.mat`, `.txt`, `.csv`, or `.npy` feature matrix
into a 90/10 train-test split:

```bash
mkdir -p work/triplets
cd Action_SPoSE_Learning

python tripletize.py \
  --in_path /path/to/action_features.npy \
  --out_path ../work/triplets \
  --method deterministic \
  --n_samples 100000
```

The resulting directory contains `train_90.npy` and `test_10.npy`. Train a
SPoSE embedding with:

```bash
python train.py \
  --task odd_one_out \
  --modality behavioral/ \
  --triplets_dir ../work/triplets \
  --results_dir ./results/ \
  --plots_dir ./plots/ \
  --learning_rate 0.001 \
  --lmbda 0.012 \
  --embed_dim 50 \
  --batch_size 100 \
  --epochs 500 \
  --device cpu \
  --rnd_seed 42 \
  --num_threads 1
```

Use `python train.py --help` for the complete set of optimization and resume
options.

### 4. Reproduce figures

Run figure scripts from `figure_plot/` so their relative paths and bundled
helper functions resolve correctly:

```bash
cd figure_plot
python draw_figure4a_community_all.py --help
python draw_figure6c_rose_step2_ROI_beta_dimension.py --help
```

MATLAB and R scripts should likewise be launched with `figure_plot/` as the
working directory. Many older analysis scripts are reproducibility snapshots
with dataset-specific path constants; update those constants or use the
available command-line path arguments before execution.

## Reproducibility notes

- Keep the category ordering, subject ordering, and train/test split fixed
  across models when comparing representational similarity matrices.
- Scripts with `step1`, `step2`, and `step3` in their names form ordered
  pipelines and may require outputs from the preceding step.
- Randomized SPoSE and triplet-generation analyses expose seed parameters or set
  seeds in the source. Record any changes to these values.
- Neuroimaging scripts depend on the bundled atlas/support files and on local
  pycortex subject databases.
- No repository-wide automated test suite or continuous-integration workflow is
  currently included.

As a lightweight syntax check:

```bash
python -m compileall Action_SPoSE_Learning Data_Collection figure_plot
```

## Citation

If you use this repository, please cite the accompanying manuscript:

> *The shared cognitive geometry of action perception in foundation models and
> the human brain.*

The full bibliographic citation will be added when the manuscript record is
available.

## Acknowledgments

The SPoSE code includes source-level attribution to Lukas Muttenthaler and the
Max Planck Institute for Human Cognitive and Brain Sciences. Bundled third-party
MATLAB utilities retain their original license files under
`figure_plot/helper_functions/external/`.

## License

This repository does not currently include a top-level license. Unless a file
states otherwise, no permission for reuse or redistribution is granted. See
the license files shipped with third-party utilities for their respective
terms.
