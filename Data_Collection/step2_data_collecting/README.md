# Data collection

This directory contains scripts used to collect odd-one-out action judgments
from text-only language models, vision-language models, and a local Qwen model.

## Configuration

The scripts resolve data and output paths relative to the repository root. To
use a separate data directory, set `HUMAN_ACTION_ROOT` before running a script:

```bash
export HUMAN_ACTION_ROOT="../human_action"
```

API credentials are read from environment variables and must never be committed:

```bash
cp data_collection/.env.example data_collection/.env
# Edit data_collection/.env, then load it into the current shell.
set -a
source data_collection/.env
set +a
```

The following variables are supported:

- `CSTCLOUD_API_KEY`: CSTCloud-hosted DeepSeek, GPT, MiniMax, and Qwen models.
- `SILICONFLOW_API_KEY`: SiliconFlow-hosted Qwen model.
- `ZHIPU_API_KEY`: Zhipu GLM model.
- `QWEN25_VL_MODEL_PATH`: optional local path to the Qwen2.5-VL model.
- `CUDA_VISIBLE_DEVICES`: optional CUDA device selection for local inference.

All credentials that previously appeared in the source files have been removed.
Rotate those credentials before publishing this directory.

## Expected project layout

```text
repository-root/
├── data/
│   ├── folder_list.txt
│   ├── transcriptions_selected/
│   └── videos_selected/
├── list/
├── run_experiment/        # Created automatically; ignored by Git
└── data_collection/
```

The exact list subdirectory depends on the selected script and experiment.

## Installation

Create an isolated Python environment, then install the shared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r data_collection/requirements.txt
```

The local Qwen script additionally requires a CUDA-compatible PyTorch setup and
FlashAttention configured for the target server.

## Usage

Most scripts take a split number as the first positional argument:

```bash
python data_collection/run_deepseek_r1_671B.py 1
python data_collection/run_qwen2.5_VL_72B.py 1
python data_collection/Qwen2.5_GPU5_train.py 18
```

Outputs are written below the configured project root. Existing choice and
failure files allow most API-based scripts to resume from completed records.

Run the repository-readiness checks with:

```bash
python -m unittest discover -s data_collection/tests -v
```
