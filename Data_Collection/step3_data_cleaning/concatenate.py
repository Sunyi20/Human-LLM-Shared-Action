import os
import glob
import pandas as pd
import re

# Base directory containing all the split folders
base_dir = 'run_experiment/results_qwen3.5_VL'

# Output directory for combined files
output_dir = "data/raw_results/results_Qwen3.5_full_sample_human/"
os.makedirs(output_dir, exist_ok=True)

# Initialize output files
combined_choices = []
combined_csv_rows = []
combined_sampled = []

# Find all folders matching the pattern
folders = sorted(glob.glob(os.path.join(base_dir, 'results_qwen3.5_VL_triplets_split_*')))
print(f"Found {len(folders)} folders to process")

# Process each folder
for folder in folders:
    # Extract the split number from the folder name
    match = re.search(r'split_(\d+)', folder)
    if not match:
        print(f"Could not extract split number from {folder}, skipping...")
        continue

    split_num = match.group(1)
    print(f"Processing folder: {folder} (split {split_num})")

    # Process choice.txt
    choice_file = os.path.join(folder, 'choice.txt')
    if os.path.exists(choice_file):
        with open(choice_file, 'r', encoding='utf-8') as f:
            choices = f.readlines()
            combined_choices.extend(choices)
        print(f"  - Added {len(choices)} lines from choice.txt")
    else:
        print(f"  - Warning: choice.txt not found in {folder}")

    # Process CSV file (assuming the split number is in the filename)
    csv_pattern = os.path.join(folder, f'train_chains&responses_qwen3.5_VL_split_{split_num}_file.csv')
    csv_files = glob.glob(csv_pattern)

    if csv_files:
        csv_file = csv_files[0]
        try:
            df = pd.read_csv(csv_file, header=None)
            combined_csv_rows.append(df)
            print(f"  - Added {len(df)} rows from {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"  - Error reading CSV file {csv_file}: {e}")
    else:
        print(f"  - Warning: No matching CSV file found in {folder}")

    # Process train_sampled_split_file.txt
    sampled_file = os.path.join(folder, 'train_sampled_split_file.txt')
    if os.path.exists(sampled_file):
        with open(sampled_file, 'r', encoding='utf-8') as f:
            sampled_lines = f.readlines()
            combined_sampled.extend(sampled_lines)
        print(f"  - Added {len(sampled_lines)} lines from train_sampled_split_file.txt")
    else:
        print(f"  - Warning: train_sampled_split_file.txt not found in {folder}")

# Save combined files
if combined_choices:
    with open(os.path.join(output_dir, 'combined_choice.txt'), 'w', encoding='utf-8') as f:
        f.writelines(combined_choices)
    print(f"Saved combined_choice.txt with {len(combined_choices)} lines")

if combined_csv_rows:
    combined_df = pd.concat(combined_csv_rows, ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, 'combined_chains_responses.csv'), index=False, header=False)
    print(f"Saved combined_chains_responses.csv with {len(combined_df)} rows")

if combined_sampled:
    with open(os.path.join(output_dir, 'combined_train_sampled.txt'), 'w', encoding='utf-8') as f:
        f.writelines(combined_sampled)
    print(f"Saved combined_train_sampled.txt with {len(combined_sampled)} lines")

print("Concatenation complete!")