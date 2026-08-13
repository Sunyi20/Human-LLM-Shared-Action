import os


def process_map_and_reshape_triplets(ref_mapped_path, target_path, ref_original_path, 
                                     output_sorted_path, output_final_path, output_reshape_path):
    """
    Combines Reordering, Reverse Mapping, and Reshaping into a single stream process.
    
    1. Reorders 'target_path' to match 'ref_mapped_path'.
    2. Maps tokens back to original IDs using 'ref_original_path'.
    3. Reshapes original IDs to Category IDs (0-40) based on specific 123-item list.
    
    Args:
        ref_mapped_path: Reference triplets in mapped ID space (0-N).
        target_path: Input triplets to be processed.
        ref_original_path: Reference triplets with Original IDs.
        output_sorted_path: Output for Step 1 (Reordered mapped IDs).
        output_final_path: Output for Step 2 (Reordered Original IDs).
        output_reshape_path: Output for Step 3 (Category IDs).
    """
    
    # --- Step 0: Define Reshape Mapping Logic ---
    # List of 123 indices to be grouped into 41 categories (3 items per category)
    sorted_indices_for_reshape = [
        3, 4, 5, 27, 28, 29, 33, 34, 35, 66, 67, 68, 69, 70, 71, 
        84, 85, 86, 135, 136, 137, 147, 148, 149, 177, 178, 179, 
        201, 202, 203, 210, 211, 212, 213, 214, 215, 222, 223, 224, 
        240, 241, 242, 270, 271, 272, 297, 298, 299, 309, 310, 311, 
        321, 322, 323, 345, 346, 347, 351, 352, 353, 360, 361, 362, 
        375, 376, 377, 399, 400, 401, 417, 418, 419, 429, 430, 431, 
        444, 445, 446, 450, 451, 452, 465, 466, 467, 483, 484, 485, 
        507, 508, 509, 540, 541, 542, 558, 559, 560, 576, 577, 578, 
        594, 595, 596, 597, 598, 599, 675, 676, 677, 690, 691, 692, 
        717, 718, 719, 741, 742, 743, 762, 763, 764, 765, 766, 767
    ]
    
    # Build {Original_ID: Category_ID} map
    reshape_map = {original_idx: i // 3 for i, original_idx in enumerate(sorted_indices_for_reshape)}

    print("Starting combined processing (Sort -> Map -> Reshape)...")
    print(f"Target Input: {target_path}")

    # --- Step 1: Load Target Data into Memory for Reordering ---
    target_dict = {}
    try:
        with open(target_path, 'r') as f:
            for line in f:
                parts = line.replace(',', ' ').strip().split() # Handle commas if present
                if len(parts) == 3:
                    triplet_ints = list(map(int, parts))
                    # Use sorted tuple as key to ignore internal order
                    key = tuple(sorted(triplet_ints))
                    if key not in target_dict:
                        target_dict[key] = []
                    target_dict[key].append(parts) # Store raw parts
    except FileNotFoundError:
        print(f"Error: Target file not found at {target_path}")
        return

    # --- Step 2: Stream Process ---
    found_count = 0
    missing_count = 0
    
    try:
        with open(ref_mapped_path, 'r') as f_ref_map, \
             open(ref_original_path, 'r') as f_ref_orig, \
             open(output_sorted_path, 'w') as f_out_sorted, \
             open(output_final_path, 'w') as f_out_final, \
             open(output_reshape_path, 'w') as f_out_reshape:
            
            # Iterate through both reference files simultaneously
            for line_map, line_orig in zip(f_ref_map, f_ref_orig):
                
                parts_map = line_map.strip().split()
                parts_orig = line_orig.strip().split()
                
                current_sorted_triplet = None # To hold the reordered triplet for this line
                original_ids = None           # To hold the calculated original IDs
                
                # Logic A: Find corresponding triplet in target data
                if len(parts_map) == 3:
                    ref_ints = list(map(int, parts_map))
                    key = tuple(sorted(ref_ints))
                    
                    if key in target_dict and target_dict[key]:
                        current_sorted_triplet = target_dict[key].pop(0) 
                        f_out_sorted.write(" ".join(current_sorted_triplet) + "\n")
                        found_count += 1
                    else:
                        f_out_sorted.write("\n")
                        missing_count += 1
                else:
                    f_out_sorted.write("\n")
                    missing_count += 1

                # Logic B: Map back to original IDs (Reverse Mapping)
                if current_sorted_triplet and len(parts_map) == 3 and len(parts_orig) == 3:
                    local_map = {}
                    map_ints = list(map(int, parts_map))
                    orig_ints = list(map(int, parts_orig))
                    
                    for m_id, o_id in zip(map_ints, orig_ints):
                        local_map[m_id] = o_id
                    
                    try:
                        target_ints = list(map(int, current_sorted_triplet))
                        original_ids = []
                        valid_map = True
                        for tid in target_ints:
                            if tid in local_map:
                                original_ids.append(local_map[tid])
                            else:
                                valid_map = False
                                break
                        
                        if valid_map:
                            f_out_final.write(" ".join(map(str, original_ids)) + "\n")
                        else:
                            f_out_final.write("\n")
                            original_ids = None # Mapping failed
                    except ValueError:
                        f_out_final.write("\n")
                        original_ids = None
                else:
                    f_out_final.write("\n")
                
                # Logic C: Reshape (Map Original IDs to Category IDs)
                if original_ids:
                    reshaped_ids = []
                    for oid in original_ids:
                        if oid in reshape_map:
                            reshaped_ids.append(str(reshape_map[oid])) # Convert to 0-40 category
                        else:
                            reshaped_ids.append("-1") # Unknown/Out of set
                    f_out_reshape.write(" ".join(reshaped_ids) + "\n")
                else:
                    f_out_reshape.write("\n") # Propagate empty line if previous steps failed

                    
    except FileNotFoundError as e:
        print(f"Error opening files: {e}")
        return

    print("-" * 30)
    print("Processing Complete.")
    print(f"Triplets Found: {found_count}")
    print(f"Triplets Missing: {missing_count}")
    print(f"1. Sorted file saved to:  {output_sorted_path}")
    print(f"2. Original IDs saved to: {output_final_path}")
    print(f"3. Reshaped IDs saved to: {output_reshape_path}")

ref_mapped_path = 'data/raw_triplets/human_full_sample_triplets_54455.txt'
ref_original_path = 'data/consistency_plot/human_triplets_full_sample_54455_filtered.txt'
target_path = 'data/consistency_plot/human_all_models_triplets_41/behavior_calculate/raw/triplets_sorted_qwen2_5_7b.txt'

output_sorted_path = 'data/consistency_plot/human_all_models_triplets_41/behavior_calculate/raw/triplets_sorted_qwen2_5_7b.txt'
output_final_path = 'data/consistency_plot/human_all_models_triplets_41/behavior_calculate/original/triplets_sorted_qwen2_5_7b.txt'
output_reshape_path = 'data/consistency_plot/human_all_models_triplets_41/behavior_calculate/reshape/triplets_sorted_qwen2_5_7b_reshape.txt'

os.makedirs(os.path.dirname(output_sorted_path), exist_ok=True)
os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
os.makedirs(os.path.dirname(output_reshape_path), exist_ok=True)

process_map_and_reshape_triplets(
    ref_mapped_path, 
    target_path, 
    ref_original_path, 
    output_sorted_path, 
    output_final_path, 
    output_reshape_path
)