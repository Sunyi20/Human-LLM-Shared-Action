
def filter_triplets(input_file, output_file, target_subset):

    target_subset_set = set(target_subset) 
    count_kept = 0
    total_lines = 0

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            
            total_lines += 1
            parts = line.replace(',', ' ').split()
            triplet = [int(p) for p in parts]
                
            if len(triplet) != 3:
                continue
            match_count = sum(1 for idx in triplet if idx in target_subset_set)
            if match_count >= 2:
                fout.write(line + "\n")
                count_kept += 1


    print("-" * 30)


input_filename = "data/raw_triplets/human_triplets_779467.txt"
output_filename = "data/consistency_plot/human_triplets_full_sample_54455.txt"
subset_filename = "data/folder_list/folder_list_MiT_human_odd_one_out_full_sample.txt"
my_subset = []
with open(subset_filename, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            idx = int(parts[0])
            my_subset.append(idx)
filter_triplets(input_filename, output_filename, my_subset)