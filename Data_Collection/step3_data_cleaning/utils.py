import os

def reindex_triplets(triplets_path, old_index_path, new_index_path, output_path):
    """
    Re-indexes a triplets file based on action name matching.
    """

    # 1. Load Old Index: Map Old_ID -> Action_Name
    # Format: "3 applauding"
    old_id_to_name = {}
    print(f"Loading old index from: {old_index_path}")
    with open(old_index_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Assuming ID is the first element, and the rest is the name
                # Adjust split logic if name contains spaces and format is strict
                idx = int(parts[0])
                name = " ".join(parts[1:])
                old_id_to_name[idx] = name

    # 2. Load New Index: Map Action_Name -> New_ID
    # Format: "5 applauding"
    name_to_new_id = {}
    print(f"Loading new index from: {new_index_path}")
    with open(new_index_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx = int(parts[0])
                name = " ".join(parts[1:])
                if name not in name_to_new_id:
                    name_to_new_id[name] = idx

    # 3. Create Mapping: Old_ID -> New_ID
    old_to_new_map = {}
    missing_actions = set()

    for old_id, action_name in old_id_to_name.items():
        if action_name in name_to_new_id:
            old_to_new_map[old_id] = name_to_new_id[action_name]
        else:
            missing_actions.add(action_name)
            old_to_new_map[old_id] = -1

    if missing_actions:
        print(f"Warning: {len(missing_actions)} actions from old index not found in new index.")
        print(f"Missing actions: {missing_actions}")

    # 4. Process Triplets
    print(f"Processing triplets from: {triplets_path}")
    converted_count = 0
    skipped_count = 0

    with open(triplets_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            parts = list(map(int, line.strip().split()))
            if len(parts) != 3:
                continue

            old_triplet = parts
            new_triplet = []
            valid_triplet = True

            for old_id in old_triplet:
                if old_id in old_to_new_map and old_to_new_map[old_id] != -1:
                    new_triplet.append(old_to_new_map[old_id])
                else:
                    valid_triplet = False
                    break

            if valid_triplet:
                f_out.write(f"{new_triplet[0]} {new_triplet[1]} {new_triplet[2]}\n")
                converted_count += 1
            else:
                skipped_count += 1

    print(f"Done. Saved to {output_path}")
    print(f"Converted: {converted_count}, Skipped (due to missing mapping): {skipped_count}")

def find_missing_triplets(full_set_path, subset_path, output_path):
    """
    Finds triplets present in full_set_path but missing in subset_path.
    Treats triplets as unordered (e.g., "0 1 2" == "2 1 0") and preserves duplicates:
    if full has a triplet 3 times and subset has it 1 time, missing will include it twice.
    """
    from collections import Counter

    print(f"Loading full set from: {full_set_path}")
    full_counter = Counter()
    with open(full_set_path, 'r') as f:
        for line in f:
            try:
                parts = list(map(int, line.strip().split()))
            except ValueError:
                continue
            if len(parts) == 3:
                parts.sort()  # treat as unordered
                full_counter[tuple(parts)] += 1

    print(f"Loading subset from: {subset_path}")
    subset_counter = Counter()
    with open(subset_path, 'r') as f:
        for line in f:
            try:
                parts = list(map(int, line.strip().split()))
            except ValueError:
                continue
            if len(parts) == 3:
                parts.sort()  # treat as unordered
                subset_counter[tuple(parts)] += 1

    # Compute missing counts per triplet (respecting multiplicity)
    missing_items = []
    for triplet, count in full_counter.items():
        missing_count = count - subset_counter.get(triplet, 0)
        if missing_count > 0:
            missing_items.extend([triplet] * missing_count)

    full_total = sum(full_counter.values())
    subset_total = sum(subset_counter.values())
    missing_total = len(missing_items)

    print(f"Full set total entries: {full_total} (unique: {len(full_counter)})")
    print(f"Subset total entries: {subset_total} (unique: {len(subset_counter)})")
    print(f"Missing count (with multiplicity): {missing_total}")

    # Save results (each missing triplet written as a line; duplicates preserved)
    print(f"Saving missing triplets to: {output_path}")
    with open(output_path, 'w') as f:
        for triplet in missing_items:
            f.write(f"{triplet[0]} {triplet[1]} {triplet[2]}\n")


def filter_triplets(input_path, output_path, exclude_ids):
    """
    Filters out triplets containing any of the exclude_ids.
    """
    print(f"Filtering triplets from: {input_path}")
    print(f"Excluding IDs: {exclude_ids}")

    kept_count = 0
    removed_count = 0

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            try:
                parts = list(map(int, line.strip().split()))
            except ValueError:
                continue

            if len(parts) != 3:
                continue

            # Check if any number in the triplet is in the exclusion set
            if any(pid in exclude_ids for pid in parts):
                removed_count += 1
            else:
                f_out.write(line)
                kept_count += 1

    print(f"Done. Saved to {output_path}")
    print(f"Kept: {kept_count}, Removed: {removed_count}")


def add_index_to_file(input_path, output_path):
    print(f"Processing file: {input_path}")

    try:
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            lines = f_in.readlines()
            count = 0
            for line in lines:
                action = line.strip()
                if action:  # 跳过空行
                    # 写入格式: 0 aiming
                    f_out.write(f"{count} {action}\n")
                    count += 1

        print(f"Successfully processed {count} lines.")
        print(f"Saved indexed file to: {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def move_first_to_last(input_path, output_path):
    """
    读取文件，将每行的第一个元素移动到最后一个位置。
    例如: "0 1 2" -> "1 2 0"
    """
    print(f"Processing file: {input_path}")

    count = 0
    try:
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()

                # 确保有数据且至少有2个元素才有移动的意义
                if len(parts) >= 2:
                    # 取出第一个元素，放到列表末尾
                    # parts[1:] 获取从第二个元素开始的所有元素
                    # [parts[0]] 将第一个元素包装成列表
                    new_parts = parts[1:] + [parts[0]]

                    # 重新组合成字符串写入
                    f_out.write(" ".join(new_parts) + "\n")
                    count += 1
                elif len(parts) == 1:
                    # 如果只有一个元素，直接写入
                    f_out.write(line)

        print(f"Successfully processed {count} lines.")
        print(f"Saved result to: {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def reorder_triplets_by_reference(reference_path, target_path, output_path):
    """
    根据 reference_path 的顺序重排 target_path 中的 triplets。
    忽略 triplet 内部元素的顺序 (即 0 1 2 == 2 1 0)。
    如果 target 中缺失某个 triplet，则在 output 中留空行。
    修改：每个 target 中的 triplet 只会被匹配一次 (消耗模式)。

    若 target_path 为文件夹，则对文件夹下所有 .txt 文件批量处理，
    输出文件命名规则：triplets_xxx.txt -> triplets_sorted_xxx.txt，保存至 output_path 文件夹。
    若 target_path 为单个文件，则行为与原来相同。
    """
    import os

    def _process_single(reference_path, target_path, output_path):
        print(f"Reference file: {reference_path}")
        print(f"Target file (to be reordered): {target_path}")

        target_dict = {}
        try:
            with open(target_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        triplet = list(map(int, parts))
                        key = tuple(sorted(triplet))
                        if key not in target_dict:
                            target_dict[key] = []
                        target_dict[key].append(line.strip())
        except FileNotFoundError:
            print(f"Error: Target file not found at {target_path}")
            return

        total_loaded = sum(len(v) for v in target_dict.values())
        print(f"Loaded {total_loaded} triplets ({len(target_dict)} unique types) from target file.")

        found_count = 0
        missing_count = 0

        try:
            with open(reference_path, 'r') as f_ref, open(output_path, 'w') as f_out:
                for line in f_ref:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        ref_triplet = list(map(int, parts))
                        key = tuple(sorted(ref_triplet))
                        if key in target_dict and target_dict[key]:
                            content = target_dict[key].pop(0)
                            f_out.write(content + "\n")
                            found_count += 1
                        else:
                            f_out.write("\n")
                            missing_count += 1
                    else:
                        f_out.write("\n")

            print(f"Processing complete.")
            print(f"Matches found: {found_count}")
            print(f"Missing (empty lines or exhausted): {missing_count}")
            print(f"Saved reordered file to: {output_path}")

        except FileNotFoundError:
            print(f"Error: Reference file not found at {reference_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    if os.path.isdir(target_path):
        os.makedirs(output_path, exist_ok=True)
        txt_files = [f for f in os.listdir(target_path) if f.endswith('.txt')]
        print(f"Found {len(txt_files)} .txt files in {target_path}")
        for filename in txt_files:
            # triplets_xxx.txt -> triplets_sorted_xxx.txt
            if filename.startswith("triplets_"):
                out_filename = filename.replace("triplets_", "triplets_sorted_", 1)
            else:
                out_filename = "sorted_" + filename
            src = os.path.join(target_path, filename)
            dst = os.path.join(output_path, out_filename)
            print(f"\n--- Processing: {filename} ---")
            _process_single(reference_path, src, dst)
    else:
        _process_single(reference_path, target_path, output_path)


def extract_subset(index_list_path, source_embedding_path, output_path):
    # 1. 读取需要提取的索引列表
    target_indices = []
    print(f"正在读取索引文件: {index_list_path}")

    with open(index_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                # 提取每行的第一个元素作为索引
                idx = int(parts[0])

                # 处理特殊情况：跳过 -1 或负数索引
                if idx >= 0:
                    target_indices.append(idx)
                else:
                    print(f"  警告: 跳过负数索引 {idx} (对应标签: {parts[1] if len(parts)>1 else 'Unknown'})")
            except ValueError:
                print(f"  警告: 无法解析行索引: {line.strip()}")

    print(f"共收集到 {len(target_indices)} 个有效索引。")

    # 2. 读取原始Embedding文件
    if not os.path.exists(source_embedding_path):
        print(f"错误: 找不到源文件 {source_embedding_path}")
        return

    print(f"正在读取源Embedding文件...")
    with open(source_embedding_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    total_source_lines = len(all_lines)
    print(f"源文件共有 {total_source_lines} 行。")

    # 3. 提取对应行
    extracted_lines = []
    for idx in target_indices:
        if idx < total_source_lines:
            # 提取对应的行
            extracted_lines.append(all_lines[idx])
        else:
            print(f"  错误: 索引 {idx} 超出了源文件的行数范围 (最大索引应为 {total_source_lines - 1})")

    # 4. 保存结果
    print(f"正在写入新文件: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(extracted_lines)

    print(f"完成! 已提取 {len(extracted_lines)} 行数据。")


def reverse_map_indices(original_ref_path, mapped_ref_path, target_path, output_path):
    """
    利用两份对应的参考文件建立映射关系 (Mapped_ID -> Original_ID)，
    并将 target_path 中的 ID 转换回原始 ID。

    参数:
    - original_ref_path: 原始索引的 triplets 文件 (参考文件 A)
    - mapped_ref_path:   转换后索引的 triplets 文件 (参考文件 B, 行数和内容应与 A 一一对应)
    - target_path:       需要被转换回来的文件或文件夹 (内容是 Mapped_ID)
    - output_path:       结果保存路径 (文件或文件夹)

    若 target_path 为文件夹，则批量处理其中所有 .txt 文件，
    命名规则：triplets_sorted_xxx.txt -> triplets_original_xxx.txt，保存至 output_path 文件夹。
    """
    import os

    def _process_single(original_ref_path, mapped_ref_path, target_path, output_path):
        print("Processing files line by line...")
        print(f"Original Ref: {original_ref_path}")
        print(f"Mapped Ref:   {mapped_ref_path}")
        print(f"Target File:  {target_path}")

        converted_count = 0
        error_count = 0

        try:
            with open(original_ref_path, 'r') as f_orig, \
                 open(mapped_ref_path, 'r') as f_map, \
                 open(target_path, 'r') as f_target, \
                 open(output_path, 'w') as f_out:

                for line_no, (line_orig, line_map, line_target) in enumerate(zip(f_orig, f_map, f_target)):

                    parts_target = line_target.strip().split()

                    if not parts_target:
                        f_out.write("\n")
                        continue

                    parts_orig = line_orig.strip().split()
                    parts_map = line_map.strip().split()

                    if len(parts_orig) == 3 and len(parts_map) == 3:
                        local_map = {}
                        orig_ints = list(map(int, parts_orig))
                        map_ints = list(map(int, parts_map))

                        for o_id, m_id in zip(orig_ints, map_ints):
                            local_map[m_id] = o_id

                        try:
                            target_ints = list(map(int, parts_target))
                            new_ids = []
                            possible = True

                            for tid in target_ints:
                                if tid in local_map:
                                    new_ids.append(local_map[tid])
                                else:
                                    possible = False
                                    break

                            if possible:
                                f_out.write(" ".join(map(str, new_ids)) + "\n")
                                converted_count += 1
                            else:
                                f_out.write("\n")
                                error_count += 1

                        except ValueError:
                            f_out.write("\n")
                            error_count += 1
                    else:
                        f_out.write("\n")
                        error_count += 1

            print(f"Done. Converted {converted_count} lines.")
            print(f"Failed/Skipped {error_count} lines.")
            print(f"Saved to: {output_path}")

        except FileNotFoundError as e:
            print(f"Error loading files: {e}")

    if os.path.isdir(target_path):
        os.makedirs(output_path, exist_ok=True)
        txt_files = [f for f in os.listdir(target_path) if f.endswith('.txt')]
        print(f"Found {len(txt_files)} .txt files in {target_path}")
        for filename in txt_files:
            # triplets_sorted_xxx.txt -> triplets_original_xxx.txt
            if filename.startswith("triplets_sorted_"):
                out_filename = filename.replace("triplets_sorted_", "triplets_original_", 1)
            else:
                out_filename = "original_" + filename
            src = os.path.join(target_path, filename)
            dst = os.path.join(output_path, out_filename)
            print(f"\n--- Processing: {filename} ---")
            _process_single(original_ref_path, mapped_ref_path, src, dst)
    else:
        _process_single(original_ref_path, mapped_ref_path, target_path, output_path)



# triplets_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_triplets_full_sample_54455_filtered.txt'
# old_index_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/data/folder_list_human_odd_one_out_expanded.txt'
# new_index_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/data/folder_list.txt'
# output_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/test.txt'
# reindex_triplets(triplets_file, old_index_file, new_index_file, output_file)


# full_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_triplets_full_sample_54455_filtered.txt'
# subset_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_all_models_triplets_41/behavior_calculate/raw/triplets_qwen2_5_7b.txt'
# output_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/missing_triplets.txt'
# find_missing_triplets(full_file,subset_file, output_file)


# input_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_triplets_full_sample_54678.txt'
# output_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_triplets_full_sample_54461_filtered.txt'
# exclude_set = {246, 247, 248}
# filter_triplets(input_file, output_file, exclude_set)


# input_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/data/folder_list_human_odd_one_out_255.txt'
# output_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/data/folder_list_human_odd_one_out_overlap.txt'
# add_index_to_file(input_file, output_file)


# input_triplets = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_all_models_triplets_41_v2/raw/triplets_qwen2.5_7B_VL_front.txt'
# output_triplets = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_all_models_triplets_41_v2/raw/triplets_qwen2.5_7B_VL.txt'
# move_first_to_last(input_triplets, output_triplets)


# reference_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/list/full_sample_human_10/human_full_sample_triplets.txt'
# target_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_all_models_triplets_41_v2/raw/'
# output_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_all_models_triplets_41_v2/sorted/'
# reorder_triplets_by_reference(reference_file, target_file, output_file)


# ref_original = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_triplets_full_sample_54455_filtered.txt'
# ref_mapped = '/data1/home/sunyi/large-files/Python_Objects/human_action/list/full_sample_human_10/human_full_sample_triplets.txt'
# target_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_all_models_triplets_41_v2/sorted/'
# output_file = '/data1/home/sunyi/large-files/Python_Objects/human_action/consistency_plot/data/human_all_models_triplets_41_v2/original/'
# reverse_map_indices(ref_original, ref_mapped, target_file, output_file)

index_list_path = '/data1/home/sunyi/large-files/Python_Objects/human_action/data/folder_list_MiT_human_odd_one_out_overlap.txt'
source_embedding_path = '/data1/home/sunyi/large-files/Python_Objects/human_action/SPoSE_calculate/data/qwen_72B_VL_spose_embedding_sorted_final.txt'
output_path = '/data1/home/sunyi/large-files/Python_Objects/human_action/shared_vs_unique/MLLM_qwen_72B_overlap_spose_embedding.txt'
extract_subset(index_list_path, source_embedding_path, output_path)