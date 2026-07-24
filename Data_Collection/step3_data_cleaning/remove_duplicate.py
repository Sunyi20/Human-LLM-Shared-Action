def find_and_remove_duplicate_lines(file_path, output_path=None):
    if output_path is None:
        output_path = file_path

    with open(file_path, 'r') as file:
        lines = file.readlines()

    original_count = len(lines)

    seen_triplets = set()
    unique_lines = []
    duplicate_count = 0

    for line in lines:
        line_stripped = line.strip()

        if line_stripped:
            try:
                triplet = tuple(sorted(map(int, line_stripped.split())))

                if triplet not in seen_triplets:
                    seen_triplets.add(triplet)
                    unique_lines.append(line)
                else:
                    duplicate_count += 1
            except ValueError:
                unique_lines.append(line)
        else:
            unique_lines.append(line)

    with open(output_path, 'w') as file:
        file.writelines(unique_lines)
    return duplicate_count

file_path = 'list/trainset_triplets.txt'
find_and_remove_duplicate_lines(file_path)