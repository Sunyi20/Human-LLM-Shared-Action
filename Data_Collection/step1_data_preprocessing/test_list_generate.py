import random
from itertools import combinations
import os

def generate_test_triplets(train_file_path, test_file_path="test_triplets.txt"):
    train_triplets = set()
    with open(train_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                triplet = (int(parts[0]), int(parts[1]), int(parts[2]))
                train_triplets.add(triplet)

    categories = list(range(355))
    all_pairs = list(combinations(categories, 2))

    pair_counts = {pair: 0 for pair in all_pairs}

    test_triplets = []

    target_count = 2
    max_triplets = 50000

    iteration = 0
    while sum(count < target_count for count in pair_counts.values()) > 0 and len(test_triplets) < max_triplets:
        iteration += 1
        eligible_pairs = [pair for pair, count in pair_counts.items() if count < target_count]
        if not eligible_pairs:
            break

        pair = random.choice(eligible_pairs)

        a, b = pair

        remaining_categories = [c for c in categories if c != a and c != b]
        c = random.choice(remaining_categories)

        new_triplet = (a, b, c)

        if new_triplet not in train_triplets and new_triplet not in test_triplets:
            test_triplets.append(new_triplet)
            pair_counts[tuple(sorted(pair))] += 1

    with open(test_file_path, 'w') as f:
        for triplet in test_triplets:
            f.write(f"{triplet[0]} {triplet[1]} {triplet[2]}\n")



def generate_triplets_to_file(filename="triplets.txt"):
    categories = list(range(355))
    all_pairs = list(combinations(categories, 2))
    pair_counts = {pair: 0 for pair in all_pairs}

    triplets = []
    flag = 0

    while sum(count < 4 for count in pair_counts.values()) > 0:
        flag += 1

        pair = random.choice([pair for pair, count in pair_counts.items() if count < 4])

        a, b = pair

        remaining_categories = [c for c in categories if c != a and c != b]
        c = random.choice(remaining_categories)

        triplet = (a, b, c)
        triplets.append(triplet)

        pair_counts[tuple(sorted(pair))] += 1

    with open(filename, 'w') as f:
        for triplet in triplets:
            f.write(f"{triplet[0]} {triplet[1]} {triplet[2]}\n")


if __name__ == "__main__":
    train_file = "/data1/home/sunyi/large-files/Python_Objects/human_action/list/train/trainset_triplets.txt"
    test_file = "/data1/home/sunyi/large-files/Python_Objects/human_action/list/testset_triplets.txt"

    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    generate_test_triplets(train_file, test_file)