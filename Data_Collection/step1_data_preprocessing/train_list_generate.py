import random
from itertools import combinations

random.seed(3)

def generate_triplets_to_file(filename="validationset.txt"):
    categories = list(range(48))
    all_pairs = list(combinations(categories, 2))
    pair_counts = {pair: 0 for pair in all_pairs}

    triplets_set = set()

    while sum(count < 1 for count in pair_counts.values()) > 0:
        pair = random.choice([pair for pair, count in pair_counts.items() if count < 1])

        a, b = pair

        remaining_categories = [c for c in categories if c != a and c != b]
        c = random.choice(remaining_categories)

        triplet_permutations = {
            tuple(sorted((a, b, c))),
        }

        if not any(perm in triplets_set for perm in triplet_permutations):
            triplets_set.add(tuple(sorted((a, b, c))))
            pair_counts[tuple(sorted(pair))] += 1

    with open(filename, 'w') as f:
        for triplet in triplets_set:
            f.write(f"{triplet[0]} {triplet[1]} {triplet[2]}\n")


if __name__ == "__main__":
    generate_triplets_to_file("triplets_noise_ceiling.txt")
