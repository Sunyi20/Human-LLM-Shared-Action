import random
import itertools
from tqdm import tqdm

def generate_triplets_from_pool(filename="triplets_from_sampling.txt", num_triplets=100000, seed=42):
    random.seed(seed)
    all_numbers = list(range(48))
    with open(filename, 'w') as f:
        for _ in tqdm(range(num_triplets)):
            triplet = random.sample(all_numbers, 3)
            f.write(f"{triplet[0]} {triplet[1]} {triplet[2]}\n")

if __name__ == "__main__":
    generate_triplets_from_pool("triplets_full_sample_v2.txt", num_triplets=100000)