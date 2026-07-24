import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def remove_highly_correlated_features(data, threshold):
    df = pd.DataFrame(data)
    n_features = df.shape[1]
    selected_features = list(range(n_features))
    removed_features = []

    while True:
        redundant = False
        for i in range(n_features):
            if i in selected_features:
                correlation_with_other_features = df[selected_features].corrwith(df[i])
                correlated_features = correlation_with_other_features[correlation_with_other_features > threshold].index.tolist()

                if len(correlated_features) > 0:
                    removed_features.extend(correlated_features[1:])
                    for feature in correlated_features[1:]:
                        selected_features.remove(feature)
                        redundant = True

        if not redundant:
            break

    return df[selected_features].values, removed_features

def sort_dim(data):
    column_sum_indices = np.argsort(np.sum(data, axis=0))
    column_sum_indices = column_sum_indices[::-1]
    data = data[:, column_sum_indices]
    return data


def calculate_rdm(data):
    similarity_matrix = cosine_similarity(data.T)
    rdm = 1 - similarity_matrix
    return rdm

def plot_rdm(rdm, output_path, title="Representational Dissimilarity Matrix (RDM)"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(rdm, annot=True, fmt=".2f", cmap='viridis', square=True,
                cbar_kws={'label': 'Dissimilarity (1 - Cosine Similarity)'},
                annot_kws={"size": 5})
    plt.title(title, fontsize=16)
    plt.xlabel("Dimension Index", fontsize=12)
    plt.ylabel("Dimension Index", fontsize=12)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

threshold = 0.2
basePath = '/data1/home/sunyi/large-files/Python_Objects/human_action/SPoSE_calculate/Qwen2.5_72B_VL/results/qwen/50d/'
output_dir = f'/data1/home/sunyi/large-files/Python_Objects/human_action/SPoSE_calculate/Qwen2.5_72B_VL/reference_models_qwen_spose_threshold_{threshold}'
os.makedirs(output_dir, exist_ok=True)
file_counter = 1
lmbda = [0.0073]

check_path = os.path.join(basePath, str(lmbda[0]))
seedIDs = []

if os.path.exists(check_path):
    for item in os.listdir(check_path):
        item_path = os.path.join(check_path, item)
        if os.path.isdir(item_path) and item.startswith('seed'):
            if os.path.exists(os.path.join(item_path, 'weights_sorted.npy')):
                seedIDs.append(item)
    seedIDs.sort()

mergedata = []
for l in lmbda:
    for index, ID in enumerate(seedIDs):
        folder = basePath + str(l) + '/' + ID + '/'
        data = np.load(folder + 'weights_sorted.npy')
        pruned_loc = data
        mergedata.append(data)
        folder_name = f"s{file_counter:02d}"
        save_folder = os.path.join(output_dir, folder_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.savetxt(save_folder + '/spose_embedding_sorted.txt', pruned_loc, fmt='%.8f')
        file_counter += 1

mergedata = np.hstack(mergedata)
mergedata_without_redundancy, removed_features = remove_highly_correlated_features(mergedata, threshold=threshold)
mergedata_without_redundancy = sort_dim(mergedata_without_redundancy)

rdm_matrix = calculate_rdm(mergedata_without_redundancy)


print(mergedata_without_redundancy.shape)
output_file_path = os.path.join(output_dir, 'spose_embedding_sorted_merge.txt')
np.savetxt(output_file_path, mergedata_without_redundancy, fmt='%.8f')
rdm_output_path = os.path.join(output_dir, 'rdm_visualization.png')
plot_rdm(rdm_matrix, rdm_output_path, title="RDM of Final SPoSE Dimensions")

rdm_npy_path = os.path.join(output_dir, 'rdm_matrix.npy')
np.save(rdm_npy_path, rdm_matrix)
