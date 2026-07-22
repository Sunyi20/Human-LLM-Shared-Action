import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def calculate_accuracy_leave_one_out(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    label_map = {label: i for i, label in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])

    predictions = np.zeros(n_samples, dtype=int)

    for i in tqdm(range(n_samples), desc="Leave-one-out classification"):
        test_feature = X[i, :].reshape(1, -1)
        train_features = np.delete(X, i, axis=0)
        train_labels = np.delete(mapped_labels, i)

        centroids = np.zeros((n_classes, X.shape[1]))
        for k in range(n_classes):
            class_mask = (train_labels == k)
            if np.any(class_mask):
                centroids[k, :] = np.mean(train_features[class_mask, :], axis=0)
            else:
                centroids[k, :] = np.inf

        distances = cdist(test_feature, centroids, 'euclidean').ravel()
        predictions[i] = np.argmin(distances)

    correct_predictions = np.sum(predictions == mapped_labels)
    accuracy = 100 * correct_predictions / n_samples
    
    return accuracy, predictions, mapped_labels

def calculate_chance_level(y_true, y_pred, num_permutations=1000):
    permuted_accuracies = np.zeros(num_permutations)
    n_samples = len(y_true)

    for i in tqdm(range(num_permutations), desc="Calculating chance level"):
        permuted_y_true = np.random.permutation(y_true)
        correct = np.sum(permuted_y_true == y_pred)
        permuted_accuracies[i] = 100 * correct / n_samples

    chance_level = np.mean(permuted_accuracies)
    lower_bound = np.percentile(permuted_accuracies, 2.5)
    upper_bound = np.percentile(permuted_accuracies, 97.5)

    return chance_level, lower_bound, upper_bound


X = np.loadtxt('data/LLM_deepseek/deepseek_spose_embedding_sorted_final.txt')
Y_labels = np.load('data/tsne/video_category_index_mit_355.npy')
accuracy, y_pred, y_true = calculate_accuracy_leave_one_out(X, Y_labels)

chance_level, lower_bound, upper_bound = calculate_chance_level(y_true, y_pred)

print("\n--- Results ---")
print(f"Original Accuracy: {accuracy:.2f}%")
print(f"Chance Level: {chance_level:.2f}%")
print(f"95% CI for Chance Level: [{lower_bound:.2f}%, {upper_bound:.2f}%]")