import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import seaborn as sns


feature_path = "data/consistency_plot/InternVideo2.5_8B.npy"
output_dir = "data/consistency_plot/models_RDM/"
os.makedirs(output_dir, exist_ok=True)

features = np.load(feature_path) 
features_zscore = stats.zscore(features, axis=1)
distances = pdist(features, metric='euclidean')
rdm = squareform(distances)
dot_product = np.dot(features, features.T)
norms = np.linalg.norm(features, axis=1)
rsm = dot_product / np.outer(norms, norms)
np.save(os.path.join(output_dir, "InternVideo2.5_8B_rdm_euclidean.npy"), rdm)
# np.save(os.path.join(output_dir, "rsm_cosine.npy"), rsm)