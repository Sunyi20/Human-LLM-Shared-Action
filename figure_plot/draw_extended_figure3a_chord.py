import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from d3blocks import D3Blocks
import asyncio
from pyppeteer import launch

plt.rc('font',family='Times New Roman')
seed = 0
np.random.seed(seed)
threshold = 0.19

embeddings = np.loadtxt('data/MLLM_qwen_7B/qwen_7B_VL_spose_embedding_sorted_final.txt')
# embeddings = np.loadtxt('data/MLLM_qwen_72B/qwen_72B_VL_spose_embedding_sorted_final.txt')
# embeddings = np.loadtxt('data/LLM_qwen_7B/qwen_7B_spose_embedding_sorted_final.txt')
# embeddings = np.loadtxt('data/LLM_deepseek/deepseek_spose_embedding_sorted_final.txt')
# embeddings = np.loadtxt('data/Human/human_odd_one_out_spose_embedding_sorted_final.txt')

corr_matrix = np.corrcoef(embeddings.T)
N = len(corr_matrix)
G = nx.from_numpy_array(np.matrix(corr_matrix))

F = G.copy()
F.remove_edges_from([(n1, n2) for n1, n2, w in F.edges(data="weight") if w < threshold or n1 == n2])  
iso_list = list(nx.isolates(F))
F.remove_nodes_from(iso_list) 

F_adj = nx.adjacency_matrix(F)
A_adj = (F_adj > 0).astype(float).todense()  
H = nx.from_numpy_array(A_adj)

idx_update = [i+1 for i in range(N) if i not in iso_list] 

N = len(H.nodes())


source, target, weight = [], [], []
emo_list = [i+1 for i in range(len(corr_matrix))]

for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        if i > j and corr_matrix[i,j] >= threshold:
            source.append(emo_list[i])
            target.append(emo_list[j])
            weight.append(corr_matrix[i,j])

data_dict = {'source': source,
             'target': target,
             'weight': weight}
df = pd.DataFrame(data_dict)


d3 = D3Blocks()
d3.chord(df, ordering=idx_update, fontsize=25, arrowhead=0)
d3.set_node_properties(df, opacity=0.5, fontsize=50)
d3.set_edge_properties(df, color='source', opacity='source', directed=False)
d3.show(filepath='MLLM_qwen_7B_chord_diagram.html')