import fastl2lir
import numpy as np
import pickle


def compute_mean_pcc(array1, array2):
    assert array1.shape == array2.shape
    pcc_values = [
        np.corrcoef(array1[i], array2[i])[0, 1] for i in range(array1.shape[0])
    ] 
    return np.mean(pcc_values)

X = np.load("data/grad_cam/text_clip_features.npy")
# X = np.load("data/grad_cam/video_clip_features_8_frames.npy")

Y = np.loadtxt("data/LLM_qwen_7B/qwen_7B_spose_embedding_sorted_final.txt")

model = fastl2lir.FastL2LiR()

model.fit(X, Y, alpha = 0.1)
Y_predicted = model.predict(X)

print(compute_mean_pcc(array1 = Y, array2 = Y_predicted))

with open("data/grad_cam/fastl2lir_LLM_qwen_7B_text.pkl", "wb") as f:
    pickle.dump(model, f)
