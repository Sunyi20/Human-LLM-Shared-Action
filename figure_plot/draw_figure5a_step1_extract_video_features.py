import os
import numpy as np
import torch
import clip
import cv2
from PIL import Image
from tqdm import tqdm

device = "cuda:2"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval() 

video_dir = "data/videos_selected"
video_count = 355 
frame_count = 8 
feature_dim = 512 
all_video_features = np.zeros((video_count, feature_dim), dtype=np.float32)


def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_total < frame_count:
        cap.release()
        return np.zeros(feature_dim, dtype=np.float32)

    indices = np.linspace(0, frame_total - 1, frame_count, dtype=int)
    frames = []

    current_idx = 0
    for idx in indices:
        while current_idx < idx:
            ret, _ = cap.read()
            if not ret:
                break
            current_idx += 1

        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(preprocess(frame).unsqueeze(0).to(device))

    cap.release()

    if not frames:
        return np.zeros(feature_dim, dtype=np.float32)

    frames = torch.cat(frames)
    with torch.no_grad():
        features = model.encode_image(frames)
        features = features / features.norm(dim=-1, keepdim=True)
        mean_feature = features.mean(dim=0).cpu().numpy()
    return mean_feature


def load_folder_mapping(folder_list_path):
    folder_mapping = {}
    with open(folder_list_path, 'r') as file:
        for line in file:
            number, folder_name = line.strip().split()
            folder_mapping[int(number)] = folder_name
    return folder_mapping

folder_list_path = "data/folder_list/folder_list.txt"
folder_mapping = load_folder_mapping(folder_list_path)
for i in tqdm(range(video_count)):
    video_path = os.path.join(video_dir, folder_mapping.get(int(i)))
    for file_name in os.listdir(video_path):
        video = os.path.join(video_path, file_name)
    all_video_features[i - 1] = extract_video_features(video)

output_path = "data/grad_cam/video_clip_features_8_frames.npy"
np.save(output_path, all_video_features)
