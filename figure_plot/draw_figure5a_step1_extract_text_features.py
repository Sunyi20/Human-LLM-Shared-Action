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


video_count = 355
feature_dim = 512
all_text_features = np.zeros((video_count, feature_dim), dtype=np.float32)

def load_folder_mapping(folder_list_path):
    folder_mapping = {}
    with open(folder_list_path, 'r') as file:
        for line in file:
            number, folder_name = line.strip().split()
            folder_mapping[int(number)] = folder_name
    return folder_mapping


def extract_text_feature(text):
    if not text:
        return np.zeros(feature_dim, dtype=np.float32)

    text_input = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features[0].cpu().numpy() 


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

folder_list_path = "data/folder_list/folder_list.txt"
transciptions_folder = 'data/transcriptions_selected'
folder_mapping = load_folder_mapping(folder_list_path)

for i in tqdm(range(video_count)):
    
    path = os.path.join(transciptions_folder, folder_mapping.get(int(i)))
    for file_name in os.listdir(path):
        text_path = os.path.join(path, file_name)
        sentence = read_txt_file(text_path)

    all_text_features[i - 1] = extract_text_feature(sentence)

output_path = "data/grad_cam/text_clip_features.npy"
np.save(output_path, all_text_features)
