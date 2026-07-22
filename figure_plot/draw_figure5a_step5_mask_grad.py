import cv2
import torch
import clip
import pickle
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
device = "cuda:5" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def load_linear_model(model_path):
    with open(model_path, "rb") as f:
        lin_model = pickle.load(f)
    W = torch.tensor(lin_model.W, dtype=torch.float32, device=device)  # (512, 30)
    b = torch.tensor(lin_model.b, dtype=torch.float32, device=device)  # (1, 30)
    W.requires_grad_(False)
    b.requires_grad_(False)
    return W, b

def preprocess_image(frame, preprocess):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    original_size = pil_image.size  # (width, height)
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    image_tensor.requires_grad_()
    return image_tensor, original_size, pil_image

def compute_grad_cam(model, image_tensor, W, b, target_dim):
    activations = {}
    gradients = {}

    def save_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook

    target_layer = model.visual.conv1 
    forward_handle = target_layer.register_forward_hook(save_activation('patch_embedding'))
    backward_handle = target_layer.register_full_backward_hook(save_gradient('patch_embedding'))

    image_features = model.encode_image(image_tensor).to(torch.float32)  # (1, 512)

    output_30d = torch.mm(image_features, W) + b  # (1, 30)

    target_idx = target_dim - 1
    model.zero_grad()
    output_30d[0, target_idx].backward(retain_graph=True)

    activation = activations['patch_embedding']  # (1, 768, 7, 7)
    gradient = gradients['patch_embedding']  # (1, 768, 7, 7)

    activation = F.interpolate(activation, scale_factor=4, mode='bilinear', align_corners=False)  # (1, 768, 28, 28)
    gradient = F.interpolate(gradient, scale_factor=4, mode='bilinear', align_corners=False)  # (1, 768, 28, 28)

    gradient_mean = gradient.mean(dim=(2, 3), keepdim=True)  # (1, 768, 1, 1)
    cam = (activation * gradient_mean).sum(dim=1)  # (1, 28, 28)
    cam = F.relu(cam).squeeze().cpu().numpy()  # (28, 28)

    cam = cam.astype(np.float32)
    cam = cv2.GaussianBlur(cam, (3, 3), 0)

    forward_handle.remove()
    backward_handle.remove()

    return cam

def generate_masked_image(cam, original_size, pil_image, threshold=0.5):
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_resized = cv2.resize(cam_normalized, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    mask = cam_resized >= threshold
    original_image_np = np.array(pil_image)
    original_image_np[mask] = 0

    result_image = Image.fromarray(original_image_np)
    
    return result_image

def create_video_from_images(image_list, output_video_path, fps=10):
    if not image_list:
        return

    width, height = image_list[0].size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for img in image_list:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        
    video_writer.release()



def extract_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot Open: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).convert("RGB")
        frames.append(img)
    
    cap.release()
    return frames, fps

def load_folder_mapping(folder_list_path):
    folder_mapping = {}
    with open(folder_list_path, 'r') as file:
        for line in file:
            number, folder_name = line.strip().split()
            folder_mapping[int(number)] = folder_name
    return folder_mapping

def main(video_index, target_dim, threshold, output_dir, model_path):
    model, preprocess = load_clip_model()
    
    W, b = load_linear_model(model_path)

    folder_list_path = "data/folder_list/folder_list.txt"
    folder_mapping = load_folder_mapping(folder_list_path)
    video_folder = os.path.join("data/videos_selected", folder_mapping.get(int(video_index)))
    video_path = None
    for file_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, file_name)
    
    if not video_path:
        return

    frames, fps = extract_all_frames(video_path)
    masked_images = []

    for frame_pil in tqdm(frames):
        frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        image_tensor, original_size, pil_image = preprocess_image(frame_cv2, preprocess)

        cam = compute_grad_cam(model, image_tensor, W, b, target_dim)
        masked_image = generate_masked_image(cam, original_size, pil_image, threshold=threshold)
        masked_images.append(masked_image)

    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"video_{video_index}_dim_{target_dim}_thresh_{threshold}.mp4")

    create_video_from_images(masked_images, output_video_path, fps=fps)



if __name__ == "__main__":
    target_video_index = 117
    target_dimension = 27
    activation_threshold = 0.2
    model = "LLM_deepseek"
    output_directory = f"data/grad_cam/gradCAM_masked_videos_{model}/"
    model_path = f"data/grad_cam/fastl2lir_{model}_video.pkl"

    main(
        video_index=target_video_index,
        target_dim=target_dimension,
        threshold=activation_threshold,
        output_dir=output_directory,
        model_path=model_path
    )