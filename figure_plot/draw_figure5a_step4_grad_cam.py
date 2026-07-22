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
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

def extract_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    for i in range(frame_index + 1):
        success, frame = cap.read()
    cap.release()
    return frame

def extract_frames_3(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    
    if total_frames < 3:
        cap.release()
    
    frame_indices = frame_index
    # frame_indices = [
    #     0,
    #     total_frames // 2,
    #     total_frames - 1
    # ]
    
    frames = []
    for target_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        success, frame = cap.read()
        if not success:
            cap.release()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).convert("RGB")
        frames.append(img)
    
    cap.release()
    return frames 

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

def generate_and_save_heatmap(cam, original_size, pil_image, show=None, output_path=None):
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, original_size, interpolation=cv2.INTER_LANCZOS4)
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    original_image = np.array(pil_image)
    superimposed_img = cv2.addWeighted(
        cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
        0.5,
        heatmap,
        0.5,
        0
    )
    result_image = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    return result_image, heatmap_pil

def concatenate_images(images):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    concatenated_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return concatenated_image

def load_folder_mapping(folder_list_path):
    folder_mapping = {}
    with open(folder_list_path, 'r') as file:
        for line in file:
            number, folder_name = line.strip().split()
            folder_mapping[int(number)] = folder_name
    return folder_mapping


def add_filmstrip_border(image, border_height=50, hole_width=30, hole_height=20, hole_spacing=50,
                         border_color=(0, 0, 0), hole_color=(255, 255, 255)):
    """
    Add a filmstrip-style border with white rectangular holes to the top and bottom of an image.
    """
    height, width, _ = image.shape
    border = np.full((border_height, width, 3), border_color, dtype=np.uint8)

    # Add white rectangular holes
    for x in range(0, width, hole_spacing):
        x_start = max(x - hole_width // 2, 0)
        x_end = min(x + hole_width // 2, width)
        border[border_height // 2 - hole_height // 2: border_height // 2 + hole_height // 2, x_start:x_end] = hole_color

    image_with_border = np.vstack([border, image, border])
    return image_with_border


def main(video_index, frame_index, target_dim, show, output_path):
    model, preprocess = load_clip_model()
    model_path = "data/grad_cam/fastl2lir_LLM_qwen_7B_video.pkl"
    W, b = load_linear_model(model_path)
    folder_list_path = "data/folder_list/folder_list.txt"
    folder_mapping = load_folder_mapping(folder_list_path)
    video_path = os.path.join("data/videos_selected", folder_mapping.get(int(video_index)))
    for file_name in os.listdir(video_path):
        video = os.path.join(video_path, file_name)
    frames = extract_frames_3(video, frame_index)

    result_images = []
    heatmap_pils = []
    
    for i, frame_pil in enumerate(frames):
        frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        image_tensor, original_size, pil_image = preprocess_image(frame_cv2, preprocess)
        cam = compute_grad_cam(model, image_tensor, W, b, target_dim)

        if output_path:
            frame_output_path = output_path.replace('.png', f'_frame_{i}.png')
        else:
            frame_output_path = None
            
        result_image, heatmap_pil = generate_and_save_heatmap(cam, original_size, pil_image, show, frame_output_path)

        result_image = result_image.resize((224, 224))
        result_images.append(result_image)
        
        heatmap_pil = heatmap_pil.resize((224, 224))
        heatmap_pils.append(heatmap_pil)
        

    return result_images, heatmap_pils


if __name__ == "__main__":
    Y = np.loadtxt(f'data/LLM_qwen_7B/qwen_7B_spose_embedding_sorted_final.txt')

    top3_indices = np.argsort(Y, axis=1)[:, -3:]
    top3_indices += 1
    heatmap_ids = [
        15
    ]
    
    for target_video_idx in tqdm(heatmap_ids, desc="Processing videos"):
        target_video_dims = top3_indices[target_video_idx]
        target_video_dims = [13]
        video_index = target_video_idx
        # frame_index = [0, 30, 45]
        frame_index = list(range(72))
        

        print(f"Processing video {video_index} with dimensions: {target_video_dims}")
        video_dir = f"data/grad_cam/LLM_qwen_7B/video_{video_index}"
        os.makedirs(video_dir, exist_ok=True)

        
        for dim in target_video_dims:
            result_images, heatmap_pils = main(video_index, frame_index, dim, False, None)    
            dim_dir = f"data/grad_cam/LLM_qwen_7B/video_{video_index}/dim_{dim-1}"
            os.makedirs(dim_dir, exist_ok=True)
            for frame_idx, result_image in zip(frame_index, result_images):
                plt.figure(figsize=(5, 5), facecolor='white')
                plt.imshow(result_image)
                plt.axis('off') 
                output_path = os.path.join(dim_dir, f"frame_{frame_idx}.png")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white')
                plt.close()
            
            
            for frame_idx, heatmap_pil in zip(frame_index, heatmap_pils):
                plt.figure(figsize=(5, 5), facecolor='white')
                plt.imshow(heatmap_pil)
                plt.axis('off') 
                output_path = os.path.join(dim_dir, f"frame_{frame_idx}_heatmap.png")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white')
                plt.close()
        print(f"Completed processing video {video_index}")