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
device = "cuda:1" if torch.cuda.is_available() else "cpu"

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


def generate_random_masked_image(cam, original_size, pil_image, threshold=0.5):
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_resized = cv2.resize(cam_normalized, original_size, interpolation=cv2.INTER_LANCZOS4)

    original_mask = cam_resized >= threshold
    num_pixels_to_mask = np.sum(original_mask)

    width, height = original_size
    total_pixels = width * height

    random_mask_flat = np.zeros(total_pixels, dtype=bool)
    
    if num_pixels_to_mask > 0:
        indices = np.arange(total_pixels)
        np.random.shuffle(indices)
        selected_indices = indices[:num_pixels_to_mask]
        random_mask_flat[selected_indices] = True

    random_mask = random_mask_flat.reshape((height, width))

    original_image_np = np.array(pil_image)

    original_image_np[random_mask] = 0

    result_image = Image.fromarray(original_image_np)
    
    return result_image


def analyze_global_distribution(frames, model, preprocess, W, b, target_dim, threshold):

    all_y_indices = []
    all_x_indices = []

    for frame_pil in tqdm(frames, desc="Phase 1: Analyzing Global Activation"):
        frame_cv2 = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        image_tensor, original_size, _ = preprocess_image(frame_cv2, preprocess)

        cam = compute_grad_cam(model, image_tensor, W, b, target_dim)

        cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_resized = cv2.resize(cam_normalized, original_size, interpolation=cv2.INTER_LANCZOS4)

        mask_indices = np.where(cam_resized >= threshold)
        if len(mask_indices[0]) > 0:
            all_y_indices.append(mask_indices[0])
            all_x_indices.append(mask_indices[1])
            

    if not all_y_indices:
        return None
        
    global_y = np.concatenate(all_y_indices)
    global_x = np.concatenate(all_x_indices)
    
    orig_mu_y = np.mean(global_y)
    orig_mu_x = np.mean(global_x)

    std_y = np.std(global_y)
    std_x = np.std(global_x)
    radius = int(max(std_x, std_y)*0.75)
    
    return (orig_mu_x, orig_mu_y), radius

def calculate_random_fixed_params(orig_center, radius, image_size):
    width, height = image_size
    orig_mu_x, orig_mu_y = orig_center
    
    max_attempts = 200
    fixed_new_x, fixed_new_y = orig_mu_x, orig_mu_y
    found = False
    
    for _ in range(max_attempts):
        margin = min(radius, width // 4)
        rand_x = np.random.randint(margin, width - margin)
        rand_y = np.random.randint(margin, height - margin)
        
        distance = np.sqrt((rand_x - orig_mu_x)**2 + (rand_y - orig_mu_y)**2)
        if distance > radius:
            fixed_new_x, fixed_new_y = rand_x, rand_y
            found = True
            break
            
    if not found:
        print("Warning: Could not find non-overlapping mask position. Using last random attempt.")
        
    return int(fixed_new_x), int(fixed_new_y), radius

def apply_fixed_mask(pil_image, center_x, center_y, radius):
    original_image_np = np.array(pil_image)
    height, width = original_image_np.shape[:2]
    
    mask_canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask_canvas, (center_x, center_y), radius, 255, -1)
    
    mask_bool = mask_canvas > 0
    original_image_np[mask_bool] = 0
    
    return Image.fromarray(original_image_np)

def main(video_index, target_dim, threshold, output_dir, model_path):
    model, preprocess = load_clip_model()
    W, b = load_linear_model(model_path)

    folder_list_path = "data/folder_list/folder_list.txt"
    folder_mapping = load_folder_mapping(folder_list_path)

    video_folder = os.path.join("data/videos_selected", folder_mapping.get(int(video_index)))
    video_path = None
    for file_name in os.listdir(video_folder):
        if file_name.endswith(('.mp4', '.avi', '.mkv')): 
            video_path = os.path.join(video_folder, file_name)
            break

    frames, fps = extract_all_frames(video_path)
    original_size = frames[0].size # (W, H)

    print(f"--- Processing Index {video_index}, Dim {target_dim} ---")
    global_stats = analyze_global_distribution(frames, model, preprocess, W, b, target_dim, threshold)
    
    orig_center, radius = global_stats
        
    print(f"Global Original Center: {orig_center}, Radius (2*Sigma): {radius}")
    fix_x, fix_y, fix_r = calculate_random_fixed_params(orig_center, radius, original_size)
    print(f"Generated Global Fixed Random Mask -> Center: ({fix_x}, {fix_y}), Radius: {fix_r}")

    masked_images = []
    for frame_pil in frames:
        masked_img = apply_fixed_mask(frame_pil, fix_x, fix_y, fix_r)
        masked_images.append(masked_img)

    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"video_{video_index}_dim_{target_dim}_thresh_{threshold}_shifted_mask.mp4")
    create_video_from_images(masked_images, output_video_path, fps=fps)

if __name__ == "__main__":
    deepseek_target_pairs = [
        (2, 20),
        (15, 17),
        (117, 27),
        (196, 5),
        (229, 10),
        (229, 21),
    ]

    qwen_target_pairs = [
        (2, 17),
        (15, 14),
        (117, 3),
        (196, 14),
        (229, 3),
        (229, 13),
    ]
    activation_threshold = 0.4
    model = "LLM_deepseek"
    output_directory = f"data/grad_cam/gradCAM_random_masked_videos_{model}/"
    model_path = f"data/grad_cam/fastl2lir_{model}_video.pkl"
    target_pairs = deepseek_target_pairs
    os.makedirs(output_directory, exist_ok=True) 
    for target_video_index, target_dimension in target_pairs:
        main(
                video_index=target_video_index,
                target_dim=target_dimension,
                threshold=activation_threshold,
                output_dir=output_directory,
                model_path=model_path
            )
