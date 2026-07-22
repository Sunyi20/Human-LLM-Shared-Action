import os
import numpy as np
import cv2
from PIL import Image


def load_folder_mapping(file_path):
    mapping = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    mapping[int(parts[0])] = parts[1]
    except FileNotFoundError:
        print(f"Error: {file_path}")
    return mapping

def extract_three_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 3:
        cap.release()
        return []

    frame_indices = [0, total_frames // 2, total_frames - 1]
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((224, 224))
            frames.append(img)
    
    cap.release()
    return frames

def add_filmstrip_border(image, border_height=30, hole_width=20, hole_height=15, hole_spacing=40,
                         border_color=(0, 0, 0), hole_color=(255, 255, 255)):
    height, width, _ = image.shape
    border = np.full((border_height, width, 3), border_color, dtype=np.uint8)

    for x in range(hole_spacing // 2, width, hole_spacing):
        x_start = max(x - hole_width // 2, 0)
        x_end = min(x + hole_width // 2, width)
        y_start = max(border_height // 2 - hole_height // 2, 0)
        y_end = min(border_height // 2 + hole_height // 2, border_height)
        border[y_start:y_end, x_start:x_end] = hole_color

    image_with_border = np.vstack([border, image, border.copy()])
    return image_with_border

def save_frames_to_pdf(frames_per_video, output_pdf_path, video_gap=10, frame_gap=3):
    all_frames_with_borders = []
    for video_frames in frames_per_video:
        gap_array = Image.new('RGB', (frame_gap, video_frames[0].height), 'white')
        concatenated_video_frames = video_frames[0]
        for frame in video_frames[1:]:
            concatenated_video_frames = Image.fromarray(np.hstack((np.array(concatenated_video_frames), np.array(gap_array), np.array(frame))))
        cv_image = cv2.cvtColor(np.array(concatenated_video_frames), cv2.COLOR_RGB2BGR)
        filmstrip_group_cv = add_filmstrip_border(cv_image)
        filmstrip_group = Image.fromarray(cv2.cvtColor(filmstrip_group_cv, cv2.COLOR_BGR2RGB))
        all_frames_with_borders.append(filmstrip_group)

    if len(all_frames_with_borders) != 6:
        return

    video_horiz_gap = Image.new('RGB', (video_gap, all_frames_with_borders[0].height), 'white')
    
    row1 = Image.fromarray(np.hstack([
        np.array(all_frames_with_borders[0]),
        np.array(video_horiz_gap),
        np.array(all_frames_with_borders[1]),
        np.array(video_horiz_gap),
        np.array(all_frames_with_borders[2])
    ]))

    row2 = Image.fromarray(np.hstack([
        np.array(all_frames_with_borders[3]),
        np.array(video_horiz_gap),
        np.array(all_frames_with_borders[4]),
        np.array(video_horiz_gap),
        np.array(all_frames_with_borders[5])
    ]))

    video_vert_gap = Image.new('RGB', (row1.width, video_gap), 'white')

    final_image = Image.fromarray(np.vstack([
        np.array(row1),
        np.array(video_vert_gap),
        np.array(row2)
    ]))

    final_image.save(output_pdf_path, "PDF", resolution=100.0, save_all=True)


embedding_file = 'data/MLLM_qwen_72B/qwen_72B_VL_spose_embedding_sorted_final.txt'
video_folder = 'data/videos_selected'
folder_list_file = 'data/folder_list/folder_list.txt'
output_pdf_folder = 'data/dim_visualization/dim_visualization_MLLM_qwen_72B/'

data = np.loadtxt(embedding_file)
pruned_loc = data
dimensions = pruned_loc.shape[1]

os.makedirs(output_pdf_folder, exist_ok=True)
folder_mapping = load_folder_mapping(folder_list_file)
for dim in range(dimensions):
    
    dim_values = pruned_loc[:, dim]
    sorted_indices = np.argsort(dim_values)[::-1]
    top_indices = sorted_indices[:6]
    
    frames_from_top_videos = []
    for index in top_indices:
        category = folder_mapping.get(int(index))
        if not category:
            continue

        video_category_path = os.path.join(video_folder, category)
        if not os.path.isdir(video_category_path):
            continue
        
        video_files = [f for f in os.listdir(video_category_path) if not f.startswith('.')]
        if not video_files:
            continue
        
        video_path = os.path.join(video_category_path, video_files[0])
        
        video_frames = extract_three_frames_from_video(video_path)
        if video_frames:
            frames_from_top_videos.append(video_frames)

    if frames_from_top_videos:
        output_pdf_path = os.path.join(output_pdf_folder, f"dim_{dim + 1}.pdf")
        save_frames_to_pdf(frames_from_top_videos, output_pdf_path)
