import numpy as np
import os
from PIL import Image, ImageOps
import shutil
import moviepy.editor as mp
from moviepy.video.compositing.CompositeVideoClip import clips_array
from tqdm import tqdm

def six_videos_to_gif(video_paths, output_gif_root, dim_idx, target_duration):
    output_gif_path = output_gif_root + 'dim_{}.gif'.format(dim_idx)
    # 加载和调整每个视频剪辑
    clips = []
    for video_path in video_paths:
        clip = mp.VideoFileClip(video_path)
        # 调整视频剪辑到目标时长
        if clip.duration > target_duration:
            clip = clip.subclip(0, target_duration)
        elif clip.duration < target_duration:
            clip = clip.loop(duration=target_duration)
        # 调整视频大小（例如，240x240）
        clip = clip.resize(height=240, width=240)
        clips.append(clip)

    # 创建6宫格排列（2行3列）
    grid = clips_array([
        [clips[0], clips[1], clips[2]],
        [clips[3], clips[4], clips[5]]
    ])

    # 设置FPS，确保GIF流畅
    grid = grid.set_fps(10)

    # 导出为GIF
    grid.write_gif(output_gif_path, fps=10)

    print(f"GIF saved at {output_gif_path}")
    return

def nine_videos_to_gif(video_paths, output_gif_root, dim_idx, target_duration):
    """
    Combine 9 videos into a 3x3 grid and save as a GIF.

    Parameters:
        video_paths (list): List of 9 video file paths.
        output_gif_root (str): Directory to save the output GIF.
        dim_idx (int): Index of the dimension for naming the GIF.
        target_duration (float): Target duration for all video clips in seconds.

    Returns:
        None
    """
    output_gif_path = output_gif_root + f'dim_{dim_idx}.gif'

    # Load and adjust each video clip
    clips = []
    for video_path in video_paths:
        clip = mp.VideoFileClip(video_path)

        # Trim or loop the video to match the target duration
        if clip.duration > target_duration:
            clip = clip.subclip(0, target_duration)
        elif clip.duration < target_duration:
            clip = clip.loop(duration=target_duration)

        # Resize the video (e.g., 240x240)
        clip = clip.resize(height=240, width=240)
        clips.append(clip)

    # Ensure there are exactly 9 videos
    if len(clips) != 9:
        raise ValueError("Exactly 9 video paths are required for a 3x3 grid.")

    # Create 3x3 grid (3 rows, 3 columns)
    grid = clips_array([
        [clips[0], clips[1], clips[2]],
        [clips[3], clips[4], clips[5]],
        [clips[6], clips[7], clips[8]]
    ])

    # Set FPS to ensure smooth playback
    grid = grid.set_fps(10)

    # Export the grid as a GIF
    grid.write_gif(output_gif_path, fps=10)

    print(f"GIF saved at {output_gif_path}")
    return

def load_folder_mapping(folder_list_path):
    folder_mapping = {}
    with open(folder_list_path, 'r') as file:
        for line in file:
            # 分割每行的号和文件夹名称
            number, folder_name = line.strip().split()
            folder_mapping[int(number)] = folder_name
    return folder_mapping

# 读取npz文件
for i in range(35,36):
    # folder = f'/data1/home/sunyi/large-files/Python_Objects/human_action/SPoSE_calculate/Deepseek-671B/results/deepseek/50d/0.014/seed{i}/'
    folder = '/data1/home/sunyi/large-files/Python_Objects/human_action/SPoSE_calculate/Deepseek-671B/reference_models_deepseek_spose_0.012/'
    # data = np.load(folder+'weights_sorted.npy')
    data = np.loadtxt(folder+'spose_embedding_sorted_final.txt')
    video_folder = '/data1/home/sunyi/large-files/Python_Objects/human_action/data/videos_selected'
    # 读取pruned_loc数组
    pruned_loc = data

    # 获取数组的维度
    dimensions = pruned_loc.shape[1]
    print(dimensions)

    # 创建保存结果图片的文件夹
    output_folder = folder+'dim_visualization_videos/'
    output_txt_path = folder + 'category.txt'
    os.makedirs(output_folder, exist_ok=True)

    
    with open(output_txt_path, "w") as file: 
        for dim in range(dimensions):
            # 获取当前维度的值
            dim_values = pruned_loc[:, dim]

            # 根据值的大小降序排序，返回排序后的索引
            sorted_indices = np.argsort(dim_values)[::-1]

            # 获取前6个索引
            top_indices = sorted_indices[:9]
            # 创建一个新的图片来容纳拼接后的图片
            concat_width = 0
            concat_height = 0
            video_list_path = []
            output_path = os.path.join(output_folder)
            sequence= 0
            category_list = []
            for index in top_indices:
                sequence += 1
                # 读取对应文件夹下的图
                folder_list = '/data1/home/sunyi/large-files/Python_Objects/human_action/data/folder_list.txt'
                folder_mapping = load_folder_mapping(folder_list)
                category = folder_mapping.get(int(index))
                category_list.append(category)
                video_path = os.path.join(video_folder, category)
                # 构建视频路径
                for file_name in os.listdir(video_path):
                    video_path = os.path.join(video_path, file_name)
                video_list_path.append(video_path)
            file.write(" ".join(category_list) + "\n")
            nine_videos_to_gif(video_paths = video_list_path, output_gif_root = output_path, dim_idx = dim+1, target_duration = 3)
            print('saved_dim_id = ', dim)

# folder = '/data1/home/sunyi/large-files/Python_Objects/human_action/SPoSE_calculate/Deepseek-671B/results/deepseek/50d/0.0089/seed12/'
# data = np.load(folder+'weights_sorted.npy')
# video_folder = '/data1/home/sunyi/large-files/Python_Objects/human_action/data/videos_selected'
# # 读取pruned_loc数组
# pruned_loc = data
# # 获取数组的维度
# dimensions = pruned_loc.shape[1]
# print(dimensions)
# # 创建保存结果图片的文件夹
# output_folder = folder+'dim_visualization_videos/'
# output_txt_path = folder + 'category.txt'
# os.makedirs(output_folder, exist_ok=True)

# with open(output_txt_path, "w") as file: 
#     for dim in range(dimensions):
#         # 获取当前维度的值
#         dim_values = pruned_loc[:, dim]
#         # 根据值的大小降序排序，返回排序后的索引
#         sorted_indices = np.argsort(dim_values)[::-1]
#         # 获取前6个索引
#         top_indices = sorted_indices[:9]
#         # 创建一个新的图片来容纳拼接后的图片
#         concat_width = 0
#         concat_height = 0
#         video_list_path = []
#         output_path = os.path.join(output_folder)
#         sequence= 0
#         category_list = []
#         for index in top_indices:
#             sequence += 1
#             # 读取对应文件夹下的图
#             folder_list = '/data1/home/sunyi/large-files/Python_Objects/human_action/data/folder_list.txt'
#             folder_mapping = load_folder_mapping(folder_list)
#             category = folder_mapping.get(int(index))
#             category_list.append(category)
#             video_path = os.path.join(video_folder, category)
#             # 构建视频路径
#             for file_name in os.listdir(video_path):
#                 video_path = os.path.join(video_path, file_name)
#             video_list_path.append(video_path)
#         file.write(" ".join(category_list) + "\n")
#         nine_videos_to_gif(video_paths = video_list_path, output_gif_root = output_path, dim_idx = dim+1, target_duration = 3)
#         print('saved_dim_id = ', dim)

