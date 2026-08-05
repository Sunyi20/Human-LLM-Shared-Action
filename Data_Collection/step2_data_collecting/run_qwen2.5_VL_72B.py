import sys
from openai import OpenAI
import os
import time
import re
import csv
from tqdm import tqdm
import cv2
import numpy as np
import base64

from config import project_path, require_env

number = int(sys.argv[1])
folder_list_path = project_path("data", "folder_list.txt")
triplet_file_path = project_path(
    "list",
    "train",
    "trainset_triplets_segments",
    f"trainset_triplets_segment_{number}.txt",
)
videos_folder = project_path("data", "videos_selected")
base_result_dir = project_path(
    "run_experiment",
    "results_qwen2.5_VL_72B",
    f"results_qwen2.5_VL_72B_triplets_split_{number}",
)

os.makedirs(base_result_dir, exist_ok=True)
choice_path = os.path.join(base_result_dir, 'choice.txt')
failure_path = os.path.join(base_result_dir, 'failure.txt')
csv_save_name = os.path.join(base_result_dir, f'train_chains&responses_qwen2.5_VL_72B_split_{number}_file.csv')
new_filename = os.path.join(base_result_dir, 'train_sampled_split_file.txt')
model_name = 'qwen2.5-vl:72b'

for filepath in [choice_path, failure_path, csv_save_name, new_filename]:
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass  # Create an empty file

client = OpenAI(
    base_url='https://uni-api.cstcloud.cn/v1/',
    api_key=require_env("CSTCLOUD_API_KEY")
)

# Enable streaming
s_value = True
def get_image_choice(prompt):
    try:
        sentences = prompt.split('.')
        for sentence in sentences:
            if "noticeably different" in sentence or "noticeably difference" in sentence:
                match = re.search(r'Video\s*\[?\s*(\d+)\s*\]?', sentence)
                if match:
                    video_number = match.group(1)
                    return int(video_number)

                else:
                    continue
        else:
            return int(9)
    except:
        return int(9)

def get_video_properties(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None, None

    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video handle
    cap.release()

    return fps, width, height

def failure_detect_and_write(lst):
    with open(failure_path, 'a') as f:
        f.write(str(lst)+ '\n')
    return


def load_folder_mapping(folder_list_path):
    folder_mapping = {}
    with open(folder_list_path, 'r') as file:
        for line in file:
            # Split each line into its numeric ID and folder name
            number, folder_name = line.strip().split()
            folder_mapping[int(number)] = folder_name
    return folder_mapping


def move_choice_to_final(lst, i):
    idx = i-1
    return lst[:idx] + lst[idx+1:] + [lst[idx]]

# Load triplets in processing order
def load_triplets(triplet_file_path):
    triplets = []
    with open(triplet_file_path, 'r') as file:
        for line in file:
            triplet = tuple(map(int, line.strip().split()))
            triplets.append(triplet)
    return triplets

def count_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return len(lines)

def write_trial_data_to_csv(filename, triplets, responses):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zip(triplets, responses))

def custom_collate_fn(batch):
    # Collect conversations and source lines separately
    conversations = [item[0] for item in batch]
    split_lines = [item[1] for item in batch]
    return conversations, split_lines

def encode_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_b64 = []

    if total_frames > 0:
        # Uniformly sample num_frames frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Encode the frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                b64_str = base64.b64encode(buffer).decode('utf-8')
                # Use image/jpeg as the MIME type
                frames_b64.append(f"data:image/jpeg;base64,{b64_str}")

    cap.release()
    return frames_b64

def save_files(sorted_images, prompt, csv_name, txt_name, response, select_choice):
    group_prompts = [None] * 1
    group_responses = [None] * 1
    group_results = [None] * 1
    group_prompts[0] = prompt
    group_responses[0] = response
    group_results[0] = sorted_images
    write_trial_data_to_csv(csv_name, group_results, group_responses)
    with open(txt_name, 'a') as f:
        f.write(' '.join(sorted_images) + '\n')
    with open(choice_path, 'a') as f:
        f.write(str(select_choice)+ '\n')
    return


line_count = count_lines(new_filename)


with open(triplet_file_path, 'r') as file:
    human_records = file.read().splitlines() #A list containing approximately 450,000 entries
total_lines = count_lines(choice_path) + count_lines(failure_path)
triplets = human_records[total_lines:]


for idx, triplet in tqdm(enumerate(triplets), total=len(triplets)):
    start_time = time.time()  # Record start time
    triplet_video_paths = [0, 0, 0]
    split_line = triplet.split(' ')
    folder_mapping = load_folder_mapping(folder_list_path)
    video_path1 = os.path.join(videos_folder, folder_mapping.get(int(split_line[0])))
    video_path2 = os.path.join(videos_folder, folder_mapping.get(int(split_line[1])))
    video_path3 = os.path.join(videos_folder, folder_mapping.get(int(split_line[2])))

    for file_name in os.listdir(video_path1):
        triplet_video_paths[0] = os.path.join(video_path1, file_name)
    for file_name in os.listdir(video_path2):
        triplet_video_paths[1] = os.path.join(video_path2, file_name)
    for file_name in os.listdir(video_path3):
        triplet_video_paths[2] = os.path.join(video_path3, file_name)


    fps1, w1, h1 = get_video_properties(triplet_video_paths[0])
    fps2, w2, h2 = get_video_properties(triplet_video_paths[1])
    fps3, w3, h3 = get_video_properties(triplet_video_paths[2])
    frames1 = encode_video_frames(triplet_video_paths[0], num_frames=8)
    frames2 = encode_video_frames(triplet_video_paths[1], num_frames=8)
    frames3 = encode_video_frames(triplet_video_paths[2], num_frames=8)
    content_list = [
        {"type": "text", "text": 'Now we need to perform a role-playing task. Here are three videos which depict actions performed by human.'}
    ]

    # Add frames from the first video
    content_list.append({"type": "text", "text": "Video 1:"})
    for f_b64 in frames1:
        content_list.append({"type": "image_url", "image_url": {"url": f_b64}})

    # Add frames from the second video
    content_list.append({"type": "text", "text": "Video 2:"})
    for f_b64 in frames2:
        content_list.append({"type": "image_url", "image_url": {"url": f_b64}})

    # Add frames from the third video
    content_list.append({"type": "text", "text": "Video 3:"})
    for f_b64 in frames3:
        content_list.append({"type": "image_url", "image_url": {"url": f_b64}})

    # Add the remaining instructions
    content_list.append({
        "type": "text",
        "text": 'Now we need to perform a role-playing task. Assume you are an action judgment expert with the ability to classify actions performed by humans. Firstly, you should describe the content of these clips individually.'
    })
    content_list.append({
        "type": "text",
        "text": 'Then you should carefully evaluate each content independently of its order and distinguish which action that is noticeably different from the other videos. Explain the reason for this difference.'
    })
    content_list.append({
        "type": "text",
        "text": 'To mitigate positional bias, follow these steps: Mentally shuffle the options into different sequences (e.g., reverse order, random permutation). Analyze each action\'s merits/demerits in each shuffled configuration. Identify if any option consistently emerges as superior across different orderings. If inconsistencies arise, conduct content-based comparisons using criteria like: Factual accuracy、Logical coherence、Contextual relevance、Completeness of information. Document your reasoning process transparently before finalizing the selection'
    })
    content_list.append({
        "type": "text",
        "text": 'Finally, give me the answer. (The answer format is "Video[number] is noticeably different from the other two".)'
    })

    messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=s_value,
        temperature=1.0
    )
    reasoning_text = ""  # Store the complete reasoning content
    final_content = ""

    for chunk in chat_completion:
        # Check that choices exists and is not empty
        if not chunk.choices or len(chunk.choices) == 0:
            continue
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            final_content += chunk.choices[0].delta.content

    print("Complete final response: ", final_content)
    print('####################################################################################### ID = ', split_line)
    select_choice = int(get_image_choice(final_content))
    if select_choice == int(9):
        print("Detected Failure!!!!!")
        failure_detect_and_write(split_line)
    else:
        sorted_images = move_choice_to_final(split_line, select_choice)
        save_files(sorted_images = sorted_images, prompt = messages, response = final_content, csv_name = csv_save_name, txt_name = new_filename, select_choice=select_choice)
        print(sorted_images)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time = {execution_time} seconds")
