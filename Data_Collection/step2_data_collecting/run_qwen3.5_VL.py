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
    "list", "full_sample_human_10", f"human_full_sample_triplets_segment_{number}.txt"
)
videos_folder = project_path("data", "videos_selected")
base_result_dir = project_path(
    "run_experiment",
    "results_qwen3.5_VL",
    f"results_qwen3.5_VL_triplets_split_{number}",
)

os.makedirs(base_result_dir, exist_ok=True)
choice_path = os.path.join(base_result_dir, 'choice.txt')
failure_path = os.path.join(base_result_dir, 'failure.txt')
csv_save_name = os.path.join(base_result_dir, f'train_chains&responses_qwen3.5_VL_split_{number}_file.csv')
new_filename = os.path.join(base_result_dir, 'train_sampled_split_file.txt')
model_name = 'qwen3.5'

for filepath in [choice_path, failure_path, csv_save_name, new_filename]:
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass

client = OpenAI(
    base_url='https://uni-api.cstcloud.cn/v1/',
    api_key=require_env("CSTCLOUD_API_KEY")
)

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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height

def failure_detect_and_write(lst):
    with open(failure_path, 'a') as f:
        f.write(str(lst) + '\n')

def load_folder_mapping(folder_list_path):
    folder_mapping = {}
    with open(folder_list_path, 'r') as file:
        for line in file:
            number, folder_name = line.strip().split()
            folder_mapping[int(number)] = folder_name
    return folder_mapping

def move_choice_to_final(lst, i):
    idx = i - 1
    return lst[:idx] + lst[idx+1:] + [lst[idx]]

def count_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return len(lines)

def write_trial_data_to_csv(filename, triplets, chains, responses):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zip(triplets, chains, responses))

def encode_video_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_b64 = []
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                b64_str = base64.b64encode(buffer).decode('utf-8')
                frames_b64.append(f"data:image/jpeg;base64,{b64_str}")
    cap.release()
    return frames_b64

def save_files(sorted_images, prompt, chains, response, csv_name, txt_name, select_choice):
    write_trial_data_to_csv(csv_name, [sorted_images], [chains], [response])
    with open(txt_name, 'a') as f:
        f.write(' '.join(sorted_images) + '\n')
    with open(choice_path, 'a') as f:
        f.write(str(select_choice) + '\n')


# ========== Main workflow ==========
folder_mapping = load_folder_mapping(folder_list_path)  # Load once

with open(triplet_file_path, 'r') as file:
    human_records = file.read().splitlines()
total_lines = count_lines(choice_path) + count_lines(failure_path)
triplets = human_records[total_lines:]

for idx, triplet in tqdm(enumerate(triplets), total=len(triplets)):
    start_time = time.time()
    triplet_video_paths = [None, None, None]
    split_line = triplet.split(' ')

    video_path1 = os.path.join(videos_folder, folder_mapping.get(int(split_line[0])))
    video_path2 = os.path.join(videos_folder, folder_mapping.get(int(split_line[1])))
    video_path3 = os.path.join(videos_folder, folder_mapping.get(int(split_line[2])))

    for file_name in os.listdir(video_path1):
        triplet_video_paths[0] = os.path.join(video_path1, file_name)
    for file_name in os.listdir(video_path2):
        triplet_video_paths[1] = os.path.join(video_path2, file_name)
    for file_name in os.listdir(video_path3):
        triplet_video_paths[2] = os.path.join(video_path3, file_name)

    frames1 = encode_video_frames(triplet_video_paths[0], num_frames=8)
    frames2 = encode_video_frames(triplet_video_paths[1], num_frames=8)
    frames3 = encode_video_frames(triplet_video_paths[2], num_frames=8)

    # ====== Construct the message payload correctly ======
    content_list = [
        {"type": "text", "text": "Here are three videos which depict actions performed by humans."}
    ]

    content_list.append({"type": "text", "text": "Video 1:"})
    for f_b64 in frames1:
        content_list.append({"type": "image_url", "image_url": {"url": f_b64}})

    content_list.append({"type": "text", "text": "Video 2:"})
    for f_b64 in frames2:
        content_list.append({"type": "image_url", "image_url": {"url": f_b64}})

    content_list.append({"type": "text", "text": "Video 3:"})
    for f_b64 in frames3:
        content_list.append({"type": "image_url", "image_url": {"url": f_b64}})

    content_list.append({
        "type": "text",
        "text": (
            'Assume you are an action judgment expert. '
            'Firstly, describe the content of these clips individually. '
            'Then carefully evaluate each content independently of its order and distinguish which action is noticeably different from the other videos. Explain the reason. '
            'To mitigate positional bias, mentally shuffle the options into different sequences, analyze each action in each shuffled configuration, '
            'and identify if any option consistently emerges as different across different orderings. '
            'Finally, give me the answer. (The answer format is "Video[number] is noticeably different from the other two".)'
        )
    })

    # ====== Messages are dictionaries with role and content fields ======
    messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]

    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=messages,  # Pass messages, not content_list
                stream=s_value,
                # max_tokens=2048,
            )

            reasoning_text = ""
            final_content = ""
            for chunk in chat_completion:
                if not chunk.choices or len(chunk.choices) == 0:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                if finish_reason is not None:
                    print(f"[DEBUG] Stream ended, finish_reason={finish_reason}")

                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_text += delta.reasoning_content

                if hasattr(delta, 'content') and delta.content:
                    final_content += delta.content

            # Remove <think>...</think> tags if the model includes them in content
            final_content_clean = re.sub(r'<think>.*?</think>', '', final_content, flags=re.DOTALL).strip()

            print("Complete reasoning content: ", reasoning_text[:500] if reasoning_text else "None")
            print("Complete final response: ", final_content_clean)
            print('##### ID = ', split_line)

            select_choice = int(get_image_choice(final_content_clean))
            if select_choice == int(9):
                print(f"No answer detected; retrying... (attempt {attempt}/{max_retries})")
                if attempt == max_retries:
                    print("Maximum retries reached; recording the item in failure.txt")
                    failure_detect_and_write(split_line)
            else:
                sorted_images = move_choice_to_final(split_line, select_choice)
                save_files(
                    sorted_images=sorted_images,
                    prompt=content_list,
                    chains=reasoning_text,
                    response=final_content_clean,
                    csv_name=csv_save_name,
                    txt_name=new_filename,
                    select_choice=select_choice
                )
                print(sorted_images)
                break
        except Exception as e:
            print(f"[ERROR] API request failed: {e}; retrying... (attempt {attempt}/{max_retries})")
            if attempt == max_retries:
                print("Maximum retries reached; recording the item in failure.txt")
                failure_detect_and_write(split_line)
            time.sleep(2)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time = {execution_time} seconds")
