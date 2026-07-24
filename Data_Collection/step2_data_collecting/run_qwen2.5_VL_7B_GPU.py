import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader, Dataset
from qwen_vl_utils import process_vision_info
import torch
import time
import re
import csv
from tqdm import tqdm
import numpy as np
import cv2
import re

from config import project_path

number = int(sys.argv[1])
folder_list_path = project_path("data", "folder_list.txt")
triplet_file_path = project_path(
    "list", "test_sgement_list", f"deepseek_r1_triplets_segment_{number}.txt"
)
videos_folder = project_path("data", "videos_selected")
result_dir = project_path(f"results_Qwen_2.5_test_triplets_{number}")
choice_path = result_dir / "choice.txt"
failure_path = result_dir / "failure.txt"
model_qwen2_5_path = os.environ.get(
    "QWEN25_VL_MODEL_PATH",
    str(project_path("models", "Qwen-2.5-7B")),
)
csv_save_name = result_dir / "train_prompts&responses_qwen2_5_vl_split_file.csv"
new_filename = result_dir / "train_sampled_split_file.txt"

if not os.path.exists(result_dir):
    print(f"Creating result directory: {result_dir}")
    os.makedirs(result_dir)
for file_path in [choice_path, failure_path, new_filename]:
    with open(file_path, 'w') as f:
        pass
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

def load_folder_mapping(folder_list_path):
    folder_mapping = {}
    with open(folder_list_path, 'r') as file:
        for line in file:
            # Split each line into its numeric ID and folder name
            number, folder_name = line.strip().split()
            folder_mapping[int(number)] = folder_name
    return folder_mapping

def failure_detect_and_write(lst):
    with open(failure_path, 'a') as f:
        f.write(str(lst)+ '\n')
    return

def move_choice_to_front(lst, i):
    idx = i-1
    return [lst[idx]] + lst[:idx] + lst[idx+1:]

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

class conversation_Dataset(Dataset):
    def __init__(self, triplet_data, line_counts):
        self.triplets = triplet_data[line_counts:]
        self.folder_mapping = load_folder_mapping(folder_list_path)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        triplet_video_paths = [0, 0, 0]
        split_line = triplet.split(' ')
        video_path1 = os.path.join(videos_folder, self.folder_mapping.get(int(split_line[0])))
        video_path2 = os.path.join(videos_folder, self.folder_mapping.get(int(split_line[1])))
        video_path3 = os.path.join(videos_folder, self.folder_mapping.get(int(split_line[2])))

        for file_name in os.listdir(video_path1):
            triplet_video_paths[0] = os.path.join(video_path1, file_name)
        for file_name in os.listdir(video_path2):
            triplet_video_paths[1] = os.path.join(video_path2, file_name)
        for file_name in os.listdir(video_path3):
            triplet_video_paths[2] = os.path.join(video_path3, file_name)


        fps1, w1, h1 = get_video_properties(triplet_video_paths[0])
        fps2, w2, h2 = get_video_properties(triplet_video_paths[1])
        fps3, w3, h3 = get_video_properties(triplet_video_paths[2])
        messages = [
            {
                "role": "user",
                "content": [
                        {"type": "text",
                            "text": 'Now we need to perform a role-playing task. Here are three videos which depict actions performed by human.'
                        },
                        {
                            "type": "video",
                            "video": triplet_video_paths[0],
                            "max_pixels": w1 * h1,
                            "fps": 5,
                        },
                        {
                            "type": "video",
                            "video": triplet_video_paths[1],
                            "max_pixels": w2 * h2,
                            "fps": 5,
                        },
                        {
                            "type": "video",
                            "video": triplet_video_paths[2],
                            "max_pixels": w3 * h3,
                            "fps": 5,
                        },
                        {
                            "type": "text",
                            "text": 'Now we need to perform a role-playing task. Assume you are an action judgment expert with the ability to classify actions performed by humans. Firstly, you should describe the content of these clips individually.'
                        },
                        {
                            "type": "text",
                            "text": 'Then you should carefully evaluate each content independently of its order and distinguish which action that is noticeably different from the other videos. Explain the reason for this difference.'
                        },
                        {
                            "type": "text",
                            "text": 'To mitigate positional bias, follow these steps: Mentally shuffle the options into different sequences (e.g., reverse order, random permutation). Analyze each action\'s merits/demerits in each shuffled configuration. Identify if any option consistently emerges as superior across different orderings. If inconsistencies arise, conduct content-based comparisons using criteria like: Factual accuracy、Logical coherence、Contextual relevance、Completeness of information. Document your reasoning process transparently before finalizing the selection'
                        },
                        {
                            "type": "text",
                            "text": 'Finally, give me the answer. (The answer format is "Video[number] is noticeably different from the other two".)'
                        }
                    ],
                }
            ]
        return messages, split_line


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_qwen2_5_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
).to("cuda")

processor = AutoProcessor.from_pretrained(model_qwen2_5_path)
processor.tokenizer.padding_side = 'left'


if not os.path.exists(new_filename):
    with open(new_filename, 'w') as new_file:
        pass
line_count = count_lines(new_filename)


with open(triplet_file_path, 'r') as file:
    human_records = file.read().splitlines() #A list containing approximately 450,000 entries
total_lines = count_lines(choice_path) + count_lines(failure_path)


train_dataset = conversation_Dataset(triplet_data = human_records, line_counts = total_lines)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)


for messages, split_lines in tqdm(train_loader):
    start_time = time.time()  # Record start time
    triplet_video_paths = []
    with torch.no_grad():
        texts = [
            processor.apply_chat_template(msg, add_generation_prompt=True, add_vision_id=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=768)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    for idx, response in enumerate(output_text):
        split_line = split_lines[idx]
        print('####################################################################################### ID = ', split_line)
        select_choice = int(get_image_choice(response))
        if select_choice == int(9):
            print("Detected Failure!!!!!")
            failure_detect_and_write(split_line)
        else:
            sorted_images = move_choice_to_front(split_line, select_choice)
            save_files(sorted_images = sorted_images, prompt = texts[0], response = response,csv_name = csv_save_name, txt_name = new_filename, select_choice=select_choice)
            print(sorted_images)
            print(response)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time = {execution_time} seconds")
    del texts, image_inputs, video_inputs, inputs,generated_ids, generated_ids_trimmed, output_text
    torch.cuda.empty_cache()
