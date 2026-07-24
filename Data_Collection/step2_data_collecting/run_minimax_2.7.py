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
transciptions_folder = project_path("data", "transcriptions_selected")
base_result_dir = project_path(
    "run_experiment",
    "results_minimax_2.7",
    f"results_minimax_2.7_triplets_split_{number}",
)

os.makedirs(base_result_dir, exist_ok=True)
choice_path = os.path.join(base_result_dir, 'choice.txt')
failure_path = os.path.join(base_result_dir, 'failure.txt')
csv_save_name = os.path.join(base_result_dir, f'train_chains&responses_minimax_2.7_split_{number}_file.csv')
new_filename = os.path.join(base_result_dir, 'train_sampled_split_file.txt')
model_name = 'minimax-m27'
for filepath in [choice_path, failure_path, csv_save_name, new_filename]:
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass

client = OpenAI(
    base_url='https://uni-api.cstcloud.cn/v1/',
    api_key=require_env("CSTCLOUD_API_KEY")
)

# Enable streaming
s_value = True
def get_choice(prompt):
    try:
        sentences = prompt.split('.')
        for sentence in sentences:
            if "noticeably different" in sentence or "noticeably difference" in sentence:
                match1 = re.search(r'Video description\s*\(?\s*(\d+)\s*\)?', sentence)
                match2 = re.search(r'Video description\s+(\*\*)?+\s?+(ID\s)?+(\*\*)?(\d+)', sentence)

                if match1:
                    video_number = match1.group(1)
                    return int(video_number)
                if match2:
                    video_number = match2.group(4)
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

def write_trial_data_to_csv(filename, prompts, chains, triplets, responses):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(zip(triplets, prompts, chains, responses))

def custom_collate_fn(batch):
    # Collect conversations and source lines separately
    conversations = [item[0] for item in batch]
    split_lines = [item[1] for item in batch]
    return conversations, split_lines

def save_files(sorted_images, prompt, chains, response, csv_name, txt_name, select_choice):
    group_prompts = [None] * 1
    group_chains = [None] * 1
    group_responses = [None] * 1
    group_results = [None] * 1

    group_prompts[0] = prompt
    group_chains[0] = chains
    group_responses[0] = response
    group_results[0] = sorted_images

    write_trial_data_to_csv(csv_name, group_prompts, group_chains, group_results, group_responses)
    with open(txt_name, 'a') as f:
        f.write(' '.join(sorted_images) + '\n')
    with open(choice_path, 'a') as f:
        f.write(str(select_choice)+ '\n')
    return

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def failure_detect_and_write(lst):
    with open(failure_path, 'a') as f:
        f.write(str(lst)+ '\n')
    return

if not os.path.exists(new_filename):
    with open(new_filename, 'w') as new_file:
        pass
line_count = count_lines(new_filename)


with open(triplet_file_path, 'r') as file:
    human_records = file.read().splitlines() #A list containing approximately 450,000 entries
total_lines = count_lines(choice_path) + count_lines(failure_path)
# total_lines = 1
triplets = human_records[total_lines:]

for idx, triplet in tqdm(enumerate(triplets), total=len(triplets)):
    start_time = time.time()  # Record start time
    triplet_paths = []
    split_line = triplet.split(' ')
    triplet_paths = [0, 0, 0]
    folder_mapping = load_folder_mapping(folder_list_path)
    category_1 = folder_mapping.get(int(split_line[0]))
    category_2 = folder_mapping.get(int(split_line[1]))
    category_3 = folder_mapping.get(int(split_line[2]))
    path1 = os.path.join(transciptions_folder, category_1)
    path2 = os.path.join(transciptions_folder, category_2)
    path3 = os.path.join(transciptions_folder, category_3)

    for file_name in os.listdir(path1):
        triplet_paths[0] = os.path.join(path1, file_name)
    for file_name in os.listdir(path2):
        triplet_paths[1] = os.path.join(path2, file_name)
    for file_name in os.listdir(path3):
        triplet_paths[2] = os.path.join(path3, file_name)
    messages=[
            {
                "role": "user",
                "content": f"Here are three video descriptions each belongs to a specific category:\
                    Video description 1 corresponds to {category_1}."
            },
            {
                "role": "user",
                "content": read_txt_file(triplet_paths[0]),
            },
            {
                "role": "user",
                "content": f"Video description 2 corresponds to {category_2}.",
            },
            {
                "role": "user",
                "content": read_txt_file(triplet_paths[1]),
            },
            {
                "role": "user",
                "content": f"Video description 3 corresponds to {category_3}.",
            },
            {
                "role": "user",
                "content": read_txt_file(triplet_paths[2]),
            },
            {
                "role": "user",
                "content": 'Now we need to perform a role-playing task. Assume you are an action judgment expert with the ability to classify actions performed by humans. Firstly, you should analyse the content and figure out the hidden dimensions individually.'
            },
            {
                "role": "user",
                "content": 'Then you should perform a second-level comparison of the pairs to identify the best option with strongly supported evidence you discover. Ensure the comparisons are not influenced by the order of priority in which the descriptions are presented.'
            },
            {
                "role": "user",
                "content": 'To mitigate positional bias, follow these steps: Mentally shuffle the options into different sequences (e.g., reverse order, random permutation). Analyze each action\'s merits/demerits in each shuffled configuration. Identify if any option consistently emerges as superior across different orderings. If inconsistencies arise, conduct content-based comparisons using criteria like: Factual accuracy、Logical coherence、Contextual relevance、Completeness of information. Document your reasoning process transparently before finalizing the selection'
            },
            {
                "role": "user",
                "content": 'Then, tell me which human action that is noticeably different from the other descriptions. Explain the reason for this difference.'

            },
            {
                "role": "user",
                "content": 'Finally, give me the answer. (The answer format is \"Video description ID is noticeably different from the other two\")'

            },

        ]
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=s_value,
            # temperature=0.1
        )
        reasoning_text = ""  # Store the complete reasoning content
        final_content = ""

        for chunk in chat_completion:
            # Check that choices exists and is not empty
            if not chunk.choices or len(chunk.choices) == 0:
                continue

            # Append reasoning content
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_text += chunk.choices[0].delta.reasoning_content

            # Append final response content
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                final_content += chunk.choices[0].delta.content

        # reasoning_text now contains the complete reasoning and final_content contains the complete answer
        print("Complete reasoning content: ", reasoning_text)
        print("Complete final response: ", final_content)
        print('####################################################################################### ID = ', split_line)
        select_choice = int(get_choice(final_content))
        if select_choice == int(9):
            print(f"No answer detected; retrying... (attempt {attempt}/{max_retries})")
            if attempt == max_retries:
                print("Maximum retries reached; recording the item in failure.txt")
                failure_detect_and_write(split_line)
        else:
            sorted_images = move_choice_to_final(split_line, select_choice)
            save_files(sorted_images = sorted_images, prompt = messages, chains=reasoning_text, response = final_content, csv_name = csv_save_name, txt_name = new_filename, select_choice=select_choice)
            print(sorted_images)
            break
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time = {execution_time} seconds")
