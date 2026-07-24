
import subprocess
import itertools

learning_rates = [0.0005]  # 设定不同的learning rate值
# lmbda = 0.0065 # 设定不同的lmbda值
lmbda_values = [0.012]  # 设定不同的lmbda值
seed_sets = list(range(1,35))
# seed_sets = [42]
model_sets = ['deepseek']  # 设定不同的模型
processes = []


for model, lr, seed, lmbda in itertools.product(model_sets, learning_rates, seed_sets, lmbda_values):
    print(f"Running model: {model}, learning_rate: {lr}, lmbda: {lmbda}")
    cmd = (
        f"python train.py --task odd_one_out --modality {model}/ "
        f"--results_dir ./results/  "
        f"--plots_dir ./plots/  "
        f"--triplets_dir /data1/home/sunyi/large-files/Python_Objects/human_action/SPoSE_calculate/data/Deepseek-671B_triplets/ "
        f"--learning_rate {lr} --lmbda {lmbda} "
        f"--embed_dim 50 --batch_size 100 --epochs 2000 "
        f"--window_size 100 --steps 100 --sampling_method normal "
        f"--device cpu --rnd_seed {seed} "
        # f"--resume "
        f"--num_threads 1"
    )
    process = subprocess.Popen(cmd, shell=True)
    processes.append(process)

# 等待所有进程完成
for process in processes:
    process.wait()    
    

