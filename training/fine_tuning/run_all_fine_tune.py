import os
import pandas as pd
import warnings
warnings.simplefilter("ignore")

# Import all data ordering functions
from data_processing.data_ordering import *

# Define the base directory for datasets
processed_dataset_dir_base = "/code/llm-fine-tuning/CLUSTERING_LEK_TRAINING_SETS"  # PROCESSED_TRAINING_SETS # CLUSTERING_MEDQA_TRAINING_SETS

# Ensure processed dataset directory exists
os.makedirs(processed_dataset_dir_base, exist_ok=True)

# Define models with different learning rates and batch sizes
models = [
       # {"name": "mistralai/Mistral-7B-v0.1", "suffix": "mistral-7B", "lr": 1e-04, "batch_size": 4, "grad_accum": 2}, 
    # {"name": "TinyLlama/TinyLlama-1.1B-Chat-v0.4", "suffix": "tinyllama-1.1B", "lr": 5e-04, "batch_size": 16, "grad_accum": 1},
    # {"name": "meta-llama/Llama-2-7b-chat-hf", "suffix": "llama2-7B", "lr": 5e-05, "batch_size": 4, "grad_accum": 2}, #  "lr": 1e-4 for ft on lek, 5e-05 for fine-tuning medqa_train
    {"name": "meta-llama/Llama-2-13b-chat-hf", "suffix": "llama2-13B", "lr": 1e-04, "batch_size": 4, "grad_accum": 2},
]

models_dir_base = "/code/llm-fine-tuning/CLUSTERING_MODELS" # CLUSTERING_MODELS
num_epochs = 1
dataloader_num_workers = 16

# Ensure the experiments.queue file exists
queue_file_path = "fine_tune_experiments.queue"
open(queue_file_path, 'a').close()

with open(queue_file_path, "a") as queue_file:
    for subdir, dirs, files in os.walk(processed_dataset_dir_base):
        for file in files:
            if file.endswith(".csv"):
                # Construct the path to the existing processed CSV file
                processed_csv_path = os.path.join(subdir, file)

                for model in models:
                    model_name = model["name"]
                    model_suffix = model["suffix"]
                    dataset_name = os.path.basename(subdir)
                    func_name = os.path.splitext(file)[0]
                    saved_model_dir = os.path.join(models_dir_base, model_suffix, dataset_name, f"{func_name}_ft_model")

                    if os.path.exists(saved_model_dir) and os.listdir(saved_model_dir):
                        # If the model directory exists and is not empty, skip fine-tuning
                        print(f"Skipping fine-tuning for existing model: {saved_model_dir}")
                        continue
                    
                    if not os.path.exists(saved_model_dir):
                        os.makedirs(saved_model_dir, exist_ok=True)

                    # Prepare the fine-tuning command with the path to the processed CSV
                    run_command = f"python3 fine_tune.py --model_name {model_name} --train_data_file {processed_csv_path} --saved_model_dir {saved_model_dir} --num_of_epochs {num_epochs} --per_device_train_batch_size {model['batch_size']} --per_device_eval_batch_size {model['batch_size']} --gradient_accumulation_steps {model['grad_accum']} --learning_rate {model['lr']} --dataloader_num_workers {dataloader_num_workers}\n"
                    queue_file.write(run_command)

print("Commands for fine-tuning tasks with different data ordering have been added to the queue.")
