import os
import pandas as pd
import warnings
warnings.simplefilter("ignore")

# Import all data ordering functions
from data_processing.data_ordering import *

# Define the base directory for datasets
dataset_dir_base = "/code/llm-fine-tuning/TRAINING_SETS"
processed_dataset_dir_base = "/code/llm-fine-tuning/PROCESSED_TRAINING_SETS"  # Directory to save processed datasets

# Ensure processed dataset directory exists
os.makedirs(processed_dataset_dir_base, exist_ok=True)

# List of CSV files
csv_files = [
    "lek_full.csv",
    "train_set_fold_1.csv",
    "train_set_fold_2.csv",
    "train_set_fold_3.csv",
    "train_set_fold_4.csv"
]

# Define models with different learning rates and batch sizes
models = [
    # {"name": "google/gemma-2b-it", "suffix": "gemma-2B", "lr": 5e-04, "batch_size": 8, "grad_accum": 1},
    # {"name": "microsoft/phi-2", "suffix": "phi2-2.7B", "lr": 5e-04, "batch_size": 8, "grad_accum": 1},
    # {"name": "microsoft/Orca-2-7b", "suffix": "Orca-2-7B", "lr": 1e-04, "batch_size": 4, "grad_accum": 2},
    {"name": "mistralai/Mixtral-8x7B-v0.1", "suffix": "mistral-8x7B", "lr": 1e-05, "batch_size": 4, "grad_accum": 2},
    # {"name": "mistralai/Mixtral-7B-v0.1", "suffix": "mistral-7B", "lr": 1e-04, "batch_size": 4, "grad_accum": 2},
    # {"name": "TinyLlama/TinyLlama-1.1B-Chat-v0.4", "suffix": "tinyllama-1.1B", "lr": 5e-04, "batch_size": 16, "grad_accum": 1},
    # {"name": "meta-llama/Llama-2-7b-chat-hf", "suffix": "llama2-7B", "lr": 1e-04, "batch_size": 4, "grad_accum": 2},
    # {"name": "meta-llama/Llama-2-13b-chat-hf", "suffix": "llama2-13B", "lr": 1e-04, "batch_size": 4, "grad_accum": 2},
    # {"name": "meta-llama/Llama-2-70b-chat-hf", "suffix": "llama2-70B", "lr": 1e-05, "batch_size": 1, "grad_accum": 2} # batch size 1 due to memory constraint
]

models_dir_base = "/code/llm-fine-tuning/MODELS"
num_epochs = 1
dataloader_num_workers = 16

# List of data ordering functions
# data_ordering_functions = [original, random_shuffle_1, random_shuffle_2, blocked_curriculum_strict_2, blocked_curriculum_strict_3, blocked_emh_2, blocked_emh_3, interleaved_curriculum_strict_2, interleaved_curriculum_strict_3, interleaved_emh_2, interleaved_emh_3]

data_ordering_functions = [original, random_shuffle_1, random_shuffle_2, curriculum_strict, curriculum_emh_1, curriculum_emh_2, blocked_1, blocked_2, blocked_3, blocked_curriculum_strict_1, blocked_curriculum_strict_2, blocked_curriculum_strict_3, blocked_emh_1, blocked_emh_2, blocked_emh_3, interleaved_1, interleaved_2, interleaved_3, interleaved_curriculum_strict_1, interleaved_curriculum_strict_2, interleaved_curriculum_strict_3, interleaved_emh_1, interleaved_emh_2, interleaved_emh_3]

# Ensure the experiments.queue file exists
queue_file_path = "fine_tune_experiments.queue"
open(queue_file_path, 'a').close()

with open(queue_file_path, "a") as queue_file:
    for csv_file in csv_files:
        dataset_path = os.path.join(dataset_dir_base, csv_file)
        dataset_name = os.path.splitext(csv_file)[0]
        df = pd.read_csv(dataset_path)  # Load the CSV file once

        for data_ordering_function in data_ordering_functions:
            func_name = data_ordering_function.__name__
            
            # Apply the data ordering function
            processed_df = data_ordering_function(df)
            
            # Prepare the directory path for the processed CSV
            processed_csv_dir = os.path.join(processed_dataset_dir_base, dataset_name)
            if not os.path.exists(processed_csv_dir):
                os.makedirs(processed_csv_dir, exist_ok=True)

            # Save the processed DataFrame to a new CSV
            processed_csv_path = os.path.join(processed_csv_dir, f"{func_name}.csv")
            processed_df.to_csv(processed_csv_path, index=False)

            for model in models:
                model_name = model["name"]
                model_suffix = model["suffix"]
                saved_model_dir = os.path.join(models_dir_base, model_suffix, dataset_name, f"{func_name}_ft_model")

                if os.path.exists(saved_model_dir) and os.listdir(saved_model_dir):
                    print(f"Skipping {model_name} for {dataset_name} with {func_name} as fine-tuning already done.")
                    continue

                if not os.path.exists(saved_model_dir):
                    os.makedirs(saved_model_dir, exist_ok=True)

                # Prepare the fine-tuning command with the path to the processed CSV
                run_command = f"python3 fine_tune.py --model_name {model_name} --train_data_file {processed_csv_path} --saved_model_dir {saved_model_dir} --num_of_epochs {num_epochs} --per_device_train_batch_size {model['batch_size']} --per_device_eval_batch_size {model['batch_size']} --gradient_accumulation_steps {model['grad_accum']} --learning_rate {model['lr']} --dataloader_num_workers {dataloader_num_workers}\n"
                queue_file.write(run_command)

print("Commands for fine-tuning tasks with different data ordering have been added to the queue.")
