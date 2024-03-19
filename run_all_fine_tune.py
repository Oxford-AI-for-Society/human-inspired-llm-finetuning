import os

# Define the base directories for datasets
datasets_dirs = [
    "/code/llm-fine-tuning/TRAINING_SETS/LEK_train_split_1",
    "/code/llm-fine-tuning/TRAINING_SETS/LEK_train_split_2",
    "/code/llm-fine-tuning/TRAINING_SETS/LEK_train_split_3",
    # "/code/llm-fine-tuning/TRAINING_SETS/LEK_train_split_4",
    "/code/llm-fine-tuning/TRAINING_SETS/LEK_full"
]

# Define models with different learning rates and batch sizes
models = [
    # {"name": "TinyLlama/TinyLlama-1.1B-Chat-v0.4", "suffix": "1.1B", "lr": 1e-06, "batch_size": 16, "grad_accum": 1},
    {"name": "meta-llama/Llama-2-7b-chat-hf", "suffix": "7B", "lr": 1e-07, "batch_size": 8, "grad_accum": 1},
    # {"name": "meta-llama/Llama-2-13b-chat-hf", "suffix": "13B", "lr": 2e-07, "batch_size": 8, "grad_accum": 2} 
]

models_dir_base = "/code/llm-fine-tuning/MODELS"

# Training parameters
num_epochs = 4
# batch_size = 16
# gradient_accumulation_steps = 2
dataloader_num_workers = 16
# learning_rate = 1e-5

# Ensure the experiments.queue file exists
queue_file_path = "fine_tune_experiments.queue"
open(queue_file_path, 'a').close()

with open(queue_file_path, "a") as queue_file:
    for model in models:
        model_name = model["name"]
        model_suffix = model["suffix"]
        model_lr = model["lr"]  # Model-specific learning rate
        model_batch_size = model["batch_size"]  # Model-specific batch size
        model_grad_accum = model["grad_accum"]  # Model-specific gradient accumulation steps
        
        for datasets_dir in datasets_dirs:
            datasets_dir_name = os.path.basename(datasets_dir)  # Extract the dataset directory name
            for dataset in os.listdir(datasets_dir):
                if dataset.endswith(".csv"):
                    dataset_path = os.path.join(datasets_dir, dataset)
                    dataset_name = os.path.splitext(dataset)[0]

                    saved_model_dir = os.path.join(models_dir_base, model_suffix, datasets_dir_name, dataset_name + "_ft_model")

                    # if os.path.exists(saved_model_dir) and os.listdir(saved_model_dir):
                        # print(f"Skipping {model_name} for {dataset_name} in {datasets_dir_name} as fine-tuning already done.")
                        # continue

                    if not os.path.exists(saved_model_dir):
                        os.makedirs(saved_model_dir, exist_ok=True)

                    run_command = f"python fine_tune.py --model_name {model_name} --train_data_file {dataset_path} --saved_model_dir {saved_model_dir} --num_of_epochs {num_epochs} --per_device_train_batch_size {model_batch_size} --per_device_eval_batch_size {model_batch_size} --gradient_accumulation_steps {model_grad_accum} --learning_rate {model_lr} --dataloader_num_workers {dataloader_num_workers}\n"
                    
                    queue_file.write(run_command)
                    exit()

print("Commands for unfinished fine-tuning tasks have been added to the queue.")







# CUDA_VISIBLE_DEVICES="0" python inference_batched.py --model_name meta-llama/Llama-2-7b-chat-hf --test_data_file /code/llm-fine-tuning/TEST_SETS/test_set_.csv --saved_result_dir /code/llm-fine-tuning/TEST_RESULTS/test_try

# CUDA_VISIBLE_DEVICES="0" python fine_tune.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v0.4 --train_data_file /code/llm-fine-tuning/TRAINING_SETS/LEK_train_split_1/train_set_fold_1.csv --saved_model_dir /code/llm-fine-tuning/MODEL_TRY --num_of_epochs 4 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --learning_rate 1e-06