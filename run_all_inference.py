import os

# Define the base directories and parameters
base_dir = "/code/llm-fine-tuning/"

# Define base model sizes
base_models = ["1.1B", "7B", "13B"]

test_datasets = [
    f"{base_dir}TEST_SETS/test_set_fold_1.csv",
    f"{base_dir}TEST_SETS/test_set_fold_2.csv",
    f"{base_dir}TEST_SETS/test_set_fold_3.csv",
    # f"{base_dir}TEST_SETS/test_set_fold_4.csv",
    f"{base_dir}TEST_SETS/medmcqa_866.csv",
    f"{base_dir}TEST_SETS/medmcqa_4183.csv",
    f"{base_dir}TEST_SETS/medqa_1276.csv"
]

saved_results_base_dir = "/code/llm-fine-tuning/TEST_RESULTS"
inference_script = "inference_batched.py"

# Ensure the experiments.queue file exists
queue_file_path = "inference_experiments.queue"
open(queue_file_path, 'a').close()

# Generate and append commands for inference
with open(queue_file_path, "a") as queue_file:
    for base_model in base_models:
        # Adjust the logic to match model_splits with corresponding test_datasets
        model_splits = ["LEK_train_split_1", "LEK_train_split_2", "LEK_train_split_3"] # "LEK_train_split_4"
        full_model_split = "LEK_full"  # This will be used with the last three test datasets
        
        for idx, test_dataset in enumerate(test_datasets):
            test_set_name = os.path.basename(test_dataset).replace(".csv", "")
            saved_results_dir = f"{saved_results_base_dir}/{base_model}/{test_set_name}"

            if not os.path.exists(saved_results_dir):
                os.makedirs(saved_results_dir, exist_ok=True)

            # Decide which model directory to use based on the index of the test_dataset
            if idx < 3:  # For the first three test datasets, use corresponding modelsplits
                models_dir = f"{base_dir}MODELS/{base_model}/{model_splits[idx]}"
            else:  # For the last three test datasets, use the full_model_split
                models_dir = f"{base_dir}MODELS/{base_model}/{full_model_split}"
            
            if not os.path.exists(models_dir):
                print(f"Models directory does not exist: {models_dir}")
                continue

            for model_dir in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_dir)
                if os.path.isdir(model_path):
                    model_results_dir = f"{saved_results_dir}/{model_dir}"

                    if not os.path.exists(model_results_dir) or not os.listdir(model_results_dir):
                        run_command = f"python {inference_script} --model_name=\"{model_path}\" --test_data_file=\"{test_dataset}\" --saved_result_dir=\"{model_results_dir}\"\n"
                        queue_file.write(run_command)
                        exit()
                    else:
                        print(f"Results already exist for model: {model_path}, skipping inference.")

print("All inference commands have been added to the queue.")







