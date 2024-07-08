import os

# Define the base directories and parameters
base_dir = "/code/llm-fine-tuning"
models_dir = f"{base_dir}/CLUSTERING_MODELS" 
results_dir = f"{base_dir}/CLUSTERING_TEST_RESULTS" 
queue_file_path = "inference_experiments.queue"

# Define base models
base_models = ["llama2-13B"] # "tinyllama-1.1B", "llama2-7B", "mistral-7B"

# Define test datasets and their corresponding model split directories
# test_datasets_with_splits = {
#     # "test_set_fold_1.csv": ["train_set_fold_1"],
#     # "test_set_fold_2.csv": ["train_set_fold_2"],
#     # "test_set_fold_3.csv": ["train_set_fold_3"],
#     # "test_set_fold_4.csv": ["train_set_fold_4"],
#     "medqa_1276.csv": ["lek_full"],
#     # "medmcqa_4183.csv": ["lek_full"]
#     # "medqa_1276_for_medqa.csv": ["medqa_train_10.2k"], 
# }

test_datasets_with_splits = {
    # "test_set_fold_1.csv": ["train_set_fold_1_hcat"],
    # "test_set_fold_2.csv": ["train_set_fold_2_hcat"],
    # "test_set_fold_3.csv": ["train_set_fold_3_hcat"],
    # "test_set_fold_4.csv": ["train_set_fold_4_hcat"],
    "medqa_1276.csv": ["lek_full_hcat"],
    # "medmcqa_4183.csv": ["lek_full_hcat"]
}

# test_datasets_with_splits = {
#     "medmcqa_4183_for_medqa.csv": ["medqa_train_10.2k"], 
#     "lek_full_for_medqa.csv": ["medqa_train_10.2k"], 
    # "medqa_1276_for_medqa.csv": ["medqa_train_10.2k"], 
# }

# Ensure the experiments.queue file exists
open(queue_file_path, 'a').close()

# Generate commands for inference and organize the test result directories
with open(queue_file_path, "a") as queue_file:
    for base_model in base_models:
        for test_csv, model_splits in test_datasets_with_splits.items():
            test_dataset_path = f"{base_dir}/TEST_SETS/{test_csv}"
            test_set_name = os.path.splitext(test_csv)[0]

            for model_split in model_splits:
                # split_dir_name = "hcat" if "hcat" in model_split else "cluster"
                model_split_dir = os.path.join(models_dir, base_model, model_split)
                print(model_split_dir)

                # Iterate over each fine-tuned model directory
                if os.path.isdir(model_split_dir):
                    for ft_model_dir in os.listdir(model_split_dir):
                        ft_model_path = os.path.join(model_split_dir, ft_model_dir)

                        # Check if this is a directory containing a model
                        if os.path.isdir(ft_model_path):
                            # Define the directory to save inference results
                            saved_results_dir = os.path.join(results_dir, base_model, test_set_name, ft_model_dir) # split_dir_name
                            
                            os.makedirs(saved_results_dir, exist_ok=True)

                            # Check if the directory is non-empty to skip inference if already done
                            if not os.listdir(saved_results_dir):  # Directory is empty
                                run_command = (
                                    f"python3 inference.py --model_name \"{ft_model_path}\" "
                                    f"--test_data_file \"{test_dataset_path}\" "
                                    f"--saved_result_dir \"{saved_results_dir}\"\n"
                                )
                                queue_file.write(run_command)
                            else:
                                # Print a message indicating that inference is being skipped
                                print(f"Skipping inference for {ft_model_dir} as results already exist in {saved_results_dir}")

print("All inference commands have been added to the queue.")
