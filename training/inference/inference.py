import argparse
import torch
import pandas as pd
import csv
import os
import torch.nn as nn
import numpy as np

from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader

import warnings
warnings.simplefilter("ignore")

from shared_utils import load_model, load_tokenizer, format_dataset


# TIP: with dataloader_num_workers >0 not getting this causes it to have a bit of a panic in the terminal logs
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set seed for reproducibility
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Perplexity(nn.Module):
    def __init__(self, reduce: bool = False):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss() # CE loss is the average negative log-likelihood of the next token appearing given the input sequence
        self.reduce = reduce # Control whether the computed perplexity should be reduced (averaged) across all examples in a batch

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous() # Remove the last token's logits because there is no next token to predict after the last one
        shift_labels = labels[..., 1:].contiguous() # Shift the labels by one position to the left, aligning each token with its subsequent token as the label for prediction

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity


def evaluate_model_on_dataset(model, tokenizer, dataset, batch_size=8):
    model.eval()
    preds = []
    
    # Use DataLoader for batch processing
    loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=False)
    data_iterator = iter(loader)
    for batch in tqdm(data_iterator, total=len(loader)):
        with torch.no_grad():
            # Prepare the samples and tokenize in batches
            samples = [batch['text'][i] + col for i in range(len(batch['text'])) for col in ["A", "B", "C", "D", "E"]] 
            inputs = tokenizer(samples, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True, max_length=500)
            
            # Get model output in batches
            output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logits = output.logits
            labels = inputs["input_ids"].clone()
            labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
            
            # Adjust labels for next-token prediction and calculate perplexity in batches
            perp_calculator = Perplexity(reduce=False)
            perps = perp_calculator(logits, labels)
            
            num_samples = logits.size(0)  # Actual number of samples in the batch
            options_per_question = 5
            perps = perps.view(num_samples // options_per_question, options_per_question)
            
            for i in range(perps.shape[0]):
                # For each set of options for a question, find the option with the lowest perplexity
                sorted_indices = torch.argsort(perps[i]).cpu().numpy()
                # Take the top-1 (lowest perplexity) prediction
                top_1_prediction = ["A", "B", "C", "D", "E"][sorted_indices[0]] 
                preds.append(top_1_prediction)
    
    return preds



def evaluate_model_on_dataset_four_options(model, tokenizer, dataset):
    model.eval()
    preds = []

    for idx in tqdm(range(len(dataset["train"])), total=len(dataset["train"])):
        with torch.no_grad():
            # Prepare the sample inputs
            cols = ["A", "B", "C", "D"] 
            perps = []
            samples = [dataset["train"][idx]['text'] + col for col in cols]

            # Tokenize the samples
            inputs = tokenizer(samples, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True, max_length=500)

            # Get model output
            output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logits = output.logits
            labels = inputs["input_ids"] # Set up the ground truth for the next-token prediction
            labels.masked_fill_(~inputs["attention_mask"].bool(), -100) # Mask out the tokens that the model should not consider for loss calculation, only keep the option label

            # Calculate perplexity for each option
            perp = Perplexity()
            for j in range(len(cols)):
                p = perp(logits[j].unsqueeze(0), labels[j].unsqueeze(0))
                perps.append(p.item())

            # Memory management: delete variables no longer needed
            del inputs, labels, output, p

            # Sort the indices based on perplexity and get the corresponding options and perplexities
            sorted_indices = np.argsort(perps)
            top_1_prediction = cols[sorted_indices[0]]

            preds.append(top_1_prediction)

    return preds


# Calculate the acccuracy and F1 score
def calculate_metrics(df, true_col, pred_col):
    correct_predictions = (df[true_col] == df[pred_col])
    accuracy = correct_predictions.sum() / len(df)
    f1 = f1_score(df[true_col], df[pred_col], average='weighted')

    return accuracy, f1


def save_metrics(accuracy, f1, model_result_dir):
    metrics_filename = os.path.join(model_result_dir, "metrics.csv")
    file_exists = os.path.isfile(metrics_filename)
    
    with open(metrics_filename, 'a', newline='') as csvfile:
        headers = ['Accuracy', 'F1 Score']
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Accuracy': accuracy,
            'F1 Score': f1
        })


def save_dataframe(test_df, model_result_dir):
    results_filename = os.path.join(model_result_dir, "results.csv")
    test_df.to_csv(results_filename, index=False)


def inference(args):
    # Set the seed for reproducibility
    set_seed()

    model = load_model(args.model_name, fine_tune=False, adapter_path=args.adapter_path)
    tokenizer = load_tokenizer(args.model_name)
    test_dataset = format_dataset(args.test_data_file, fine_tune=False) 

    # Decide how many options to handle based on the test set name
    if 'medmcqa_4183' in args.test_data_file:
        evaluate_function = evaluate_model_on_dataset_four_options
    else:
        evaluate_function = evaluate_model_on_dataset

    preds = evaluate_function(model, tokenizer, test_dataset) # args.batch_size

    test_df = pd.read_csv(args.test_data_file)
    test_df['Prediction'] = preds
    accuracy, f1 = calculate_metrics(test_df, 'Answer', 'Prediction')

    # Directory for saving model's results
    model_result_dir = os.path.join(args.saved_result_dir, os.path.basename(args.model_name))
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)

    # Save the DataFrame
    save_dataframe(test_df, model_result_dir)
    print(f"Results saved.")

    # Save the metrics
    save_metrics(accuracy, f1, model_result_dir)
    print(f"Metrics saved.")


def main():
    parser = argparse.ArgumentParser(description='Run inference using a fine-tuned model.')
    parser.add_argument("--model_name", type=str, required=True, help="The name/path of the fine-tuned model")
    parser.add_argument("--adapter_path", type=str, required=False, help="The path to the adapter")
    parser.add_argument('--test_data_file', type=str, required=True, help='File path to the test dataset')
    parser.add_argument("--saved_result_dir", type=str, required=True, help="Directory to save the inference results")
    # parser.add_argument("--batch_size", type=int, default=4, help="Batch size for model evaluation")
    # # parser.add_argument("--options_count", type=int, default=5, help="Number of option indexes for question")

    args = parser.parse_args()

    inference(args)

if __name__ == "__main__":
    main()



