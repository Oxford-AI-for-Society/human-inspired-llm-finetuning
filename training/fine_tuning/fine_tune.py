import warnings
warnings.simplefilter("ignore")

import os
import argparse
import torch
import bitsandbytes as bnb
from torch.utils.data import SequentialSampler
from transformers import TrainingArguments,set_seed
from transformers.trainer_utils import has_length
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from typing import Optional

from shared_utils import load_model, load_tokenizer, format_dataset


# TIP: with dataloader_num_workers >0 not getting this causes it to have a bit of a panic in the terminal logs
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set seed for the Transformer library (i.e. Trainer)
set_seed(42)


# Disable data shuffling in PyTorch to ensure data ordering is strictly applied
class CustomSFTTrainer(SFTTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # Check if the dataset is set and supports indexing
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        
        # Regardless of the args.group_by_length setting or other considerations,
        # always use SequentialSampler for a deterministic order.
        return SequentialSampler(self.train_dataset)
    

# Function to find linear layers for QLoRA adjustments
def find_linear_layers(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


def fine_tune(args):
    model = load_model(args.model_name, fine_tune=True)
    tokenizer = load_tokenizer(args.model_name)
    train_dataset = format_dataset(args.train_data_file, fine_tune=True)

    target_modules = find_linear_layers(model)
    qlora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=args.saved_model_dir,
        num_train_epochs=args.num_of_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.saved_model_dir}/logs",
        logging_steps=10,
        logging_strategy="steps",
        save_strategy="epoch",
        # warmup_steps=2, # Learning rate warm-up steps
        optim="paged_adamw_8bit",
        fp16=True,
        # save_total_limit=1,  # The number of saved checkpoints
        report_to="none",
        dataloader_num_workers=args.dataloader_num_workers,
    )

    # Initialize custom trainer
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        peft_config=qlora_config,
        dataset_text_field="text",
        max_seq_length=500, 
        data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template="Answer:")
    )

    # Start training
    print('Start training.')
    trainer.train()

    # Save the model
    trainer.save_model(args.saved_model_dir)
    print('Model saved.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on a dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--train_data_file", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--saved_model_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_of_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, required=True, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training")
    parser.add_argument("--dataloader_num_workers", type=int, required=False, default=16, help="Number of workers for data loading")

    args = parser.parse_args()

    fine_tune(args)


# Terminal code example
# CUDA_VISIBLE_DEVICES=0 python fine_tune.py --model_name meta-llama/Llama-2-7b-chat-hf --saved_model_dir /code/llm-fine-tuning/MODELS_TRY/2e --num_of_epochs 2 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 1e-04 --train_data_file /code/llm-fine-tuning/TRAINING_SETS/train_set_fold_2.csv 