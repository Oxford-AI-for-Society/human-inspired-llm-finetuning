
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from string import Template
from pathlib import Path

import os

import warnings
warnings.simplefilter("ignore")

from tqdm.notebook import tqdm

# for training
from peft import LoraConfig, get_peft_model, LoraModel, PeftModel
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# for traing set
from datasets import load_dataset
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
import bitsandbytes as bnb
import numpy as np

from IPython.display import Markdown, display
import json



with open('/home/blac0817/tokens.json') as f:
    tokens = json.load(f)
hf_token = tokens['hugging_face']

# Base model name for training
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Directory of the training file
train_data_file = "/data/blac0817/medqa/train.jsonl"

# Directory to save the fine-tuned model locally
saved_model_dir = '/data/blac0817/finetuning'

# Hyperparameters for training
num_of_epochs = 4 # 4
per_device_train_batch_size=4 # 4
per_device_eval_batch_size=8 # 8
gradient_accumulation_steps=2 # Set to 2 to increase the batch size without increasing the memory requirement
learning_rate=1e-5 #2e-5


### Setup model for training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtyp=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
    token=hf_token
)
# this should be set as False for finetuning
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

def find_linear_layers(model):
    """ Find linear layers in the given transformer model """
    lora_module_names = set()
    for name, module in model.named_modules():
        # 4 bits for qlora
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])  # can this just be names[-1]?

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


# Only apply QLoRA on linear layers of the model
target_modules = find_linear_layers(model)

# For llama 2 (they need different target module)
qlora_config = LoraConfig(
    r=16,  # dimension of the updated matrices
    lora_alpha=64,  # parameter for scaling
    target_modules=target_modules, # this chooses on which modules inside our model needs to be amended with LoRA matrices
    lora_dropout=0.1,  # dropout probability for layers
    bias="none",
    task_type="CAUSAL_LM",
)

# Set up training arguments
# "max_steps=1" is just for testing execution
training_args = TrainingArguments(
    output_dir=saved_model_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps, # Set to 2 to increase the batch size without increasing the memory requirement
    learning_rate=learning_rate, # Lower the lr to prevent overshooting
    logging_steps=10,
    logging_strategy="steps",
    warmup_steps=2,
    num_train_epochs=num_of_epochs,
    # max_steps=1, # This overrides "num_train_epochs"
    optim="paged_adamw_8bit",
    fp16=True,
    save_total_limit=1,  # The number of saved checkpoints, can be increased, but beware of kaggle notebook output size limit
    report_to="none"
    # push_to_hub=True # Push the model to hub
)

response_template="Answer:"

supervised_finetuning_trainer = SFTTrainer(
    model,
    train_dataset=train_dataset['train'],
    args=training_args,
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="text",
    max_seq_length=3000, # The maximum number of tokens that will be considered from each sequence when processing text data
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,
                                                  response_template=response_template) # Train on completions
)

# load training data\
train_dataset = load_dataset("json", data_files=train_data_file)


# Prepare template - alternatively add "###Instruction:" etc
template = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]

    Question: {prompt}\n
    A) {a}\n
    B) {b}\n
    C) {c}\n
    D) {d}\n
    E) {e}\n

    ### Answer: {answer}"""

prompt = PromptTemplate(template=template, input_variables=['prompt', 'a', 'b', 'c', 'd', 'e', 'answer'])
# Make the training dataset to have this format
train_dataset = train_dataset.map(format_text)

supervised_finetuning_trainer.train()
# Save model locally
supervised_finetuning_trainer.save_model(saved_model_dir)