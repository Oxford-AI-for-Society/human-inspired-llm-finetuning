import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import PeftModel
import pandas as pd

from data_processing.data_ordering import blocked_1

#nf4 - 30 steps = 1:02  on 2 gpu
#bfloat16 about the same
#bfloat16 - 70 steps = 1:03
#num_workers=16 about the same,perhaps a little faster - dataloading not a bottleneck as we don't do much processing and loading text is fast
# batch_size = 16 - 28 steps, same as 112 steps with batch_size=4 as above in the same time

def load_model(model_name, fine_tune, adapter_path=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # TIP: quantizing to 4bit makes it smaller but might reduce performance, 8bit would be a good compromise if bfloat16 doesn't fit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto", # "cuda:0"
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    # This should be set as False for fine-tuning
    if fine_tune:
        model.config.use_cache = False

    if adapter_path:
        model = PeftModel.from_pretrained(
            model, 
            adapter_path
        )
        # model = model.merge_and_unload()
    return model


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, parallelism=False)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
    

def format_example(example, fine_tune):
    # Define the templates
    template_with_e = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E].

Question: {prompt}
A) {a}
B) {b}
C) {c}
D) {d}
E) {e}

### Answer: {answer}"""

    template_without_e = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D].

Question: {prompt}
A) {a}
B) {b}
C) {c}
D) {d}

### Answer: {answer}"""

    # Decide what to put in the answer section based on the fine_tune flag
    answer = example['Answer'] if fine_tune else ''

    # Check if the 'E' option is present in the example
    if 'E' in example:
        # If 'E' option is present, use the template with 'E'
        text = template_with_e.format(
            prompt=example['prompt'],
            a=example['A'],
            b=example['B'],
            c=example['C'],
            d=example['D'],
            e=example['E'],  
            answer=answer)
    else:
        # If 'E' option is not present, use the template without 'E'
        text = template_without_e.format(
            prompt=example['prompt'],
            a=example['A'],
            b=example['B'],
            c=example['C'],
            d=example['D'],
            answer=answer)

    return {"text": text}


def format_dataset(data_file, fine_tune, few_shot=False, dataset_name=None):
    dataset = load_dataset("csv", data_files=data_file)

    # Apply the formatting function to each example in the dataset, passing the fine_tune flag
    # formatted_dataset = dataset.map(format_text, num_proc=32)
    formatted_dataset = dataset.map(lambda example: format_example(example, fine_tune))

    return formatted_dataset


def main():
    # Define a single question example manually for demonstration
    example_question = {
        "prompt": "What is the primary function of the placenta?",
        "A": "Facilitate fetal nutrition.",
        "B": "Produce red blood cells for the fetus.",
        "C": "Generate antibodies for the fetus.",
        "D": "Excrete waste products for the fetus.",
        "E": "Regulate temperature for the fetus.",
        "Answer": "A"
    }

    # Define some example parameters
    fine_tune = False
    few_shot = True
    dataset = 'lek'
    
    # Format the single example using your predefined function
    formatted_example = format_example(example_question, fine_tune, few_shot, dataset)

    # Print the formatted example
    print(formatted_example)

if __name__ == "__main__":
    main()