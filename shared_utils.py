
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from langchain.prompts import PromptTemplate
from peft import PeftModel

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
        device_map="cuda:0",
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
        model = model.merge_and_unload()
    return model


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, parallelism=False)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def format_dataset(data_file):
    dataset = load_dataset("csv", data_files=data_file)
    
    # Define two templates: one with the 'E' option and one without
    template_with_e = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]

    Question: {prompt}\n
    A) {a}\n
    B) {b}\n
    C) {c}\n
    D) {d}\n
    E) {e}\n

    ### Answer: {answer}"""

    template_without_e = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D]

    Question: {prompt}\n
    A) {a}\n
    B) {b}\n
    C) {c}\n
    D) {d}\n

    ### Answer: {answer}"""

    def format_text(example):
        # Check if 'E' option exists in the example
        if 'E' in example:
            prompt = PromptTemplate(template=template_with_e, input_variables=['prompt', 'a', 'b', 'c', 'd', 'e', 'answer'])
            text = prompt.format(
                prompt=example['prompt'],
                a=example['A'],
                b=example['B'],
                c=example['C'],
                d=example['D'],
                e=example['E'],
                answer='')
        else:
            prompt = PromptTemplate(template=template_without_e, input_variables=['prompt', 'a', 'b', 'c', 'd', 'answer'])
            text = prompt.format(
                prompt=example['prompt'],
                a=example['A'],
                b=example['B'],
                c=example['C'],
                d=example['D'],
                answer='')
        return {"text": text}
    
    # Applying the formatting function to each example in the dataset
    formatted_dataset = dataset.map(format_text, num_proc=32)
    # train_dataset.set_transform(format_text)
    # TIP: if the dataset is large you can use `num_proc=4` to parallelize the processing
    # set_transform is similar but applied when the data is loaded in DataLoader, so would mean no waiting here before we start training
    return formatted_dataset






