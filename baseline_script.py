from dataclasses import dataclass
from dataclasses_json import dataclass_json
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from datasets import load_dataset_builder, load_dataset, get_dataset_config_names
import pandas as pd
import torch
import numpy as np
import re
from tqdm import tqdm
import random
import os


## Options to change regularly

with open('../../home/blac0817/tokens.json') as f:
    tokens = json.load(f)
hf_token = tokens['hugging_face']


models = { 'Mistral 8x7B': 'mistralai/Mixtral-8x7B-v0.1',
           'Meditron 7B': 'epfl-llm/meditron-7b',
           'Llama 2 7B':  'meta-llama/Llama-2-7b-chat-hf',
           'Gemma 7B': 'google/gemma-7b-it',
          }

model_name = models['Gemma 7B']

cache_dir = '../../data/blac0817/huggingface'
os.putenv("HF_HOME", cache_dir)
#questions_path = "../../data/blac0817/data/master_questions.csv"
questions_path = 'medqa'

batch_size = 10
question_limit = 100000  # for testing
randomize_choices = False

out_folder = "../../data/blac0817/human-learning-strategies/responses"
filename = f"{model_name.split('/')[-1]}_medqa.json"

## Data structure for outputs

@dataclass_json
@dataclass
class QA:
    question: str
    correct_answer: str
    question_index: int
    shuffle: list[int]
    response: str
    top3: list[str]
    clls: list[float]


@dataclass_json
@dataclass
class QAs:
    questions: list[QA]


## Functions for running and parsing the results

def rank_multichoice(prompt, max_length, choice_tokens):
    
    samples = [p + ' Answer: ' for p in prompt]

    inputs = tokenizer(samples, add_special_tokens=False, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True).to(device)

    output = model.generate(**inputs, max_length=max_length+1, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_logits=True)
    response_ids = output['sequences'][:, inputs["input_ids"].shape[1]:] 
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    logits = [output['logits'][0][:,c] for c in choice_tokens]

    logits = torch.cat(logits, dim=1)

    log_likelihood = torch.log_softmax(logits, dim=1)

    return log_likelihood

def make_batch(questions, answers, indices, shuffle, batch_size=64):
    batches = []
    for i in range(0, len(questions), batch_size):
        batches.append({'questions':questions[i : i + batch_size], 'answers':answers[i : i + batch_size], 'index':indices[i : i + batch_size], 'shuffles':shuffle[i : i + batch_size]})
    return batches

def load_medqa(randomize_choices=False):
    ### Loads the medqa dataset and parses it into the desired format
    med_qa = load_dataset('bigbio/med_qa','med_qa_en_source',trust_remote_code=True,split='train')
    df = med_qa.to_pandas()
    choices = ['A','B','C','D','E']

    random.seed(0)

    choices_df = df['options'].apply(lambda x: pd.Series([x[0]['value'],x[1]['value'],x[2]['value'],x[3]['value'],x[4]['value'],]))
    choices_df.columns = choices
    df = pd.concat([df,choices_df],axis=1)

    def format_prompt(row, choices):
        answer_choices = [row[c] for c in choices]
        keys = list(range(len(choices)))
        if randomize_choices:
            random.shuffle(keys)
        text = f"Answer the following multiple choice question by giving the most appropriate response. The answer should be one of [A, B, C, D, E]. Question: {row['question']} A) {answer_choices[keys[0]]} B) {answer_choices[keys[1]]} C) {answer_choices[keys[2]]} D) {answer_choices[keys[3]]} E) {answer_choices[keys[4]]}"
        answer = choices[keys.index(choices.index(row['answer_idx']))]
        return pd.Series([text, answer, keys],index=['question','answer', 'shuffle'])
    
    shuffled_df = df[['question', 'A', 'B', 'C', 'D', 'E','answer_idx']].apply(lambda x: format_prompt(x,choices),axis=1)
    questions = shuffled_df['question'].values
    correct_answers = shuffled_df['answer'].values
    shuffle = shuffled_df['shuffle'].values
    q_idx = list(df.index.values)
    return questions, correct_answers, q_idx, shuffle, choices

def load_medmcqa(randomize_choices=False):
    ### Loads the medqa dataset and parses it into the desired format
    medmcqa = load_dataset("openlifescienceai/medmcqa",split='train')
    df = medmcqa.to_pandas()
    choices = ['A','B','C','D']

    df = df.rename(columns={'opa': 'A', 'opb': 'B', 'opc': 'C', 'opd': 'D'})

    df['answer'] = df['cop'].apply(lambda x: choices[x])

    random.seed(0)

    def format_prompt(row, choices):
        answer_choices = [row[c] for c in choices]
        keys = list(range(len(choices)))
        if randomize_choices:
            random.shuffle(keys)
        text = f"Answer the following multiple choice question by giving the most appropriate response. The answer should be one of [A, B, C, D]. Question: {row['question']} A) {answer_choices[keys[0]]} B) {answer_choices[keys[1]]} C) {answer_choices[keys[2]]} D) {answer_choices[keys[3]]}"
        answer = choices[keys.index(choices.index(row['answer']))]
        return pd.Series([text, answer, keys],index=['question','answer', 'shuffle'])
    
    shuffled_df = df[['question', 'A', 'B', 'C', 'D','answer']].apply(lambda x: format_prompt(x,choices),axis=1)
    questions = shuffled_df['question'].values
    correct_answers = shuffled_df['answer'].values
    shuffle = shuffled_df['shuffle'].values
    q_idx = list(df.index.values)
    return questions, correct_answers, q_idx, shuffle, choices



def load_questions(questions_path, randomize_choices=False):

    if questions_path == 'medqa':
        questions, correct_answers, q_idx, shuffle, choices = load_medqa(randomize_choices)

    elif questions_path 'medmcqa':
        questions, correct_answers, q_idx, shuffle, choices = load_medqa(randomize_choices)

    else:

        ### Loads a csv file with columns labelled 'Question' and 'Answer' and returns two lists with the questions and answers
        choices=["A", "B", "C", "D", "E"]
        df = pd.read_csv(questions_path, ).set_index('Unnamed: 0')
        random.seed(0)

        def format_prompt(row, choices):
            answer_choices = [row[c] for c in choices]
            keys = list(range(len(choices)))
            if randomize_choices:
                random.shuffle(keys)
            text = f"Answer the following multiple choice question by giving the most appropriate response. The answer should be one of [A, B, C, D, E]. Question: {row['prompt']} A) {answer_choices[keys[0]]} B) {answer_choices[keys[1]]} C) {answer_choices[keys[2]]} D) {answer_choices[keys[3]]} E) {answer_choices[keys[4]]}"
            answer = choices[keys.index(choices.index(row['Answer']))]
            return pd.Series([text, answer, keys],index=['question','answer', 'shuffle'])
        
        shuffled_df = df[['prompt', 'A', 'B', 'C', 'D', 'E','Answer']].apply(lambda x: format_prompt(x,choices),axis=1)
        questions = shuffled_df['question'].values
        correct_answers = shuffled_df['answer'].values
        shuffle = shuffled_df['shuffle'].values
        q_idx = list(df.index.values)

    return questions, correct_answers, q_idx, shuffle, choices


##################################
## Actual running code
##################################


## Loading the model 

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    model_name, padding_side="left", trust_remote_code=True, token=hf_token
)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    #bnb_4bit_compute_dtyp=torch.bfloat16,
)

# faster on single gpu if the model can fit, change "auto" to "cuda:0"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    #torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir,
    # use_flash_attention_2=True,
    trust_remote_code=True,
    quantization_config=bnb_config,
)

# not yet available on python 3.12
# model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

questions, correct_answers, q_idx, shuffles, choices = load_questions(questions_path, randomize_choices=randomize_choices)
questions, correct_answers, q_idx, shuffles = questions[:question_limit], correct_answers[:question_limit], q_idx[:question_limit], shuffles[:question_limit]

max_length = np.max([len(q) for q in questions])
qas = QAs([])
choice_tokens = tokenizer(choices, add_special_tokens=False, return_tensors="pt", max_length=1, padding="max_length", truncation=True).input_ids.tolist()


for batch in tqdm(make_batch(questions, correct_answers, q_idx, shuffles, batch_size=batch_size)):
    
    response = rank_multichoice(batch['questions'], max_length, choice_tokens)

    rs = [
            QA(q, a, int(idx), s, choices[r.argmax()], [choices[i] for i in r.argsort().flip(0)[:3]], r.tolist())
            for q, a, idx, s, r in zip(
                batch['questions'], batch['answers'], batch['index'], batch['shuffles'], response
            )
        ]

    qas.questions.extend(rs)

        
folder = Path(out_folder)
folder.mkdir(exist_ok=True)
(folder / filename).write_text(json.dumps(qas.to_dict(), indent=4))
