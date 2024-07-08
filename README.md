# Fine-tuning LLMs with Human-inspired learning Strategies

This repository provides the fine-tuning experiments in Fine-tuning Large Language Models with Human-inspired Learning Strategies in Medical Question Answering.

## LLM difficulty measure
The code that measures question difficulty based on LLMs' responses is contained in `./measure_difficulty`. To use LLMs to measure difficulty of questions, see `./measure_difficulty/baseline_script.py`.

## Fine-tuning
The learning orders inspired by human-learning strategies are contained in `./training/data_ordering.py`.

To fine-tune a LLM for medical question answering with multiple choices, see `./training/fine-tuning/fine-tune.py`. For inference, see `./training/inference/inference.py`. 

## Clustering 
To cluster question categories based on semantic similarity using UMAP + HDBSCAN, see `./text_clustering.py`.

## How to cite
If you find our work relevant, please cite as following:
