# Evaluating Fine-Tuning Efficiency of Human-Inspired Learning Strategies in Medical Question Answering

This repository contains the fine-tuning experiments from the paper 'Fine-tuning Large Language Models with Human-inspired Learning Strategies in Medical Question Answering'.

## Measuring Question Difficulty with LLMs

To use LLMs for measuring the difficulty of questions, refer to:
- `./measure_difficulty/baseline_script.py`

To score questions based on LLM-defined difficulty, use:
- `./measure_difficulty/scoring.py`

## Fine-tuning LLMs with QLora

The scripts for learning orders inspired by human-learning strategies are located in:
- `./training/data_ordering.py`

To fine-tune a LLM for multiple-choice medical question answering, refer to:
- `./training/fine-tuning/fine-tune.py`

For inference, use:
- `./training/inference/inference.py`

## Clustering Questions

To cluster question categories based on semantic similarity using UMAP and HDBSCAN, see:
- `./text_clustering.py`

## How to Cite

If you find our work relevant, please cite it as follows:

