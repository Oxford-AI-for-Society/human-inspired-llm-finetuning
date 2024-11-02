# Evaluating Fine-Tuning Efficiency of Human-Inspired Learning Strategies in Medical Question Answering

This repository contains code for the fine-tuning experiments in [Evaluating Fine-Tuning Efficiency of Human-Inspired Learning Strategies in Medical Question Answering](https://arxiv.org/abs/2408.07888).

The paper is presented at NeurIPS 2024 in the workshop on [Fine-Tuning in Modern Machine Learning: Principles and Scalability (FITML)]([https://adaptive-foundation-models.org/](https://sites.google.com/view/neurips2024-ftw)).

![Fine-tuning with human-inspired learning strategies](learning_orders.png)


## Measuring question difficulty with LLMs

To use LLMs for measuring the difficulty of questions, run:
- `./measure_difficulty/baseline_script.py`

To score questions based on LLM-defined difficulty, run:
- `./measure_difficulty/scoring.py`

## Fine-tuning LLMs with QLora

The scripts for learning orders inspired by human-learning strategies are located in:
- `./training/data_ordering.py`

To fine-tune a LLM for multiple-choice medical question answering, run:
- `./training/fine-tuning/fine-tune.py`

For inference, run:
- `./training/inference/inference.py`

## Clustering for question categories

To cluster question categories based on semantic similarity using UMAP and HDBSCAN, run:
- `./text_clustering.py`

## How to cite

If you find our work relevant, please cite it as follows:

## Citation
```bibtex
@misc{yang2024finetuninglargelanguagemodels,
      title={Fine-tuning Large Language Models with Human-inspired Learning Strategies in Medical Question Answering}, 
      author={Yushi Yang and Andrew M. Bean and Robert McCraith and Adam Mahdi},
      year={2024},
      eprint={2408.07888},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.07888}, 
}
```

