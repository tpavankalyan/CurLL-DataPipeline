# CurLL-DataPipeline

This repository contains the data generation pipeline for the paper "Curriculum learning of Language Models". The pipeline is designed to process and prepare datasets for training language models in a curriculum learning setup.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data](#data)

## Overview

The CurLL-DataPipeline is a series of scripts and notebooks that work together to generate, process, and format data for training language models. The pipeline is divided into several stages, with each stage building upon the previous one to create a curriculum of increasing complexity.

The main steps in the pipeline are:
1.  **Seed Preparation**: Generating initial seed data.
2.  **Data Processing**: Cleaning, parsing, and transforming raw data.
3.  **Dataset Creation**: Formatting and combining datasets into training, validation, and test splits.
4.  **Hugging Face Hub Integration**: Uploading the final datasets to the Hugging Face Hub.

## Repository Structure

Here is a breakdown of the files in this repository and their purposes:

-   **`prepare_seed.ipynb`**: A Jupyter notebook for preparing the initial seed data required for the data generation pipeline.
-   **`skill_graph.ipynb`**: A Jupyter notebook that deals with creating and visualizing a skill graph, generating the edges, which is a key component of the curriculum learning strategy.
-   **`templates.ipynb`**: A Jupyter notebook that contains code for generating templates from skill graph.
-   **`process_context_instruct_outputs.py`**: A Python script that processes the outputs of the LLM. It parses the generated context/instruct and prepares the context for CQA, CSQA.
-   **`process_cqa_csqa_outputs.ipynb`**: A Jupyter notebook for processing outputs of CQA, CSQA generations.
-   **`format_train.py`**: A Python script that formats the processed data into a consistent format for training. It combines different datasets, applies formatting, and prepares the data for the final stage.
-   **`create_train_val_test.ipynb`**: A Jupyter notebook that takes entire data and splits it into training, validation, and test sets.
-   **`results.ipynb`**: This contains code for all analysis and plots presented in the paper
-   **`run_inference.py`**: Use this python script to run inference using Gemma3-27B-IT

By default the script uses `Gemma3-27B-IT`. If the vllm is giving error, then create a seaparate environment specifcally for this file and try the following setup:
```bash
pip install vllm==0.9.1 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -U "ray[data,train,tune,serve]"
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```

## Data
1. The prompts are mentioned in the corresponding .ipynb files and in the folder `./prompts`
2. The data related to skill graph is present in `./skill_graph_data`
3. All datasets are listed here:
    - Context, CQA, CSQA related datasets:
        * [Stage 0](https://huggingface.co/datasets/Pavankalyan/stage0_c_all)
        * [Stage 1](https://huggingface.co/datasets/Pavankalyan/stage1_c_all)
        * [Stage 2](https://huggingface.co/datasets/Pavankalyan/stage2_c_all)
        * [Stage 3](https://huggingface.co/datasets/Pavankalyan/stage3_c_all)
        * [Stage 4](https://huggingface.co/datasets/Pavankalyan/stage4_c_all)
    - Instruct datasets:
        * [Stage 0](https://huggingface.co/datasets/Pavankalyan/stage0_instruct)
        * [Stage 1](https://huggingface.co/datasets/Pavankalyan/stage1_instruct)
        * [Stage 2](https://huggingface.co/datasets/Pavankalyan/stage2_instruct)
        * [Stage 3](https://huggingface.co/datasets/Pavankalyan/stage3_instruct)
        * [Stage 4](https://huggingface.co/datasets/Pavankalyan/stage4_instruct)
    - Data formatted for training
        * [Stage 0](https://huggingface.co/datasets/Pavankalyan/stage0_train)
        * [Stage 1](https://huggingface.co/datasets/Pavankalyan/stage1_train)
        * [Stage 2](https://huggingface.co/datasets/Pavankalyan/stage2_train)
        * [Stage 3](https://huggingface.co/datasets/Pavankalyan/stage3_train)
        * [Stage 4](https://huggingface.co/datasets/Pavankalyan/stage4_train)
4. The skill graph can be found here: [SkillGraph](https://huggingface.co/spaces/Pavankalyan/CurrLL-metadata)


