# CurLL-DataPipeline

This repository contains the data generation pipeline for the paper "Curriculum learning of Language Models". The pipeline is designed to process and prepare datasets for training language models in a curriculum learning setup.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Data Pipeline](#data-pipeline)
- [Configuration](#configuration)

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
-   **`skill_graph.ipynb`**: A Jupyter notebook that likely deals with creating or visualizing a skill graph, which is a key component of the curriculum learning strategy.
-   **`templates.ipynb`**: A Jupyter notebook that contains templates used in the data generation process.
-   **`process_context_instruct_outputs.py`**: A Python script that processes the outputs of the data generation model. It parses the generated text, extracts relevant information, and prepares it for the next stage of the pipeline.
-   **`process_cqa_csqa_outputs.ipynb`**: A Jupyter notebook for processing outputs related to CQA (Conversational Question Answering) and CSQA (Commonsense Question Answering).
-   **`format_train.py`**: A Python script that formats the processed data into a consistent format for training. It combines different datasets, applies formatting, and prepares the data for the final stage.
-   **`create_train_val_test.ipynb`**: A Jupyter notebook that takes the formatted training data and splits it into training, validation, and test sets.
-   **`.gitignore`**: A file that specifies which files and directories to ignore in a Git repository.
-   **`README.md`**: This file, which provides an overview of the project and instructions on how to use it.

## Setup

To run this data pipeline, you need to have Python 3 installed, along with the required packages.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tpavankalyan/CurLL-DataPipeline.git
    cd CurLL-DataPipeline
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not currently present in the repository. It is recommended to create one to ensure consistent environments.)*

## Data Pipeline

The data pipeline is executed as a series of steps, starting from seed preparation and ending with the creation of the final training datasets.

1.  **Prepare Seed Data**: Run the `prepare_seed.ipynb` notebook to generate the initial seed data.
2.  **Process Outputs**: Use the `process_context_instruct_outputs.py` script and the `process_cqa_csqa_outputs.ipynb` notebook to process the raw outputs from the data generation model.
3.  **Format Training Data**: Run the `format_train.py` script to format the processed data into a consistent training format.
4.  **Create Splits**: Use the `create_train_val_test.ipynb` notebook to create the final training, validation, and test splits.

## Configuration

The scripts and notebooks in this repository may require some configuration, such as setting file paths, API keys, or other parameters.

-   **`format_train.py`**: This script has a configuration section at the top where you can set the `STAGE`, `MODEL_NAME`, `SAVE_PATH`, and `HF_TOKEN`.
-   **`process_context_instruct_outputs.py`**: This script requires you to set the `base_path` variable to the location of your raw data.

Please refer to the individual files for more details on their specific configurations.
