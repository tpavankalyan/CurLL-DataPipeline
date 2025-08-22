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
4.  **Hugging Face Hub Integration
