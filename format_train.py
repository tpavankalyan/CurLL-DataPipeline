from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import os
import json
from tqdm import tqdm

"""
This script processes and formats datasets for a specific training stage in the CurLL project.

It performs the following main steps:
1.  Loads two datasets from the Hugging Face Hub: one with CQA/CSQA data and another with instruction-response pairs.
2.  Formats the CQA/CSQA data into conversational turns and the instruction data into a system-user-assistant format.
3.  Flattens nested text lists from the CQA/CSQA dataset for consistent processing.
4.  Combines the formatted datasets into a single final dataset.
5.  Shuffles the combined dataset.
6.  Optionally pushes the final dataset to the Hugging Face Hub (requires a valid token).
7.  Saves a local backup of the final dataset.
8.  Cleans up temporary files created during the process.

This script is designed to be run for different stages of the curriculum learning pipeline,
as specified by the `STAGE` variable.
"""

# Configuration
STAGE = 3
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
SAVE_PATH = "/datadrive/pavan/experiments" # Path to save temporary files and final dataset
HF_TOKEN = "your_huggingface_token_here"  # Replace with your actual Hugging Face token for pushing to hub

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def format_cqa_csqa(batch):
    """
    Formats CQA/CSQA dataset examples into a conversational turn structure.

    Args:
        batch (dict): A dictionary containing 'cqa', 'csqa', and 'output' (context text).

    Returns:
        dict: A dictionary with a 'text' key containing a list of formatted conversation strings.
    """
    cqa, csqa, context_text = batch['cqa'], batch['csqa'], batch['output']
    eos_token = tokenizer.eos_token
    
    # Format CQA turns: system context, user question, assistant answer
    form_cqa = [
        f"<|system|>\n{context_text}\n<|user|>\n{turns['question']}\n<|assistant|>\n{turns['answer']}\n{eos_token}" 
        for turns in cqa
    ]
    
    # Format CSQA turns: system context, user question, assistant answer
    form_csqa = [
        f"<|system|>\n{context_text}\n<|user|>\n{turns['question']}\n<|assistant|>\n{turns['answer']}\n{eos_token}" 
        for turns in csqa
    ]
    
    # Combine all formatted texts from both CQA and CSQA
    texts = form_cqa + form_csqa
    return {"text": texts}

def format_ir(batch):
    """
    Formats instruction-response dataset examples into a system-user-assistant structure.

    Args:
        batch (dict): A dictionary containing 'instruction' and 'response'.

    Returns:
        dict: A dictionary with a 'text' key containing the single formatted string.
    """
    ins = batch['instruction']
    res = batch['response']
    # Format instruction-response: system (empty), user instruction, assistant response
    formatted = f"<|system|>\n\n<|user|>\n{ins}\n<|assistant|>\n{res}\n{tokenizer.eos_token}"
    return {"text": formatted}

def flatten_nested_texts(dataset):
    """
    Flattens a dataset where the 'text' column might contain nested lists of strings
    into a single list of strings. This is useful after formatting functions that
    return multiple texts per input example.

    Args:
        dataset (datasets.Dataset): The dataset with a 'text' column, potentially containing lists.

    Returns:
        list: A flat list of all text strings from the dataset.
    """
    flat_texts = []
    for example in tqdm(dataset, desc="Flattening texts"):
        if isinstance(example['text'], list):
            flat_texts.extend(example['text'])
        else:
            flat_texts.append(example['text'])
    return flat_texts

def main():
    """
    Main function to execute the data processing and formatting pipeline.
    """
    print(f"Processing stage {STAGE} datasets...")
    
    # Load datasets from Hugging Face Hub
    # IMPORTANT: Ensure these datasets exist under the 'Pavankalyan' namespace
    # or update the dataset paths accordingly.
    print("Loading datasets...")
    ds1 = load_dataset(f"Pavankalyan/stage{STAGE}_c_all", split="train")
    ds2 = load_dataset(f"Pavankalyan/stage{STAGE}_instruct", split="train")
    
    print(f"Dataset 1 (CQA/CSQA) size: {len(ds1)}")
    print(f"Dataset 2 (Instruct) size: {len(ds2)}")
    
    # Format datasets in parallel using available CPU cores
    print("Formatting CQA/CSQA dataset...")
    ds1_formatted = ds1.map(
        format_cqa_csqa, 
        num_proc=os.cpu_count(), 
        remove_columns=ds1.column_names, # Remove original columns to keep only 'text'
        desc="Formatting CQA/CSQA"
    )
    
    print("Formatting instruction dataset...")
    ds2_formatted = ds2.map(
        format_ir, 
        num_proc=os.cpu_count(), 
        remove_columns=ds2.column_names, # Remove original columns to keep only 'text'
        desc="Formatting instructions"
    )
    
    # Flatten the nested texts from the CQA/CSQA formatted dataset
    print("Flattening CQA/CSQA texts...")
    flat_texts = flatten_nested_texts(ds1_formatted)
    
    print(f"Total flattened texts from CQA/CSQA: {len(flat_texts)}")
    print(f"Sample text:\n{flat_texts[0][:200]}...")
    
    # Create a temporary directory and JSONL file for memory-efficient loading
    os.makedirs(SAVE_PATH, exist_ok=True)
    jsonl_path = f"{SAVE_PATH}/stage{STAGE}_cqa_csqa.jsonl"
    
    print("Writing flattened texts to JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for text in tqdm(flat_texts, desc="Writing JSONL"):
            f.write(json.dumps({"text": text}) + "\n")
    
    # Load the flattened dataset from the temporary JSONL file
    print("Loading flattened dataset...")
    ds1_flat = Dataset.from_json(jsonl_path)
    
    # Combine the flattened CQA/CSQA dataset with the formatted instruction dataset
    print("Combining datasets...")
    final_dataset = concatenate_datasets([ds1_flat, ds2_formatted])
    
    print(f"Combined dataset size: {len(final_dataset)}")
    
    # Shuffle the final dataset for randomness
    print("Shuffling dataset...")
    final_dataset = final_dataset.shuffle(seed=42)
    
    # Push the final dataset to Hugging Face Hub if a valid token is provided
    if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
        print(f"Pushing to hub: Pavankalyan/stage{STAGE}_train")
        final_dataset.push_to_hub(f"Pavankalyan/stage{STAGE}_train", token=HF_TOKEN)
        print("Dataset pushed successfully!")
    else:
        print("No valid HF token provided. Skipping push to hub.")
        print("To push to hub, set your HuggingFace token in the HF_TOKEN variable.")
    
    # Save the final dataset locally as a backup
    local_save_path = f"{SAVE_PATH}/stage{STAGE}_final_dataset"
    print(f"Saving locally to {local_save_path}")
    final_dataset.save_to_disk(local_save_path)
    
    # Clean up the temporary JSONL file
    os.remove(jsonl_path)
    print("Temporary JSONL file cleaned up.")
    
    # Print a sample from the final dataset for verification
    print("\nSample from final dataset:")
    print(final_dataset[0]['text'][:300] + "...")
    
    print(f"\nProcessing complete! Final dataset has {len(final_dataset)} examples.")

if __name__ == "__main__":
    main()
