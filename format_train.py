from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import os
import json
from tqdm import tqdm

# Configuration
STAGE = 3
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
SAVE_PATH = "/datadrive/pavan/experiments"
HF_TOKEN = "your_huggingface_token_here"  # Replace with your actual token

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def format_cqa_csqa(batch):
    """Format CQA/CSQA dataset with conversation turns"""
    cqa, csqa, context_text = batch['cqa'], batch['csqa'], batch['output']
    eos_token = tokenizer.eos_token
    
    # Format CQA turns
    form_cqa = [
        f"<|system|>\n{context_text}\n<|user|>\n{turns['question']}\n<|assistant|>\n{turns['answer']}\n{eos_token}" 
        for turns in cqa
    ]
    
    # Format CSQA turns
    form_csqa = [
        f"<|system|>\n{context_text}\n<|user|>\n{turns['question']}\n<|assistant|>\n{turns['answer']}\n{eos_token}" 
        for turns in csqa
    ]
    
    # Combine all formatted texts
    texts = form_cqa + form_csqa
    return {"text": texts}

def format_ir(batch):
    """Format instruction-response dataset"""
    ins = batch['instruction']
    res = batch['response']
    formatted = f"<|system|>\n\n<|user|>\n{ins}\n<|assistant|>\n{res}\n{tokenizer.eos_token}"
    return {"text": formatted}

def flatten_nested_texts(dataset):
    """Flatten nested text lists from CQA/CSQA formatting"""
    flat_texts = []
    for example in tqdm(dataset, desc="Flattening texts"):
        if isinstance(example['text'], list):
            flat_texts.extend(example['text'])
        else:
            flat_texts.append(example['text'])
    return flat_texts

def main():
    print(f"Processing stage {STAGE} datasets...")
    
    # Load datasets
    print("Loading datasets...")
    ds1 = load_dataset(f"Pavankalyan/stage{STAGE}_c_all", split="train")
    ds2 = load_dataset(f"Pavankalyan/stage{STAGE}_instruct", split="train")
    
    print(f"Dataset 1 (CQA/CSQA) size: {len(ds1)}")
    print(f"Dataset 2 (Instruct) size: {len(ds2)}")
    
    # Format datasets
    print("Formatting CQA/CSQA dataset...")
    ds1_formatted = ds1.map(
        format_cqa_csqa, 
        num_proc=os.cpu_count(), 
        remove_columns=ds1.column_names,
        desc="Formatting CQA/CSQA"
    )
    
    print("Formatting instruction dataset...")
    ds2_formatted = ds2.map(
        format_ir, 
        num_proc=os.cpu_count(), 
        remove_columns=ds2.column_names,
        desc="Formatting instructions"
    )
    
    # Flatten the nested texts from ds1
    print("Flattening CQA/CSQA texts...")
    flat_texts = flatten_nested_texts(ds1_formatted)
    
    print(f"Total flattened texts from CQA/CSQA: {len(flat_texts)}")
    print(f"Sample text:\n{flat_texts[0][:200]}...")
    
    # Create temporary JSONL file for memory efficiency
    os.makedirs(SAVE_PATH, exist_ok=True)
    jsonl_path = f"{SAVE_PATH}/stage{STAGE}_cqa_csqa.jsonl"
    
    print("Writing flattened texts to JSONL...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for text in tqdm(flat_texts, desc="Writing JSONL"):
            f.write(json.dumps({"text": text}) + "\n")
    
    # Load flattened dataset from JSONL
    print("Loading flattened dataset...")
    ds1_flat = Dataset.from_json(jsonl_path)
    
    # Combine datasets
    print("Combining datasets...")
    final_dataset = concatenate_datasets([ds1_flat, ds2_formatted])
    
    print(f"Combined dataset size: {len(final_dataset)}")
    
    # Shuffle dataset
    print("Shuffling dataset...")
    final_dataset = final_dataset.shuffle(seed=42)
    
    # Push to hub
    if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
        print(f"Pushing to hub: Pavankalyan/stage{STAGE}_train")
        final_dataset.push_to_hub(f"Pavankalyan/stage{STAGE}_train", token=HF_TOKEN)
        print("Dataset pushed successfully!")
    else:
        print("No valid HF token provided. Skipping push to hub.")
        print("To push to hub, set your HuggingFace token in the HF_TOKEN variable.")
    
    # Save locally as backup
    local_save_path = f"{SAVE_PATH}/stage{STAGE}_final_dataset"
    print(f"Saving locally to {local_save_path}")
    final_dataset.save_to_disk(local_save_path)
    
    # Clean up temporary file
    os.remove(jsonl_path)
    print("Temporary JSONL file cleaned up.")
    
    # Print sample from final dataset
    print("\nSample from final dataset:")
    print(final_dataset[0]['text'][:300] + "...")
    
    print(f"\nProcessing complete! Final dataset has {len(final_dataset)} examples.")

if __name__ == "__main__":
    main()
