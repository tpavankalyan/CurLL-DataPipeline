"""
This script is responsible for processing raw outputs from a data generation model,
specifically for "context" and "instruct" data types within the CurLL data pipeline.

It performs the following key operations:
1.  **Loads Raw Data**: Reads parquet files from a specified base path, typically
    containing outputs from an LLM.
2.  **Parses Answers**: Extracts structured information (instructions, responses,
    expanded topics, generated text) from the raw text outputs, handling both
    JSON and regex-based parsing for robustness.
3.  **Extracts Seeds**: Compares the final prompts with a template to extract
    seed values that were used to generate the data.
4.  **Uploads to Hugging Face Hub**: Pushes the processed and enriched dataset
    to the Hugging Face Hub for further use in training.

This script is a critical step in transforming raw model outputs into a clean,
structured dataset suitable for curriculum learning.
"""
import os
import pandas as pd
from tqdm import tqdm
import json
import re
from datasets import load_dataset, concatenate_datasets

def load_all_files(base_path):
    """
    Loads all parquet files from specified chunks within a base path and concatenates them into a single dataset.

    Args:
        base_path (str): The base directory where raw data chunks are stored.
                         Expected structure: `{base_path}/raw/chunk_i/*.parquet`

    Returns:
        datasets.Dataset: A concatenated dataset containing data from all specified chunks.
    """
    root_dir = base_path + "/raw"
    dataframes = []
    print("Loading all files from", root_dir)

    # Iterate through chunks (assuming 4 chunks as per original code)
    for i in range(4):
        fol1 = f"{root_dir}/chunk_{i}/"
        # Load parquet files from the current chunk
        hf_df = load_dataset(
            "parquet",
            data_files=os.path.join(fol1, "*.parquet"),
            streaming=False
        )['train']
        dataframes.append(hf_df)
                
    print("Completed: loading all files from", root_dir)
    print("Concatenating all dataframes...")

    return concatenate_datasets(dataframes)

def parse_answer_instruct(text: str):
    """
    Parses the answer text for instruction-response pairs, handling JSON and regex-based extraction.

    Args:
        text (str): The raw text containing instruction and response.

    Returns:
        dict: A dictionary with 'invalid_case' (1 if JSON parsing failed, 0 otherwise),
              'instruction', and 'response'.
    """
    invalid_case = 0
    parsed = {}

    # Strip whitespace and common markdown code block wrappers
    cleaned = text.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^```', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        invalid_case = 1

    # If JSON parsing fails, attempt to extract using regex
    if not parsed:
        instr_match = re.search(r'"instruction"\s*:\s*"(.+?)"', cleaned, re.DOTALL)
        resp_match = re.search(r'"response"\s*:\s*"(.+?)"', cleaned, re.DOTALL)

        instruction = instr_match.group(1).strip() if instr_match else ""
        response = resp_match.group(1).strip() if resp_match else ""

        # Handle escaped newlines and quotes
        response = response.replace('\\n', '\n').replace('\\"', '"')

        parsed = {
            "instruction": instruction,
            "response": response
        }

    return {
        "invalid_case": invalid_case,
        "instruction": parsed.get("instruction", ""),
        "response": parsed.get("response", "")
    }

def parse_answer_context(text: str):
    """
    Parses the answer text for context-based outputs, handling JSON and regex-based extraction.

    Args:
        text (str): The raw text containing expanded topic and generated text.

    Returns:
        dict: A dictionary with 'invalid_case' (1 if JSON parsing failed, 0 otherwise),
              'expanded_topic', and 'output' (generated text).
    """
    invalid_cases = 0
    parsed = {}

    # Remove common wrappers
    cleaned = text.strip()
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^```', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        invalid_cases = 1

    # If JSON parsing fails, attempt to extract using regex
    if not parsed:
        exp_match = re.search(r'"expanded_topic"\s*:\s*"([^"]+)"', cleaned, re.DOTALL)
        gen_match = re.search(r'"generated_text"\s*:\s*"(.+?)"\s*\}$', cleaned, re.DOTALL)

        expanded_topic = exp_match.group(1).strip() if exp_match else ""
        generated_text = gen_match.group(1).strip() if gen_match else ""

        # Handle escaped newlines and quotes
        generated_text = generated_text.replace('\\n', '\n').replace('\\"', '"')

        parsed = {
            "expanded_topic": expanded_topic,
            "generated_text": generated_text
        }

    return {
        "invalid_case": invalid_cases,
        "expanded_topic": parsed.get("expanded_topic", ""),
        "output": parsed.get("generated_text", "")
    }

def extract_seeds_from_template(template: str, final_prompt: str, end_string: str) -> dict:
    """
    Extracts seed values from a final prompt by comparing it against a template.
    This function assumes that the template and final prompt share a common header
    before the `end_string` marker (e.g., "Instructions:").

    Args:
        template (str): The template string with placeholders (e.g., "{id}").
        final_prompt (str): The actual prompt with values filled in.
        end_string (str): A string that marks the end of the header section in the prompt.

    Returns:
        dict: A dictionary where keys are placeholder names and values are the extracted seed values.
    """
    # Keep only the lines before 'Instructions:'
    template_header = template.split(end_string, 1)[0].strip()
    final_header = final_prompt.split(end_string, 1)[0].strip()

    template_lines = template_header.splitlines()
    final_lines = final_header.splitlines()
    
    seeds = {}
    
    for tmpl_line, final_line in zip(template_lines, final_lines):
        # Look for placeholders in the template line
        placeholders = re.findall(r'\{(\w+)\}', tmpl_line)
        if not placeholders:
            continue  # Skip lines with no placeholders
        
        # Extract value: template => e.g., "- ID: {id}" ; final => "- ID: i215"
        tmpl_prefix = tmpl_line.split('{')[0]
        value = final_line.replace(tmpl_prefix, '').strip()
        
        if len(placeholders) == 1:
            seeds[placeholders[0]] = value
        else:
            # For lines like "- (Word, Part of speech): {word_list}"
            # This part might need more sophisticated parsing depending on the actual template structure
            for ph in placeholders:
                seeds[ph] = value
    
    return seeds

# --- Main execution starts here ---
# Define the type of data and stage to process
data_type = "instruct"  # Can be "instruct" or "context"
stage = 4               # The stage number for the data pipeline

print(f"Processing {data_type} data for stage {stage}")

# Construct the base path for loading raw data
# IMPORTANT: Update this path if your data is stored elsewhere.
# Example: "/path/to/your/CurLL_data/stages/stage{stage}/{data_type}"
base_path = f"/datadrive/pavan/az_storage/CurLL_data/stages/stage{stage}/{data_type}"
df_all = load_all_files(base_path)

# Remove unnecessary columns from the dataset to reduce memory footprint and simplify data
df_all = df_all.remove_columns([
    'batch_uuid', 'embeddings', 'generated_text', 'generated_tokens', 'messages', 
    'metrics', 'num_generated_tokens', 'num_input_tokens', 'params', 'prompt', 
    'prompt_token_ids', 'request_id', 'system', 'time_taken_llm'
])

# Apply the appropriate parsing function based on the data_type
if data_type == "instruct":
    df_all = df_all.map(
        lambda x: parse_answer_instruct(x['answer']),
        num_proc=os.cpu_count(),
        desc="Parsing answers in parallel"
    )
else:
    df_all = df_all.map(
        lambda x: parse_answer_context(x['answer']),
        num_proc=os.cpu_count(),
        desc="Parsing answers in parallel"
    )

# NOTE: There seems to be a redundant call to parse_answer_context here.
# Depending on the intended logic, this might need to be removed or conditionally applied.
df_all = df_all.map(
    lambda x: parse_answer_context(x['answer']),
    num_proc=os.cpu_count(),
    desc="Parsing answers in parallel"
)

# Load the user prompt template from a JSON file
# IMPORTANT: Ensure 'prompt.json' exists in the `base_path` directory.
with open(base_path + "/prompt.json") as f:
    user_prompt_template = json.load(f)['user']
end_string = "Instructions:" # Marker to identify the end of the template header

# Extract seed values from the prompts using the template
df_all = df_all.map(
    lambda x: extract_seeds_from_template(user_prompt_template, x['user'], end_string),
    num_proc=os.cpu_count(),
    desc="Extracting seeds in parallel"
)

# Define the dataset name for Hugging Face Hub
dataset_name = f"Pavankalyan/stage{stage}_{data_type}" # Assuming 'Pavankalyan' is the user's HF namespace

# Push the processed dataset to Hugging Face Hub
# Ensure you are logged in to Hugging Face and have write permissions to the namespace.
df_all.push_to_hub(dataset_name)
print(f"Successfully uploaded: {dataset_name}")
