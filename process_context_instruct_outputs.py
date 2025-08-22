import os
import pandas as pd
from tqdm import tqdm
import json
import re
from datasets import load_dataset, concatenate_datasets

def load_all_files(base_path):
    root_dir = base_path+ "/raw"
    dataframes = []
    print("Loading all files from", root_dir)

    for i in range(4):
        fol1 = f"{root_dir}/chunk_{i}/"
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

    if not parsed:
        instr_match = re.search(r'"instruction"\s*:\s*"(.+?)"', cleaned, re.DOTALL)
        resp_match = re.search(r'"response"\s*:\s*"(.+?)"', cleaned, re.DOTALL)

        instruction = instr_match.group(1).strip() if instr_match else ""
        response = resp_match.group(1).strip() if resp_match else ""

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

    if not parsed:
        exp_match = re.search(r'"expanded_topic"\s*:\s*"([^"]+)"', cleaned, re.DOTALL)
        gen_match = re.search(r'"generated_text"\s*:\s*"(.+?)"\s*\}$', cleaned, re.DOTALL)

        expanded_topic = exp_match.group(1).strip() if exp_match else ""
        generated_text = gen_match.group(1).strip() if gen_match else ""

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

def extract_seeds_from_template(template: str, final_prompt: str, end_string) -> dict:
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
            for ph in placeholders:
                seeds[ph] = value
    
    return seeds

data_type = "instruct"
stage = 4
print(f"Processing {data_type} data for stage {stage}")

base_path = f"/datadrive/pavan/az_storage/CurLL_data/stages/stage{stage}/{data_type}"
df_all = load_all_files(base_path)

df_all = df_all.remove_columns(['batch_uuid', 'embeddings', 'generated_text', 'generated_tokens', 'messages', 'metrics', 'num_generated_tokens', 'num_input_tokens', 'params', 'prompt', 'prompt_token_ids', 'request_id', 'system', 'time_taken_llm'])

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

df_all = df_all.map(
    lambda x: parse_answer_context(x['answer']),
    num_proc=os.cpu_count(),
    desc="Parsing answers in parallel"
)

with open(base_path + "/prompt.json") as f:
    user_prompt_template = json.load(f)['user']
end_string = "Instructions:"

df_all = df_all.map(
    lambda x: extract_seeds_from_template(user_prompt_template, x['user'], end_string),
    num_proc=os.cpu_count(),
    desc="Extracting seeds in parallel"
)

dataset_name = f"{stage}_{data_type}"

df_all.push_to_hub(dataset_name)
print(f"Successfully uploaded: {dataset_name}")
