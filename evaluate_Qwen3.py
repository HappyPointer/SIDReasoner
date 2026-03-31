
import pandas as pd
import fire
import torch
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from transformers import GenerationConfig,  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper
from data_Qwen3 import EvalSidDataset
from accelerate import Accelerator
import random
import bitsandbytes as bnb



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
P = 998244353
MOD = int(1e9 + 9)
import numpy as np

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def main(
    base_model: str = "./output_dir/7Task-End2End_Qwen3_Games/final_checkpoint",
    train_file: str = "./data/Amazon_Games/train/Video_Games_5_2016-10-2018-11.csv",
    info_file: str = "./data/Amazon_Games/info/Video_Games_5_2016-10-2018-11.txt",
    category: str = "Video_Games",
    test_data_path: str = "./data/Amazon_Games/test/Video_Games_5_2016-10-2018-11_for_test.csv",
    result_json_data: str = "./temp/test_results_Qwen3.json",
    batch_size: int = 16,
    K: int = 0,
    seed: int = 42,
    length_penalty: float=0.0,
    max_new_tokens: int = 256,
    num_beams: int = 10,
    padding_side: str = "left",
):
    random.seed(seed)
    set_seed(seed)
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books", "Video_Games": "video games"}
    if category in category_dict:
        category = category_dict[category]
    else:
        category = "items"
    print(category)

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prefix_prompt = "\n</think>\n\n"
    prefix_index = 3
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Parse new format: semantic_id \t item_title \t item_id
        semantic_ids = [line.split('\t')[0].strip() for line in info]        
        # Format for tokenization
        info_semantic = [f'''{prefix_prompt}{_}\n''' for _ in semantic_ids]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # Create prefixID for semantic IDs (existing functionality)
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
    
    # Build hash_dict for semantic IDs (existing functionality)
    hash_dict = dict()
    # print(f"eos token: {tokenizer.eos_token_id}")
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])


    # Convert sets to lists for both dictionaries
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])


    def find_last_sublist(lst, sub):
        """Find the last occurrence of sublist in list"""
        if not sub:
            return None
        n, m = len(lst), len(sub)
        for start in range(n - m, -1, -1):
            if lst[start:start + m] == sub:
                return start
        return None
        

    sep_ids = tokenizer(prefix_prompt, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id

    # Define prefix constraint functions
    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        input_ids = input_ids.tolist()
        pos = find_last_sublist(input_ids, sep_ids)
        if pos is None:
            # "\n</think>\n\n" not detected
            raise Exception(f"Error: Prefix prompt not found in input IDs - {tokenizer.decode(input_ids)}.")
        
        # Calculate position after "\n</think>\n\n"
        pos_after_sep = pos + len(sep_ids)
        generated_after_sep = input_ids[pos_after_sep:]
        current_pos = len(generated_after_sep)

        if current_pos == 0:
            # First token after prefix prompt, the hash key should be predix.
            hash_number = get_hash(sep_ids)
            if hash_number in hash_dict:
                return hash_dict[hash_number]
            else:
                return [eos_id]
        else:
            # Subsequent tokens, the hash key should be generated tokens after prefix.
            hash_number = get_hash(generated_after_sep)
            if hash_number in hash_dict:
                return hash_dict[hash_number]
            else:
                return [eos_id]

        
    # Default to semantic constraints (backward compatibility)
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
    # prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_title
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    
    # val_dataset = EvalD3Dataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    val_dataset = EvalSidDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    
    encodings = [val_dataset[i] for i in range(len(val_dataset))]

    # encodings = encodings[:100]
    # encodings = [val_dataset[i] for i in indexes]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate(
            encodings,
            num_beams=10,
            max_new_tokens=64,
            length_penalty=1.0,
            padding_side="right",
            **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for  _ in encodings:
            L = len(_["input_ids"])
            if padding_side == "left":
                padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
                attention_mask.append([0] * (maxLen - L) + [1] * L) 
            elif padding_side == "right":
                padding_encodings["input_ids"].append(_["input_ids"] + [tokenizer.pad_token_id] * (maxLen - L))
                attention_mask.append([1] * L + [0] * (maxLen - L))
            else:
                raise ValueError("Invalid padding_side. Choose 'left' or 'right'.")
            
    
        with torch.no_grad():
            generate_kwargs = {
                "input_ids": torch.tensor(padding_encodings["input_ids"]).to(device),
                "attention_mask": torch.tensor(attention_mask).to(device),
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "num_return_sequences": num_beams,
                "output_scores": True,
                "return_dict_in_generate": True,
                "early_stopping": True,
                "length_penalty": length_penalty,
                # "temperature": args.temperature,
                "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            }

            generation_output = model.generate(
                **generate_kwargs
            )
       
        batched_completions = generation_output.sequences[:, maxLen:]
       
        
        if base_model.lower().find("llama") > -1:
            output_raw = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output_raw = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
            
        output = [_.split("Response:\n")[-1].strip() for _ in output_raw]
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]

        if "" in real_outputs:
            print("Warning: Empty string detected in outputs.")
        return real_outputs
    
    model = model.to(device)

    from tqdm import tqdm
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])

    
    for idx, encodings in enumerate(tqdm(new_encodings)):
        # Use standard evaluation
        output = evaluate(encodings, max_new_tokens=max_new_tokens, num_beams=num_beams, length_penalty=length_penalty, padding_side=padding_side)
        
        outputs = outputs + output
       
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
  

    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')  
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)





