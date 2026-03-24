import argparse
import os
import re
import datasets
from verl.utils.hdfs_io import copy, makedirs
from torch.utils.data import Dataset
import random
import pandas as pd
import json
from tqdm import tqdm

# This dataset is used for reasoning activation task.
# The learning objective is to generate reasoning and answer given user history.
class Reasoning_RL_Dataset(Dataset):
    def __init__(
        self,
        data_file,
        item_file,
        index_file,
        tokenizer,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="",
        dedup=False,
    ):
        """
        Fusion dataset combining sequence recommendation with item features.
        Uses semantic IDs for user history, outputs item titles or descriptions.
        
        Args:
            train_file: Path to CSV file with sequence data
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)
        
        # Load sequence data
        self.data = pd.read_csv(data_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        
        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        
        self.tokenizer = tokenizer
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        # Build sid2title and sid2description mappings
        self.sid2title = {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']                                
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
        
        self.get_inputs()
    
    def __len__(self):
        return len(self.data)
    
    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Let's think step by step before making recommendation. Directly output the item SID after thinking."
    
    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        history_str = ", ".join(history_item_sid)
        
        target_sid = row['item_sid']
        
        # Use the new sid2title and sid2description mappings
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid
        
        # Check for deduplication
        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid
        
        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_sid": target_sid,
            "dedup": is_duplicate,
        }
    
    def generate_formatted_prompt(self, prompt, response):
        return f"""{prompt}"""
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Can you recommend the next item for the user based on their interaction history?
"""  
        # tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history_data = self.get_history(self.data.iloc[idx])
        
        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None
        
        # Randomly choose between title and description tasks
        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['target_sid']
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        assistant_response = f"{target}"

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": formatted_prompt},
        ]
        return {
            "input": messages,
            "target": assistant_response,
        }
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.pre(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs
    
    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []
    
    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.inputs[idx]
        return self.pre(idx)



# Convert torch dataset to parquet manually
def convert_to_verl_format(ds, split, out_path):
    rows = []
    for idx in range(len(ds)):
        example = ds[idx]
        question_raw = example["input"]
        answer_raw = example["target"]

        rows.append({
            "data_source": data_source,
            "prompt": question_raw,      # must be list[dict]
            "ability": "Recommendation",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer_raw
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            }
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}")


# Extract the model output after <\think>
def extract_solution(solution_str):
    solution = re.search("<\\think> (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("<\think>")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", default="./data/Amazon_mix3_IOV/index/Amazon_mix3_IOV.integrated_narrative.csv")
    parser.add_argument("--eval_data_dir", default="./data/Amazon_mix3_IOV/test/Amazon_mix3_IOV.csv")
    parser.add_argument("--local_dir", default="./data/Amazon_mix3_IOV/rec_reasoning_verl/Amazon_mix3_IOV")
    parser.add_argument("--item_file", default="./data/Amazon_mix3_IOV/index/Amazon_mix3_IOV.item.json")
    parser.add_argument("--index_file", default="./data/Amazon_mix3_IOV/index/Amazon_mix3_IOV.index.json")
    args = parser.parse_args()

    data_source = "rec/Amazon_mix3_IOV"
    train_dataset = Reasoning_RL_Dataset(
        data_file=args.train_data_dir,
        item_file=args.item_file,
        index_file=args.index_file,
        tokenizer=None,
        max_len=2048,
        sample=-1,
        test=False,
        seed=0,
        category="Amazon_mix3_IOV",
        dedup=False,
    )

    eval_dataset = Reasoning_RL_Dataset(
        data_file=args.eval_data_dir,
        item_file=args.item_file,
        index_file=args.index_file,
        tokenizer=None,
        max_len=2048,
        sample=-1,
        test=True,
        seed=0,
        category="Amazon_mix3_IOV",
        dedup=False,
    )
    local_dir = args.local_dir
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    train_save_path = os.path.join(args.local_dir, "train.parquet")
    test_save_path = os.path.join(args.local_dir, "test.parquet")

    convert_to_verl_format(train_dataset, split="train", out_path=train_save_path)
    convert_to_verl_format(eval_dataset, split="test", out_path=test_save_path)


    # Debugging
    df_train = pd.read_parquet(train_save_path)
    df_test = pd.read_parquet(test_save_path)

    print("=== First 3 Data in Training Set ===")
    print(df_train.head(3).to_dict(orient="records"))

    print("\n=== First 3 Data in Test Set ===")
    print(df_test.head(3).to_dict(orient="records"))
