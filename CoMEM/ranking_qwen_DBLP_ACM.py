import argparse
import json
import time
import torch
from pathlib import Path
import pandas as pd
from rich import print
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from comparing_qwen_DBLP_ACM import ComparingSQ
from matching_qwen_DBLP_ACM import MatchingSQ

import os
from pathlib import Path

project_root = Path("").resolve()
os.chdir(project_root)

def load_instances(filepath: str) -> list[dict]:
    df = pd.read_csv(filepath)
    # Group by id_left, aggregate lists of id_right, record_right, label
    # Note: record_left is the same for all rows of a given id_left
    groupby = list(df.groupby("id_left")[["id_right", "record_right", "label"]].apply(lambda x: x.to_dict("list")).to_dict().items())
    return [
        {
            "anchor_id": anchor_id,
            "anchor": df[df["id_left"] == anchor_id]["record_left"].iloc[0],
            "candidates": v["record_right"],
            "candidate_ids": v["id_right"],
            "labels": v["label"]  # optional, not used by MatchingSQ but kept for completeness
        }
        for anchor_id, v in groupby
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--strategy", type=str, default="matching", choices=["matching", "comparing"])
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    # 1. Load Model and Tokenizer (Using BFloat16 for stability)
    print(f"[bold yellow]Loading model to GPU: {args.model}...[/bold yellow]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )

    # 2. Initialize Ranker
    if args.strategy == "matching":
        ranker = MatchingSQ(model, tokenizer, device, model_name=args.model)
    else:
        ranker = ComparingSQ(model, tokenizer, device, model_name=args.model)

    # Adjust dataset file name as needed
    dataset_files = [Path("./CoMEM/data/sample_DBLP_ACM_top5.csv")]  # <-- CHANGE THIS TO YOUR CSV
    all_rankings = {}

    for file in dataset_files:
        if not file.exists(): continue
        print(f"[bold magenta]Ranking Dataset: {file.stem}[/bold magenta]")
        instances = load_instances(str(file))

        results = []
        for instance in tqdm(instances, desc="Ranking Instances"):
            if args.strategy == "matching":
                ranked_items = ranker.pointwise_rank(instance)
            else:
                # For comparing strategy, catch integer indexes and map them to string IDs
                outputs = ranker.pairwise_rank(instance, topK=args.topk)
                if outputs and isinstance(outputs[0], int):
                    ranked_items = [instance["candidate_ids"][idx] for idx in outputs]
                else:
                    ranked_items = outputs
            
            results.append((str(instance["anchor_id"]), ranked_items[:args.topk]))

        all_rankings[file.stem] = {anchor_id: list_ids for anchor_id, list_ids in results}

    output_file = f"./CoMEM/data/qwen_rankings_{args.strategy}_DBLP_ACM.json"
    with open(output_file, "w") as f:
        json.dump(all_rankings, f, indent=2)
    print(f"[bold green]Success![/bold green] Results saved to {output_file}")