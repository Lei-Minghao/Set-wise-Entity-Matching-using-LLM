import argparse
import json
import time
import torch
from pathlib import Path
import pandas as pd
from rich import print
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from comparing_qwen_solute import ComparingSQ
from matching_qwen_solute import MatchingSQ

import os
from pathlib import Path

project_root = Path("").resolve()
os.chdir(project_root)

def load_instances(filepath: str) -> list[dict]:
    df = pd.read_csv(filepath)
    groupby = list(df.groupby("id_left")[["id_right", "record_left", "record_right", "label"]].apply(lambda x: x.to_dict("list")).to_dict().items())
    return [{"anchor_id": anchor_id, "anchor": v["record_left"][0], "candidates": v["record_right"]} for anchor_id, v in groupby]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--strategy", type=str, default="matching", choices=["matching", "comparing"])
    parser.add_argument("--topk", type=int, default=10)
    # ADDED: Command line argument for input file so it stops skipping silently
    parser.add_argument("--input", type=str, default="./CoMEM/data/sample_solute_top25.csv") # <-- CHANGE THIS TO YOUR CSV
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

    # Use the parameter from the command line argument
    target_file = Path(args.input)
    all_rankings = {}

    if not target_file.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: The file '{args.input}' does not exist in this directory!")

    print(f"[bold magenta]Ranking Dataset: {target_file.stem}[/bold magenta]")
    instances = load_instances(str(target_file))

    results = []
    # We use tqdm here to see progress in your Slurm .out file
    for instance in tqdm(instances, desc="Ranking Instances"):
        if args.strategy == "matching":
            indexes = ranker.pointwise_rank(instance)
        else:
            indexes = ranker.pairwise_rank(instance, topK=args.topk)
        
        results.append((str(instance["anchor_id"]), indexes[:args.topk]))

    all_rankings[target_file.stem] = {anchor_id: indexes for anchor_id, indexes in results}

    output_file = f"./CoMEM/data/qwen_rankings_{args.strategy}_solute.json"
    with open(output_file, "w") as f:
        json.dump(all_rankings, f, indent=2)
    print(f"[bold green]Success![/bold green] Results saved to {output_file}")