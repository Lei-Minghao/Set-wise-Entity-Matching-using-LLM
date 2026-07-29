import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Literal

import pandas as pd
from rich import print
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.contrib.concurrent import thread_map

from selecting_DBLP_ACM import Selecting
from utils import TokenUsage, set_gemini_rate_limit
import os
from pathlib import Path

project_root = Path("").resolve()
os.chdir(project_root)

# Maintain your original rate limit settings
set_gemini_rate_limit(60)

# ══════════════════════════════════════════════════════════════════════════════
# Union-Find for transitive closure (from compound.py)
# ══════════════════════════════════════════════════════════════════════════════

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def clusters(self) -> list[list]:
        groups: dict = {}
        for x in self.parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return list(groups.values())


def build_clusters(
    instances: list[dict],
    preds_lst: list[list[bool]],
    id_rights_per_instance: list[list],
) -> list[list]:
    """
    Apply transitive closure over all predicted matches to produce
    final entity clusters.
    """
    uf = UnionFind()

    # Register every ID so singletons are not lost
    for instance in instances:
        uf.add(instance["anchor_id"])
    for cand_ids in id_rights_per_instance:
        for cid in cand_ids:
            uf.add(cid)

    # Union anchor with every predicted match
    for instance, preds, cand_ids in zip(instances, preds_lst, id_rights_per_instance):
        for pred, cid in zip(preds, cand_ids):
            if pred:
                uf.union(instance["anchor_id"], cid)

    return uf.clusters()


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (from compound.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_instances(filepath: str) -> tuple[list[dict], pd.DataFrame, list[list]]:
    df = pd.read_csv(filepath)
    groupby = list(
        df.groupby("id_left")[["id_right", "record_left", "record_right", "label"]]
        .apply(lambda x: x.to_dict("list"))
        .to_dict()
        .items()
    )
    instances              = []
    id_rights_per_instance = []

    for anchor_id, v in groupby:
        instances.append({
            "anchor_id":  anchor_id,
            "anchor":     v["record_left"][0],
            "candidates": v["record_right"],
            "labels":     [bool(l) for l in v["label"]],
        })
        id_rights_per_instance.append(v["id_right"])

    return instances, df, id_rights_per_instance


# ══════════════════════════════════════════════════════════════════════════════
# Report builder (from compound.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_output_report(
    file: Path,
    instances: list[dict],
    preds_lst: list[list[bool]],
    df_raw: pd.DataFrame,
    labels_flat: list[bool],
    preds_flat: list[bool],
    timing: dict,
    ranking_usage: TokenUsage,
    selecting_usage: TokenUsage,
    final_clusters: list[list],
) -> str:
    lines = []
    sep  = "=" * 70
    dash = "-" * 70

    lines.append(sep)
    lines.append(f"DATASET: {file.stem}")
    lines.append(sep)

    # ── Classification report ────────────────────────────────────────────────
    lines.append("\nCLASSIFICATION REPORT")
    lines.append(dash)
    lines.append(classification_report(
        labels_flat[: len(preds_flat)], preds_flat,
        digits=4, labels=[False, True],
    ))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(labels_flat[: len(preds_flat)], preds_flat)
    lines.append("CONFUSION MATRIX")
    lines.append(dash)
    lines.append(f"                Predicted False   Predicted True")
    lines.append(f"  Actual False       {cm[0][0]:<18}{cm[0][1]}")
    lines.append(f"  Actual True        {cm[1][0]:<18}{cm[1][1]}")

    # ── Timing ───────────────────────────────────────────────────────────────
    lines.append(f"\n\nTIMING")
    lines.append(dash)
    lines.append(f"  Ranking (Precomputed): {timing['ranking_s']:.2f}s")
    lines.append(f"  Selecting (Gemini)   : {timing['selecting_s']:.2f}s")
    lines.append(f"  Total                : {timing['total_s']:.2f}s")

    # ── Token usage ──────────────────────────────────────────────────────────
    lines.append(f"\n\nTOKEN USAGE")
    lines.append(dash)

    if ranking_usage is not None:
        lines.append(f"  Ranking   ({ranking_usage.total_tokens} total)")
        lines.append(f"    Prompt     : {ranking_usage.prompt_tokens}")
        lines.append(f"    Completion : {ranking_usage.completion_tokens}")
    else:
        lines.append(f"  Ranking   (Loaded from JSON / no API tokens)")
        
    lines.append(f"  Selecting (Gemini / {selecting_usage.total_tokens} total)")
    lines.append(f"    Prompt     : {selecting_usage.prompt_tokens}")
    lines.append(f"    Completion : {selecting_usage.completion_tokens}")
    grand_total = (ranking_usage.total_tokens if ranking_usage is not None else 0) + selecting_usage.total_tokens
    lines.append(f"  Grand Total        : {grand_total}")

    # ── Per-anchor cluster detail ─────────────────────────────────────────────
    lines.append(f"\n\nPER-ANCHOR CLUSTER DETAIL")
    lines.append(sep)

    for instance, preds in zip(instances, preds_lst):
        anchor_id   = instance["anchor_id"]
        anchor_rows = df_raw[df_raw["id_left"] == anchor_id]
        anchor_cluster = anchor_rows["cluster_left"].iloc[0]

        true_match_ids = sorted(anchor_rows[anchor_rows["label"] == 1]["id_right"].tolist())
        true_clusters  = sorted(anchor_rows[anchor_rows["label"] == 1]["cluster_right"].unique().tolist())

        id_rights      = anchor_rows["id_right"].tolist()
        pred_match_ids = sorted([id_rights[i] for i, p in enumerate(preds) if p])
        pred_clusters  = sorted(
            anchor_rows[anchor_rows["id_right"].isin(pred_match_ids)]["cluster_right"]
            .unique().tolist()
        )

        true_set = set(true_match_ids)
        pred_set = set(pred_match_ids)
        missed   = sorted(true_set - pred_set)
        fp_ids   = sorted(pred_set - true_set)

        tp = len(true_set & pred_set)
        fp = len(fp_ids)
        fn = len(missed)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        lines.append(f"\nAnchor ID      : {anchor_id}")
        lines.append(f"Anchor Cluster : {anchor_cluster}")
        lines.append(dash)
        lines.append(f"  Ground Truth")
        lines.append(f"    Cluster(s)  : {true_clusters}")
        lines.append(f"    Matched IDs : {true_match_ids}")
        lines.append(f"  Prediction")
        lines.append(f"    Cluster(s)  : {pred_clusters}")
        lines.append(f"    Matched IDs : {pred_match_ids}")
        if missed:
            lines.append(f"  Missed  (FN)  : {missed}")
        if fp_ids:
            lines.append(f"  False + (FP)  : {fp_ids}")
        lines.append(f"  Scores  →  Precision: {precision*100:.2f}%  "
                     f"Recall: {recall*100:.2f}%  F1: {f1*100:.2f}%  "
                     f"(TP={tp}  FP={fp}  FN={fn})")

    # ── Final merged clusters (transitivity) ─────────────────────────────────
    lines.append(f"\n\nFINAL MERGED CLUSTERS (TRANSITIVITY)")
    lines.append(sep)
    sorted_clusters = sorted(final_clusters, key=len, reverse=True)
    for i, cluster in enumerate(sorted_clusters):
        lines.append(f"  Cluster {i+1} ({len(cluster)} members): {sorted(cluster)}")

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main Logic with JSON Integration
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking_file", type=str, default="./CoMEM/data/qwen_rankings_matching_walmart_amazon.json")
    parser.add_argument("--selecting-model", type=str, default=os.environ.get("GEMINI_MODEL", ""))
    parser.add_argument("--output", type=str, default="./CoMEM/results/compound_results_walmart_amazon.txt")
    args = parser.parse_args()

    # 1. Load the pre-computed rankings from the server
    if not Path(args.ranking_file).exists():
        raise FileNotFoundError(f"Could not find ranking file: {args.ranking_file}")
    
    with open(args.ranking_file, "r") as f:
        precomputed_rankings = json.load(f)

    # 2. Initialize your existing Selecting model
    selector = Selecting(model_name=args.selecting_model) 
    
    # Target dataset
    dataset_files = [Path("./CoMEM/data/sample_walmart_amazon_top5.csv")]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    results = {}
    all_report_blocks = []

    for file in dataset_files:
        dataset = file.stem
        print(f"\n[bold magenta]Processing Dataset: {dataset}[/bold magenta]")
        
        # Load raw data with IDs for transitive closure reporting
        instances, df_raw, id_rights_per_instance = load_instances(str(file))
        
        # Get specific rankings for this dataset
        rank_data = precomputed_rankings.get(dataset, {})

        if hasattr(selector, "token_usage"):
            selector.token_usage = TokenUsage()

        # FIX: Build a mapping from anchor_id to its list of candidate IDs
        # This allows conversion of string IDs from JSON back to integer indices.
        anchor_to_cand_ids = {
            inst["anchor_id"]: ids
            for inst, ids in zip(instances, id_rights_per_instance)
        }

        def process_instance(instance):
            anchor_id = str(instance["anchor_id"])
            original_candidates = instance["candidates"]
            
            # INTEGRATION: Fall back to original order if ID not found in JSON
            if anchor_id in rank_data:
                # rank_data[anchor_id] is a list of candidate ID strings, e.g., ["walmart_507", ...]
                candidate_ids_for_anchor = anchor_to_cand_ids[anchor_id]
                indexes_k = []
                for rid in rank_data[anchor_id]:
                    try:
                        idx = candidate_ids_for_anchor.index(rid)
                        indexes_k.append(idx)
                    except ValueError:
                        # ID from JSON not in current candidate list – skip
                        pass
            else:
                indexes_k = list(range(len(original_candidates)))
            
            # Reorder candidates based on LLM's previous JSON ranking
            instance_k = {
                "anchor": instance["anchor"],
                "candidates": [original_candidates[i] for i in indexes_k],
            }
            
            # Selection using Vertex AI model
            preds_k = selector(instance_k)
            
            # Core Fix from compound.py: Map the predictions back to the original index positions 
            # so they correctly align with df_raw in build_output_report
            preds = [False] * len(original_candidates)
            for i, pred in enumerate(preds_k):
                if i < len(indexes_k): # Ensure we map exactly K results
                    preds[indexes_k[i]] = pred
                    
            return preds

        print(f"  Running Selection via Vertex AI for {len(instances)} instances...")
        wall_start = time.perf_counter()
        
        preds_lst = list(thread_map(process_instance, instances, max_workers=10))
        
        wall_total = time.perf_counter() - wall_start

        preds  = [p for p_list in preds_lst for p in p_list]
        labels = [l for inst in instances for l in inst["labels"]]

        print(classification_report(labels[: len(preds)], preds, digits=4, labels=[False, True]))
        print(confusion_matrix(labels[: len(preds)], preds))
        print(f"  Selecting time : {wall_total:.2f}s")
        print(f"  Wall time      : {wall_total:.2f}s")
        if hasattr(selector, "token_usage"):
            print(f"  Selecting tokens: {selector.token_usage}")

        # 3. Transitive Closure (Clustering) using specific compound.py method
        final_clusters = build_clusters(instances, preds_lst, id_rights_per_instance)
        final_clusters.sort(key=len, reverse=True)

        print(f"\n  [bold cyan]Final merged clusters (transitivity):[/bold cyan]")
        for i, cluster in enumerate(final_clusters[:5]):
            print(f"    Cluster {i+1} ({len(cluster)} members): {sorted(cluster)}")

        report = classification_report(
            labels[: len(preds)], preds, output_dict=True, labels=[False, True]
        )["True"]
        
        if "support" in report:
            report.pop("support")
        results[dataset] = {k: v * 100 for k, v in report.items()}

        timing = {
            "ranking_s":   0.0, # Pre-computed via JSON
            "selecting_s": wall_total,
            "total_s":     wall_total,
        }
        
        # 4. Generate identical output report formats
        all_report_blocks.append(build_output_report(
            file, instances, preds_lst, df_raw, labels, preds,
            timing,
            None, # Pass None for ranking_usage since it doesn't apply here
            getattr(selector, "token_usage", TokenUsage()),
            final_clusters,
        ))

    # Final summary for all datasets
    if len(results) > 1:
        results["mean"] = {
            "precision": sum(v["precision"] for v in results.values()) / len(results),
            "recall":    sum(v["recall"]    for v in results.values()) / len(results),
            "f1-score":  sum(v["f1-score"]  for v in results.values()) / len(results),
        }

    if len(results) > 0:
        df_summary = pd.DataFrame.from_dict(results, orient="index")
        print("\n[bold green]Summary Table:[/bold green]")
        print(df_summary)

    # Output detailed report file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_report_blocks))
    print(f"\n[green]Output report saved to:[/green] {args.output}")