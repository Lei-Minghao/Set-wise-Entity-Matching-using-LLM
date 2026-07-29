import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path

project_root = Path("").resolve()
os.chdir(project_root)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = "./dataset/sample_DBLP_ACM.csv"
OUTPUT_PATH  = "./CoMEM/data/sample_DBLP_ACM_top5.csv"
TOP_K        = 5        # number of candidates to retrieve per anchor
ID_COL       = "id"
CLUSTER_COL  = "cluster_id"
SKIP_COLS    = {"cluster_id", "id"}

# If set, only these IDs will be used as anchors.
# If empty, every record is used as an anchor.
ANCHOR_IDS   = []

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def make_record(row: pd.Series, columns: list[str]) -> str:
    """Concatenate non-empty field values as 'col: value' strings."""
    parts = []
    for col in columns:
        val = str(row[col]).strip()
        if val and val.lower() != "nan":
            parts.append(f"{col}: {val}")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Blocking
# ══════════════════════════════════════════════════════════════════════════════

def blocking(
    df: pd.DataFrame,
    anchor_ids: list,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    text_cols = [c for c in df.columns if c not in SKIP_COLS]

    # Build record text for every row
    df = df.copy()
    df["record"] = df.apply(lambda row: make_record(row, text_cols), axis=1)
    df = df.set_index(ID_COL)

    all_ids   = df.index.tolist()
    id_to_idx = {rid: i for i, rid in enumerate(all_ids)}

    # TF-IDF vectorise all records once
    vectorizer  = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["record"])

    rows = []
    for anchor_id in anchor_ids:
        anchor_idx = id_to_idx[anchor_id]
        anchor_vec = tfidf_matrix[anchor_idx]

        sims = cosine_similarity(anchor_vec, tfidf_matrix).flatten()
        sims[anchor_idx] = -1  # exclude self

        top_k_indices = np.argsort(sims)[::-1][:top_k]

        anchor_row = df.loc[anchor_id]
        for j in top_k_indices:
            cand_id  = all_ids[j]
            cand_row = df.iloc[j]
            label    = int(anchor_row[CLUSTER_COL] == cand_row[CLUSTER_COL])
            rows.append({
                "id_left":       anchor_id,
                "id_right":      cand_id,
                "record_left":   anchor_row["record"],
                "record_right":  cand_row["record"],
                "cluster_left":  anchor_row[CLUSTER_COL],
                "cluster_right": cand_row[CLUSTER_COL],
                "label":         label,
            })

    result = pd.DataFrame(rows)

    # Drop duplicate pairs — (A, B) and (B, A) are the same pair, keep only one
    
    result["_pair_key"] = result.apply(
        lambda r: tuple(sorted([r["id_left"], r["id_right"]])), axis=1
    )
    result = result.drop_duplicates(subset="_pair_key").drop(columns="_pair_key")
    result = result.reset_index(drop=True)

    return result
    


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Loading dataset from: {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} records.")

    anchor_ids = ANCHOR_IDS if ANCHOR_IDS else df[ID_COL].tolist()
    print(f"Anchors: {len(anchor_ids)}  |  Top-K: {TOP_K}")

    result = blocking(df, anchor_ids=anchor_ids, top_k=TOP_K)

    print(f"\nOutput shape : {result.shape}")
    print(f"Label distribution:\n{result['label'].value_counts().to_string()}")

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()