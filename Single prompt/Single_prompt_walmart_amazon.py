import os
import re
import time
import pandas as pd
from google import genai
from pathlib import Path

project_root = Path("").resolve()
os.chdir(project_root)
# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════════

class SimpleRateLimiter:
    """Throttles API requests to stay within rate limits."""

    def __init__(self, max_requests_per_minute: int = 60):
        self.max_rpm = max_requests_per_minute
        self.request_times = []

    def wait_if_needed(self):
        """Check rate limits and wait if necessary."""
        current_time = time.time()
        # Remove request times older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        # If approaching limit, wait
        if len(self.request_times) >= self.max_rpm:
            sleep_time = 60 - (current_time - self.request_times[0]) + 0.1
            print(f"[RATE LIMIT] Approaching limit ({len(self.request_times)}/{self.max_rpm} requests/min). "
                  f"Waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            self.request_times = []

        self.request_times.append(current_time)


# Global rate limiter
_rate_limiter = SimpleRateLimiter(max_requests_per_minute=30)


def set_rate_limit(requests_per_minute: int):
    """Configure global rate limiting."""
    global _rate_limiter
    _rate_limiter = SimpleRateLimiter(max_requests_per_minute=requests_per_minute)
    print(f"[CONFIG] Rate limit set to {requests_per_minute} requests/minute")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these or override via environment variables / CLI args
# ══════════════════════════════════════════════════════════════════════════════

GOOGLE_CLOUD_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT",  "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "")
GEMINI_MODEL          = os.getenv("GEMINI_MODEL",           "")
RATE_LIMIT_RPM        = int(os.getenv("RATE_LIMIT_RPM",     ""))

# Path to the dataset (must contain an 'id' column and a 'cluster_id' column)
DATASET_PATH      = "./dataset/sample_walmart_amazon.csv"
GROUND_TRUTH_PATH = "./dataset/sample_walmart_amazon_gt.csv"

# Fields to use for entity matching (must exist in the dataset)
MATCHING_FIELDS = ['title', 'brand', 'modelno', 'category', 'name', 'desc']

# ══════════════════════════════════════════════════════════════════════════════
# Gemini client
# ══════════════════════════════════════════════════════════════════════════════

client = genai.Client(
    vertexai=True,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
)

# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path):
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='utf-8-sig')


def get_id_column(df):
    for col in df.columns:
        if str(col).lower() == 'id':
            return col
    return None


def load_ground_truth_from_df(df, id_col, cluster_col='cluster_id'):
    """
    Build ground truth clusters from the cluster_id column.
    Returns a list of lists, each inner list contains record IDs that belong to the same cluster.
    """
    # Drop rows with missing cluster_id (should not happen)
    df_clean = df.dropna(subset=[cluster_col])
    clusters = {}
    for _, row in df_clean.iterrows():
        cid = row[cluster_col]
        rid = str(row[id_col])
        clusters.setdefault(cid, []).append(rid)
    # Return as list of sorted ID lists
    return [sorted(v) for v in clusters.values()]


# ══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(df):
    id_col = get_id_column(df)

    # Only use the specified matching fields
    available_fields = [f for f in MATCHING_FIELDS if f in df.columns]
    if not available_fields:
        raise ValueError(f"Dataset must contain at least one of: {MATCHING_FIELDS}\n"
                        f"Available columns: {list(df.columns)}")

    print(f"[INFO] Using fields for entity matching: {available_fields}")

    lines = []
    for _, row in df.iterrows():
        record_id = row[id_col] if id_col else _
        attrs_list = []
        for field in available_fields:
            value = row[field]
            if pd.isna(value):
                value = "(none)"
            else:
                value = str(value).strip()
            attrs_list.append(f"{field}: {value}")

        attrs = ", ".join(attrs_list)
        lines.append(f"Record {record_id}: {attrs}")

    return (
        f"""
        You are an expert in Product Entity Resolution.

        Task:
        Partition the product records below into clusters such that records referring to the exact same real‑world product are in the same cluster.

        Use ONLY these product attributes for matching:
        {", ".join(available_fields)}

        Hard requirements:
        1. Every input record ID must appear exactly once in the output.
        2. No record ID may be omitted in the output.
        3. No record ID may appear more than once.
        4. The union of all predicted clusters must equal exactly the input IDs above.
        5. Return ONLY a JSON‑style two‑dimensional list of integer/string IDs.
        6. No explanation, no markdown, no extra text.

        Before finalizing, internally verify:
        - no missing IDs
        - no duplicate IDs

        Example output:
        [[111, 222, 333], [444, 555], [666]]

        Records:
        """ + "\n".join(lines)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Response parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_clusters(response_text):
    # Extract everything between square brackets, preserving content
    text = response_text.replace('\n', '').strip()
    clusters = []
    for match in re.findall(r'\[(.*?)\]', text):
        inner = match.strip()
        if not inner:
            continue
        items = [x.strip().strip('"\'') for x in inner.split(',')]
        clusters.append(items)
    return clusters


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def get_pairs(clusters):
    pairs = set()
    for c in clusters:
        sc = sorted(c)
        for i in range(len(sc)):
            for j in range(i + 1, len(sc)):
                pairs.add((sc[i], sc[j]))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Report builder
# ══════════════════════════════════════════════════════════════════════════════

def build_report(pred_clusters, true_clusters,
                 elapsed_time, prompt_tokens,
                 completion_tokens, total_tokens,
                 fields_used=None):
    SEP = "=" * 80
    DASH = "-" * 80

    true_pairs = get_pairs(true_clusters)
    pred_pairs = get_pairs(pred_clusters)
    tp_g = len(true_pairs & pred_pairs)
    fp_g = len(pred_pairs - true_pairs)
    fn_g = len(true_pairs - pred_pairs)
    precision = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0.0
    recall = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    lines = []

    # Header
    lines += ["", SEP, "LLMCER - MODEL OUTPUT SUMMARY", SEP,
              f"\nTotal Predicted Clusters:    {len(pred_clusters)}",
              f"Total Ground Truth Clusters: {len(true_clusters)}"]

    # Predicted clusters
    lines += ["", DASH, "PREDICTED CLUSTERS (Record IDs):", DASH]
    for i, c in enumerate(pred_clusters):
        lines.append(f"\nCluster {i+1} ({len(c)} records): {sorted(c)}")

    # Ground truth clusters
    lines += ["", DASH, "GROUND TRUTH CLUSTERS (Record IDs):", DASH]
    for i, c in enumerate(true_clusters):
        lines.append(f"\nCluster {i+1} ({len(c)} records): {sorted(c)}")

    # Cluster comparison
    lines += ["", DASH, "CLUSTER COMPARISON (Predicted vs Ground Truth):", DASH]
    for i, pred in enumerate(pred_clusters):
        pred_set = set(pred)
        best_j, _ = max(
            ((j, len(pred_set & set(gt_c))) for j, gt_c in enumerate(true_clusters)),
            key=lambda x: x[1],
            default=(0, 0)
        )
        gt_set = set(true_clusters[best_j])
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        lines.append(f"\nPredicted Cluster {i+1}  <->  Ground Truth Cluster {best_j+1}")
        lines.append(f"  Correct (TP)  : {sorted(pred_set & gt_set)}")
        if fp > 0:
            lines.append(f"  Extra   (FP)  : {sorted(pred_set - gt_set)}")
        if fn > 0:
            lines.append(f"  Missed  (FN)  : {sorted(gt_set - pred_set)}")
        lines.append(
            f"  Scores  ->  Precision: {p*100:.2f}%  Recall: {r*100:.2f}%  "
            f"F1: {f*100:.2f}%  (TP={tp}  FP={fp}  FN={fn})"
        )

    # Performance metrics
    lines += ["", SEP, "PERFORMANCE METRICS (OVERALL)", SEP,
              f"\nPrecision      : {precision:.4f} ({precision*100:.2f}%)",
              f"Recall         : {recall:.4f} ({recall*100:.2f}%)",
              f"Pairwise F1    : {f1:.4f}"]

    # Execution statistics
    lines += ["", SEP, "EXECUTION STATISTICS", SEP, f"""
Method         : Single-prompt Gemini benchmark
Model          : {GEMINI_MODEL}
API Calls      : 1
Rate Limit     : {_rate_limiter.max_rpm} requests/minute
Fields Used    : {', '.join(fields_used) if fields_used else ', '.join(MATCHING_FIELDS)}
Total Time     : {elapsed_time:.2f}s

Tokens (total) : {total_tokens}
  - Input      : {prompt_tokens}
  - Output     : {completion_tokens}"""]

    lines += ["", SEP, "END", SEP, ""]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Benchmark — Single-Prompt Gemini (Product Entity Resolution)")
    print("=" * 60)
    print(f"  Model    : {GEMINI_MODEL}")
    print(f"  Project  : {GOOGLE_CLOUD_PROJECT}")
    print(f"  Location : {GOOGLE_CLOUD_LOCATION}")
    print(f"  Dataset  : {DATASET_PATH}")
    print(f"  Fields   : {', '.join(MATCHING_FIELDS)}")
    print("=" * 60)

    set_rate_limit(RATE_LIMIT_RPM)

    # Load dataset (contains both records and ground truth cluster_id)
    df = load_dataset(DATASET_PATH)
    id_col = get_id_column(df)
    if id_col is None:
        raise ValueError("Dataset must contain an 'id' column.")
    print(f"Dataset loaded: {len(df)} records.")

    # Ground truth clusters from the cluster_id column
    if 'cluster_id' not in df.columns:
        raise ValueError("Dataset must contain a 'cluster_id' column for ground truth.")
    ground_truth = load_ground_truth_from_df(df, id_col, 'cluster_id')
    print(f"Ground truth clusters loaded: {len(ground_truth)}.")

    # Check rate limit before API call
    print("\nChecking rate limits...")
    _rate_limiter.wait_if_needed()

    # Single Gemini API call
    print("Calling Gemini...")
    prompt = build_prompt(df)
    start_time = time.time()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            "You are an expert in Entity Resolution. "
            "You specialize in clustering product records that refer to the same real-world product.",
            prompt,
        ],
    )
    elapsed_time = time.time() - start_time

    prompt_tokens = response.usage_metadata.prompt_token_count
    completion_tokens = response.usage_metadata.candidates_token_count
    total_tokens = response.usage_metadata.total_token_count

    print(f"Response received in {elapsed_time:.2f}s")
    print(f"Tokens — total: {total_tokens}  input: {prompt_tokens}  output: {completion_tokens}")

    # Parse clusters from response
    pred_clusters = parse_clusters(response.text)
    print(f"Parsed {len(pred_clusters)} predicted clusters.")

    # Build and save report
    dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
    output_path = f"./Single prompt/results/{dataset_name}_single_prompt_walmart_amazon_Summary_pro.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    available_fields = [f for f in MATCHING_FIELDS if f in df.columns]
    report = build_report(
        pred_clusters=pred_clusters,
        true_clusters=ground_truth,
        elapsed_time=elapsed_time,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        fields_used=available_fields,
    )

    print(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()