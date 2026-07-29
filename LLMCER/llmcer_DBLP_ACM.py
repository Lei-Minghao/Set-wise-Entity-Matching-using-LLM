#!/usr/bin/env python3
"""
LLM‑CER pipeline for DBLP‑ACM publication dataset.
  1. LSH blocking
  2. LLM separation + MDG (whole block processing, no chunking)
  3. Iterative conservative merge_2 with BATCHED LLM calls (only confirmed pairs merged)
"""

import ast, csv, math, os, re, sys, time, random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.types import GenerateContentConfig
import os
from pathlib import Path

project_root = Path("").resolve()
os.chdir(project_root)

# ======================== PARAMETERS ========================
LSH_THRESHOLD = 0.5
STRONG_SIM_THRESHOLD = 0.52
BLOCK_MEDOID_KEEP_THRESHOLD = 0.33

MDG_MARGIN = 0.0
MAX_MDG_RETRIES = 2

# Merge thresholds – only consider high‑similarity pairs
BLOCK_THRESHOLD = 0.90               # upper bound of highest similarity band
MERGE_THRESHOLD = 0.70               # lower bound – ignore anything below this

# Optimisation parameters for merge_2
MAX_MERGE_CALLS_PER_ROUND = 50       # maximum number of LLM calls per merge round
BATCH_SIZE = 5                       # number of pairs evaluated in one call
MIN_MERGE_SIM = 0.75                 # ignore pairs with similarity below this

# Round limitation for iterative merging
MAX_MERGE_ROUNDS = 10                # maximum number of merge passes

REQUESTS_PER_MINUTE = 60

# Columns used for publication representation
COLUMNS_TO_USE = ['title', 'authors', 'venue', 'year']

# ======================== UTILITIES ========================
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        if x not in self.parent: self.parent[x] = x
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry: self.parent[rx] = ry
    def add(self, x):
        if x not in self.parent: self.parent[x] = x

def safe_print(msg):
    try: print(msg, flush=True)
    except: print(str(msg).encode('ascii','replace').decode(), flush=True)

def get_id_column(df):
    for col in ['ID', 'id']:
        if col in df.columns: return col
    return None

def get_pairs(clusters):
    pairs = set()
    for c in clusters:
        sc = sorted(set(c))
        for i in range(len(sc)):
            for j in range(i+1, len(sc)):
                a, b = sc[i], sc[j]
                if a > b: a, b = b, a
                pairs.add((a, b))
    return pairs

# ======================== VECTORIZATION ========================
def load_data_and_vectorize(data_path):
    local = os.getenv("EMBEDDING_MODEL_PATH","all-MiniLM-L6-v2")
    model_name = local if os.path.isdir(local) else "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252']
    data = None
    for enc in encodings_to_try:
        try:
            data = pd.read_csv(data_path, encoding=enc, dtype=str)
            safe_print(f"Read CSV with encoding: {enc}")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    if data is None:
        raise ValueError(f"Could not read CSV file with any common encoding: {data_path}")

    id_col = get_id_column(data)
    if not id_col: raise ValueError("No ID column found")
    data[id_col] = data[id_col].astype(str)

    def combine(row):
        parts = []
        for col in COLUMNS_TO_USE:
            if col in row.index and pd.notna(row[col]):
                parts.append(str(row[col]))
        return ' '.join(parts) if parts else "unknown"
    data['combined_text'] = data.apply(combine, axis=1)
    vectors = [model.encode(t) for t in data['combined_text']]
    simi_matrix = cosine_similarity(vectors)
    safe_print(f"[OK] Similarity matrix shape={simi_matrix.shape}")
    return vectors, simi_matrix, data, id_col

# ======================== LSH BLOCKING ========================
def _tokenize(text):
    if not isinstance(text, str): return set()
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2}

def _author_overlap_ok(a1, a2):
    if not isinstance(a1, str) or not isinstance(a2, str):
        return False
    set1 = {name.strip().lower() for name in a1.split(',')}
    set2 = {name.strip().lower() for name in a2.split(',')}
    if not set1 or not set2:
        return False
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter >= 1 and (inter / union) >= 0.3

def _title_overlap_ok(t1, t2):
    tok1, tok2 = _tokenize(t1), _tokenize(t2)
    if not tok1 or not tok2: return False
    inter = len(tok1 & tok2); union = len(tok1 | tok2)
    return inter >= 3 and (inter / union) >= 0.35

def _lexical_rescue(data, i, j):
    rowi, rowj = data.iloc[i], data.iloc[j]
    ti = str(rowi.get('title', '')).strip().lower()
    tj = str(rowj.get('title', '')).strip().lower()
    ai = str(rowi.get('authors', '')).strip().lower()
    aj = str(rowj.get('authors', '')).strip().lower()
    if _title_overlap_ok(ti, tj):
        return True
    if _author_overlap_ok(ai, aj) and _tokenize(ti) & _tokenize(tj):
        return True
    return False

def lsh_block(vectors, data):
    from lshashpy3 import LSHash
    lsh = LSHash(hash_size=15, input_dim=384, num_hashtables=8)
    for ix, vec in enumerate(vectors): lsh.index(vec, extra_data=ix)

    cand = defaultdict(dict)
    for ix, vec in enumerate(vectors):
        for r in lsh.query(vec, num_results=None, distance_func='cosine'):
            if r[0][1] is None: continue
            jx = int(r[0][1])
            if jx == ix: continue
            sim = 1 - r[1]
            if sim > cand[ix].get(jx, -1): cand[ix][jx] = sim

    graph = defaultdict(set)
    n = len(vectors)
    for ix in range(n):
        neigh = sorted(cand[ix].items(), key=lambda x: x[1], reverse=True)[:6]
        top_ids = {nid for nid, _ in neigh}
        for jx, sim in neigh:
            rev = sorted(cand[jx].items(), key=lambda x: x[1], reverse=True)[:6]
            rev_ids = {nid for nid, _ in rev}
            mutual = (jx in top_ids) and (ix in rev_ids)
            add = False
            if sim >= STRONG_SIM_THRESHOLD: add = True
            elif mutual and sim >= LSH_THRESHOLD: add = True
            elif sim >= LSH_THRESHOLD + 0.04 and (jx in top_ids or ix in rev_ids): add = True
            elif _lexical_rescue(data, ix, jx): add = True
            if add:
                graph[ix].add(jx); graph[jx].add(ix)

    # Title token groups
    token_groups = defaultdict(list)
    for ix in range(n):
        toks = sorted(_tokenize(str(data.iloc[ix].get('title', ''))))[:8]
        for tok in toks: token_groups[tok].append(ix)
    pair_counts = defaultdict(int)
    for members in token_groups.values():
        if len(members) < 2 or len(members) > 36*3: continue
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                a, b = members[i], members[j]
                if a > b: a, b = b, a
                pair_counts[(a, b)] += 1
    for (a, b), cnt in pair_counts.items():
        if cnt < 3: continue
        ta = _tokenize(str(data.iloc[a].get('title', '')))
        tb = _tokenize(str(data.iloc[b].get('title', '')))
        union = len(ta | tb); jacc = len(ta & tb) / union if union else 0.0
        if jacc >= 0.28:
            graph[a].add(b); graph[b].add(a)

    # Author groups (first author)
    author_groups = defaultdict(list)
    for ix in range(n):
        authors = str(data.iloc[ix].get('authors', '')).strip().lower()
        if authors and authors not in {'nan', 'none', 'missing'}:
            first = authors.split(',')[0].strip()
            if first:
                author_groups[first].append(ix)
    for members in author_groups.values():
        if len(members) < 2: continue
        members = members[:36*2]
        for a_pos, a in enumerate(members):
            ta = _tokenize(str(data.iloc[a].get('title', '')))
            for b in members[a_pos+1:]:
                tb = _tokenize(str(data.iloc[b].get('title', '')))
                if len(ta & tb) >= 2 or _title_overlap_ok(str(data.iloc[a].get('title', '')),
                                                           str(data.iloc[b].get('title', ''))):
                    graph[a].add(b); graph[b].add(a)

    visited = set(); comps = []
    for node in range(n):
        if node in visited: continue
        stack = [node]; comp = []
        while stack:
            cur = stack.pop()
            if cur in visited: continue
            visited.add(cur); comp.append(cur)
            stack.extend(graph[cur] - visited)
        comps.append(sorted(comp))

    purified = []
    for comp in comps:
        if len(comp) <= 2:
            purified.append(comp)
            continue
        comp_vecs = np.array([vectors[i] for i in comp])
        centroid = np.mean(comp_vecs, axis=0)
        norms = np.linalg.norm(comp_vecs, axis=1) * (np.linalg.norm(centroid) + 1e-12)
        sims = (comp_vecs @ centroid) / np.maximum(norms, 1e-12)
        keep = [comp[i] for i, s in enumerate(sims) if s >= BLOCK_MEDOID_KEEP_THRESHOLD]
        fringe = [comp[i] for i, s in enumerate(sims) if s < BLOCK_MEDOID_KEEP_THRESHOLD]
        if len(keep) <= 1:
            purified.append(comp)
        else:
            purified.append(sorted(keep))
            for idx in fringe: purified.append([idx])
    safe_print("lsh done")
    return [c for c in purified if c]

# ======================== LLM API (Vertex AI) ========================
_request_timestamps = []
_client = None

def get_vertex_client():
    global _client
    if _client is None:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        if not project or not location:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set")
        _client = genai.Client(vertexai=True, project=project, location=location)
    return _client

def call_gemini(system, prompt):
    client = get_vertex_client()
    now = time.time()
    global _request_timestamps
    _request_timestamps = [t for t in _request_timestamps if now - t < 60]
    if len(_request_timestamps) >= REQUESTS_PER_MINUTE:
        time.sleep(60 - (now - _request_timestamps[0]) + 0.05)
    _request_timestamps.append(time.time())
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=GenerateContentConfig(system_instruction=system)
    )
    in_tok = getattr(resp.usage_metadata, "prompt_token_count", 0)
    out_tok = getattr(resp.usage_metadata, "candidates_token_count", 0)
    return resp.text, in_tok, out_tok

# ======================== SEPARATION PROMPT ========================
def build_record_lines(indices, df):
    id_col = get_id_column(df)
    lines = []
    for idx in indices:
        row = df.iloc[idx]
        rid = row[id_col] if id_col else idx
        parts = [f"{col}={str(row[col])}" for col in COLUMNS_TO_USE if col in df.columns and pd.notna(row[col])]
        lines.append(f"Record {rid}: " + ", ".join(parts))
    return lines

def separation_prompt(indices, df):
    lines = build_record_lines(indices, df)
    available_fields = COLUMNS_TO_USE
    prompt = f"""You are an expert in Entity Resolution.

        Task:
        Partition the publication records below into clusters such that records referring to the same real-world publication are in the same cluster.

        Use ONLY these fields for matching:
        {", ".join(available_fields)}

        Hard requirements:
        1. Every input record ID must appear exactly once in the output.
        2. No record ID may be omitted in the output.
        3. No record ID may appear more than once.
        4. The union of all predicted clusters must equal exactly the input IDs above.
        5. Return ONLY a JSON-style two-dimensional list of record IDs.
        6. No explanation, no markdown, no extra text.

        Before finalizing, internally verify:
        - no missing IDs
        - no duplicate IDs

        Example output:
        [[id1, id2], [id3, id4, id5]]

        Records:
        """ + "\n".join(lines)
    return prompt

# ======================== PARSING ========================
def parse_llm_clusters(text, id_map):
    content = text.strip()
    try:
        parsed = ast.literal_eval(content)
        if not isinstance(parsed, list): raise ValueError
    except:
        matches = re.findall(r'\[([^\]]*)\]', content)
        parsed = []
        for m in matches:
            items = [p.strip() for p in m.split(',') if p.strip()]
            if items: parsed.append(items)
    result = []
    for cl in parsed:
        rows = []
        for cid in cl:
            if cid in id_map: rows.append(id_map[cid])
        if rows: result.append(rows)
    return result

# ======================== MDG ========================
def mdg_check(clusters, sim_matrix):
    for ci, cluster in enumerate(clusters):
        for ridx in cluster:
            intra = [sim_matrix[ridx][r2] for r2 in cluster if r2 != ridx]
            if not intra: continue
            min_intra = min(intra)
            max_inter = -1.0
            for cj, other in enumerate(clusters):
                if cj == ci: continue
                inter = [sim_matrix[ridx][r2] for r2 in other]
                if inter: max_inter = max(max_inter, max(inter))
            if max_inter > min_intra - MDG_MARGIN:
                return False
    return True

def find_misclustered(clusters, sim_matrix):
    for ci, cluster in enumerate(clusters):
        for ridx in cluster:
            intra = [sim_matrix[ridx][r2] for r2 in cluster if r2 != ridx]
            if not intra: continue
            min_intra = min(intra)
            max_inter = -1.0; target_ci = None
            for cj, other in enumerate(clusters):
                if cj == ci: continue
                inter = [sim_matrix[ridx][r2] for r2 in other]
                if inter and max(inter) > max_inter:
                    max_inter = max(inter); target_ci = cj
            if max_inter > min_intra - MDG_MARGIN:
                return ridx, ci, target_ci
    return None

def reorder_list(indices, clusters, sim_matrix):
    viol = find_misclustered(clusters, sim_matrix)
    if viol is None: return None
    ridx, ci, target_ci = viol
    target = clusters[target_ci]
    ordered = [r for r in target]
    ordered.append(ridx)
    for idx, cl in enumerate(clusters):
        if idx == target_ci: continue
        for r in cl:
            if r != ridx: ordered.append(r)
    return ordered

def cluster_one_set(indices, df, sim_matrix, id_map):
    current_order = indices[:]
    total_in = 0; total_out = 0; calls = 0
    for attempt in range(1 + MAX_MDG_RETRIES):
        prompt = separation_prompt(current_order, df)
        text, in_t, out_t = call_gemini("", prompt)
        total_in += in_t; total_out += out_t; calls += 1
        clusters = parse_llm_clusters(text, id_map)
        flat = [r for cl in clusters for r in cl]
        missing = set(current_order) - set(flat)
        for m in missing: clusters.append([m])
        extra = set(flat) - set(current_order)
        if extra: clusters = [[r] for r in current_order]
        if mdg_check(clusters, sim_matrix):
            return clusters, total_in, total_out, calls
        ordered = reorder_list(current_order, clusters, sim_matrix)
        if ordered is not None:
            current_order = ordered
        else:
            return clusters, total_in, total_out, calls
    return clusters, total_in, total_out, calls

# ======================== SEPARATION (NO CHUNKING) ========================
def llm_separate_block(block_indices, df, sim_matrix, id_map):
    if len(block_indices) == 1:
        return [[block_indices]], 0, 0, 0
    clusters, in_t, out_t, calls = cluster_one_set(block_indices, df, sim_matrix, id_map)
    return [clusters], calls, in_t, out_t

# ======================== MERGE_2 (BATCHED, OPTIMISED) ========================
PRE_PROMPT_MERGE = (
    "You are an expert in publication entity resolution. "
    "Two candidate clusters are shown below. Each cluster already contains publication records that refer to the same publication. "
    "Decide whether the two clusters should be merged into a single cluster (i.e., they represent the same publication). "
    "Consider title, authors, venue, and year. "
    "Answer only YES or NO for each pair.\n"
)

def get_most_simi(list1, list2, simi_matrix):
    max_simi = 0
    for a in list1:
        for b in list2:
            if simi_matrix[a][b] > max_simi:
                max_simi = simi_matrix[a][b]
    return max_simi

def pick_elements(list1, list2, simi_matrix, n=2):
    combined = list1 + list2
    if len(combined) <= n:
        return combined
    best = (list1[0], list2[0], -1.0)
    for a in list1:
        for b in list2:
            s = simi_matrix[a][b]
            if s > best[2]:
                best = (a, b, s)
    result = [best[0], best[1]]
    remaining = [x for x in combined if x not in result]
    if len(result) < n:
        result += random.sample(remaining, min(n - len(result), len(remaining)))
    return result

def get_prompt_from_indices(indices, df):
    id_col = get_id_column(df)
    lines = []
    for idx in indices:
        row = df.iloc[idx]
        rid = row[id_col] if id_col else idx
        parts = [f"{col}={str(row[col])}" for col in COLUMNS_TO_USE if col in df.columns and pd.notna(row[col])]
        lines.append(f"Record {rid}: " + ", ".join(parts))
    return '\n'.join(lines)

def merge_2(clusters, simi_matrix, df, block_threshold, merge_threshold):
    """
    Merge clusters using batched LLM calls.
    Only pairs that receive a "yes" are merged (transitively via Union-Find).
    """
    n = len(clusters)
    if n <= 1:
        return clusters, 0, 0, 0, 0, 0

    # Build cluster‑level similarity matrix (max similarity between any two members)
    batch_simi = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            batch_simi[i][j] = get_most_simi(clusters[i], clusters[j], simi_matrix)
            batch_simi[j][i] = batch_simi[i][j]

    # Collect candidate pairs with similarity >= MIN_MERGE_SIM
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if batch_simi[i][j] >= MIN_MERGE_SIM:
                candidates.append((i, j, batch_simi[i][j]))
    # Sort by descending similarity
    candidates.sort(key=lambda x: x[2], reverse=True)
    # Limit total number of pairs to (MAX_MERGE_CALLS_PER_ROUND * BATCH_SIZE)
    candidates = candidates[:MAX_MERGE_CALLS_PER_ROUND * BATCH_SIZE]

    confirmed_pairs = []
    total_calls = 0
    total_in = 0
    total_out = 0
    start_time = time.time()

    # Process candidates in batches
    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[batch_start:batch_start + BATCH_SIZE]
        if not batch:
            break

        # Build a single prompt for all pairs in the batch
        batch_prompt = ""
        for idx, (i, j, _) in enumerate(batch):
            reps = pick_elements(clusters[i], clusters[j], simi_matrix)
            batch_prompt += f"\n--- Pair {idx+1} ---\n"
            batch_prompt += get_prompt_from_indices(reps, df) + "\n"

        full_prompt = PRE_PROMPT_MERGE + batch_prompt + "\nAnswer with a comma‑separated list of YES/NO in the same order as the pairs."

        # Call Gemini (same model as separation)
        text, in_t, out_t = call_gemini(
            "You are a worker with rich experience performing Entity Resolution tasks.",
            full_prompt
        )
        total_calls += 1
        total_in += in_t
        total_out += out_t

        answer_text = text.strip().lower()
        answers = [a.strip() for a in answer_text.split(',')]
        for idx, (i, j, _) in enumerate(batch):
            if idx < len(answers) and 'yes' in answers[idx]:
                confirmed_pairs.append([i, j])

    # Merge confirmed pairs using Union‑Find (transitive closure)
    if not confirmed_pairs:
        new_clusters = clusters
    else:
        uf = UnionFind()
        for i in range(n):
            uf.add(i)
        for i, j in confirmed_pairs:
            uf.union(i, j)
        groups = {}
        for i in range(n):
            root = uf.find(i)
            groups.setdefault(root, []).extend(clusters[i])
        new_clusters = [sorted(set(v)) for v in groups.values()]

    merge_time = time.time() - start_time
    safe_print(f"merge_2 done: {len(new_clusters)} clusters (LLM calls: {total_calls})")
    return new_clusters, total_calls, total_in, total_out, merge_time

def iterative_merge(initial_clusters, simi_matrix, df, block_threshold, merge_threshold, max_rounds):
    clusters = initial_clusters
    total_calls = 0
    total_in = 0
    total_out = 0
    total_time = 0.0

    for round_idx in range(max_rounds):
        prev_len = len(clusters)
        safe_print(f"Merge round {round_idx+1}/{max_rounds} (clusters: {prev_len})")
        clusters, calls, in_t, out_t, t = merge_2(clusters, simi_matrix, df, block_threshold, merge_threshold)
        total_calls += calls
        total_in += in_t
        total_out += out_t
        total_time += t
        if len(clusters) == prev_len:
            safe_print(f"No change after round {round_idx+1}, stopping.")
            break
    return clusters, total_calls, total_in, total_out, total_time

# ======================== COVERAGE DIAGNOSTICS ========================
def compute_coverage(clusters, ground_truth, data):
    if not ground_truth: return None
    id_col = get_id_column(data)
    if not id_col: return None
    valid_ids = set(data[id_col].tolist())
    row_to_id = {i: data.iloc[i][id_col] for i in range(len(data))}

    cluster_sets = []
    for cluster in clusters:
        mapped = set()
        for x in cluster:
            if x is None: continue
            if x in row_to_id: mapped.add(row_to_id[x])
        cluster_sets.append(mapped)

    total_true_pairs = 0; covered_true_pairs = 0
    gt_cluster_count = 0; fully_covered_clusters = 0

    for cluster in ground_truth:
        cluster_ids = [str(x) for x in cluster if str(x) in valid_ids]
        if len(cluster_ids) < 2: continue
        gt_cluster_count += 1
        cluster_total_pairs = 0; cluster_covered_pairs = 0
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                a, b = cluster_ids[i], cluster_ids[j]
                cluster_total_pairs += 1; total_true_pairs += 1
                if any(a in cl and b in cl for cl in cluster_sets):
                    cluster_covered_pairs += 1; covered_true_pairs += 1
        if cluster_total_pairs > 0 and cluster_covered_pairs == cluster_total_pairs:
            fully_covered_clusters += 1

    return {
        "pair_completeness": covered_true_pairs / total_true_pairs if total_true_pairs else 0.0,
        "full_cluster_coverage": fully_covered_clusters / gt_cluster_count if gt_cluster_count else 0.0,
    }

def print_coverage(label, cov):
    if cov:
        safe_print(f"{label}: pair completeness={cov['pair_completeness']:.4f}, "
                   f"full cluster coverage={cov['full_cluster_coverage']:.4f}")

# ======================== EVALUATION ========================
def load_ground_truth(file_path):
    def merge_coords(coords):
        uf = UnionFind(); ids = set()
        for l, r in coords:
            l, r = str(l), str(r)
            uf.add(l); uf.add(r); uf.union(l, r); ids.update([l, r])
        groups = defaultdict(list)
        for i in ids: groups[uf.find(i)].append(i)
        return [v for v in groups.values()]
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            clusters = []
            for line in f:
                parts = line.strip().split()
                if parts:
                    clusters.append(parts)
            return clusters
    elif file_path.endswith('.csv'):
        try: df = pd.read_csv(file_path, encoding='utf-8-sig', dtype=str)
        except: df = pd.read_csv(file_path, encoding='utf-8', dtype=str)
        if df.shape[1] >= 2:
            pairs = df.iloc[:, :2].values.tolist()
            return merge_coords(pairs)
    return []

def compute_metrics(gt, pred):
    true_pairs = get_pairs(gt); pred_pairs = get_pairs(pred)
    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    total_pred = sum(len(c) for c in pred)
    correct_pur = 0
    for pc in pred:
        cnt = defaultdict(int)
        for sid in pc:
            for tc in gt:
                if sid in tc: cnt[tuple(tc)] += 1
        if cnt: correct_pur += max(cnt.values())
    pur = correct_pur/total_pred if total_pred else 0
    total_true = sum(len(c) for c in gt)
    correct_inv = 0
    for tc in gt:
        if tc:
            cnt = defaultdict(int)
            for sid in tc:
                for pc2 in pred:
                    if sid in pc2: cnt[tuple(pc2)] += 1
            if cnt: correct_inv += max(cnt.values())
    inv_pur = correct_inv/total_true if total_true else 0
    fpm = 2*pur*inv_pur/(pur+inv_pur) if pur+inv_pur else 0
    all_ids = sorted(set(s for c in gt for s in c) | set(s for c in pred for s in c))
    if not all_ids: ari = 0
    else:
        id_map = {sid: i for i, sid in enumerate(all_ids)}
        true_labels = [-1]*len(all_ids); pred_labels = [-1]*len(all_ids)
        for i, c in enumerate(gt):
            for sid in c: true_labels[id_map[sid]] = i
        for i, c in enumerate(pred):
            for sid in c: pred_labels[id_map[sid]] = i
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(true_labels, pred_labels)
    return prec, rec, f1, tp, fp, fn, pur, inv_pur, fpm, ari

def write_report(pred_clusters, gt, out_path, extra_info=""):
    with open(out_path, 'w', encoding='utf-8') as f:
        sep, dash = "="*80, "-"*80
        f.write(f"{sep}\nLLMCER - MODEL OUTPUT SUMMARY\n{sep}\n\n")
        f.write(f"Total Predicted Clusters:    {len(pred_clusters)}\n")
        f.write(f"Total Ground Truth Clusters: {len(gt) if gt else 'N/A'}\n\n")
        f.write(f"{dash}\nPREDICTED CLUSTERS (Record IDs):\n{dash}\n")
        for i, c in enumerate(pred_clusters):
            f.write(f"\nCluster {i+1} ({len(c)} records): {sorted(c)}\n")
        if gt:
            prec, rec, f1, tp, fp, fn, pur, inv, fpm, ari = compute_metrics(gt, pred_clusters)
            f.write(f"\n{sep}\nPERFORMANCE METRICS (OVERALL)\n{sep}\n")
            f.write(f"\nPrecision      : {prec:.4f} ({prec*100:.2f}%)\n")
            f.write(f"Recall         : {rec:.4f} ({rec*100:.2f}%)\n")
            f.write(f"Pairwise F1    : {f1:.4f}\n")
            f.write(f"TP             : {tp}\nFP             : {fp}\nFN             : {fn}\n")
            f.write(f"Purity         : {pur:.4f}\nInverse Purity : {inv:.4f}\n")
            f.write(f"FP-measure     : {fpm:.4f}\nARI            : {ari:.4f}\n")
        f.write(f"\n{sep}\nEXECUTION INFORMATION\n{sep}\n{extra_info}\n{sep}\nEND\n")

# ======================== MAIN ========================
def main():
    total_start = time.time()
    data_path = os.getenv("DATASET_PATH", "sample_DBLP_ACM.csv")
    gt_path = os.getenv("GROUND_TRUTH_PATH", "sample_DBLP_ACM_gt.csv")
    vectors, sim_matrix, data, id_col = load_data_and_vectorize(data_path)

    blocks = lsh_block(vectors, data)
    safe_print(f"Step 1 – LSH blocks: {len(blocks)}")

    gt = []
    if os.path.exists(gt_path):
        try: gt = load_ground_truth(gt_path)
        except Exception as e: safe_print(f"Warning: could not load ground truth: {e}")

    if gt:
        cov_lsh = compute_coverage(blocks, gt, data)
        print_coverage("After LSH", cov_lsh)

    id_map = {data.iloc[i][id_col]: i for i in range(len(data))}

    safe_print("Step 2 – LLM separation + MDG (NO chunking)")
    all_sets = []
    sep_calls = 0; sep_in = 0; sep_out = 0
    for bi, block in enumerate(blocks):
        if not block: continue
        safe_print(f"  Block {bi+1}/{len(blocks)} (size={len(block)})")
        sets, calls, in_t, out_t = llm_separate_block(block, data, sim_matrix, id_map)
        all_sets.extend(sets)
        sep_calls += calls; sep_in += in_t; sep_out += out_t

    safe_print(f"After separation: {len(all_sets)} record sets (LLM calls: {sep_calls})")

    if gt:
        sep_clusters = []
        for s in all_sets: sep_clusters.extend(s)
        cov_sep = compute_coverage(sep_clusters, gt, data)
        print_coverage("After separation", cov_sep)

    safe_print("Step 3 – Iterative conservative merge_2 (batched, confirmed pairs only)")
    initial_clusters = []
    for s in all_sets: initial_clusters.extend(s)
    safe_print(f"  Starting with {len(initial_clusters)} clusters")
    final_clusters, merge_calls, merge_in, merge_out, merge_time = iterative_merge(
        initial_clusters, sim_matrix, data, BLOCK_THRESHOLD, MERGE_THRESHOLD, MAX_MERGE_ROUNDS
    )
    safe_print(f"  After merging: {len(final_clusters)} clusters (LLM calls: {merge_calls})")

    if gt:
        cov_merge = compute_coverage(final_clusters, gt, data)
        print_coverage("After merge", cov_merge)

    pred_clusters = []
    for cl in final_clusters:
        ids = [data.iloc[idx][id_col] for idx in cl]
        if ids: pred_clusters.append(sorted(ids))

    total_time = time.time() - total_start
    total_tokens_in = sep_in + merge_in
    total_tokens_out = sep_out + merge_out
    total_tokens = total_tokens_in + total_tokens_out

    if gt:
        prec, rec, f1, tp, fp, fn, _, _, _, _ = compute_metrics(gt, pred_clusters)
        safe_print("\n" + "="*80)
        safe_print("PERFORMANCE METRICS (OVERALL)")
        safe_print(f"Precision      : {prec:.4f} ({prec*100:.2f}%)")
        safe_print(f"Recall         : {rec:.4f} ({rec*100:.2f}%)")
        safe_print(f"Pairwise F1    : {f1:.4f}")
        safe_print(f"TP             : {tp}")
        safe_print(f"FP             : {fp}")
        safe_print(f"FN             : {fn}")

    extra = (f"Total time      : {total_time:.1f}s\n"
             f"Separation LLM calls : {sep_calls}\n"
             f"Separation tokens (in)  : {sep_in}\n"
             f"Separation tokens (out) : {sep_out}\n"
             f"Merge LLM calls : {merge_calls}\n"
             f"Merge tokens (in)  : {merge_in}\n"
             f"Merge tokens (out) : {merge_out}\n"
             f"Total tokens (in)  : {total_tokens_in}\n"
             f"Total tokens (out) : {total_tokens_out}\n"
             f"Total tokens (in+out) : {total_tokens}")
    safe_print(f"\nToken usage:\n{extra}")

    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    out_path = f"./LLMCER/results/{dataset_name}_llmcer_results.txt"
    write_report(pred_clusters, gt, out_path, extra)
    safe_print(f"\nResults written to {out_path}")

if __name__ == "__main__":
    main()