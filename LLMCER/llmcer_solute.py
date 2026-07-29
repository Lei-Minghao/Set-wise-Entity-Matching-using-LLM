#!/usr/bin/env python3
"""
LLM‑CER pipeline – anti‑transitivity merge + conservative prompt + higher threshold.
Adapted for sample_dataset_1 (columns: id, brand, name, desc, price, product_id, cluster_id).
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
LSH_THRESHOLD = 0.32
STRONG_SIM_THRESHOLD = 0.52
BLOCK_MEDOID_KEEP_THRESHOLD = 0.33

MDG_MARGIN = 0.0
MAX_MDG_RETRIES = 2

CANDIDATE_SIM_THRESH = 0.70          # centroid cosine ≥ this creates candidate edge
MAX_CLUSTERS_PER_PROMPT = 30
MERGE_MAX_ROUNDS = 15

REQUESTS_PER_MINUTE = 60

# Columns used for product representation (dataset specific)
COLUMNS_TO_USE = ['brand', 'name', 'desc', 'price']

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
    for col in ['ID','id']:
        if col in df.columns: return col
    return None

def get_pairs(clusters):
    pairs = set()
    for c in clusters:
        sc = sorted(set(c))
        for i in range(len(sc)):
            for j in range(i+1,len(sc)):
                a,b = sc[i],sc[j]
                if a>b: a,b = b,a
                pairs.add((a,b))
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
            data = pd.read_csv(data_path, encoding=enc)
            safe_print(f"Read CSV with encoding: {enc}")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    if data is None:
        raise ValueError(f"Could not read CSV file with any common encoding: {data_path}")

    id_col = get_id_column(data)
    if not id_col: raise ValueError("No ID column found")
    def combine(row):
        parts = [str(row[col]) for col in COLUMNS_TO_USE if col in row.index and pd.notna(row[col])]
        return ' '.join(parts) if parts else "unknown"
    data['combined_text'] = data.apply(combine, axis=1)
    vectors = [model.encode(t) for t in data['combined_text']]
    simi_matrix = cosine_similarity(vectors)
    safe_print(f"[OK] Similarity matrix shape={simi_matrix.shape}")
    return vectors, simi_matrix, data, id_col

# ======================== LSH BLOCKING ========================
MODEL_RE = re.compile(r"\b[a-z]*\d[a-z\d-]{2,}\b", re.I)
def _tokenize(text):
    if not isinstance(text,str): return set()
    return {t for t in re.findall(r"[a-z0-9]+",text.lower()) if len(t)>=2}
def _model_tokens(text):
    if not isinstance(text,str): return set()
    return set(MODEL_RE.findall(text.lower()))
def _title_overlap_ok(t1,t2):
    tok1,tok2 = _tokenize(t1),_tokenize(t2)
    if not tok1 or not tok2: return False
    inter = len(tok1&tok2); union = len(tok1|tok2)
    return inter>=3 and (inter/union)>=0.35
def _lexical_rescue(data,i,j):
    rowi,rowj = data.iloc[i],data.iloc[j]
    bi = str(rowi.get('brand','')).strip().lower()
    bj = str(rowj.get('brand','')).strip().lower()
    ni = str(rowi.get('name','')).strip().lower()
    nj = str(rowj.get('name','')).strip().lower()
    mi = _model_tokens(ni); mj = _model_tokens(nj)
    shared = mi&mj
    if bi and bj and bi==bj and shared: return True
    if shared and _title_overlap_ok(ni,nj): return True
    return False

def lsh_block(vectors, data):
    from lshashpy3 import LSHash
    lsh = LSHash(hash_size=15, input_dim=384, num_hashtables=8)
    for ix,vec in enumerate(vectors): lsh.index(vec, extra_data=ix)

    cand = defaultdict(dict)
    for ix,vec in enumerate(vectors):
        for r in lsh.query(vec, num_results=None, distance_func='cosine'):
            if r[0][1] is None: continue
            jx = int(r[0][1])
            if jx==ix: continue
            sim = 1-r[1]
            if sim>cand[ix].get(jx,-1): cand[ix][jx]=sim

    graph = defaultdict(set)
    n = len(vectors)
    for ix in range(n):
        neigh = sorted(cand[ix].items(),key=lambda x:x[1],reverse=True)[:6]
        top_ids = {nid for nid,_ in neigh}
        for jx,sim in neigh:
            rev = sorted(cand[jx].items(),key=lambda x:x[1],reverse=True)[:6]
            rev_ids = {nid for nid,_ in rev}
            mutual = (jx in top_ids) and (ix in rev_ids)
            add = False
            if sim>=STRONG_SIM_THRESHOLD: add = True
            elif mutual and sim>=LSH_THRESHOLD: add = True
            elif sim>=LSH_THRESHOLD+0.04 and (jx in top_ids or ix in rev_ids): add = True
            elif _lexical_rescue(data,ix,jx): add = True
            if add:
                graph[ix].add(jx); graph[jx].add(ix)

    brand_groups = defaultdict(list)
    for ix in range(n):
        brand = str(data.iloc[ix].get('brand','')).strip().lower()
        if brand and brand not in {'nan','none','missing'}: brand_groups[brand].append(ix)
    for members in brand_groups.values():
        if len(members)<2: continue
        members = members[:36*2]
        for a_pos,a in enumerate(members):
            ma = _model_tokens(str(data.iloc[a].get('name','')))
            for b in members[a_pos+1:]:
                mb = _model_tokens(str(data.iloc[b].get('name','')))
                if len(ma&mb)>=1 or _title_overlap_ok(str(data.iloc[a].get('name','')),
                                                       str(data.iloc[b].get('name',''))):
                    graph[a].add(b); graph[b].add(a)
    token_groups = defaultdict(list)
    for ix in range(n):
        toks = sorted(_tokenize(str(data.iloc[ix].get('name',''))))[:8]
        for tok in toks: token_groups[tok].append(ix)
    pair_counts = defaultdict(int)
    for members in token_groups.values():
        if len(members)<2 or len(members)>36*3: continue
        for i in range(len(members)):
            for j in range(i+1,len(members)):
                a,b = members[i],members[j]
                if a>b: a,b=b,a
                pair_counts[(a,b)] += 1
    for (a,b),cnt in pair_counts.items():
        if cnt<3: continue
        ta = _tokenize(str(data.iloc[a].get('name','')))
        tb = _tokenize(str(data.iloc[b].get('name','')))
        union = len(ta|tb); jacc = len(ta&tb)/union if union else 0.0
        if jacc>=0.28: graph[a].add(b); graph[b].add(a)

    for ix in range(n): graph[ix]

    visited = set(); comps = []
    for node in range(n):
        if node in visited: continue
        stack = [node]; comp = []
        while stack:
            cur = stack.pop()
            if cur in visited: continue
            visited.add(cur); comp.append(cur)
            stack.extend(graph[cur]-visited)
        comps.append(sorted(comp))

    purified = []
    for comp in comps:
        if len(comp)<=2:
            purified.append(comp)
            continue
        comp_vecs = np.array([vectors[i] for i in comp])
        centroid = np.mean(comp_vecs,axis=0)
        norms = np.linalg.norm(comp_vecs,axis=1)*(np.linalg.norm(centroid)+1e-12)
        sims = (comp_vecs @ centroid)/np.maximum(norms,1e-12)
        keep = [comp[i] for i,s in enumerate(sims) if s>=BLOCK_MEDOID_KEEP_THRESHOLD]
        fringe = [comp[i] for i,s in enumerate(sims) if s<BLOCK_MEDOID_KEEP_THRESHOLD]
        if len(keep)<=1:
            purified.append(comp)
        else:
            purified.append(sorted(keep))
            for idx in fringe: purified.append([idx])
    safe_print("lsh done")
    return [c for c in purified if c]

# ======================== DIAGNOSTICS ========================
def compute_coverage(clusters, ground_truth, data):
    if not ground_truth: return None
    id_col = get_id_column(data)
    if not id_col: return None
    valid_ids = set(int(x) for x in data[id_col].tolist())
    row_to_id = {int(i): int(v) for i, v in enumerate(data[id_col].tolist())}
    cluster_sets = []
    for cluster in clusters:
        mapped = set()
        for x in cluster:
            if x is None: continue
            try: ix = int(x)
            except: continue
            if ix in row_to_id: mapped.add(row_to_id[ix])
            elif ix in valid_ids: mapped.add(ix)
        cluster_sets.append(mapped)

    total_true_pairs = 0; covered_true_pairs = 0
    gt_cluster_count = 0; fully_covered = 0
    for cluster in ground_truth:
        cluster_ids = [int(x) for x in cluster if int(x) in valid_ids]
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
            fully_covered += 1
    return {
        "pair_completeness": covered_true_pairs / total_true_pairs if total_true_pairs else 0.0,
        "full_cluster_coverage": fully_covered / gt_cluster_count if gt_cluster_count else 0.0,
    }

def print_coverage(label, cov):
    if cov:
        safe_print(f"{label}: pair completeness={cov['pair_completeness']:.4f}, "
                   f"full cluster coverage={cov['full_cluster_coverage']:.4f}")

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
    _request_timestamps = [t for t in _request_timestamps if now-t<60]
    if len(_request_timestamps) >= REQUESTS_PER_MINUTE:
        time.sleep(60-(now-_request_timestamps[0])+0.05)
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

# ======================== SEPARATION PROMPT (unchanged) ========================
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
    prompt = (
        "Cluster the following records into groups that refer to the same real-world entity.\n"
        "Output a Python list of lists of record IDs.\n"
        "Every record ID must appear exactly once.\n"
        "Example: [[101, 102], [103, 104, 105]]\n\n"
        "Records:\n" + "\n".join(lines)
    )
    return prompt

# ======================== PARSING ========================
def parse_llm_clusters(text, id_map):
    content = text.strip()
    try:
        parsed = ast.literal_eval(content)
        if not isinstance(parsed,list): raise ValueError
    except:
        matches = re.findall(r'\[([^\]]*)\]', content)
        parsed = []
        for m in matches:
            nums = [int(p.strip()) for p in m.split(',') if p.strip().lstrip('-').isdigit()]
            if nums: parsed.append(nums)
    result = []
    for cl in parsed:
        rows = []
        for cid in cl:
            if cid in id_map: rows.append(id_map[cid])
        if rows: result.append(rows)
    return result

# ======================== MDG ========================
def mdg_check(clusters, sim_matrix):
    for ci,cluster in enumerate(clusters):
        for ridx in cluster:
            intra = [sim_matrix[ridx][r2] for r2 in cluster if r2!=ridx]
            if not intra: continue
            min_intra = min(intra)
            max_inter = -1.0
            for cj,other in enumerate(clusters):
                if cj==ci: continue
                inter = [sim_matrix[ridx][r2] for r2 in other]
                if inter: max_inter = max(max_inter, max(inter))
            if max_inter > min_intra - MDG_MARGIN:
                return False
    return True

def find_misclustered(clusters, sim_matrix):
    for ci,cluster in enumerate(clusters):
        for ridx in cluster:
            intra = [sim_matrix[ridx][r2] for r2 in cluster if r2!=ridx]
            if not intra: continue
            min_intra = min(intra)
            max_inter = -1.0; target_ci = None
            for cj,other in enumerate(clusters):
                if cj==ci: continue
                inter = [sim_matrix[ridx][r2] for r2 in other]
                if inter and max(inter)>max_inter:
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
    for idx,cl in enumerate(clusters):
        if idx==target_ci: continue
        for r in cl:
            if r!=ridx: ordered.append(r)
    return ordered

def cluster_one_set(indices, df, sim_matrix, id_map):
    current_order = indices[:]
    total_in = 0; total_out = 0; calls = 0
    for attempt in range(1+MAX_MDG_RETRIES):
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

def llm_separate_block(block_indices, df, sim_matrix, id_map):
    if len(block_indices) == 1:
        return [[block_indices]], 0, 0, 0
    clusters, in_t, out_t, calls = cluster_one_set(block_indices, df, sim_matrix, id_map)
    return [clusters], calls, in_t, out_t

# ======================== CHANGE 4: HIERARCHICAL ANTI‑TRANSITIVITY MERGE ========================
def centroid_of(cluster, vectors):
    return np.mean([vectors[i] for i in cluster], axis=0)

def representative(cluster, vectors):
    c = centroid_of(cluster, vectors)
    dists = [np.linalg.norm(vectors[i]-c) for i in cluster]
    return cluster[np.argmin(dists)]

def merge_prompt(rep_indices, df):
    id_col = get_id_column(df)
    lines = []
    for cid, ridx in enumerate(rep_indices):
        row = df.iloc[ridx]
        rid = row[id_col] if id_col else ridx
        parts = [f"{col}={str(row[col])}" for col in COLUMNS_TO_USE if col in df.columns and pd.notna(row[col])]
        lines.append(f"Cluster_{cid}: " + ", ".join(parts))
    prompt = (
        "You are an expert in Entity Resolution.\n"
        "Merge the clusters below that refer to the same real-world product.\n"
        "Look at brand, model numbers, and name similarity.\n"
        "Output a Python list of lists of cluster indices (0‑based).\n"
        "Every index must appear exactly once.\n"
        "Example: [[0,2], [1], [3,4]]\n\n"
        "Clusters:\n" + "\n".join(lines)
    )
    return prompt

def hierarchical_merge_antitrans(initial_clusters, vectors, df, sim_matrix, id_map):
    """
    Anti‑transitivity aware hierarchical merge with conservative LLM instruction.
    """
    clusters = [sorted(c) for c in initial_clusters if c]
    if len(clusters) <= 1:
        return clusters, 0, 0, 0

    total_calls = 0
    total_in = 0
    total_out = 0
    round_no = 0

    if not hasattr(hierarchical_merge_antitrans, "cannot_merge"):
        hierarchical_merge_antitrans.cannot_merge = set()

    # CHANGE 2: Conservative system instruction for merge calls
    conservative_system = (
        "You are an expert in Entity Resolution. "
        "Only merge clusters if you are absolutely certain they refer to the same product. "
        "If there is any doubt, keep them separate."
    )

    while round_no < MERGE_MAX_ROUNDS:
        round_no += 1
        centroids = [centroid_of(c, vectors) for c in clusters]
        n = len(clusters)
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                sim = float(np.dot(centroids[i], centroids[j]) / (
                    np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]) + 1e-9))
                if sim >= CANDIDATE_SIM_THRESH:
                    edges.append((i, j))
        if not edges:
            break

        graph = defaultdict(set)
        for i, j in edges:
            graph[i].add(j)
            graph[j].add(i)

        visited = set()
        components = []
        for node in range(n):
            if node in visited:
                continue
            comp = []
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                stack.extend(graph[cur] - visited)
            components.append(comp)

        merge_happened = False

        for comp in components:
            allowed_edges = [(i, j) for i in comp for j in comp if i < j and (i, j) not in hierarchical_merge_antitrans.cannot_merge]
            if not allowed_edges:
                continue

            sub_graph = defaultdict(set)
            for i, j in allowed_edges:
                sub_graph[i].add(j)
                sub_graph[j].add(i)

            sub_visited = set()
            groups = []
            for node in comp:
                if node in sub_visited:
                    continue
                group = []
                queue = [node]
                sub_visited.add(node)
                while queue and len(group) < MAX_CLUSTERS_PER_PROMPT:
                    cur = queue.pop(0)
                    group.append(cur)
                    for nb in sub_graph[cur]:
                        if nb not in sub_visited and len(group) < MAX_CLUSTERS_PER_PROMPT:
                            sub_visited.add(nb)
                            queue.append(nb)
                groups.append(group)

            for group in groups:
                reps = [representative(clusters[idx], vectors) for idx in group]
                prompt = merge_prompt(reps, df)
                text, in_t, out_t = call_gemini(conservative_system, prompt)   # conservative system instruction
                total_calls += 1
                total_in += in_t
                total_out += out_t
                try:
                    merge_groups = ast.literal_eval(text.strip())
                    if not isinstance(merge_groups, list):
                        merge_groups = [[i] for i in range(len(group))]
                except:
                    merge_groups = [[i] for i in range(len(group))]

                global_merge_groups = []
                for mg in merge_groups:
                    global_merge_groups.append([group[i] for i in mg])

                new_clusters = {}
                for global_group in global_merge_groups:
                    if len(global_group) == 1:
                        new_clusters[global_group[0]] = clusters[global_group[0]]
                    else:
                        merged = []
                        for idx in global_group:
                            merged.extend(clusters[idx])
                        merged = sorted(set(merged))
                        new_clusters[global_group[0]] = merged
                        for other in global_group[1:]:
                            new_clusters[other] = None

                # Update anti‑transitivity
                for i in range(len(global_merge_groups)):
                    for j in range(i+1, len(global_merge_groups)):
                        for a in global_merge_groups[i]:
                            for b in global_merge_groups[j]:
                                hierarchical_merge_antitrans.cannot_merge.add((a, b))
                                hierarchical_merge_antitrans.cannot_merge.add((b, a))

                for idx, cl in new_clusters.items():
                    if cl is not None:
                        clusters[idx] = cl
                    else:
                        clusters[idx] = []
                merge_happened = True

        clusters = [c for c in clusters if c]
        if not merge_happened:
            break

    return clusters, total_calls, total_in, total_out

# ======================== EVALUATION ========================
def load_ground_truth(file_path):
    def merge_coords(coords):
        uf = UnionFind(); ids = set()
        for l,r in coords:
            try: l,r = int(l),int(r)
            except: continue
            uf.add(l); uf.add(r); uf.union(l,r); ids.update([l,r])
        groups = defaultdict(list)
        for i in ids: groups[uf.find(i)].append(i)
        return [v for v in groups.values()]
    if file_path.endswith('.txt'):
        with open(file_path,'r',encoding='utf-8') as f:
            clusters = []
            for line in f:
                parts = line.strip().split()
                if parts:
                    try: clusters.append([int(p) for p in parts])
                    except: continue
            return clusters
    elif file_path.endswith('.csv'):
        try: df = pd.read_csv(file_path,encoding='utf-8-sig')
        except: df = pd.read_csv(file_path,encoding='utf-8')
        if df.shape[1]>=2: return merge_coords(df.iloc[:,:2].values.tolist())
    return []

def compute_metrics(gt,pred):
    true_pairs = get_pairs(gt); pred_pairs = get_pairs(pred)
    tp = len(true_pairs&pred_pairs)
    fp = len(pred_pairs-true_pairs)
    fn = len(true_pairs-pred_pairs)
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn) if tp+fn else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    total_pred = sum(len(c) for c in pred)
    correct_pur = 0
    for pc in pred:
        cnt = defaultdict(int)
        for sid in pc:
            for tc in gt:
                if sid in tc: cnt[tuple(tc)]+=1
        if cnt: correct_pur += max(cnt.values())
    pur = correct_pur/total_pred if total_pred else 0
    total_true = sum(len(c) for c in gt)
    correct_inv = 0
    for tc in gt:
        if tc:
            cnt = defaultdict(int)
            for sid in tc:
                for pc2 in pred:
                    if sid in pc2: cnt[tuple(pc2)]+=1
            if cnt: correct_inv += max(cnt.values())
    inv_pur = correct_inv/total_true if total_true else 0
    fpm = 2*pur*inv_pur/(pur+inv_pur) if pur+inv_pur else 0
    all_ids = sorted(set(s for c in gt for s in c)|set(s for c in pred for s in c))
    if not all_ids: ari = 0
    else:
        id_map = {sid:i for i,sid in enumerate(all_ids)}
        true_labels = [-1]*len(all_ids); pred_labels = [-1]*len(all_ids)
        for i,c in enumerate(gt):
            for sid in c: true_labels[id_map[sid]] = i
        for i,c in enumerate(pred):
            for sid in c: pred_labels[id_map[sid]] = i
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(true_labels,pred_labels)
    return prec,rec,f1,tp,fp,fn,pur,inv_pur,fpm,ari

def write_report(pred_clusters, gt, out_path, extra_info=""):
    with open(out_path,'w',encoding='utf-8') as f:
        sep,dash = "="*80, "-"*80
        f.write(f"{sep}\nLLMCER - MODEL OUTPUT SUMMARY\n{sep}\n\n")
        f.write(f"Total Predicted Clusters:    {len(pred_clusters)}\n")
        f.write(f"Total Ground Truth Clusters: {len(gt) if gt else 'N/A'}\n\n")
        f.write(f"{dash}\nPREDICTED CLUSTERS (Product IDs):\n{dash}\n")
        for i,c in enumerate(pred_clusters):
            f.write(f"\nCluster {i+1} ({len(c)} products): {sorted(c)}\n")
        if gt:
            prec,rec,f1,tp,fp,fn,pur,inv,fpm,ari = compute_metrics(gt,pred_clusters)
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
    data_path = os.getenv("DATASET_PATH", "sample_solute.csv")
    gt_path = os.getenv("GROUND_TRUTH_PATH", "sample_solute_gt.csv")
    vectors, sim_matrix, data, id_col = load_data_and_vectorize(data_path)

    blocks = lsh_block(vectors, data)
    safe_print(f"Step 1 – LSH blocks: {len(blocks)}")

    gt = []
    if os.path.exists(gt_path):
        try: gt = load_ground_truth(gt_path)
        except: pass

    if gt:
        cov = compute_coverage(blocks, gt, data)
        print_coverage("After LSH", cov)

    id_map = {int(data.iloc[i][id_col]): i for i in range(len(data))}

    safe_print("Step 2 – LLM separation + MDG (whole blocks)")
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
        for s in all_sets:
            sep_clusters.extend(s)
        cov_sep = compute_coverage(sep_clusters, gt, data)
        print_coverage("After separation", cov_sep)

    # Step 3 – Anti‑transitivity merge
    safe_print("Step 3 – Hierarchical anti‑transitivity merge (conservative)")
    initial_clusters = []
    for s in all_sets:
        initial_clusters.extend(s)
    safe_print(f"  Starting with {len(initial_clusters)} clusters")
    final_clusters, merge_calls, merge_in, merge_out = hierarchical_merge_antitrans(
        initial_clusters, vectors, data, sim_matrix, id_map
    )
    safe_print(f"  After merging: {len(final_clusters)} clusters (LLM calls: {merge_calls})")

    # No sanitization step

    if gt:
        cov_merge = compute_coverage(final_clusters, gt, data)
        print_coverage("After merge", cov_merge)

    pred_clusters = []
    for cl in final_clusters:
        ids = [int(data.iloc[idx][id_col]) for idx in cl]
        if ids: pred_clusters.append(sorted(ids))

    total_time = time.time()-total_start
    total_tokens_in = sep_in + merge_in
    total_tokens_out = sep_out + merge_out
    total_tokens = total_tokens_in + total_tokens_out

    if gt:
        prec,rec,f1,tp,fp,fn,_,_,_,_ = compute_metrics(gt, pred_clusters)
        safe_print("\n"+"="*80)
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

if __name__=="__main__":
    main()