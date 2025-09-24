#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triple Evaluator
"""
import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Set

Triple = Tuple[str, str, str]

def _strip_comments_and_trailing_commas(text: str) -> str:
    text = re.sub(r'(?m)//.*$', '', text)
    text = re.sub(r'(?m)#.*$',  '', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text

def load_any_json(path: str) -> Dict[str, Any]:
    raw = open(path, 'r', encoding='utf-8').read()
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        cleaned = _strip_comments_and_trailing_commas(raw)
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        from json_repair import repair_json 
        repaired = repair_json(raw)
        return json.loads(repaired)
    except Exception:
        pass
    raise ValueError(f"Cannot parse the JSON")

def normalize_text(s: str, case_sensitive: bool, keep_underscores: bool, strip_punct: bool) -> str:
    if not isinstance(s, str):
        s = str(s)
    t = s.strip()
    if not case_sensitive:
        t = t.lower()
    if not keep_underscores:
        t = t.replace('_', ' ')
    t = re.sub(r'\s+', ' ', t)
    t = t.strip('\'"“”‘’')
    if strip_punct:
        t = re.sub(r'[.,;:]+\s*$', '', t)
    return t

def normalize_triples(triples: Iterable[Sequence[str]],
                      case_sensitive: bool,
                      keep_underscores: bool,
                      strip_punct: bool) -> List[Triple]:
    out: List[Triple] = []
    for t in triples:
        if len(t) != 3:
            continue
        s, p, o = t[0], t[1], t[2]
        out.append((
            normalize_text(s, case_sensitive, keep_underscores, strip_punct),
            normalize_text(p, case_sensitive, keep_underscores, strip_punct),
            normalize_text(o, case_sensitive, keep_underscores, strip_punct),
        ))
    return out

# handling mutiple formats such as tuple, or triples using subject/predicate/object
def extract_edges(obj: Dict[str, Any]) -> List[Triple]:
    edges = obj['edges']
    triples: List[Triple] = []
    if isinstance(edges, list):
        if not edges:
            return []
        first = edges[0]
        if isinstance(first, dict):
            for e in edges:
                s = e.get('subject', e.get('subj'))
                p = e.get('predicate', e.get('pred'))
                o = e.get('object', e.get('obj'))
                if s is None or p is None or o is None:
                    continue
                triples.append((str(s), str(p), str(o)))
        elif isinstance(first, (list, tuple)):
            for e in edges:
                if not isinstance(e, (list, tuple)) or len(e) < 3:
                    continue
                triples.append((str(e[0]), str(e[1]), str(e[2])))
        else:
            for e in edges:
                if isinstance(e, str) and '|' in e:
                    parts = [x.strip() for x in e.split('|')]
                    if len(parts) >= 3:
                        triples.append((parts[0], parts[1], parts[2]))
    return triples

# for OpenIE
def coerce_obj_to_graph(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        if 'edges' in obj and isinstance(obj['edges'], list):
            return obj
        for k in ('triples','relations','edges_list'):
            if k in obj and isinstance(obj[k], list):
                return {'edges': obj[k]}
    elif isinstance(obj, list):
        edges = []
        for e in obj:
            if isinstance(e, dict):
                s = e.get('subject', e.get('subj'))
                p = e.get('predicate', e.get('pred'))
                o = e.get('object', e.get('obj'))
                if s is not None and p is not None and o is not None:
                    edges.append((str(s), str(p), str(o)))
            elif isinstance(e, (list, tuple)) and len(e) >= 3:
                edges.append((str(e[0]), str(e[1]), str(e[2])))
        if edges:
            return {'edges': edges}
    raise ValueError("Cannot get edges from the list")

def load_graph_file(path: str) -> Dict[str, Any]:
    # JSON first
    try:
        obj = load_any_json(path)
        return coerce_obj_to_graph(obj)
    except Exception:
        pass
    # Try OpenIE tuple-per-line
    import ast
    edges = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if line.endswith(','):
                line = line[:-1]
            if line.startswith('(') and ')' in line:
                try:
                    tup = ast.literal_eval(line)
                    if isinstance(tup, (list, tuple)) and len(tup) >= 3:
                        s, p, o = tup[0], tup[1], tup[2]
                        edges.append((str(s), str(p), str(o)))
                except Exception:
                    continue
    return {'edges': edges}

def try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer, util, CrossEncoder
        return SentenceTransformer, util, CrossEncoder
    except Exception:
        return None, None, None

def embed_texts(texts: List[str], model_name: str):
    SentenceTransformer, util, _ = try_import_sentence_transformers()
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embs, util

def cosine_sim_matrix(embA, embB, util):
    return util.cos_sim(embA, embB).cpu().numpy()

# Matching
def greedy_max_matching(scores, threshold: float) -> List[Tuple[int, int, float]]:
    import numpy as np
    scores = scores.copy()
    m, n = scores.shape
    matched = []
    used_rows = set()
    used_cols = set()
    while True:
        best = None
        best_val = threshold
        for i in range(m):
            if i in used_rows: continue
            for j in range(n):
                if j in used_cols: continue
                val = scores[i, j]
                if val >= best_val:
                    best_val = val
                    best = (i, j, val)
        if best is None:
            break
        i, j, val = best
        matched.append(best)
        used_rows.add(i); used_cols.add(j)
    return matched

def match_with_policy(scores, threshold: float, policy: str) -> List[Tuple[int,int,float]]:
    if policy == "greedy1to1":
        return greedy_max_matching(scores, threshold)
    else:
        return greedy_max_matching(scores, threshold)

def build_vocab_and_embeddings(triplesA: List[Triple], triplesB: List[Triple], model_name: str):
    texts = []
    for t in triplesA + triplesB:
        texts.extend([t[0], t[1], t[2]])
    unique = sorted(set(texts))
    embs, util = embed_texts(unique, model_name)
    index = {s: k for k, s in enumerate(unique)}
    return unique, embs, index, util

def triple_scores_min(pred_triples: List[Triple], gold_triples: List[Triple], model_name: str):
    import numpy as np
    unique, embs, index, util = build_vocab_and_embeddings(pred_triples, gold_triples, model_name)
    def idx(s): return index[s]
    sA = np.array([idx(t[0]) for t in pred_triples]); pA = np.array([idx(t[1]) for t in pred_triples]); oA = np.array([idx(t[2]) for t in pred_triples])
    sB = np.array([idx(t[0]) for t in gold_triples]); pB = np.array([idx(t[1]) for t in gold_triples]); oB = np.array([idx(t[2]) for t in gold_triples])
    S = cosine_sim_matrix(embs[sA], embs[sB], util)
    P = cosine_sim_matrix(embs[pA], embs[pB], util)
    O = cosine_sim_matrix(embs[oA], embs[oB], util)
    scores = (S if S is not None else 0)
    scores = np.minimum(np.minimum(S, P), O)
    return scores, (S, P, O)

def triple_scores_pred_exact(pred_triples: List[Triple], gold_triples: List[Triple], model_name: str):
    import numpy as np
    unique, embs, index, util = build_vocab_and_embeddings(pred_triples, gold_triples, model_name)
    def idx(s): return index[s]
    sA = np.array([idx(t[0]) for t in pred_triples]); pA = [t[1] for t in pred_triples]; oA = np.array([idx(t[2]) for t in pred_triples])
    sB = np.array([idx(t[0]) for t in gold_triples]); pB = [t[1] for t in gold_triples]; oB = np.array([idx(t[2]) for t in gold_triples])
    S = cosine_sim_matrix(embs[sA], embs[sB], util)
    O = cosine_sim_matrix(embs[oA], embs[oB], util)
    scores = np.zeros_like(S, dtype=float)
    for i, pa in enumerate(pA):
        for j, pb in enumerate(pB):
            if pa == pb:
                scores[i, j] = min(S[i, j], O[i, j])
    return scores, (S, None, O)

def triple_scores_concat(pred_triples: List[Triple], gold_triples: List[Triple], model_name: str, delim_p: str, delim_o: str):
    SentenceTransformer, util, _ = try_import_sentence_transformers()
    sb = SentenceTransformer(model_name)
    def to_str(t: Triple) -> str:
        return f"{t[0]}{delim_p}{t[1]}{delim_o}{t[2]}"
    A = [to_str(t) for t in pred_triples]
    B = [to_str(t) for t in gold_triples]
    embA = sb.encode(A, convert_to_tensor=True, normalize_embeddings=True)
    embB = sb.encode(B, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(embA, embB).cpu().numpy()
    return scores, None

def triple_scores_weighted(pred_triples: List[Triple], gold_triples: List[Triple], model_name: str, weights: Tuple[float,float,float]):
    import numpy as np
    unique, embs, index, util = build_vocab_and_embeddings(pred_triples, gold_triples, model_name)
    def idx(s): return index[s]
    sA = np.array([idx(t[0]) for t in pred_triples]); pA = np.array([idx(t[1]) for t in pred_triples]); oA = np.array([idx(t[2]) for t in pred_triples])
    sB = np.array([idx(t[0]) for t in gold_triples]); pB = np.array([idx(t[1]) for t in gold_triples]); oB = np.array([idx(t[2]) for t in gold_triples])
    S = cosine_sim_matrix(embs[sA], embs[sB], util)
    P = cosine_sim_matrix(embs[pA], embs[pB], util)
    O = cosine_sim_matrix(embs[oA], embs[oB], util)
    ws, wp, wo = weights
    scores = ws*S + wp*P + wo*O
    return scores, (S, P, O)

def triple_scores_cross_encoder(pred_triples: List[Triple], gold_triples: List[Triple], ce_model: str):
    import numpy as np
    SentenceTransformer, _, CrossEncoder = try_import_sentence_transformers()
    ce = CrossEncoder(ce_model)
    def t2s(t: Triple) -> str:
        return f"{t[0]} [P] {t[1]} [O] {t[2]}"
    A = [t2s(t) for t in pred_triples]
    B = [t2s(t) for t in gold_triples]
    m, n = len(A), len(B)
    scores = np.zeros((m, n), dtype=float)
    batch = 128
    for i in range(m):
        pairs = [(A[i], B[j]) for j in range(n)]
        for k in range(0, n, batch):
            sub = pairs[k:k+batch]
            s = ce.predict(sub)  # numpy array
            scores[i, k:k+batch] = s
    return scores, None

def compute_semantic_triple_scores(pred_triples: List[Triple], gold_triples: List[Triple],
                                   mode: str, model_name: str, weights, ce_model,
                                   min_subj: float, min_pred: float, min_obj: float,
                                   delim_p: str, delim_o: str):
    if mode == "min":
        scores, parts = triple_scores_min(pred_triples, gold_triples, model_name)
        S, P, O = parts
        import numpy as np
        mask = (S >= min_subj) & (P >= min_pred) & (O >= min_obj)
        scores = np.where(mask, scores, -1.0)
        return scores
    elif mode == "pred-exact":
        scores, parts = triple_scores_pred_exact(pred_triples, gold_triples, model_name)
        S, _, O = parts
        import numpy as np
        mask = (S >= min_subj) & (O >= min_obj)
        scores = np.where(mask, scores, -1.0)
        return scores
    elif mode == "concat":
        scores, _ = triple_scores_concat(pred_triples, gold_triples, model_name, delim_p, delim_o)
        return scores
    elif mode == "weighted":
        scores, parts = triple_scores_weighted(pred_triples, gold_triples, model_name, weights)
        S, P, O = parts
        import numpy as np
        mask = (S >= min_subj) & (P >= min_pred) & (O >= min_obj)
        scores = np.where(mask, scores, -1.0)
        return scores
    elif mode == "cross-enc":
        scores, _ = triple_scores_cross_encoder(pred_triples, gold_triples, ce_model)
        return scores
    else:
        raise ValueError("Unknown sem-mode")

def metrics_from_scores(scores, threshold: float, policy: str, pred_items: List, gold_items: List):
    matches = match_with_policy(scores, threshold, policy)
    matched_pred = set(i for i, j, s in matches)
    matched_gold = set(j for i, j, s in matches)
    TP = len(matched_pred)
    FP = len(pred_items) - TP
    FN = len(gold_items) - len(matched_gold)
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2*precision*recall / (precision+recall)) if (precision+recall) else 0.0
    return TP, FP, FN, precision, recall, f1

def nodes_from_edges(triples: List[Triple]) -> List[str]:
    seen = set(); out = []
    for s,p,o in triples:
        if s not in seen: seen.add(s); out.append(s)
        if o not in seen: seen.add(o); out.append(o)
    return out

def parse_list_arg(val: str) -> List[str]:
    if not val: return []
    parts = []
    for tok in val.replace(',', ' ').split():
        parts.append(tok)
    return parts

def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Ontology triple evaluator: multi-gold + multi-pred averaging & dual-group plot")
    ap.add_argument("pred", nargs="?", help="optional: --pred-list）")
    ap.add_argument("--pred-list", type=str, default=None, help="mutiple files, using comma to seperate")
    ap.add_argument("--pred1-label", type=str, default=None, help="label in the figure, e.g. pred-avg or pred）")

    ap.add_argument("--pred2", type=str, default=None, help="single file for openIE")
    ap.add_argument("--pred2-list", type=str, default=None, help="optional")
    ap.add_argument("--pred2-label", type=str, default=None, help="label: pred2-avg or pred2）")

    ap.add_argument("--gold-short", type=str, required=True)
    ap.add_argument("--gold-medium", type=str, required=True)
    ap.add_argument("--gold-long", type=str, required=True)

    ap.add_argument("--case-sensitive", action="store_true")
    ap.add_argument("--keep-underscores", action="store_true")
    ap.add_argument("--no-strip-punct", action="store_true")

    ap.add_argument("--node-sim", action="store_true", help="Compute semantic metrics for the node set (by default, nodes are derived from edges).")
    ap.add_argument("--use-nodes-field", action="store_true", help="Use the nodes passed from the files")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--match-policy", type=str, default="greedy1to1",
                    choices=["greedy1to1"])

    ap.add_argument("--semantic", action="store_true", help="")
    ap.add_argument("--sem-mode", type=str, default="concat", choices=["min","pred-exact","concat","weighted","cross-enc"])
    ap.add_argument("--weights", type=str, default="0.3,0.4,0.3")
    ap.add_argument("--min-subj", type=float, default=0.0)
    ap.add_argument("--min-pred", type=float, default=0.0)
    ap.add_argument("--min-obj",  type=float, default=0.0)
    ap.add_argument("--delim-p", type=str, default=" [P] ")
    ap.add_argument("--delim-o", type=str, default=" [O] ")

    ap.add_argument("--t-start", type=float, default=0.10)
    ap.add_argument("--t-end",   type=float, default=0.90)
    ap.add_argument("--t-step",  type=float, default=0.05)
    ap.add_argument("--out-csv", type=str, default="sweep_metrics.csv")
    ap.add_argument("--plot-out", type=str, default=None, help="save figure to .png/.pdf")
    ap.add_argument("--plot-ci", action="store_true", help="show ±std shadow")
    ap.add_argument("--plot-ymin", type=float, default=0.0, help="y's min（default 0.0）")
    ap.add_argument("--plot-ymax", type=float, default=1.0, help="y's max（default 1.0）")
    ap.add_argument("--plot-tick-step", type=float, default=0.1, help="y step（default 0.1）")

    args = ap.parse_args(argv[1:])

    group1 = []
    if args.pred: group1.append(args.pred)
    group1 += parse_list_arg(args.pred_list) if args.pred_list else []

    group2 = []
    if args.pred2: group2.append(args.pred2)
    group2 += parse_list_arg(args.pred2_list) if args.pred2_list else []

    label1 = args.pred1_label or ("pred-avg" if len(group1)>1 else "pred")
    label2 = args.pred2_label or (("pred2-avg" if len(group2)>1 else "pred2") if group2 else "")

    gold_paths = {"short": args.gold_short, "medium": args.gold_medium, "long": args.gold_long}

    def norm_triples_from_obj(obj):
        edges_raw = extract_edges(obj)
        return normalize_triples(edges_raw, args.case_sensitive, args.keep_underscores, not args.no_strip_punct)

    def load_graph(path: str) -> Dict[str, Any]:
        try:
            return load_graph_file(path)
        except Exception as e:
            print(f"Read failed {path}: {e}", file=sys.stderr); sys.exit(3)

    ts = []
    t = args.t_start
    while t <= args.t_end + 1e-9:
        ts.append(round(t, 5))
        t += args.t_step

    try:
        ws, wp, wo = [float(x) for x in args.weights.split(",")]
    except Exception:
        ws, wp, wo = 0.3, 0.4, 0.3
    ssum = ws + wp + wo
    if ssum <= 0:
        ws, wp, wo = 0.33, 0.34, 0.33
    else:
        ws, wp, wo = ws/ssum, wp/ssum, wo/ssum

    gold_triple_sets = {}; gold_node_sets = {}
    for tag, path in gold_paths.items():
        obj = load_graph(path)
        tri = norm_triples_from_obj(obj)
        gold_triple_sets[tag] = tri
        if args.use_nodes_field and "nodes" in obj and isinstance(obj["nodes"], list):
            nodes = []
            for x in obj["nodes"]:
                tnorm = normalize_text(str(x), args.case_sensitive, args.keep_underscores, not args.no_strip_punct)
                if tnorm not in nodes: nodes.append(tnorm)
        else:
            nodes = nodes_from_edges(tri)
        gold_node_sets[tag] = nodes

    SentenceTransformer, util, _ = try_import_sentence_transformers()
    from sentence_transformers import SentenceTransformer as STModel
    sbert = STModel(args.model)
    def embed(list_text):
        return sbert.encode(list_text, convert_to_tensor=True, normalize_embeddings=True)

    def per_run_metrics_for_group(pred_path: str, tag: str):
        obj = load_graph(pred_path)
        tri = norm_triples_from_obj(obj)
        nodes_pred = nodes_from_edges(tri)

        if args.node_sim:
            emb_pn = embed(nodes_pred)
            emb_gn = embed(gold_node_sets[tag])
            sims_nodes = util.cos_sim(emb_pn, emb_gn).cpu().numpy()
        else:
            sims_nodes = None

        if args.semantic:
            scores_tri = compute_semantic_triple_scores(
                tri, gold_triple_sets[tag], mode=args.sem_mode, model_name=args.model, weights=(ws,wp,wo),
                ce_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                min_subj=args.min_subj, min_pred=args.min_pred, min_obj=args.min_obj,
                delim_p=args.delim_p, delim_o=args.delim_o
            )
        else:
            scores_tri = None

        return {
            "pred_triples": tri,
            "pred_nodes": nodes_pred,
            "sims_nodes": sims_nodes,
            "scores_triples": scores_tri,
        }

    import numpy as np
    def sweep_group(group_files: List[str], label: str):
        out = {"label": label, "kinds": {"node": {}, "triple": {}}}
        for tag in gold_paths.keys():
            if args.node_sim:
                runs = []
                pred_counts = []
                gold_count = len(gold_node_sets[tag])
                for f in group_files:
                    r = per_run_metrics_for_group(f, tag)
                    sims = r["sims_nodes"]
                    pn = r["pred_nodes"]
                    if sims is None:
                        continue
                    pred_counts.append(len(pn))
                    metrics_each_thr = [metrics_from_scores(sims, thr, args.match_policy, pn, gold_node_sets[tag]) for thr in ts]
                    runs.append(np.array([[m[3], m[4], m[5]] for m in metrics_each_thr]))
                if runs:
                    arr = np.stack(runs, axis=0)
                    mean = arr.mean(axis=0); std = arr.std(axis=0)
                    out["kinds"]["node"][tag] = {"mean": mean, "std": std, "pred_count_mean": float(np.mean(pred_counts)), "gold_count": gold_count}
            if args.semantic:
                runs = []
                pred_counts = []
                gold_count = len(gold_triple_sets[tag])
                for f in group_files:
                    r = per_run_metrics_for_group(f, tag)
                    scores = r["scores_triples"]
                    pt = r["pred_triples"]
                    if scores is None:
                        continue
                    pred_counts.append(len(pt))
                    metrics_each_thr = [metrics_from_scores(scores, thr, args.match_policy, pt, gold_triple_sets[tag]) for thr in ts]
                    runs.append(np.array([[m[3], m[4], m[5]] for m in metrics_each_thr]))
                if runs:
                    arr = np.stack(runs, axis=0)
                    mean = arr.mean(axis=0); std = arr.std(axis=0)
                    out["kinds"]["triple"][tag] = {"mean": mean, "std": std, "pred_count_mean": float(np.mean(pred_counts)), "gold_count": gold_count}
        return out

    group1_stats = sweep_group(group1, label1)
    group2_stats = sweep_group(group2, label2) if group2 else None

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pred_label","kind","gold","threshold","precision_mean","recall_mean","f1_mean",
                    "precision_std","recall_std","f1_std","pred_count_mean","gold_count","policy","model","sem_mode"])
        def dump_group(gstats):
            if not gstats: return
            for kind in ("node","triple"):
                if kind not in gstats["kinds"]: continue
                for tag, rec in gstats["kinds"][kind].items():
                    mean = rec["mean"]; std = rec["std"]
                    for idx, thr in enumerate(ts):
                        Pm, Rm, Fm = mean[idx]
                        Ps, Rs, Fs = std[idx]
                        w.writerow([gstats["label"], kind, tag, thr, Pm, Rm, Fm, Ps, Rs, Fs,
                                    rec.get("pred_count_mean", 0.0), rec.get("gold_count", 0), args.match_policy, args.model, args.sem_mode])
        dump_group(group1_stats); dump_group(group2_stats)
    print(f"[OK] CSV is ready：{args.out_csv}")

    if args.plot_out:
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import matplotlib.ticker as mticker
            import numpy as np
        except Exception as e:
            print(f"[WARN] plot failed：{e}.")
            return 0
        df = pd.read_csv(args.out_csv)
        orders = ["short","medium","long"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

        groups = [label1] + ([label2] if group2_stats else [])
        color_map = {"precision_mean": "C0", "recall_mean": "C1", "f1_mean": "C2"}
        ls_map = {label1: "-"}
        if group2_stats: ls_map[label2] = "--"

        def plot_block(ax, sub_df, title):
            if sub_df.empty:
                ax.set_visible(False)
                return
            for glabel in groups:
                gdf = sub_df[sub_df["pred_label"]==glabel].sort_values("threshold")
                if gdf.empty: continue
                ls = ls_map.get(glabel, "-")
                x = gdf["threshold"].values
                for metric_key, color in color_map.items():
                    y = gdf[metric_key].values
                    ax.plot(x, y, linestyle=ls, color=color, label=f"{glabel}: {metric_key.split('_')[0].capitalize()}")
                    if args.plot_ci:
                        ystd = gdf[metric_key.replace("_mean","_std")].values
                        ax.fill_between(x, y-ystd, y+ystd, color=color, alpha=0.12)
            ax.set_title(title)
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.set_ylim(args.plot_ymin, args.plot_ymax)
            if args.plot_tick_step > 0:
                ax.yaxis.set_major_locator(mticker.MultipleLocator(base=args.plot_tick_step))
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2, fontsize=8)

        for idx, tag in enumerate(orders):
            ax = axes[0, idx]
            sub = df[(df["kind"]=="node") & (df["gold"]==tag)]
            plot_block(ax, sub, f"Nodes vs {tag}")
        for idx, tag in enumerate(orders):
            ax = axes[1, idx]
            sub = df[(df["kind"]=="triple") & (df["gold"]==tag)]
            plot_block(ax, sub, f"Triples vs {tag}")
        fig.suptitle(f"Similarity sweep (mean±std) — policy={args.match_policy}, model={args.model}, sem={args.sem_mode}", fontsize=12)
        fig.savefig(args.plot_out, dpi=300)
        print(f"[OK] image is saved：{args.plot_out}")

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
