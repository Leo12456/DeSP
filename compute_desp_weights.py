#!/usr/bin/env python
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_minmax_norm(matrix):
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = np.nanmin(matrix)
    max_value = np.nanmax(matrix)
    if abs(max_value - min_value) < 1e-12:
        return np.zeros_like(matrix)
    return (matrix - min_value) / (max_value - min_value)


def cosine_distance_matrix(embeddings):
    return np.nan_to_num(pairwise_distances(embeddings, embeddings, metric="cosine"))


def extract_output_item(output):
    matches = re.findall(r'"([^"]*)"', output or "")
    if matches:
        return matches[-1]
    return (output or "").strip().strip('"')


def aggregate_neighbors(neighbors_idx, neighbor_similarity, lambda_val, initial_scores, num_iterations):
    rho = np.array(initial_scores, dtype=float)
    for iteration in range(num_iterations):
        new_rho = np.zeros_like(rho)
        for item_idx, neighbor_ids in enumerate(neighbors_idx):
            sim = np.array(neighbor_similarity[item_idx], dtype=float)
            neighbor_values = rho[neighbor_ids]
            neighbor_contribution = np.sum(sim * neighbor_values)
            new_rho[item_idx] = (1 - lambda_val) * neighbor_contribution + lambda_val * initial_scores[item_idx]

        change = np.abs(new_rho - rho).mean()
        rho = new_rho
        print(f"iteration {iteration + 1}/{num_iterations}, mean change: {change:.6f}")
        if change < 1e-6:
            break
    return rho


def compute_weights(args):
    data_dir = os.path.join(args.data_dir, args.dataset)
    train_path = os.path.join(data_dir, "train", args.train_file)
    id2name_path = os.path.join(data_dir, args.id2name_file)
    llm_embedding_path = os.path.join(data_dir, args.item_embedding_file)
    output_path = args.output_file or os.path.join(
        data_dir,
        "train",
        f"new_semantic_weight_lambda{args.lambda_val}.json",
    )

    train_data = load_json(train_path)
    id2name_raw = load_json(id2name_path)
    id2name = {int(key): value for key, value in id2name_raw.items()}
    name2id = {value: key for key, value in id2name.items()}

    raw_counts = defaultdict(int)
    for entry in train_data:
        item_name = extract_output_item(entry.get("output", ""))
        if item_name in name2id:
            raw_counts[name2id[item_name]] += 1

    if not raw_counts:
        raise ValueError("No training output items were matched in id2name.")

    llm_embeddings = torch.load(llm_embedding_path, map_location="cpu").float().numpy()
    sasrec_embeddings = None
    if args.sasrec_weight > 0:
        if not args.sasrec_embedding_path or not os.path.exists(args.sasrec_embedding_path):
            print("SASRec embedding not found; setting sasrec_weight to 0.")
            args.sasrec_weight = 0.0
        else:
            sasrec_embeddings = torch.load(args.sasrec_embedding_path, map_location="cpu").float().numpy()

    valid_ids = []
    names = []
    for item_id in sorted(raw_counts):
        if item_id not in id2name:
            continue
        if item_id >= llm_embeddings.shape[0]:
            continue
        if sasrec_embeddings is not None and item_id >= sasrec_embeddings.shape[0]:
            continue
        valid_ids.append(item_id)
        names.append(id2name[item_id])

    if not valid_ids:
        raise ValueError("No valid item ids remain after matching embeddings and id2name.")

    valid_ids = np.array(valid_ids, dtype=int)
    counts = np.array([raw_counts[int(item_id)] for item_id in valid_ids], dtype=float)
    popularity_scores = counts ** args.alpha

    llm_dist = cosine_distance_matrix(llm_embeddings[valid_ids])
    tfidf_matrix = TfidfVectorizer(
        max_features=args.tfidf_max_features,
        ngram_range=(1, 2),
        stop_words="english",
    ).fit_transform(names)
    tfidf_dist = pairwise_distances(tfidf_matrix, tfidf_matrix, metric="cosine")

    if sasrec_embeddings is not None:
        sasrec_dist = cosine_distance_matrix(sasrec_embeddings[valid_ids])
    else:
        sasrec_dist = np.zeros_like(llm_dist)

    fused_dist = (
        args.llm_weight * safe_minmax_norm(llm_dist)
        + args.tfidf_weight * safe_minmax_norm(tfidf_dist)
        + args.sasrec_weight * safe_minmax_norm(sasrec_dist)
    )

    neighbors_idx = []
    neighbors_sim = []
    for item_idx in range(len(valid_ids)):
        sorted_idx = np.argsort(fused_dist[item_idx])
        topk = [idx for idx in sorted_idx if idx != item_idx][: args.k]
        if not topk:
            topk = [item_idx]
        distances = fused_dist[item_idx][topk]
        sim = np.exp(-distances / args.tau)
        sim = sim / (sim.sum() + args.eps)
        neighbors_idx.append(topk)
        neighbors_sim.append(sim)

    print(
        f"computing DeSP weights: dataset={args.dataset}, items={len(valid_ids)}, "
        f"lambda={args.lambda_val}, k={args.k}"
    )
    rho = aggregate_neighbors(
        neighbors_idx,
        neighbors_sim,
        args.lambda_val,
        popularity_scores,
        args.num_iterations,
    )

    weights = {str(item_id): 1.0 for item_id in id2name}
    for idx, item_id in enumerate(valid_ids):
        weights[str(int(item_id))] = float(rho[idx])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(weights, file, indent=2, ensure_ascii=False)

    if len(rho) > 1:
        corr = np.corrcoef(rho, popularity_scores)[0, 1]
        rho_entropy = entropy(rho / rho.sum() + args.eps)
        pop_entropy = entropy(popularity_scores / popularity_scores.sum() + args.eps)
        print(f"rho/popularity correlation: {corr:.4f}")
        print(f"rho entropy: {rho_entropy:.4f}; popularity entropy: {pop_entropy:.4f}")

    print(f"saved weights to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute DeSP semantic debiasing weights.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data_dir.")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--train_file", default="train_4096.json")
    parser.add_argument("--id2name_file", default="id2name4Rec.json")
    parser.add_argument("--item_embedding_file", default="item_embedding.pt")
    parser.add_argument("--sasrec_embedding_path", default="")
    parser.add_argument("--output_file", default="")
    parser.add_argument("--lambda_val", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--llm_weight", type=float, default=4.0)
    parser.add_argument("--tfidf_weight", type=float, default=1.0)
    parser.add_argument("--sasrec_weight", type=float, default=0.0)
    parser.add_argument("--tfidf_max_features", type=int, default=1028)
    return parser.parse_args()


if __name__ == "__main__":
    compute_weights(parse_args())
