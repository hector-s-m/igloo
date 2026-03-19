"""
Split loop data into train/val sets using pre-computed structural clusters
or sequence clustering. Testing is done separately with a benchmark dataset.

Usage:
    # Using pre-computed TM-score clusters (recommended)
    python process_data/split_data.py \
        --input loops.jsonl \
        --output_dir splits/ \
        --cluster_csv SNAC-DataBase/cluster_information/ab_complexes/tm_threshold_0p90_summary.csv \
        --seed 42

    # Using MMseqs2 sequence clustering
    python process_data/split_data.py \
        --input loops.jsonl \
        --output_dir splits/ \
        --identity 0.9 \
        --seed 42

    # Simple random split by parent antibody (no clustering)
    python process_data/split_data.py \
        --input loops.jsonl \
        --output_dir splits/ \
        --no_cluster \
        --seed 42

Requires: MMseqs2 (pip install mmseqs) only if not using --cluster_csv
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
from collections import defaultdict

import pandas as pd


def load_loops(input_path: str) -> list[dict]:
    """Load loop data from JSONL or Parquet."""
    if input_path.endswith(".jsonl"):
        data = []
        with open(input_path) as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
        return df.to_dict(orient="records")
    else:
        raise ValueError("Unsupported format. Use .jsonl or .parquet.")


def load_structural_clusters(cluster_csv: str) -> dict[str, str]:
    """
    Load pre-computed TM-score clusters from a CSV.

    Expected format (from cluster_information/):
        Representative_Complex,Complexes,Size_of_Cluster
        1CE1-...,  "1CE1-...,1E4X-...,3CXD-...",  14

    Returns:
        dict mapping complex_name -> cluster_representative
    """
    df = pd.read_csv(cluster_csv)
    name_to_cluster = {}
    for _, row in df.iterrows():
        rep = row["Representative_Complex"]
        members = row["Complexes"].split(",")
        for member in members:
            name_to_cluster[member.strip()] = rep
    return name_to_cluster


def cluster_with_mmseqs2(
    sequences: dict[str, str], identity: float = 0.9, coverage: float = 0.8
) -> dict[str, str]:
    """
    Cluster sequences using MMseqs2.

    Returns:
        dict mapping sequence_id -> cluster_representative_id
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "seqs.fasta")
        with open(fasta_path, "w") as f:
            for seq_id, seq in sequences.items():
                f.write(f">{seq_id}\n{seq}\n")

        db_path = os.path.join(tmpdir, "seqDB")
        cluster_path = os.path.join(tmpdir, "clusterDB")
        tsv_path = os.path.join(tmpdir, "clusters.tsv")

        subprocess.run(
            ["mmseqs", "createdb", fasta_path, db_path],
            check=True, capture_output=True,
        )
        subprocess.run(
            [
                "mmseqs", "cluster", db_path, cluster_path,
                os.path.join(tmpdir, "tmp"),
                "--min-seq-id", str(identity),
                "-c", str(coverage),
                "--cov-mode", "0",
            ],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["mmseqs", "createtsv", db_path, db_path, cluster_path, tsv_path],
            check=True, capture_output=True,
        )

        seq_to_cluster = {}
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    seq_to_cluster[parts[1]] = parts[0]

    return seq_to_cluster


def split_by_clusters(
    cluster_ids: list[str], train_frac: float = 0.9, seed: int = 42
) -> dict[str, str]:
    """
    Assign clusters to train/val splits.

    Returns:
        dict mapping cluster_id -> "train" or "val"
    """
    unique_clusters = list(set(cluster_ids))
    random.seed(seed)
    random.shuffle(unique_clusters)

    n_train = int(len(unique_clusters) * train_frac)

    cluster_to_split = {}
    for i, c in enumerate(unique_clusters):
        cluster_to_split[c] = "train" if i < n_train else "val"

    return cluster_to_split


def get_parent_id(loop_id: str) -> str:
    """Extract parent antibody ID from loop_id (e.g., '5L6Y-ASU0_H1' -> '5L6Y-ASU0')."""
    parts = loop_id.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else loop_id


def write_jsonl(data: list[dict], path: str):
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split loop data into train/val with structural or sequence clustering. "
                    "Testing is done with a separate benchmark dataset."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSONL or Parquet file from process_dihedrals.py.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write train/val JSONL files.",
    )
    parser.add_argument(
        "--cluster_csv", type=str, default=None,
        help="Pre-computed TM-score cluster CSV (e.g., tm_threshold_0p90_summary.csv). "
             "Recommended over MMseqs2 for structural similarity-aware splits.",
    )
    parser.add_argument(
        "--identity", type=float, default=0.9,
        help="MMseqs2 sequence identity threshold (default: 0.9). "
             "Only used if --cluster_csv is not provided.",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.8,
        help="MMseqs2 coverage threshold (default: 0.8).",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.9,
        help="Fraction of clusters for training (default: 0.9, rest goes to val).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no_cluster", action="store_true",
        help="Skip all clustering; split by parent antibody ID instead.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading data from {args.input}...")
    data = load_loops(args.input)
    print(f"  Loaded {len(data)} loops")

    if args.cluster_csv:
        # --- Structural clustering from pre-computed TM-score CSV ---
        print(f"Loading structural clusters from {args.cluster_csv}...")
        name_to_cluster = load_structural_clusters(args.cluster_csv)

        # Map loop_ids to clusters via their parent complex name
        loop_to_cluster = {}
        unmatched = 0
        for d in data:
            parent = get_parent_id(d["loop_id"])
            if parent in name_to_cluster:
                loop_to_cluster[d["loop_id"]] = name_to_cluster[parent]
            else:
                # Unmatched loops get their own singleton cluster
                loop_to_cluster[d["loop_id"]] = parent
                unmatched += 1

        unique_clusters = list(set(loop_to_cluster.values()))
        print(f"  {len(name_to_cluster)} complexes in cluster CSV")
        print(f"  {len(unique_clusters)} clusters covering input data ({unmatched} loops unmatched)")

        cluster_to_split = split_by_clusters(unique_clusters, args.train_frac, args.seed)
        loop_to_split = {
            lid: cluster_to_split[cluster]
            for lid, cluster in loop_to_cluster.items()
        }

    elif args.no_cluster:
        # --- Split by parent antibody ID ---
        print("Splitting by parent antibody ID (no clustering)...")
        parent_ids = list(set(get_parent_id(d["loop_id"]) for d in data))
        cluster_to_split = split_by_clusters(parent_ids, args.train_frac, args.seed)
        loop_to_split = {
            d["loop_id"]: cluster_to_split[get_parent_id(d["loop_id"])]
            for d in data
        }

    else:
        # --- MMseqs2 sequence clustering ---
        print(f"Clustering loop sequences at {args.identity:.0%} identity...")
        seq_to_id = {}
        for d in data:
            seq_to_id.setdefault(d["loop_sequence"], d["loop_id"])

        unique_seqs_for_clustering = {seq_to_id[seq]: seq for seq in seq_to_id}
        seq_to_cluster = cluster_with_mmseqs2(
            unique_seqs_for_clustering, identity=args.identity, coverage=args.coverage
        )

        loop_to_cluster = {}
        for d in data:
            rep_lid = seq_to_id[d["loop_sequence"]]
            loop_to_cluster[d["loop_id"]] = seq_to_cluster.get(rep_lid, rep_lid)

        unique_clusters = list(set(loop_to_cluster.values()))
        print(f"  {len(unique_seqs_for_clustering)} unique sequences -> {len(unique_clusters)} clusters")
        cluster_to_split = split_by_clusters(unique_clusters, args.train_frac, args.seed)
        loop_to_split = {
            lid: cluster_to_split[cluster]
            for lid, cluster in loop_to_cluster.items()
        }

    # 2. Partition data
    splits = defaultdict(list)
    for d in data:
        splits[loop_to_split[d["loop_id"]]].append(d)

    # 3. Remove sequence overlaps from val (extra safety)
    train_seqs = set(d["loop_sequence"] for d in splits["train"])
    splits["val"] = [d for d in splits["val"] if d["loop_sequence"] not in train_seqs]

    # 4. Write outputs
    for split_name in ["train", "val"]:
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        write_jsonl(splits[split_name], out_path)
        n_unique = len(set(d["loop_sequence"] for d in splits[split_name]))
        print(f"  {split_name}: {len(splits[split_name])} loops ({n_unique} unique sequences) -> {out_path}")

    # 5. Report
    final_train = set(d["loop_sequence"] for d in splits["train"])
    final_val = set(d["loop_sequence"] for d in splits["val"])
    print(f"\nSequence overlap train/val after filtering: {len(final_train & final_val)}")

    for split_name in ["train", "val"]:
        types = defaultdict(int)
        for d in splits[split_name]:
            lt = d.get("loop_type", d["loop_id"].rsplit("_", 1)[-1])
            types[lt] += 1
        print(f"  {split_name} loop types: {dict(sorted(types.items()))}")


if __name__ == "__main__":
    main()
