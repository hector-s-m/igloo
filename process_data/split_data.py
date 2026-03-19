"""
Split loop data into train/val/test sets using sequence clustering.

Clusters loop sequences at a given identity threshold using MMseqs2,
then splits by cluster so no similar sequences leak between sets.

Usage:
    # With MMseqs2 clustering (recommended for high-quality splits)
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

Requires: MMseqs2 installed (unless --no_cluster is used)
    conda install -c conda-forge -c bioconda mmseqs2
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


def cluster_with_mmseqs2(
    sequences: dict[str, str], identity: float = 0.9, coverage: float = 0.8
) -> dict[str, str]:
    """
    Cluster sequences using MMseqs2.

    Args:
        sequences: dict mapping sequence_id -> amino acid sequence
        identity: minimum sequence identity threshold (0-1)
        coverage: minimum coverage threshold (0-1)

    Returns:
        dict mapping sequence_id -> cluster_representative_id
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write FASTA
        fasta_path = os.path.join(tmpdir, "seqs.fasta")
        with open(fasta_path, "w") as f:
            for seq_id, seq in sequences.items():
                f.write(f">{seq_id}\n{seq}\n")

        db_path = os.path.join(tmpdir, "seqDB")
        cluster_path = os.path.join(tmpdir, "clusterDB")
        tsv_path = os.path.join(tmpdir, "clusters.tsv")

        # Create MMseqs2 database
        subprocess.run(
            ["mmseqs", "createdb", fasta_path, db_path],
            check=True, capture_output=True,
        )

        # Cluster
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

        # Convert to TSV
        subprocess.run(
            ["mmseqs", "createtsv", db_path, db_path, cluster_path, tsv_path],
            check=True, capture_output=True,
        )

        # Parse TSV: columns are [representative_id, member_id]
        seq_to_cluster = {}
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep_id, member_id = parts[0], parts[1]
                    seq_to_cluster[member_id] = rep_id

    return seq_to_cluster


def split_by_clusters(
    cluster_ids: list[str], train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42
) -> dict[str, str]:
    """
    Assign clusters to train/val/test splits.

    Returns:
        dict mapping cluster_id -> split name ("train", "val", "test")
    """
    unique_clusters = list(set(cluster_ids))
    random.seed(seed)
    random.shuffle(unique_clusters)

    n_train = int(len(unique_clusters) * train_frac)
    n_val = int(len(unique_clusters) * val_frac)

    cluster_to_split = {}
    for i, c in enumerate(unique_clusters):
        if i < n_train:
            cluster_to_split[c] = "train"
        elif i < n_train + n_val:
            cluster_to_split[c] = "val"
        else:
            cluster_to_split[c] = "test"

    return cluster_to_split


def get_parent_id(loop_id: str) -> str:
    """Extract parent antibody ID from loop_id (e.g., '5L6Y-ASU0_H1' -> '5L6Y-ASU0')."""
    # Split on last underscore + loop type suffix
    parts = loop_id.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else loop_id


def write_jsonl(data: list[dict], path: str):
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split loop data into train/val/test with sequence clustering."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSONL or Parquet file from process_dihedrals.py.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write train/val/test JSONL files.",
    )
    parser.add_argument(
        "--identity", type=float, default=0.9,
        help="MMseqs2 sequence identity threshold (default: 0.9).",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.8,
        help="MMseqs2 coverage threshold (default: 0.8).",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8,
        help="Fraction of clusters for training (default: 0.8).",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1,
        help="Fraction of clusters for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no_cluster", action="store_true",
        help="Skip MMseqs2 clustering; split by parent antibody ID instead.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading data from {args.input}...")
    data = load_loops(args.input)
    print(f"  Loaded {len(data)} loops")

    if args.no_cluster:
        # Split by parent antibody ID (all loops from one antibody stay together)
        print("Splitting by parent antibody ID (no clustering)...")
        parent_ids = list(set(get_parent_id(d["loop_id"]) for d in data))
        cluster_to_split = split_by_clusters(
            parent_ids, args.train_frac, args.val_frac, args.seed
        )
        loop_to_split = {}
        for d in data:
            parent = get_parent_id(d["loop_id"])
            loop_to_split[d["loop_id"]] = cluster_to_split[parent]
    else:
        # 2. Deduplicate sequences for clustering
        print(f"Clustering loop sequences at {args.identity:.0%} identity...")
        seq_to_id = {}
        id_to_seq = {}
        for d in data:
            lid = d["loop_id"]
            seq = d["loop_sequence"]
            id_to_seq[lid] = seq
            seq_to_id.setdefault(seq, lid)  # deduplicate: one representative per unique sequence

        # Use unique sequences for clustering
        unique_seqs = {v: k for k, v in seq_to_id.items()}  # id -> seq
        unique_seqs_for_clustering = {seq_to_id[seq]: seq for seq in seq_to_id}

        # 3. Run MMseqs2
        seq_to_cluster = cluster_with_mmseqs2(
            unique_seqs_for_clustering, identity=args.identity, coverage=args.coverage
        )

        # Map all loop_ids to their cluster representative
        loop_to_cluster = {}
        for d in data:
            lid = d["loop_id"]
            seq = d["loop_sequence"]
            # Find the representative for this sequence
            rep_lid = seq_to_id[seq]
            cluster_rep = seq_to_cluster.get(rep_lid, rep_lid)
            loop_to_cluster[lid] = cluster_rep

        # 4. Split clusters
        unique_clusters = list(set(loop_to_cluster.values()))
        print(f"  {len(unique_seqs_for_clustering)} unique sequences -> {len(unique_clusters)} clusters")
        cluster_to_split = split_by_clusters(
            unique_clusters, args.train_frac, args.val_frac, args.seed
        )

        loop_to_split = {
            lid: cluster_to_split[cluster]
            for lid, cluster in loop_to_cluster.items()
        }

    # 5. Partition data
    splits = defaultdict(list)
    for d in data:
        split_name = loop_to_split[d["loop_id"]]
        splits[split_name].append(d)

    # 6. Remove sequence overlaps from val/test (extra safety)
    train_seqs = set(d["loop_sequence"] for d in splits["train"])
    splits["val"] = [d for d in splits["val"] if d["loop_sequence"] not in train_seqs]
    val_seqs = set(d["loop_sequence"] for d in splits["val"])
    splits["test"] = [
        d for d in splits["test"]
        if d["loop_sequence"] not in train_seqs and d["loop_sequence"] not in val_seqs
    ]

    # 7. Write outputs
    for split_name in ["train", "val", "test"]:
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        write_jsonl(splits[split_name], out_path)
        n_unique = len(set(d["loop_sequence"] for d in splits[split_name]))
        print(f"  {split_name}: {len(splits[split_name])} loops ({n_unique} unique sequences) -> {out_path}")

    # 8. Report overlaps
    final_train = set(d["loop_sequence"] for d in splits["train"])
    final_val = set(d["loop_sequence"] for d in splits["val"])
    final_test = set(d["loop_sequence"] for d in splits["test"])
    print(f"\nSequence overlaps after filtering:")
    print(f"  train/val:  {len(final_train & final_val)}")
    print(f"  train/test: {len(final_train & final_test)}")
    print(f"  val/test:   {len(final_val & final_test)}")

    # 9. Loop type distribution
    for split_name in ["train", "val", "test"]:
        types = defaultdict(int)
        for d in splits[split_name]:
            lt = d.get("loop_type", d["loop_id"].rsplit("_", 1)[-1])
            types[lt] += 1
        print(f"  {split_name} loop types: {dict(sorted(types.items()))}")


if __name__ == "__main__":
    main()
