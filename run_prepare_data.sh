#!/bin/bash
set -euo pipefail

###############################################################################
# Igloo Data Preparation Pipeline
# Usage: bash run_prepare_data.sh
#
# Run this ONCE before training. Produces splits/train.jsonl and splits/val.jsonl
###############################################################################

# ── Configuration ────────────────────────────────────────────────────────────
PDB_DIR="training/ab_complexes"
CLUSTER_CSV="cluster_information/ab_complexes/tm_threshold_0p90_summary.csv"
NCPU=16
SEED=42

# Intermediate outputs
DATASET_CSV="training_ab.csv"
LOOPS_JSONL="training_loops.jsonl"
SPLITS_DIR="splits"

# ── Step 1: Prepare dataset from PDB files ────────────────────────────────────
echo "=== Step 1: Parsing PDB files → ${DATASET_CSV} ==="
python process_data/prepare_pdb_dataset.py \
    --pdb_dir "${PDB_DIR}" \
    --output_csv "${DATASET_CSV}" \
    --ncpu "${NCPU}"

# ── Step 2: Extract dihedral angles ───────────────────────────────────────────
echo "=== Step 3: Extracting dihedrals → ${LOOPS_JSONL} ==="
python process_data/process_dihedrals.py \
    --df_path "${DATASET_CSV}" \
    --id_key id \
    --aho_heavy_key fv_heavy_aho \
    --aho_light_key fv_light_aho \
    --heavy_chain_id_key heavy_chain_id \
    --light_chain_id_key light_chain_id \
    --jsonl_output_path "${LOOPS_JSONL}" \
    --num_workers "${NCPU}"

# ── Step 3: Split into train/val ──────────────────────────────────────────────
echo "=== Step 4: Splitting data → ${SPLITS_DIR}/ ==="
python process_data/split_data.py \
    --input "${LOOPS_JSONL}" \
    --output_dir "${SPLITS_DIR}" \
    --cluster_csv "${CLUSTER_CSV}" \
    --seed "${SEED}"

echo ""
echo "=== Data preparation complete ==="
echo "Train samples: $(wc -l < "${SPLITS_DIR}/train.jsonl")"
echo "Val samples:   $(wc -l < "${SPLITS_DIR}/val.jsonl")"
echo ""
echo "Next: bash run_train.sh"
