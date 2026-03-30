#!/bin/bash
set -euo pipefail

###############################################################################
# Igloo Training (Phase 1 + Phase 2)
# Usage: bash run_train.sh
#
# Requires: splits/train.jsonl and splits/val.jsonl from run_prepare_data.sh
###############################################################################

# ── Configuration ────────────────────────────────────────────────────────────
# Make model/ importable both as bare modules (dataset, trainer, etc.)
# and as the 'igloo' package (mapped via pyproject.toml: igloo -> model/)
_IGLOO_TMP=$(mktemp -d)
ln -s "$(pwd)/model" "${_IGLOO_TMP}/igloo"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd):$(pwd)/model:${_IGLOO_TMP}"
SPLITS_DIR="splits"
MODEL_DIR="Igloo_models"
DEVICE="cuda"

# Verify data exists
if [[ ! -f "${SPLITS_DIR}/train.jsonl" ]] || [[ ! -f "${SPLITS_DIR}/val.jsonl" ]]; then
    echo "ERROR: Missing ${SPLITS_DIR}/train.jsonl or val.jsonl"
    echo "Run 'bash run_prepare_data.sh' first."
    exit 1
fi

# ── Phase 1: Pretrain ─────────────────────────────────────────────────────────
echo "=== Phase 1: Pretraining ==="
python train.py \
    --train_data_path "${SPLITS_DIR}/train.jsonl" \
    --val_data_path "${SPLITS_DIR}/val.jsonl" \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --device "${DEVICE}" \
    --num_epochs 100 \
    --codebook_size 8192 \
    --num_encoder_layers 4 \
    --embedding_dim 128 \
    --commit_loss_weight 0.5 \
    --unit_circle_transform_weight 0.01 \
    --loop_length_tolerance 0 \
    --dihedral_loss \
    --learnable_codebook \
    --save_dir "${MODEL_DIR}" \
    --project_name "Phase 1: pretrain" \
    --use_wandb

# Find the best checkpoint from Phase 1
PHASE1_DIR="${MODEL_DIR}/version_1"
BEST_EPOCH=$(awk '{if(NR==1 || $2<min){min=$2; epoch=$1}} END{print epoch}' "${PHASE1_DIR}/model_loss.txt")
PHASE1_WEIGHTS="${PHASE1_DIR}/checkpoints/model_epoch_${BEST_EPOCH}.pt"
PHASE1_CONFIG="${PHASE1_DIR}/model_config.json"
echo "Best Phase 1 epoch: ${BEST_EPOCH} → ${PHASE1_WEIGHTS}"

# ── Phase 2: Finetune ────────────────────────────────────────────────────────
echo "=== Phase 2: Finetuning ==="
python train.py \
    --train_data_path "${SPLITS_DIR}/train.jsonl" \
    --val_data_path "${SPLITS_DIR}/val.jsonl" \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --device "${DEVICE}" \
    --num_epochs 100 \
    --codebook_size 8192 \
    --num_encoder_layers 4 \
    --embedding_dim 128 \
    --commit_loss_weight 0.5 \
    --unit_circle_transform_weight 0.01 \
    --loop_length_tolerance 0 \
    --dihedral_loss \
    --learnable_codebook \
    --codebook_learning_rate 1e-3 \
    --weight_decay 1e-5 \
    --pretrained_model_weights "${PHASE1_WEIGHTS}" \
    --pretrained_model_config "${PHASE1_CONFIG}" \
    --save_dir "${MODEL_DIR}" \
    --project_name "Phase 2: finetune" \
    --use_wandb

echo ""
echo "=== Training complete ==="
echo "Weights saved in ${MODEL_DIR}/"
