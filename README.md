# Igloo: Tokenizing Loops of Antibodies

<img src="assets/igloo_logo.png" alt="Igloo" width="300"/>

[Preprint](https://arxiv.org/abs/2509.08707)

Authors
* Ada Fang
* Rob Alberstein
* Simon Kelow
* Frédéric Dreyer

## Getting started

Clone the repo and install dependencies:
```
git clone https://github.com/hector-s-m/igloo.git
cd igloo
pip install -r requirements.txt
pip install -e .
```

## Repository structure

```
training/
  ab_complexes/       # Antibody PDB structures for training
  nb_complexes/       # Nanobody PDB structures for training
benchmark/            # Held-out PDB structures for evaluation
process_data/         # Data preprocessing scripts
model/                # Igloo VQ-VAE model (importable as `igloo`)
finetune_igbert/      # IgBert fine-tuning (importable as `igloo.plm`)
```

## Training Igloo from PDB structures

### 1. Prepare dataset

Given a directory of PDB files with chain IDs encoded in filenames
(e.g., `1A2Y-ASU0-VH_B-VL_A-Ag_C.pdb`), extract sequences and run
AHO alignment:

```
python process_data/prepare_pdb_dataset.py \
    --pdb_dir training/ab_complexes/ \
    --output_csv training_ab.csv \
    --ncpu 16
```

If you have a master CSV with a `Name` column, you can pass it with `--input_csv`.

### 2. Extract dihedral angles

```
python process_data/process_dihedrals.py \
    --df_path training_ab.csv \
    --id_key id \
    --aho_heavy_key fv_heavy_aho \
    --aho_light_key fv_light_aho \
    --heavy_chain_id_key heavy_chain_id \
    --light_chain_id_key light_chain_id \
    --jsonl_output_path training_loops.jsonl \
    --num_workers 16
```

For PDB files with standard `H`/`L` chain IDs, the `--heavy_chain_id_key`
and `--light_chain_id_key` flags can be omitted.

### 3. Split into train/val

Using pre-computed TM-score structural clusters (recommended):
```
python process_data/split_data.py \
    --input training_loops.jsonl \
    --output_dir splits/ \
    --cluster_csv cluster_information/ab_complexes/tm_threshold_0p90_summary.csv \
    --seed 42
```

Alternatively, using MMseqs2 sequence clustering:
```
python process_data/split_data.py \
    --input training_loops.jsonl \
    --output_dir splits/ \
    --identity 0.9 \
    --seed 42
```

Testing is done separately using the structures in `benchmark/`.

### 4. Train

**Phase 1: Pretrain**
```
python train.py \
    --train_data_path splits/train.jsonl \
    --val_data_path splits/val.jsonl \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --device cuda \
    --num_epochs 100 \
    --codebook_size 8192 \
    --num_encoder_layers 4 \
    --embedding_dim 128 \
    --commit_loss_weight 0.5 \
    --unit_circle_transform_weight 0.01 \
    --loop_length_tolerance 0 \
    --dihedral_loss \
    --learnable_codebook \
    --save_dir Igloo_models \
    --use_wandb
```

**Phase 2: Finetune**
```
python train.py \
    --train_data_path splits/train.jsonl \
    --val_data_path splits/val.jsonl \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --device cuda \
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
    --pretrained_model_weights Igloo_models/best_phase1.pt \
    --pretrained_model_config Igloo_models/model_config.json \
    --save_dir Igloo_models \
    --use_wandb
```

## Run Igloo (inference)

### With sequences and structures (recommended)

**1. Prepare input**

Prepare a CSV file (see `example/sample_igloo_sequences.csv`) with columns:
* `fv_heavy_aho` and `fv_light_aho`: AHO-aligned sequences. See [ANARCI](https://github.com/oxpig/ANARCI).
* `id`: unique identifier matching `<id>.pdb` in the structure directory.

```
python process_data/process_dihedrals.py \
    --id_key "id" --aho_light_key "fv_light_aho" --aho_heavy_key "fv_heavy_aho" \
    --structure_dir my_pdbs/ \
    --df_path example/sample_igloo_sequences.csv \
    --parquet_output_path example/sample_igloo_input.parquet
```

**2. Inference**
```
python run_igloo.py \
    --model_ckpt checkpoints/igloo_weights.pt \
    --model_config checkpoints/igloo_config.json \
    --loop_dataset_path example/sample_igloo_input.parquet \
    --output_path example/sample_igloo_output.parquet
```

### With sequences and predicted structures

Generate structures with [Ibex](https://github.com/prescient-design/ibex):
```
pip install prescient-ibex
ibex --csv example/sample_igloo_sequences.csv --output ibex_predictions_dir/
```

Then run steps 1-2 above, pointing `--structure_dir` at `ibex_predictions_dir/`.

### Sequence only (no structures)

Prepare a CSV with columns `loop_id` and `loop_sequence` (see `example/sample_igloo_input_sequence_only.csv`):
```
python run_igloo.py \
    --model_ckpt checkpoints/igloo_weights.pt \
    --model_config checkpoints/igloo_config.json \
    --loop_dataset_path example/sample_igloo_input_sequence_only.csv \
    --output_path example/sample_igloo_out_sequence_only.parquet
```

### Output format

The output parquet file contains:
* `loop_id`
* `encoded`: Continuous Igloo representation
* `quantized`: Discrete Igloo representation (after Vector Quantize layer)
* `quantized_indices`: Integer indicating the discrete Igloo token

## IglooLM and IglooALM

See [finetune_igbert/README.md](finetune_igbert/README.md).

## Citation

```bibtex
@misc{fang2025tokenizingloopsantibodies,
      title={Tokenizing Loops of Antibodies},
      author={Ada Fang and Robert G. Alberstein and Simon Kelow and Frédéric A. Dreyer},
      year={2025},
      eprint={2509.08707},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2509.08707},
}
```
