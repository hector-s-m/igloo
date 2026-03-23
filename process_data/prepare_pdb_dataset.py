"""
Preprocess a PDB dataset for Igloo training.

Scans a directory of PDB files (or reads a master CSV), extracts VH/VL
sequences, runs ANARCI for AHO alignment, and outputs an Igloo-compatible CSV.

PDB filenames must encode chain IDs in the format:
    VH_<chain>-VL_<chain> (e.g., '1A2Y-ASU0-VH_B-VL_A-Ag_C.pdb')
    VHH_<chain> for nanobodies (e.g., '8FSL-ASU1-VHH_A-Ag_C.pdb')

Usage:
    # From a directory of PDB files (no CSV needed)
    python process_data/prepare_pdb_dataset.py \
        --pdb_dir training/ab_complexes/ \
        --output_csv dataset.csv \
        --ncpu 16

    # From a master CSV with a 'Name' column
    python process_data/prepare_pdb_dataset.py \
        --input_csv complexes_curated.csv \
        --pdb_dir training/ab_complexes/ \
        --output_csv dataset.csv \
        --ncpu 16
"""

import argparse
import glob
import re
import os
import pandas as pd
import numpy as np
import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_reader
from biotite.structure.info import one_letter_code, amino_acid_names
from tqdm import tqdm
from multiprocessing import Pool

# Amino acid 3-to-1 mapping (matches biotoolkit.py)
_AA_3TO1 = {res: one_letter_code(res) for res in amino_acid_names() if one_letter_code(res) is not None}
_AA_3TO1.update({
    "HOH": "0", "SOL": "0",
    "SLL": "K", "DV7": "G", "PCA": "Q",
})


def aa_3to1(res: str) -> str:
    return _AA_3TO1.get(res, "X")


def parse_pdb_filename(name: str) -> dict:
    """
    Parse a PDB complex filename to extract chain IDs.

    Expects naming convention: VH_<chain>-VL_<chain>[-Ag_<chain>]
    or VHH_<chain>[-Ag_<chain>] for nanobodies.

    Examples:
        '1A2Y-ASU0-VH_B-VL_A-Ag_C' -> {'heavy_chain_id': 'B', 'light_chain_id': 'A'}
        '7y7m-ASU1-frame0-VH_F-VL_G-Ag_C' -> {'heavy_chain_id': 'F', 'light_chain_id': 'G'}
        '8FSL-ASU1-VHH_A-Ag_C' -> {'heavy_chain_id': 'A', 'light_chain_id': None}  (nanobody)
    """
    vh_match = re.search(r'VH_(\w)', name)
    vl_match = re.search(r'VL_(\w)', name)
    vhh_match = re.search(r'VHH_(\w)', name)

    result = {
        'heavy_chain_id': None,
        'light_chain_id': None,
        'is_nanobody': False,
    }

    if vhh_match:
        result['heavy_chain_id'] = vhh_match.group(1)
        result['is_nanobody'] = True
    elif vh_match:
        result['heavy_chain_id'] = vh_match.group(1)

    if vl_match:
        result['light_chain_id'] = vl_match.group(1)

    return result


def extract_chain_sequence(pdb_path: str, chain_id: str) -> str:
    """Extract amino acid sequence for a given chain from a PDB file."""
    try:
        pdb_file = pdb_reader.PDBFile.read(pdb_path)
        # Try model=1 first; if the file has no MODEL/ENDMDL records
        # biotite reports 0 models, so fall back to reading all atoms.
        try:
            atom_array = pdb_file.get_structure(model=1)
        except (biotite.InvalidFileError, IndexError, Exception):
            atom_array = pdb_file.get_structure()
            if hasattr(atom_array, 'stack_depth') and atom_array.stack_depth() > 0:
                atom_array = atom_array[0]
        # Filter to target chain
        chain_atoms = atom_array[atom_array.chain_id == chain_id]
        if len(chain_atoms) == 0:
            return None
        # Get residue-level sequence
        _, res_names = struc.get_residues(chain_atoms)
        seq = "".join(aa_3to1(r) for r in res_names)
        # Remove water/non-standard
        seq = seq.replace("0", "")
        return seq if len(seq) > 0 else None
    except Exception as e:
        print(f"Warning: Failed to extract chain {chain_id} from {pdb_path}: {e}")
        return None


def run_anarci_alignment(sequences: list, ncpu: int = 1) -> list:
    """
    Run ANARCI AHO alignment on a list of sequences.
    Returns list of AHO-aligned sequences (with gaps as '-'), or None for failures.
    """
    try:
        from anarci import anarci as run_anarci
    except ImportError:
        raise ImportError(
            "ANARCI is required for AHO alignment. Install with: pip install anarci"
        )

    # ANARCI expects list of (name, sequence) tuples
    input_seqs = [(str(i), seq) for i, seq in enumerate(sequences)]
    results = run_anarci(input_seqs, scheme="aho", output=False, ncpu=ncpu)

    numberings, alignments, hit_tables = results
    aho_seqs = []

    for i, numbering in enumerate(numberings):
        if numbering is None or len(numbering) == 0:
            aho_seqs.append(None)
            continue

        # Take the best hit (first domain)
        domain = numbering[0]
        if domain is None or len(domain) == 0:
            aho_seqs.append(None)
            continue

        # domain is (numbering_list, start_idx, end_idx)
        numbering_list = domain[0]

        # Get chain type from alignments
        chain_type = "H"
        if alignments[i] and len(alignments[i]) > 0:
            chain_type = alignments[i][0].get("chain_type", "H")

        # Determine max AHO position based on chain type
        max_pos = 148 if chain_type in ("L", "K") else 149

        # Build gapped AHO sequence
        pos_to_aa = {}
        for (pos_num, insertion), aa in numbering_list:
            if aa != "-":
                pos_to_aa[pos_num] = aa

        aho_seq = ""
        for pos in range(1, max_pos + 1):
            aho_seq += pos_to_aa.get(pos, "-")

        aho_seqs.append(aho_seq)

    return aho_seqs


def find_pdb_file(name: str, pdb_dir: str) -> str:
    """Find the PDB file for a given entry name."""
    pdb_path = os.path.join(pdb_dir, f"{name}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    cif_path = os.path.join(pdb_dir, f"{name}.cif")
    if os.path.exists(cif_path):
        return cif_path
    return None


def scan_pdb_dir(pdb_dir: str) -> pd.DataFrame:
    """Build a DataFrame from PDB filenames in a directory (no CSV needed)."""
    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in pdb_files]
    return pd.DataFrame({"Name": names})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess a PDB complex dataset for Igloo training."
    )
    parser.add_argument(
        "--input_csv", type=str, default=None,
        help="Path to master CSV with a 'Name' column. If not provided, "
             "PDB filenames are scanned from --pdb_dir."
    )
    parser.add_argument(
        "--pdb_dir", type=str, required=True,
        help="Directory containing PDB files.",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Path to output Igloo-compatible CSV.",
    )
    parser.add_argument(
        "--ncpu", type=int, default=1,
        help="Number of CPUs for ANARCI alignment.",
    )
    parser.add_argument(
        "--include_nanobodies", action="store_true",
        help="Include VHH (nanobody) entries (default: antibodies only).",
    )
    parser.add_argument(
        "--resume_csv", type=str, default=None,
        help="Path to a partial output CSV from a previous run. "
             "Already-extracted entries are skipped (resumes from where it left off).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Build entry list from CSV or by scanning PDB directory
    if args.input_csv:
        print(f"Reading CSV: {args.input_csv}")
        input_df = pd.read_csv(args.input_csv)
    else:
        print(f"Scanning PDB files in: {args.pdb_dir}")
        input_df = scan_pdb_dir(args.pdb_dir)
    print(f"  Total entries: {len(input_df)}")

    # 2. Parse filenames to extract chain IDs
    print("Parsing filenames for chain IDs...")
    parsed = input_df["Name"].apply(parse_pdb_filename).apply(pd.Series)
    input_df = pd.concat([input_df, parsed], axis=1)

    # 3. Filter for antibodies (VH + VL) or optionally include nanobodies
    if args.include_nanobodies:
        mask = input_df["heavy_chain_id"].notna()
    else:
        mask = input_df["heavy_chain_id"].notna() & input_df["light_chain_id"].notna() & ~input_df["is_nanobody"]
    input_df = input_df[mask].reset_index(drop=True)
    print(f"  After filtering: {len(input_df)} entries")

    # Load previously extracted entries to skip (resume from sequences-only CSV)
    intermediate_path = args.output_csv.replace(".csv", "_sequences_only.csv")
    already_done = set()
    resumed_records = []
    if os.path.exists(intermediate_path):
        prev_df = pd.read_csv(intermediate_path)
        already_done = set(prev_df["id"].tolist())
        resumed_records = prev_df.to_dict("records")
        print(f"  Resuming from {intermediate_path}: {len(already_done)} entries already extracted, skipping them")
    elif args.resume_csv and os.path.exists(args.resume_csv):
        prev_df = pd.read_csv(args.resume_csv)
        already_done = set(prev_df["id"].tolist())
        resumed_records = prev_df.to_dict("records")
        print(f"  Resuming from {args.resume_csv}: {len(already_done)} entries already extracted, skipping them")

    # 4. Find PDB files and extract sequences
    print("Extracting sequences from PDB files...")
    records = list(resumed_records)
    missing_pdbs = 0
    failed_extractions = 0
    skipped = 0
    new_since_last_save = 0

    for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Extracting sequences"):
        name = row["Name"]
        if name in already_done:
            skipped += 1
            continue

        pdb_path = find_pdb_file(name, args.pdb_dir)
        if pdb_path is None:
            missing_pdbs += 1
            continue

        heavy_chain_id = row["heavy_chain_id"]
        light_chain_id = row["light_chain_id"]

        heavy_seq = extract_chain_sequence(pdb_path, heavy_chain_id)
        light_seq = extract_chain_sequence(pdb_path, light_chain_id) if light_chain_id else None

        # Fallback: if filename chain IDs don't match, try standardized H/L
        if heavy_seq is None and heavy_chain_id != "H":
            heavy_seq = extract_chain_sequence(pdb_path, "H")
            if heavy_seq is not None:
                heavy_chain_id = "H"
        if light_seq is None and light_chain_id and light_chain_id != "L":
            light_seq = extract_chain_sequence(pdb_path, "L")
            if light_seq is not None:
                light_chain_id = "L"

        if heavy_seq is None:
            failed_extractions += 1
            continue
        if light_chain_id and light_seq is None:
            failed_extractions += 1
            continue

        records.append({
            "id": name,
            "fv_heavy": heavy_seq,
            "fv_light": light_seq,
            "heavy_chain_id": heavy_chain_id,
            "light_chain_id": light_chain_id,
            "pdb_path": pdb_path,
        })
        new_since_last_save += 1

        # Save checkpoint every 1000 new extractions
        if new_since_last_save >= 1000:
            pd.DataFrame(records).to_csv(intermediate_path, index=False)
            print(f"\n  Checkpoint: saved {len(records)} entries to {intermediate_path}")
            new_since_last_save = 0

    new_extracted = len(records) - len(resumed_records)
    print(f"  New: {new_extracted} | Resumed: {len(resumed_records)} | Skipped: {skipped} | Missing PDBs: {missing_pdbs} | Failed: {failed_extractions}")

    if len(records) == 0:
        print("Error: No sequences extracted. Check PDB directory and paths.")
        return

    result_df = pd.DataFrame(records)

    # Save intermediate CSV (sequences extracted, before alignment)
    # so that if ANARCI fails, we can resume without re-extracting
    intermediate_path = args.output_csv.replace(".csv", "_sequences_only.csv")
    result_df.to_csv(intermediate_path, index=False)
    print(f"  Saved intermediate sequences to {intermediate_path}")

    # 5. Run ANARCI for AHO alignment
    print(f"Running ANARCI AHO alignment (ncpu={args.ncpu})...")

    # Align heavy chains
    heavy_seqs = result_df["fv_heavy"].tolist()
    heavy_aho = run_anarci_alignment(heavy_seqs, ncpu=args.ncpu)
    result_df["fv_heavy_aho"] = heavy_aho

    # Align light chains (if present)
    if result_df["fv_light"].notna().any():
        light_seqs = result_df["fv_light"].tolist()
        light_seqs_clean = [s if pd.notna(s) else "" for s in light_seqs]
        light_aho = run_anarci_alignment(light_seqs_clean, ncpu=args.ncpu)
        result_df["fv_light_aho"] = light_aho
    else:
        result_df["fv_light_aho"] = None

    # 6. Filter out entries where ANARCI alignment failed
    before_filter = len(result_df)
    result_df = result_df[result_df["fv_heavy_aho"].notna()]
    if result_df["fv_light"].notna().any():
        result_df = result_df[result_df["fv_light_aho"].notna()]
    result_df = result_df.reset_index(drop=True)
    print(f"  After alignment filtering: {len(result_df)} (dropped {before_filter - len(result_df)} failed alignments)")

    # 7. Output CSV
    output_cols = ["id", "fv_heavy", "fv_light", "fv_heavy_aho", "fv_light_aho",
                   "heavy_chain_id", "light_chain_id", "pdb_path"]
    result_df[output_cols].to_csv(args.output_csv, index=False)
    print(f"Saved {len(result_df)} entries to {args.output_csv}")


if __name__ == "__main__":
    main()
