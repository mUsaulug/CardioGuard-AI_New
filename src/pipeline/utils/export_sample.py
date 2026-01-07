"""
Export a single sample (raw signal) from validation set for testing inference.

Usage:
    python -m src.pipeline.export_sample --index 0 --output sample_test.npy
"""

import argparse
import json
from pathlib import Path
import numpy as np
import wfdb

from src.config import get_default_config
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.labels_superclass import add_superclass_labels_derived
from src.data.splits import get_standard_split
from src.pipeline.training.train_superclass_cnn import filter_missing_files, SUPERCLASS_LABELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Index in validation set to export")
    parser.add_argument("--output", type=Path, default=Path("sample_test.npy"))
    args = parser.parse_args()
    
    # Load data
    config = get_default_config()
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)
    df = add_superclass_labels_derived(df, scp_df)
    df = filter_missing_files(df, config.data_root, config.filename_column)
    
    # Get val split
    _, val_idx, _ = get_standard_split(df)
    val_df = df.loc[val_idx].reset_index(drop=True)
    
    if args.index >= len(val_df):
        print(f"Index {args.index} out of bounds (size {len(val_df)})")
        return

    row = val_df.iloc[args.index]
    
    # Load signal
    str_path = str(config.data_root / row[config.filename_column])
    record = wfdb.rdrecord(str_path)
    signal = record.p_signal # (timesteps, channels)
    
    # Save signal
    np.save(args.output, signal)
    print(f"Saved signal shape {signal.shape} to {args.output}")
    
    # Save absolute ground truth for verification
    meta = {
        "ecg_id": int(row.name) if hasattr(row, "name") else int(row["ecg_id"]),
        "superclass_pathologies": row["superclass_pathologies"],
        "diagnostic_superclass": row.get("diagnostic_superclass", row.get("diagnostic_superclass_derived")),
        "y_multi": row["y_multi4"].tolist()
    }
    
    meta_path = args.output.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Saved metadata to {meta_path}")
    print(f"Ground Truth: {meta['superclass_pathologies']}")

if __name__ == "__main__":
    main()
