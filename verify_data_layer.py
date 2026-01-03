"""
CardioGuard-AI Data Layer Verification Script

Run this script to verify the data layer implementation.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_default_config
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.labels import add_binary_mi_labels, add_5class_labels, filter_valid_samples
from src.data.signals import load_single_signal
from src.data.splits import get_standard_split, verify_no_patient_leakage


def main():
    print("=" * 60)
    print("CardioGuard-AI Data Layer Verification")
    print("=" * 60)
    
    # Load config
    config = get_default_config(Path("."))
    print(f"Config loaded: {config.data_root}")
    
    # Load metadata
    df = load_ptbxl_metadata(config.metadata_path)
    patient_count = df["patient_id"].nunique()
    print(f"Loaded {len(df)} ECG records")
    print(f"Unique patients: {patient_count}")
    
    # Load SCP statements
    scp = load_scp_statements(config.scp_statements_path)
    print(f"Loaded {len(scp)} SCP statement types")
    
    # Add binary labels
    df = add_binary_mi_labels(df, scp, min_likelihood=config.min_likelihood)
    print()
    print("Binary MI vs NORM label distribution:")
    counts = df["label_mi_norm"].value_counts()
    for label in sorted(counts.index):
        count = counts[label]
        label_name = {0: "NORM", 1: "MI", -1: "Excluded"}[label]
        pct = count / len(df) * 100
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    # Filter valid samples
    df_valid = filter_valid_samples(df, "label_mi_norm")
    print(f"After filtering excluded: {len(df_valid)} samples")
    
    # Get splits
    train_idx, val_idx, test_idx = get_standard_split(df_valid)
    print()
    print("Data split (using strat_fold):")
    total = len(df_valid)
    print(f"  Train: {len(train_idx)} samples ({len(train_idx)/total*100:.1f}%)")
    print(f"  Val:   {len(val_idx)} samples ({len(val_idx)/total*100:.1f}%)")
    print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/total*100:.1f}%)")
    
    # Verify no leakage
    verify_no_patient_leakage(df_valid, train_idx, val_idx, test_idx)
    print()
    print("✓ NO PATIENT LEAKAGE - All splits have disjoint patients")
    
    # Check label distribution in each split
    print()
    print("Label distribution per split:")
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        split_df = df_valid.loc[idx]
        norm_count = (split_df["label_mi_norm"] == 0).sum()
        mi_count = (split_df["label_mi_norm"] == 1).sum()
        print(f"  {name}: NORM={norm_count}, MI={mi_count}")

    # Cache benchmark
    cache_dir = Path("verification_output") / "signal_cache"
    sample_df = df_valid.head(20)
    if not sample_df.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for cache_file in cache_dir.glob("*"):
            if cache_file.is_file():
                cache_file.unlink()
        write_start = time.perf_counter()
        for _, row in sample_df.iterrows():
            load_single_signal(
                row[config.filename_column],
                config.data_root,
                cache_dir=cache_dir,
                use_cache=True
            )
        write_time = time.perf_counter() - write_start

        read_start = time.perf_counter()
        for _, row in sample_df.iterrows():
            load_single_signal(
                row[config.filename_column],
                config.data_root,
                cache_dir=cache_dir,
                use_cache=True
            )
        read_time = time.perf_counter() - read_start

        print()
        print(f"Cache benchmark ({len(sample_df)} samples):")
        print(f"  Write (populate cache): {write_time:.3f}s")
        print(f"  Read (from cache):      {read_time:.3f}s")
    
    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE - All checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
