import numpy as np
import pandas as pd
from pathlib import Path
import wfdb
import shutil
import subprocess
import sys

# 1. Load Validation Data to find an MI sample
print("Finding MI sample...")
sys.path.append(str(Path(__file__).parent.parent.parent)) # Add project root to path
from src.config import get_default_config
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.labels_superclass import add_superclass_labels_derived
from src.data.mi_localization import add_mi_localization_labels

config = get_default_config()
df = load_ptbxl_metadata(config.metadata_path)
scp_df = load_scp_statements(config.scp_statements_path)
df = add_superclass_labels_derived(df, scp_df)
df = add_mi_localization_labels(df)

# Filter: Fold 9 (Validation), MI=1, Localization exists
val_mi = df[
    (df["strat_fold"] == 9) & 
    (df["label_MI"] == 1) & 
    (df["has_mi_localization"] == 1)
]

if len(val_mi) == 0:
    print("No validation MI samples found!")
    sys.exit(1)

sample = val_mi.iloc[0]
print(f"Selected Sample: {sample.name}")
print(f"True Localization: {sample['y_loc']}")

# 2. Export Sample to .npz
print(f"Exporting sample...")
record_path = config.data_root / sample[config.filename_column]
record = wfdb.rdrecord(str(record_path))
signal = record.p_signal

# Ensure float32
signal = signal.astype(np.float32)

output_path = Path("test_mi_sample.npz")
np.savez_compressed(output_path, signal=signal)
print(f"Saved to {output_path}")

# 3. Run Inference Pipeline
print("\nRunning Inference...")
cmd = [
    sys.executable, "-m", "src.pipeline.run_inference_superclass",
    "--input", str(output_path),
    "--output", "test_mi_result.json",
    "--device", "cpu", # Force CPU for safety
    "--explain" # Let's see explanation too
]

subprocess.run(cmd, check=True)

# 4. Print Result
import json
with open("test_mi_result.json") as f:
    res = json.load(f)

print("\n--- INFERENCE RESULT ---")
print(f"Primary Label: {res['primary']['label']}")
print(f"Multi Labels: {res['multi']['predicted_labels']}")
if "mi_localization" in res:
    print(f"Localization: {res['mi_localization']['predicted_regions']}")
    print(f"Loc Probs: {res['mi_localization']}")
else:
    print("Localization: NONE (MI not detected or model missing)")

print("\n--- DONE ---")
