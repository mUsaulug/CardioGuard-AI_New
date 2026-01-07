
import torch
from pathlib import Path

def main():
    path = Path("checkpoints/ecgcnn.pt")
    if not path.exists():
        print(f"File not found: {path}")
        return

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        keys = list(checkpoint["model_state_dict"].keys())
        print(f"Found {len(keys)} keys in model_state_dict. First 10:")
    elif isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f"Found {len(keys)} keys in checkpoint dict. First 10:")
    else:
        # Unexpected format or direct state_dict
        # Try iterating
        try:
            keys = list(checkpoint.keys())
            print(f"Found {len(keys)} keys in checkpoint object. First 10:")
        except:
             print("Checkpoint is not a dict-like object.")
             return

    for k in keys[:10]:
        print(k)

if __name__ == "__main__":
    main()
