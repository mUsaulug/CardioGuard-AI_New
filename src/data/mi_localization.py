"""
MI Localization Label Extraction Module.

Derives 5 anatomical MI region labels from PTB-XL SCP codes.
Target labels: [AMI, ASMI, ALMI, IMI, LMI]

NOTE: These are DERIVED labels from MI_CODE_TO_REGIONS mapping:
- 13+ PTB-XL codes (ILMI, IPLMI, INJXX, etc.) -> 5 targets
- ILMI/IPLMI -> multi-hot IMI + LMI (posterior ignored)
- PMI (posterior) excluded in Phase 1

Label space: "ptbxl_derived_anatomical_v1"
Fingerprint: 8ab274e06afa1be8 (locked at training time)

Variants are mapped to combinations:
- ILMI -> IMI + LMI
- IPLMI -> IMI + LMI (posterior ignored for now)
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Any

import numpy as np
import pandas as pd


# Core MI localization regions
MI_LOCALIZATION_REGIONS = ["AMI", "ASMI", "ALMI", "IMI", "LMI"]
NUM_MI_REGIONS = len(MI_LOCALIZATION_REGIONS)

# Mapping from PTB-XL codes to our regions
# Some codes map to multiple regions
MI_CODE_TO_REGIONS = {
    # Direct mappings
    "AMI": ["AMI"],        # Anterior MI
    "ASMI": ["ASMI"],      # Anteroseptal MI
    "ALMI": ["ALMI"],      # Anterolateral MI
    "IMI": ["IMI"],        # Inferior MI
    "LMI": ["LMI"],        # Lateral MI
    
    # Combination mappings
    "ILMI": ["IMI", "LMI"],           # Inferolateral -> Inferior + Lateral
    "IPLMI": ["IMI", "LMI"],          # Inferoposterolateral -> Inferior + Lateral
    "IPMI": ["IMI"],                   # Inferoposterior -> just Inferior
    
    # Subendocardial injury codes (map to anatomical regions)
    "INJIN": ["IMI"],                  # Subendocardial injury inferior
    "INJAL": ["ALMI"],                 # Subendocardial injury anterolateral
    "INJAS": ["ASMI"],                 # Subendocardial injury anteroseptal
    "INJIL": ["IMI", "LMI"],           # Subendocardial injury inferolateral
    "INJLA": ["LMI"],                  # Subendocardial injury lateral
}

# Codes to exclude (too rare or no clear anatomical mapping)
EXCLUDED_MI_CODES = ["PMI"]  # Posterior MI - excluded for Phase 1


def extract_mi_regions(
    scp_codes: Dict[str, float],
    min_likelihood: float = 0.0
) -> List[str]:
    """
    Extract MI localization regions from a single record's SCP codes.
    
    Args:
        scp_codes: Dictionary of {code: likelihood}
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        List of MI regions present (subset of MI_LOCALIZATION_REGIONS)
    """
    regions = set()
    
    for code, likelihood in scp_codes.items():
        if likelihood > min_likelihood and code in MI_CODE_TO_REGIONS:
            regions.update(MI_CODE_TO_REGIONS[code])
    
    return list(regions)


def extract_mi_localization_labels(
    df: pd.DataFrame,
    min_likelihood: float = 0.0
) -> np.ndarray:
    """
    Extract multi-hot MI localization labels for all records.
    
    Only records with MI should have localization labels.
    
    Args:
        df: DataFrame with 'scp_codes' column
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        y_loc: (N, 5) array with multi-hot labels for [AMI, ASMI, ALMI, IMI, LMI]
    """
    n_samples = len(df)
    y_loc = np.zeros((n_samples, NUM_MI_REGIONS), dtype=np.float32)
    
    for i, scp_codes in enumerate(df["scp_codes"]):
        regions = extract_mi_regions(scp_codes, min_likelihood)
        for region in regions:
            if region in MI_LOCALIZATION_REGIONS:
                idx = MI_LOCALIZATION_REGIONS.index(region)
                y_loc[i, idx] = 1.0
    
    return y_loc


def add_mi_localization_labels(
    df: pd.DataFrame,
    min_likelihood: float = 0.0
) -> pd.DataFrame:
    """
    Add MI localization labels to DataFrame.
    
    Args:
        df: DataFrame with 'scp_codes' column
        min_likelihood: Minimum likelihood threshold
        
    Returns:
        DataFrame with added columns:
        - 'mi_regions': list of regions
        - 'label_AMI', 'label_ASMI', etc.: binary columns
        - 'y_loc': (5,) array for each row
        - 'has_mi_localization': whether any MI region is present
    """
    df = df.copy()
    
    # Extract y_loc
    y_loc = extract_mi_localization_labels(df, min_likelihood)
    
    # Add region list
    df["mi_regions"] = [
        [MI_LOCALIZATION_REGIONS[j] for j in range(NUM_MI_REGIONS) if y_loc[i, j] == 1]
        for i in range(len(df))
    ]
    
    # Add binary columns
    for j, region in enumerate(MI_LOCALIZATION_REGIONS):
        df[f"label_{region}"] = y_loc[:, j].astype(int)
    
    # Add y_loc as list
    df["y_loc"] = [y_loc[i] for i in range(len(df))]
    
    # Has any MI localization
    df["has_mi_localization"] = (y_loc.sum(axis=1) > 0).astype(int)
    
    return df


def get_mi_localization_mask(
    y_multi4: np.ndarray,
    mi_class_index: int = 0
) -> np.ndarray:
    """
    Get mask for samples that have MI (for localization loss masking).
    
    During training, localization loss should be 0 for MI-negative samples.
    
    Args:
        y_multi4: (N, 4) superclass labels [MI, STTC, CD, HYP]
        mi_class_index: Index of MI in y_multi4 (default 0)
        
    Returns:
        mask: (N,) boolean array, True if sample has MI
    """
    return y_multi4[:, mi_class_index] == 1


def compute_mi_localization_stats(
    df: pd.DataFrame,
    y_loc: np.ndarray
) -> Dict[str, Any]:
    """
    Compute statistics for MI localization labels.
    
    Args:
        df: DataFrame
        y_loc: (N, 5) localization labels
        
    Returns:
        Statistics dictionary
    """
    n_total = len(df)
    n_with_mi_loc = int((y_loc.sum(axis=1) > 0).sum())
    
    stats = {
        "n_total": n_total,
        "n_with_mi_localization": n_with_mi_loc,
        "mi_localization_rate": n_with_mi_loc / n_total if n_total > 0 else 0,
        "per_region": {},
        "avg_regions_per_mi_sample": 0,
    }
    
    for j, region in enumerate(MI_LOCALIZATION_REGIONS):
        count = int(y_loc[:, j].sum())
        stats["per_region"][region] = {
            "count": count,
            "rate": count / n_total if n_total > 0 else 0,
        }
    
    # Average regions per MI sample
    mi_mask = y_loc.sum(axis=1) > 0
    if mi_mask.sum() > 0:
        stats["avg_regions_per_mi_sample"] = float(y_loc[mi_mask].sum(axis=1).mean())
    
    return stats


def compute_mi_cooccurrence(y_loc: np.ndarray) -> pd.DataFrame:
    """
    Compute co-occurrence matrix for MI localization regions.
    
    Args:
        y_loc: (N, 5) localization labels
        
    Returns:
        Co-occurrence matrix as DataFrame
    """
    cooccur = np.zeros((NUM_MI_REGIONS, NUM_MI_REGIONS), dtype=int)
    
    for i in range(len(y_loc)):
        for j in range(NUM_MI_REGIONS):
            for k in range(NUM_MI_REGIONS):
                if y_loc[i, j] == 1 and y_loc[i, k] == 1:
                    cooccur[j, k] += 1
    
    return pd.DataFrame(
        cooccur,
        index=MI_LOCALIZATION_REGIONS,
        columns=MI_LOCALIZATION_REGIONS
    )
