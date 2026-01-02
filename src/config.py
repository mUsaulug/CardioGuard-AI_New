"""
CardioGuard-AI Configuration Module

Central configuration for the CardioGuard-AI project.
All paths, constants, and hyperparameters should be defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PTBXLConfig:
    """Configuration for PTB-XL dataset processing."""
    
    # Dataset paths (relative to project root by default)
    data_root: Path = field(default_factory=lambda: Path("physionet.org/files/ptb-xl/1.0.3"))
    
    # Sampling rate: 100 or 500 Hz
    sampling_rate: int = 100
    
    # Standard PTB-XL split using strat_fold
    # Train: folds 1-8, Val: fold 9, Test: fold 10
    train_folds: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    val_folds: List[int] = field(default_factory=lambda: [9])
    test_folds: List[int] = field(default_factory=lambda: [10])
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Minimum likelihood threshold for label assignment
    # PTB-XL uses 0-100 scale for likelihood
    min_likelihood: float = 50.0
    
    # Label task type
    task: str = "binary"  # "binary" for MI vs NORM, "multiclass" for 5-class
    
    @property
    def metadata_path(self) -> Path:
        """Path to ptbxl_database.csv"""
        return self.data_root / "ptbxl_database.csv"
    
    @property
    def scp_statements_path(self) -> Path:
        """Path to scp_statements.csv"""
        return self.data_root / "scp_statements.csv"
    
    @property
    def records_path(self) -> Path:
        """Path to signal records directory"""
        if self.sampling_rate == 100:
            return self.data_root / "records100"
        else:
            return self.data_root / "records500"
    
    @property
    def filename_column(self) -> str:
        """Column name for signal file paths"""
        if self.sampling_rate == 100:
            return "filename_lr"
        else:
            return "filename_hr"


# Diagnostic class mappings
DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
DIAGNOSTIC_PRIORITY = ["MI", "STTC", "CD", "HYP", "NORM"]

# All MI-related SCP codes from scp_statements.csv
# These include both definite MI and subendocardial injury patterns
MI_CODES = [
    "IMI",    # Inferior MI
    "ASMI",   # Anteroseptal MI
    "AMI",    # Anterior MI
    "ALMI",   # Anterolateral MI
    "LMI",    # Lateral MI
    "ILMI",   # Inferolateral MI
    "IPLMI",  # Inferoposterolateral MI
    "IPMI",   # Inferoposterior MI
    "PMI",    # Posterior MI
    # Subendocardial injury codes (also MI class)
    "INJIN",  # Subendocardial injury inferior
    "INJAL",  # Subendocardial injury anterolateral
    "INJAS",  # Subendocardial injury anteroseptal
    "INJIL",  # Subendocardial injury inferolateral
    "INJLA",  # Subendocardial injury lateral
]


def get_default_config(project_root: Optional[Path] = None) -> PTBXLConfig:
    """
    Get default configuration with optional project root override.
    
    Args:
        project_root: Optional path to project root. If None, uses current working directory.
        
    Returns:
        PTBXLConfig instance with paths resolved relative to project_root.
    """
    if project_root is None:
        project_root = Path.cwd()
    
    return PTBXLConfig(
        data_root=project_root / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
    )
