"""
CardioGuard-AI Signal Loading Module

Functions for loading and preprocessing ECG signals from PTB-XL.
Implements lazy loading for memory efficiency with large datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator

import numpy as np
import pandas as pd

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("Warning: wfdb not installed. Signal loading will not work.")
    print("Install with: pip install wfdb")


def _sanitize_cache_key(filename: str) -> str:
    return filename.replace("/", "_")


def _cache_paths(cache_dir: Path, cache_key: str) -> Tuple[Path, Path]:
    return cache_dir / f"{cache_key}.npz", cache_dir / f"{cache_key}.npy"


def _load_cached_signal(cache_dir: Path, cache_key: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    npz_path, npy_path = _cache_paths(cache_dir, cache_key)
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as data:
            signal = data["signal"]
            fields = data["fields"].item() if "fields" in data.files else None
        return signal, fields
    if npy_path.exists():
        return np.load(npy_path, allow_pickle=True), None
    return None, None


def _save_cached_signal(cache_dir: Path, cache_key: str, signal: np.ndarray, fields: Optional[dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, _ = _cache_paths(cache_dir, cache_key)
    np.savez_compressed(npz_path, signal=signal, fields=fields)


def load_single_signal(
    filename: str,
    base_path: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Load a single ECG signal from disk.
    
    Args:
        filename: Relative path from metadata (e.g., 'records100/00000/00001_lr')
        base_path: Base path to PTB-XL data directory
        cache_dir: Optional cache directory for .npy/.npz signal storage
        use_cache: If True, read/write from cache when cache_dir is provided
        
    Returns:
        Tuple of (signal_array, metadata_dict)
        - signal_array: shape (samples, channels) e.g., (1000, 12) for 100Hz
        - metadata_dict: wfdb record metadata
        
    Example:
        >>> signal, meta = load_single_signal("records100/00000/00001_lr", base_path)
        >>> print(signal.shape)  # (1000, 12)
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library required for signal loading")
    
    base_path = Path(base_path)
    full_path = base_path / filename

    cache_dir_path = Path(cache_dir) if cache_dir is not None else None
    cache_key = _sanitize_cache_key(filename)
    if use_cache and cache_dir_path is not None:
        cached_signal, cached_fields = _load_cached_signal(cache_dir_path, cache_key)
        if cached_signal is not None:
            return cached_signal, cached_fields or {}
    
    # wfdb.rdsamp returns (signal, fields)
    signal, fields = wfdb.rdsamp(str(full_path))
    
    if use_cache and cache_dir_path is not None:
        _save_cached_signal(cache_dir_path, cache_key, signal, fields)

    return signal, fields


def load_signals_batch(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    filename_column: str = "filename_lr",
    max_samples: Optional[int] = None,
    progress: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True
) -> Tuple[np.ndarray, List[int]]:
    """
    Load multiple ECG signals into a single array.
    
    WARNING: This loads all signals into memory. For large datasets,
    use SignalDataset for lazy loading.
    
    Args:
        df: DataFrame with signal file paths
        base_path: Base path to PTB-XL data directory
        filename_column: Column containing relative file paths
        max_samples: Optional limit on number of samples to load
        progress: If True, print progress updates
        cache_dir: Optional cache directory for .npy/.npz signal storage
        use_cache: If True, read/write from cache when cache_dir is provided
        
    Returns:
        Tuple of (signals_array, ecg_ids)
        - signals_array: shape (n_samples, n_timesteps, n_channels)
        - ecg_ids: list of ecg_id values corresponding to loaded signals
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb library required for signal loading")
    
    base_path = Path(base_path)
    cache_dir_path = Path(cache_dir) if cache_dir is not None else None
    
    # Limit samples if specified
    if max_samples is not None:
        df = df.head(max_samples)
    
    signals = []
    ecg_ids = []
    total = len(df)
    
    for i, (ecg_id, row) in enumerate(df.iterrows()):
        if progress and (i + 1) % 1000 == 0:
            print(f"Loading: {i + 1}/{total}")
        
        try:
            signal, _ = load_single_signal(
                row[filename_column],
                base_path,
                cache_dir=cache_dir_path,
                use_cache=use_cache
            )
            signals.append(signal)
            ecg_ids.append(ecg_id)
        except Exception as e:
            print(f"Error loading ecg_id={ecg_id}: {e}")
            continue
    
    if signals:
        signals_array = np.array(signals)
    else:
        signals_array = np.array([])
    
    return signals_array, ecg_ids


def build_npz_cache(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    cache_path: Union[str, Path],
    filename_column: str = "filename_lr",
    max_samples: Optional[int] = None,
    progress: bool = True
) -> Path:
    """
    Build a single NPZ cache file containing signals and ecg_ids.

    Args:
        df: DataFrame with signal file paths
        base_path: Base path to PTB-XL data directory
        cache_path: Output path for the npz file
        filename_column: Column containing relative file paths
        max_samples: Optional limit on number of samples to cache
        progress: If True, print progress updates

    Returns:
        Path to the written cache file
    """
    signals, ecg_ids = load_signals_batch(
        df,
        base_path=base_path,
        filename_column=filename_column,
        max_samples=max_samples,
        progress=progress
    )
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, signals=signals, ecg_ids=np.array(ecg_ids))
    return cache_path


def load_npz_cache(cache_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load signals and ecg_ids from an NPZ cache.

    Args:
        cache_path: Path to npz cache created by build_npz_cache

    Returns:
        Tuple of (signals, ecg_ids)
    """
    cache_path = Path(cache_path)
    data = np.load(cache_path, allow_pickle=False)
    return data["signals"], data["ecg_ids"]


class CachedSignalDataset:
    """
    Dataset backed by a prebuilt NPZ cache.

    Example:
        >>> signals, ecg_ids = load_npz_cache(cache_path)
        >>> dataset = CachedSignalDataset(signals, ecg_ids, labels)
    """

    def __init__(
        self,
        signals: np.ndarray,
        ecg_ids: np.ndarray,
        labels: Optional[Dict[int, int]] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize the cached dataset.

        Args:
            signals: Array of shape (n_samples, n_timesteps, n_channels)
            ecg_ids: Array of ecg_id values aligned with signals
            labels: Optional mapping {ecg_id: label}
            transform: Optional transform function applied to each signal
        """
        self.signals = signals
        self.ecg_ids = ecg_ids
        self.labels = labels or {}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[int], int]:
        signal = self.signals[idx]
        ecg_id = int(self.ecg_ids[idx])
        if self.transform is not None:
            signal = self.transform(signal)
        label = self.labels.get(ecg_id)
        return signal, label, ecg_id


class SignalDataset:
    """
    Lazy-loading dataset for ECG signals.
    
    Loads signals on-demand to minimize memory usage.
    Suitable for training loops and large-scale processing.
    
    Example:
        >>> dataset = SignalDataset(df, base_path, config)
        >>> for signal, label, ecg_id in dataset:
        ...     # Process signal
        ...     pass
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        base_path: Union[str, Path],
        filename_column: str = "filename_lr",
        label_column: Optional[str] = None,
        transform: Optional[callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True
    ):
        """
        Initialize the signal dataset.
        
        Args:
            df: DataFrame with signal metadata
            base_path: Base path to PTB-XL data
            filename_column: Column with relative file paths
            label_column: Optional column with labels
            transform: Optional transform function applied to each signal
            cache_dir: Optional cache directory for .npy/.npz signal storage
            use_cache: If True, read/write from cache when cache_dir is provided
        """
        self.df = df
        self.base_path = Path(base_path)
        self.filename_column = filename_column
        self.label_column = label_column
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        if self.use_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store index for iteration
        self._indices = list(df.index)
    
    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[int], int]:
        """
        Get a single sample by index position.
        
        Args:
            idx: Integer index (position in dataset)
            
        Returns:
            Tuple of (signal, label, ecg_id)
            - signal: numpy array of shape (samples, channels)
            - label: label value if label_column specified, else None
            - ecg_id: original ecg_id from the DataFrame index
        """
        ecg_id = self._indices[idx]
        row = self.df.loc[ecg_id]

        # Load signal (with optional cache)
        signal = None
        if self.cache_dir is not None:
            cache_path = self._cache_path(ecg_id, row[self.filename_column])
            if cache_path.exists():
                signal = np.load(cache_path)
        if signal is None:
            signal, _ = load_single_signal(row[self.filename_column], self.base_path)
            if self.cache_dir is not None:
                np.save(cache_path, signal)
        if self.use_cache and self.cache_dir is not None:
            cache_key = self._cache_key(ecg_id, row[self.filename_column])
            signal, _ = _load_cached_signal(self.cache_dir, cache_key)
        if signal is None:
            signal, _ = load_single_signal(row[self.filename_column], self.base_path)
            if self.use_cache and self.cache_dir is not None:
                _save_cached_signal(self.cache_dir, cache_key, signal, None)
        
        # Apply transform if specified
        if self.transform is not None:
            signal = self.transform(signal)
        
        # Get label if specified
        label = None
        if self.label_column is not None:
            label = row[self.label_column]
        
        return signal, label, ecg_id
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[int], int]]:
        """Iterate over all samples."""
        for idx in range(len(self)):
            yield self[idx]

    def _cache_path(self, ecg_id: int, filename: str) -> Path:
        safe_name = filename.replace("/", "_")
        return self.cache_dir / f"{ecg_id}_{safe_name}.npy"
    def _cache_key(self, ecg_id: int, filename: str) -> str:
        safe_name = _sanitize_cache_key(filename)
        return f"{ecg_id}_{safe_name}"


# ============================================================================
# Basic Preprocessing Functions
# ============================================================================

def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore"
) -> np.ndarray:
    """
    Normalize ECG signal.
    
    Args:
        signal: Input signal of shape (samples, channels) or (batch, samples, channels)
        method: Normalization method - "zscore", "minmax", or "robust"
        
    Returns:
        Normalized signal with same shape
    """
    if method == "zscore":
        # Z-score normalization per channel
        mean = signal.mean(axis=-2, keepdims=True)
        std = signal.std(axis=-2, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (signal - mean) / std
    
    elif method == "minmax":
        # Min-max normalization per channel
        min_val = signal.min(axis=-2, keepdims=True)
        max_val = signal.max(axis=-2, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        return (signal - min_val) / range_val
    
    elif method == "robust":
        # Robust scaling using median and IQR per channel
        median = np.median(signal, axis=-2, keepdims=True)
        q75 = np.percentile(signal, 75, axis=-2, keepdims=True)
        q25 = np.percentile(signal, 25, axis=-2, keepdims=True)
        iqr = q75 - q25
        iqr = np.where(iqr == 0, 1, iqr)
        return (signal - median) / iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_channel_stats(
    signals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std using training signals.

    Args:
        signals: Array of shape (n_samples, n_timesteps, n_channels)

    Returns:
        Tuple of (mean, std) arrays of shape (1, 1, n_channels)
    """
    mean = signals.mean(axis=(0, 1), keepdims=True)
    std = signals.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1, std)
    return mean, std


def normalize_with_stats(
    signal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Normalize signal using precomputed per-channel stats.

    Args:
        signal: Input signal of shape (samples, channels) or
            (batch, samples, channels)
        mean: Mean array broadcastable to signal
        std: Std array broadcastable to signal

    Returns:
        Normalized signal
    """
    return (signal - mean) / std


def resample_signal(
    signal: np.ndarray,
    original_freq: int,
    target_freq: int
) -> np.ndarray:
    """
    Resample signal to target frequency.
    
    Args:
        signal: Input signal of shape (samples, channels)
        original_freq: Original sampling frequency in Hz
        target_freq: Target sampling frequency in Hz
        
    Returns:
        Resampled signal
    """
    from scipy import signal as scipy_signal
    
    # Calculate resampling ratio
    n_samples = signal.shape[0]
    n_target = int(n_samples * target_freq / original_freq)
    
    # Resample each channel
    resampled = scipy_signal.resample(signal, n_target, axis=0)
    
    return resampled


def get_lead_names() -> List[str]:
    """
    Get standard 12-lead ECG lead names in PTB-XL order.
    
    Returns:
        List of lead names ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                           'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    """
    return ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
