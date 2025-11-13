import pandas as pd
import numpy as np

# Label mapping (robust to string or numeric labels)
LABEL_MAP = {"no_pain": 0, "low_pain": 1, "high_pain": 2}

def _detect_joint_cols(df):
    return sorted([c for c in df.columns if c.startswith("joint_")])

def _get_data_cols(df):
    cols = _detect_joint_cols(df)
    if not cols:
        raise ValueError("No 'joint_*' columns found in df.")
    return cols

# Load labels if not already present
if "target" not in globals():
    try:
        target = pd.read_csv("pirate_pain_train_labels.csv")
    except FileNotFoundError:
        print("Warning: 'target' not defined and 'pirate_pain_train_labels.csv' not found.")
    else:
        if "label" in target.columns:
            # Map strings to ints if needed
            if target["label"].dtype == object:
                target["label"] = target["label"].map(lambda x: LABEL_MAP.get(x, x))

# --- Window builder ---
def build_windows(
    df: pd.DataFrame,
    target: pd.DataFrame | None,
    window: int = 300,
    stride: int = 75,
    padding: str = "zero",      # 'zero' or 'drop_last'
    feature: str = "3d",        # '3d' (for RNNs) or 'flatten' (for traditional ML)
    data_cols: list | None = None,
):
    """
    Builds sliding windows from df and returns (X, y, groups).
    
    Args:
        df: DataFrame with time series data
        target: DataFrame with labels
        window: Window size (number of timesteps)
        stride: Stride for sliding window
        padding: 'zero' to pad with zeros, 'drop_last' to drop incomplete windows
        feature: '3d' returns (samples, timesteps, features) for RNNs,
                 'flatten' returns (samples, timesteps*features) for traditional ML
        data_cols: List of columns to use as features
    
    Returns:
        X: numpy array of shape (n_samples, window, n_features) if feature='3d'
           or (n_samples, window*n_features) if feature='flatten'
        y: numpy array of labels
        groups: numpy array of sample indices
    """
    if data_cols is None:
        data_cols = _get_data_cols(df)
    X, y, groups = [], [], []
    for sid in df["sample_index"].unique():
        temp = df[df["sample_index"] == sid][data_cols].values
        # get label for this id
        if target is not None:
            lab_arr = target[target["sample_index"] == sid]["label"].values
            if len(lab_arr) == 0:
                # if missing label, skip this id
                continue
            lab = lab_arr[0]
            if isinstance(lab, str):
                lab = LABEL_MAP.get(lab, lab)
        # padding computation
        pad = (window - (len(temp) % window)) % window
        if padding == "zero" and pad:
            temp = np.concatenate([temp, np.zeros((pad, temp.shape[1]), dtype=temp.dtype)], axis=0)
        L = len(temp)
        start = 0
        while start + window <= L:
            seg = temp[start:start + window]  # shape: (window, n_features)
            if feature == "flatten":
                feat = seg.reshape(-1)  # shape: (window * n_features,)
            else:
                feat = seg  # Keep 3D: (window, n_features)
            X.append(feat)
            if target is not None:
                y.append(lab)
            groups.append(sid)
            start += stride
    if not X:
        raise ValueError("No windows were created. Check your data and parameters.")
    return np.asarray(X), np.asarray(y), np.asarray(groups)