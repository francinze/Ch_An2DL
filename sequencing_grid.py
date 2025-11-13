from windows import build_windows
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np


# --- Strategy evaluator ---
def eval_strategy(df: pd.DataFrame, target: pd.DataFrame, params: dict, n_splits: int = 5):
    X, y, groups = build_windows(df, target, **params)
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, te in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        scores.append(f1_score(y[te], pred, average="macro"))
    return np.array(scores)

# ---- Configurazione griglia ----
WINDOWS = list(range(10, 601, 10))   # 10..600 step 10 (0 escluso perché invalido)
STRIDES = list(range(10, 601, 10))
PADDING_RUNS = ["zero", "drop_last"]  # due conti separati
N_SPLITS_MAX = 5                      # max fold per GroupKFold

# Se vuoi forzare le feature, imposta DATA_COLS (altrimenti usa quelle auto-rilevate in build_windows)
# DATA_COLS = ['has_prosthetics'] + [c for c in df_train.columns if c.startswith('joint_')]
DATA_COLS = None

def _effective_splits(df, n_splits_max=N_SPLITS_MAX):
    n_groups = int(df["sample_index"].nunique())
    return max(2, min(n_splits_max, n_groups))

def run_grid(df, target, padding="zero", data_cols, n_splits_max, windows, strides):
    """Valuta tutte le combinazioni (window, stride) ∈ WINDOWS×STRIDES con il padding richiesto.
    Ritorna un DataFrame con mean_macroF1/std per ciascuna coppia."""
    if "eval_strategy" not in globals():
        raise RuntimeError("Serve eval_strategy()/build_windows(). Esegui prima le celle dell'harness.")
    rows = []
    n_splits = _effective_splits(df, n_splits_max)
    for w in windows:
        for s in strides:
            params = {"window": w, "stride": s, "labeling": "id", "padding": padding}
            if data_cols is not None:
                params["data_cols"] = data_cols
            try:
                scores = eval_strategy(df, target, params, n_splits=n_splits)
                rows.append({"window": w, "stride": s, "padding": padding,
                             "mean_macroF1": float(scores.mean()), "std": float(scores.std())})
            except Exception as e:
                rows.append({"window": w, "stride": s, "padding": padding,
                             "mean_macroF1": np.nan, "std": np.nan, "error": str(e)})
    return pd.DataFrame(rows)

def run_sequencing_grid(df_train, target, n_splits_max=N_SPLITS_MAX, windows=WINDOWS, strides=STRIDES, data_cols=DATA_COLS):
    # ---- Pre-flight check ----
    needed = ["df_train", "target", "build_windows", "eval_strategy"]
    missing = [n for n in needed if n not in globals()]
    if missing:
        print("⚠️ Mancano variabili/funzioni:", missing)
        print("Esegui le celle che definiscono df_train/target/build_windows/eval_strategy e riprova.")
    else:
        # ---- Esecuzione dei due conti separati (padding diverso) ----
        all_results = []
        for pad in PADDING_RUNS:
            print(f"\n>>> Running 2D grid with padding = {pad}")
            df_res = run_grid(df_train, target, padding=pad, data_cols=data_cols, n_splits_max=n_splits_max, windows=windows, strides=strides)
            all_results.append(df_res)
            out_csv = f"grid_window_stride_results_padding_{pad}.csv"
            df_res.to_csv(out_csv, index=False)
            # Mostra le migliori 15 combinazioni
            try:
                top = (df_res.dropna(subset=["mean_macroF1"])  # noqa
                            .sort_values("mean_macroF1", ascending=False)
                            .head(15))
                print("Top-15:") 
                print(top.to_string(index=False))
            except Exception:
                pass

        # ---- Unione e best complessivo ----
        results = pd.concat(all_results, ignore_index=True)
        print("\n=== BEST OVERALL (top 20 su entrambi i padding) ===")
        print(results.dropna(subset=["mean_macroF1"])  # noqa
                    .sort_values("mean_macroF1", ascending=False)
                    .head(20)
                    .to_string(index=False))
        
    return results