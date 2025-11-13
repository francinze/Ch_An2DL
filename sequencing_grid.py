from windows import build_windows
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from preprocessing import run_preprocessing


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

def _effective_splits(df, n_splits_max):
    n_groups = int(df["sample_index"].nunique())
    return max(2, min(n_splits_max, n_groups))

def run_grid(df, target, padding, data_cols, n_splits_max, windows, strides, df_res):
    """Valuta tutte le combinazioni (window, stride) ∈ WINDOWS×STRIDES con il padding richiesto.
    Ritorna un DataFrame con mean_macroF1/std per ciascuna coppia."""
    if "eval_strategy" not in globals():
        raise RuntimeError("Serve eval_strategy()/build_windows(). Esegui prima le celle dell'harness.")
    rows = []
    n_splits = _effective_splits(df, n_splits_max)
    print(f"Using n_splits={n_splits} for GroupKFold (max {n_splits_max})")
    for w in windows:
        print(f"\nWindow={w}")
        for s in strides:
            print(f"    Stride={s}...", end="", flush=True)
            params = {"window": w, "stride": s, "padding": padding, "feature": "flatten"}
            if data_cols is not None:
                params["data_cols"] = data_cols
            try:
                scores = eval_strategy(df, target, params, n_splits=n_splits)
                rows.append({"window": w, "stride": s, "padding": padding,
                             "mean_macroF1": float(scores.mean()), "std": float(scores.std())})
            except Exception as e:
                rows.append({"window": w, "stride": s, "padding": padding,
                             "mean_macroF1": np.nan, "std": np.nan, "error": str(e)})
    df_res = pd.concat([df_res, pd.DataFrame(rows)], ignore_index=True)
    return df_res

def run_sequencing_grid(df_train, target, n_splits_max, windows, strides, data_cols=None, padding_runs=["zero", "edge"]):
    # ---- Pre-flight check ----
    needed = ["df_train", "target", "build_windows", "eval_strategy"]
    missing = [n for n in needed if n not in globals()]
    if missing:
        print("⚠️ Mancano variabili/funzioni:", missing)
        print("Esegui le celle che definiscono df_train/target/build_windows/eval_strategy e riprova.")
        return None
    else:
        # ---- Esecuzione dei due conti separati (padding diverso) ----
        df_res = pd.DataFrame()
        for pad in padding_runs:
            print(f"\n>>> Running 2D grid with padding = {pad}, windows = {windows}, strides = {strides}...")
            df_res = run_grid(df_train, target, padding=pad, data_cols=data_cols, n_splits_max=n_splits_max, windows=windows, strides=strides, df_res=df_res)

            # Mostra le migliori 15 combinazioni
            try:
                top = (df_res.dropna(subset=["mean_macroF1"])  # noqa
                            .sort_values("mean_macroF1", ascending=False)
                            .head(15))
                print("Top-15:") 
                print(top.to_string(index=False))
            except Exception:
                pass
        out_csv = f"grid_sequencing_results.csv"
        df_res.to_csv(out_csv, index=False)

        # ---- Unione e best complessivo ----
        print("\n=== BEST OVERALL (top 20 su entrambi i padding) ===")
        print(df_res.dropna(subset=["mean_macroF1"])  # noqa
                    .sort_values("mean_macroF1", ascending=False)
                    .head(20)
                    .to_string(index=False))
        
        return df_res

if __name__ == "__main__":
    df_train, df_val, target, val_target = run_preprocessing()

    # Hyperparameters
    N_SPLITS_MAX = 2
    WINDOWS = [150]
    STRIDES = [150]
    PADDING_RUNS = ["zero", "edge"]

    run_sequencing_grid(df_train, target, n_splits_max=N_SPLITS_MAX, windows=WINDOWS, strides=STRIDES, padding_runs=PADDING_RUNS)