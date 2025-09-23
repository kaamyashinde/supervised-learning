from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay

# Modellfabrikker: nye instanser hver gang
MODEL_FNS = {
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=None, max_features="sqrt", n_jobs=-1, random_state=42
    ),
}

def make_X(train: pd.DataFrame,
           test: pd.DataFrame,
           stream_cols: List[str],
           methods: Dict[str, callable],
           col: str,
           mname: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    bit_train = methods[mname](train[col]).rename(f"{col}__{mname}_bit")
    bit_test  = methods[mname](test[col]).rename(f"{col}__{mname}_bit")
    other = [c for c in stream_cols if c != col]
    Xtr = pd.concat([bit_train, train[other]], axis=1) if other else bit_train.to_frame()
    Xte = pd.concat([bit_test,  test[other]],  axis=1) if other else bit_test.to_frame()
    Xtr = Xtr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xte = Xte.apply(pd.to_numeric,  errors="coerce").fillna(0.0)
    return Xtr, Xte, list(Xtr.columns)

def auc_safe(y_true, proba) -> float:
    try:
        return roc_auc_score(y_true, proba)
    except Exception:
        return np.nan

def run_grid(train: pd.DataFrame,
             test: pd.DataFrame,
             y_train,
             y_test,
             stream_cols: List[str],
             methods: Dict[str, callable]) -> Tuple[dict, pd.DataFrame]:
    """Kjører kolonne×metode×modell, returnerer best-resultat og tabell med alle AUC."""
    from itertools import product

    results = []
    best = {"auc": -np.inf}

    for col, (mname, _), model_label in product(stream_cols, methods.items(), MODEL_FNS):
        Xtr, Xte, feat = make_X(train, test, stream_cols, methods, col, mname)
        clf = MODEL_FNS[model_label]()   # ny instans
        clf.fit(Xtr, y_train)
        auc = auc_safe(y_test, clf.predict_proba(Xte)[:, 1])
        results.append({"column": col, "method": mname, "model": model_label, "auc": auc})
        if np.isfinite(auc) and auc > best.get("auc", -np.inf):
            best = {"auc": auc, "col": col, "method": mname, "model": model_label,
                    "clf": clf, "X_test": Xte, "feat_names": feat}

    res_df = pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
    return best, res_df

def plot_best_roc(best: dict, y_test) -> None:
    plt.figure()
    RocCurveDisplay.from_estimator(best["clf"], best["X_test"], y_test)
    plt.title(f"Best ROC – {best['col']} [{best['method']}], {best['model']}, AUC={best['auc']:.3f}")
    plt.show()

def compare_dt_vs_rf(train: pd.DataFrame, test: pd.DataFrame, y_train, y_test,
                     stream_cols: List[str], methods: Dict[str, callable], best: dict) -> None:
    Xtr, Xte, _ = make_X(train, test, stream_cols, methods, best["col"], best["method"])
    dt = MODEL_FNS["DecisionTree"]().fit(Xtr, y_train)
    rf = MODEL_FNS["RandomForest"]().fit(Xtr, y_train)

    plt.figure()
    RocCurveDisplay.from_estimator(dt, Xte, y_test)
    RocCurveDisplay.from_estimator(rf, Xte, y_test)
    plt.title(f"ROC comparison on {best['col']} [{best['method']}] — DT vs RF")
    plt.legend(["DecisionTree", "RandomForest"])
    plt.show()

def visualize_best_model(best: dict) -> None:
    if best["model"] == "DecisionTree":
        plt.figure(figsize=(12, 6))
        plot_tree(best["clf"], filled=True,
                  feature_names=best["feat_names"],
                  class_names=["0", "1"], rounded=True)
        plt.title(f"Decision Tree (best overall): {best['col']} [{best['method']}]")
        plt.tight_layout()
        plt.show()
    else:
        rf = best["clf"]
        topk = pd.Series(rf.feature_importances_, index=best["feat_names"]).sort_values(ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        topk.plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title(f"Random Forest – Top 20 feature importances\n{best['col']} [{best['method']}]")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
