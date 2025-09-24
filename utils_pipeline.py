from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.pipeline import Pipeline
from itertools import product

from stream_transformers import StreamFeatureTransformer

# Model factories: new instances each time
MODEL_FNS = {
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=None, max_features="sqrt", n_jobs=-1, random_state=42
    ),
}

def create_pipeline(target_column: str, method: str, other_columns: List[str], 
                   model_label: str) -> Pipeline:
    """
    Create a pipeline with stream feature transformer and classifier.
    """
    transformer = StreamFeatureTransformer(
        target_column=target_column,
        method=method,
        other_columns=other_columns
    )
    
    classifier = MODEL_FNS[model_label]()
    
    pipeline = Pipeline([
        ('transformer', transformer),
        ('classifier', classifier)
    ])
    
    return pipeline

def auc_safe(y_true, proba) -> float:
    """Safe AUC calculation with error handling."""
    try:
        return roc_auc_score(y_true, proba)
    except Exception:
        return np.nan

def run_grid(train: pd.DataFrame,
             test: pd.DataFrame,
             y_train,
             y_test,
             stream_cols: List[str],
             methods: Dict[str, str]) -> Tuple[dict, pd.DataFrame]:
    """
    Run grid search over column×method×model combinations using pipelines.
    Returns best result and table with all AUC scores.
    """
    results = []
    best = {"auc": -np.inf}

    for col, method_name in product(stream_cols, methods.keys()):
        # Get other columns (excluding the target column)
        other_cols = [c for c in stream_cols if c != col]
        
        for model_label in MODEL_FNS.keys():
            # Create pipeline
            pipeline = create_pipeline(col, method_name, other_cols, model_label)
            
            # Fit and predict
            pipeline.fit(train, y_train)
            auc = auc_safe(y_test, pipeline.predict_proba(test)[:, 1])
            
            results.append({
                "column": col, 
                "method": method_name, 
                "model": model_label, 
                "auc": auc
            })
            
            if np.isfinite(auc) and auc > best.get("auc", -np.inf):
                best = {
                    "auc": auc, 
                    "col": col, 
                    "method": method_name, 
                    "model": model_label,
                    "pipeline": pipeline,
                    "X_test": test,
                    "feat_names": pipeline.named_steps['transformer'].transform(test).columns.tolist()
                }

    res_df = pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
    return best, res_df

def plot_best_roc(best: dict, y_test) -> None:
    """Plot ROC curve for the best model."""
    plt.figure()
    RocCurveDisplay.from_estimator(best["pipeline"], best["X_test"], y_test)
    plt.title(f"Best ROC – {best['col']} [{best['method']}], {best['model']}, AUC={best['auc']:.3f}")
    plt.show()

def compare_dt_vs_rf(train: pd.DataFrame, test: pd.DataFrame, y_train, y_test,
                     stream_cols: List[str], methods: Dict[str, str], best: dict) -> None:
    """Compare DecisionTree vs RandomForest performance."""
    other_cols = [c for c in stream_cols if c != best["col"]]
    
    dt_pipeline = create_pipeline(best["col"], best["method"], other_cols, "DecisionTree")
    rf_pipeline = create_pipeline(best["col"], best["method"], other_cols, "RandomForest")
    
    dt_pipeline.fit(train, y_train)
    rf_pipeline.fit(train, y_train)

    plt.figure()
    RocCurveDisplay.from_estimator(dt_pipeline, test, y_test)
    RocCurveDisplay.from_estimator(rf_pipeline, test, y_test)
    plt.title(f"ROC comparison on {best['col']} [{best['method']}] — DT vs RF")
    plt.legend(["DecisionTree", "RandomForest"])
    plt.show()

def visualize_best_model(best: dict) -> None:
    """Visualize the best model (tree structure or feature importance)."""
    if best["model"] == "DecisionTree":
        plt.figure(figsize=(12, 6))
        plot_tree(best["pipeline"].named_steps['classifier'], filled=True,
                  feature_names=best["feat_names"],
                  class_names=["0", "1"], rounded=True)
        plt.title(f"Decision Tree (best overall): {best['col']} [{best['method']}]")
        plt.tight_layout()
        plt.show()
    else:
        rf = best["pipeline"].named_steps['classifier']
        topk = pd.Series(rf.feature_importances_, index=best["feat_names"]).sort_values(ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        topk.plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title(f"Random Forest – Top 20 feature importances\n{best['col']} [{best['method']}]")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()