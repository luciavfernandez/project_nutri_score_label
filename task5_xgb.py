"""
Task 5 (Optional): Machine-learning model combining Nutri-Score + Green-Score
-------------------------------------------------------------------------------
Train an XGBoost classifier to predict Nutri-Score labels (A–E) from the
nutritional criteria + Green-Score, then compare its assignments to the
weighted-sum model (WSM) from Task 4.4.

Usage:
    python task5_xgb.py \
        --data food_database.csv \
        --wsm weighted_sum_results.csv \
        --output xgb_results.csv

Outputs:
    - xgb_results.csv : dataset with true Nutri-Score, XGB prediction, WSM class
    - feature_importances.csv : gain-based feature importances
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from weighted_sum_model import WeightedSumModel, DEFAULT_WEIGHTS


TARGET_COL = "nutri_score_label"
CLASSES = ["A", "B", "C", "D", "E"]


@dataclass
class DataBundle:
    df: pd.DataFrame
    with_wsm: bool


def load_with_wsm(data_path: Path, wsm_path: Optional[Path]) -> DataBundle:
    """
    Load the main dataset and, if needed, compute or read WSM outputs.
    """
    df = pd.read_csv(data_path)

    # Align column names to weighted_sum_model expectations
    rename_map = {
        "energy_100g": "energy_kj",
        "saturated_fat_100g": "saturated_fat_g",
        "sugars_100g": "sugars_g",
        "salt_100g": "salt_g",
        "proteins_100g": "proteins_g",
        "fiber_100g": "fiber_g",
        "fvl_percent": "fruit_veg_pct",
        "green_score_label": "ecoscore_grade",
    }
    df = df.rename(columns=rename_map)

    # If WSM file provided, read it; otherwise compute on the fly.
    if wsm_path and wsm_path.exists():
        wsm_df = pd.read_csv(wsm_path)
    else:
        model = WeightedSumModel(weights=DEFAULT_WEIGHTS)
        wsm_df, _ = model.score_dataframe(df)

    # Choose join key
    key = "code" if "code" in df.columns and "code" in wsm_df.columns else "product_name"
    merged = df.merge(
        wsm_df[[key, "wsm_score", "wsm_class"]],
        on=key,
        how="left",
        validate="m:1",
    )
    return DataBundle(df=merged, with_wsm="wsm_class" in merged.columns)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    Select and clean feature columns for XGB.
    """
    feature_cols: List[str] = [
        "energy_kj",
        "saturated_fat_g",
        "sugars_g",
        "salt_g",
        "proteins_g",
        "fiber_g",
        "fruit_veg_pct",
        "green_score_value",
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    # Impute missing with column medians
    X = X.fillna(X.median(numeric_only=True))
    return X, available


def fit_xgb(X: pd.DataFrame, y: np.ndarray) -> XGBClassifier:
    """
    Train an XGBoost classifier with sane defaults to avoid overfitting.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X, y)
    return model


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    preds_int = model.predict(X_test)
    preds = np.vectorize(lambda i: CLASSES[int(i)])(preds_int)
    y_labels = np.vectorize(lambda i: CLASSES[int(i)])(y_test)

    acc = accuracy_score(y_labels, preds)
    cm = confusion_matrix(y_labels, preds, labels=CLASSES)
    report = classification_report(
        y_labels, preds, labels=CLASSES, zero_division=0
    )
    print("\n=== XGBoost Performance (held-out test) ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=list("ABCDE"), columns=list("ABCDE")))
    print("\nClassification Report:")
    print(report)
    return {"preds": preds, "acc": acc}


def compare_to_wsm(df: pd.DataFrame, preds: np.ndarray) -> None:
    if "wsm_class" not in df.columns:
        print("\nNo WSM classes available for comparison.")
        return
    agreement = (df["wsm_class"] == preds).mean()
    print(f"\nXGB vs WSM agreement: {agreement*100:.2f}%")
    cm = pd.crosstab(df["wsm_class"], preds, dropna=False).reindex(
        index=list("ABCDE"), columns=list("ABCDE")
    ).fillna(0).astype(int)
    print("\nXGB vs WSM crosstab (rows=WSM, cols=XGB):")
    print(cm)


def save_outputs(
    df: pd.DataFrame, preds: np.ndarray, model: XGBClassifier, output: Path, feature_cols: List[str]
) -> None:
    out_df = df.copy()
    out_df["xgb_pred"] = preds
    out_df.to_csv(output, index=False)
    print(f"\nSaved predictions to: {output}")

    # Feature importances (gain)
    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    fi.to_csv(output.with_name("feature_importances.csv"), index=False)
    print(f"Saved feature importances to: {output.with_name('feature_importances.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: XGB classifier for Nutri-Score + Green-Score")
    parser.add_argument("--data", default="food_database.csv", help="Input CSV with Nutri-Score and Green-Score fields")
    parser.add_argument("--wsm", default="weighted_sum_results.csv", help="Optional WSM results CSV for comparison")
    parser.add_argument("--output", default="xgb_results.csv", help="Output CSV for predictions")
    args = parser.parse_args()

    data_path = Path(args.data)
    wsm_path = Path(args.wsm) if args.wsm else None

    bundle = load_with_wsm(data_path, wsm_path)
    df = bundle.df

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X, feature_cols = prepare_features(df)
    y_labels = df[TARGET_COL].str.upper()
    label_to_int = {label: idx for idx, label in enumerate(CLASSES)}
    y = y_labels.map(label_to_int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_labels
    )

    model = fit_xgb(X_train, y_train)
    eval_res = evaluate(model, X_test, y_test)

    # Full-dataset predictions for comparison against WSM
    full_preds_int = model.predict(X)
    full_preds = np.vectorize(lambda i: CLASSES[int(i)])(full_preds_int)
    compare_to_wsm(df, full_preds)
    save_outputs(df, full_preds, model, Path(args.output), feature_cols)


if __name__ == "__main__":
    main()
