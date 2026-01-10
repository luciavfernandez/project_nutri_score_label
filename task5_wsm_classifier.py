"""
Task 5 (Reframed): Learn a WSM-like classifier and compare to Nutri-Score
-----------------------------------------------------------------------
Instead of predicting the official Nutri-Score, we train a classifier to
replicate our Weighted Sum Model (WSM) classes (A–E) from the raw criteria.
We then compare the learned assignments to:
  - the WSM target (agreement rate, confusion)
  - the official Nutri-Score, to highlight differences

Usage:
    python task5_wsm_classifier.py \
        --data food_database.csv \
        --wsm weighted_sum_results.csv \
        --output wsm_classifier_results.csv

Outputs:
    - wsm_classifier_results.csv : dataset with WSM class, ML predicted class
    - wsm_classifier_feature_importances.csv : gain-based importances
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

from weighted_sum_model import DEFAULT_WEIGHTS, WeightedSumModel


CLASSES = ["A", "B", "C", "D", "E"]
GREEN_GRADE_TO_VALUE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


@dataclass
class DataBundle:
    df: pd.DataFrame
    joined_on: str


def load_and_attach_wsm(data_path: Path, wsm_path: Optional[Path]) -> DataBundle:
    """
    Load the base dataset and ensure WSM scores/classes are present
    (by reading an existing file or computing them).
    """
    df = pd.read_csv(data_path)

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

    if merged["wsm_class"].isna().any():
        raise ValueError("WSM classes missing after merge; check join key or input files.")

    return DataBundle(df=merged, joined_on=key)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    Build numeric feature matrix aligned with WSM inputs.
    """
    # Create numeric eco value if only grade is present
    if "green_score_value" not in df.columns and "ecoscore_grade" in df.columns:
        df = df.copy()
        df["green_score_value"] = (
            df["ecoscore_grade"]
            .astype(str)
            .str.upper()
            .map(GREEN_GRADE_TO_VALUE)
        )

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
    X = X.fillna(X.median(numeric_only=True))
    return X, available


def fit_model(X: pd.DataFrame, y: np.ndarray) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=250,
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


def evaluate_vs_wsm(model: XGBClassifier, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    preds_int = model.predict(X_test)
    preds = np.vectorize(lambda i: CLASSES[int(i)])(preds_int)
    y_labels = np.vectorize(lambda i: CLASSES[int(i)])(y_test)

    acc = accuracy_score(y_labels, preds)
    cm = confusion_matrix(y_labels, preds, labels=CLASSES)
    print("\n=== ML vs WSM (held-out test) ===")
    print(f"Agreement with WSM: {acc*100:.2f}%")
    print("\nConfusion Matrix (rows=WSM target, cols=ML pred):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
    print("\nClassification Report:")
    print(classification_report(y_labels, preds, labels=CLASSES, zero_division=0))
    return {"preds": preds, "acc": acc}


def compare_to_nutriscore(df: pd.DataFrame, preds: np.ndarray) -> None:
    if "nutri_score_label" not in df.columns:
        print("\nNutri-Score labels not present; skipping comparison.")
        return
    base = df["nutri_score_label"].str.upper()
    cm = pd.crosstab(base, preds, dropna=False).reindex(index=CLASSES, columns=CLASSES)
    print("\nML WSM-like vs official Nutri-Score (rows=Nutri, cols=ML pred):")
    print(cm.fillna(0).astype(int))


def save_outputs(
    df: pd.DataFrame, preds: np.ndarray, model: XGBClassifier, feature_cols: List[str], output: Path
) -> None:
    out_df = df.copy()
    out_df["wsm_ml_pred"] = preds
    out_df.to_csv(output, index=False)
    print(f"\nSaved predictions to: {output}")

    fi = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    fi.to_csv(output.with_name("wsm_classifier_feature_importances.csv"), index=False)
    print(f"Saved feature importances to: {output.with_name('wsm_classifier_feature_importances.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 5 redo: ML classifier mimicking WSM classes, then compare to Nutri-Score"
    )
    parser.add_argument("--data", default="food_database.csv", help="Input CSV with Nutri + Green fields")
    parser.add_argument("--wsm", default="weighted_sum_results.csv", help="WSM results CSV (will compute if absent)")
    parser.add_argument("--output", default="wsm_classifier_results.csv", help="Output CSV for predictions")
    args = parser.parse_args()

    bundle = load_and_attach_wsm(Path(args.data), Path(args.wsm) if args.wsm else None)
    df = bundle.df

    y_labels = df["wsm_class"].str.upper()
    label_to_int = {label: idx for idx, label in enumerate(CLASSES)}
    y = y_labels.map(label_to_int).to_numpy()

    X, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_labels
    )

    model = fit_model(X_train, y_train)
    eval_res = evaluate_vs_wsm(model, X_test, y_test)

    # Full-dataset predictions for comparisons
    full_preds_int = model.predict(X)
    full_preds = np.vectorize(lambda i: CLASSES[int(i)])(full_preds_int)
    compare_to_nutriscore(df, full_preds)
    save_outputs(df, full_preds, model, feature_cols, Path(args.output))


if __name__ == "__main__":
    main()
