"""
Optuna hyperparameter search for XGBoost (Task 5 extension)
-----------------------------------------------------------
Tunes an XGBClassifier to predict Nutri-Score labels and compares the tuned
model to the Weighted Sum Model (WSM).

Usage:
    python task5_xgb_optuna.py \
        --data food_database.csv \
        --wsm weighted_sum_results.csv \
        --trials 25 \
        --output xgb_optuna_results.csv

Outputs:
    - xgb_optuna_results.csv : dataset with tuned XGB predictions + WSM class
    - xgb_optuna_feature_importances.csv : gain importances of tuned model
    - study_best_params.json : best hyperparameters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from weighted_sum_model import DEFAULT_WEIGHTS, WeightedSumModel

TARGET_COL = "nutri_score_label"
CLASSES = ["A", "B", "C", "D", "E"]


def load_with_wsm(data_path: Path, wsm_path: Path) -> pd.DataFrame:
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

    # WSM scores/classes
    if wsm_path and wsm_path.exists():
        wsm_df = pd.read_csv(wsm_path)
    else:
        model = WeightedSumModel(weights=DEFAULT_WEIGHTS)
        wsm_df, _ = model.score_dataframe(df)

    key = "code" if "code" in df.columns and "code" in wsm_df.columns else "product_name"
    merged = df.merge(
        wsm_df[[key, "wsm_score", "wsm_class"]],
        on=key,
        how="left",
        validate="m:1",
    )
    return merged


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, List[str]]:
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

    y_labels = df[TARGET_COL].str.upper()
    label_to_int = {label: idx for idx, label in enumerate(CLASSES)}
    y = y_labels.map(label_to_int).to_numpy()
    return X, y, available


def build_model(trial: optuna.Trial) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=trial.suggest_int("n_estimators", 150, 450),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 8.0),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 2.0),
    )


def objective(trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray) -> float:
    model = build_model(trial)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accuracies = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        accuracies.append(acc)
    return float(np.mean(accuracies))


def evaluate_final(model: XGBClassifier, X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    preds = model.predict(X_test)
    preds_lbl = np.vectorize(lambda i: CLASSES[int(i)])(preds)
    y_lbl = np.vectorize(lambda i: CLASSES[int(i)])(y_test)
    acc = accuracy_score(y_lbl, preds_lbl)
    print(f"\nHeld-out accuracy (tuned XGB): {acc*100:.2f}%")


def compare_to_wsm(df: pd.DataFrame, preds_lbl: np.ndarray) -> None:
    if "wsm_class" not in df.columns:
        print("No WSM classes available for comparison.")
        return
    agreement = (df["wsm_class"] == preds_lbl).mean()
    print(f"XGB (tuned) vs WSM agreement: {agreement*100:.2f}%")
    cm = pd.crosstab(df["wsm_class"], preds_lbl, dropna=False).reindex(
        index=CLASSES, columns=CLASSES
    ).fillna(0).astype(int)
    print("\nXGB vs WSM crosstab (rows=WSM, cols=XGB):")
    print(cm)


def save_outputs(
    df: pd.DataFrame,
    preds_lbl: np.ndarray,
    model: XGBClassifier,
    feature_cols: List[str],
    output: Path,
    best_params: dict,
) -> None:
    out_df = df.copy()
    out_df["xgb_tuned_pred"] = preds_lbl
    out_df.to_csv(output, index=False)
    print(f"Saved predictions to: {output}")

    fi = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    fi_path = output.with_name("xgb_optuna_feature_importances.csv")
    fi.to_csv(fi_path, index=False)
    print(f"Saved feature importances to: {fi_path}")

    bp_path = output.with_name("study_best_params.json")
    bp_path.write_text(json.dumps(best_params, indent=2))
    print(f"Saved best params to: {bp_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for XGB Nutri-Score model")
    parser.add_argument("--data", default="food_database.csv", help="Input dataset")
    parser.add_argument("--wsm", default="weighted_sum_results.csv", help="WSM results for comparison")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna trials")
    parser.add_argument("--output", default="xgb_optuna_results.csv", help="Output CSV for tuned predictions")
    args = parser.parse_args()

    df = load_with_wsm(Path(args.data), Path(args.wsm))
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'.")

    X, y, feature_cols = prepare_features(df)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=args.trials, show_progress_bar=True)

    print("\nBest trial accuracy:", study.best_value)
    print("Best params:", study.best_params)

    # Train final model on full data with best params
    best_model = build_model(study.best_trial)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    best_model.fit(X_train, y_train)
    evaluate_final(best_model, X_test, y_test)

    # Full predictions and WSM comparison
    full_preds = best_model.predict(X)
    full_preds_lbl = np.vectorize(lambda i: CLASSES[int(i)])(full_preds)
    compare_to_wsm(df, full_preds_lbl)
    save_outputs(df, full_preds_lbl, best_model, feature_cols, Path(args.output), study.best_params)


if __name__ == "__main__":
    main()
