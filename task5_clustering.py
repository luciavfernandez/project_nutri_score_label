"""
Task 4.5 (ML variant): Clustering Nutri + Eco criteria
------------------------------------------------------
Unsupervised clustering (K-Means) on Nutri-Score criteria + Green-Score,
then map clusters to ordered labels (A–E) using the mean WSM score.
Finally, compare cluster labels to official Nutri-Score and WSM classes,
and export quick plots (counts + confusion heatmaps).

Usage:
    python task5_clustering.py \
        --data food_database.csv \
        --wsm weighted_sum_results.csv \
        --output clustering_results.csv \
        --clusters 5 \
        --plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from weighted_sum_model import DEFAULT_WEIGHTS, WeightedSumModel


CLASSES = ["A", "B", "C", "D", "E"]
GREEN_GRADE_TO_VALUE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}


def load_and_prepare(
    data_path: Path, wsm_path: Optional[Path]
) -> tuple[pd.DataFrame, pd.Series]:
    """Load data, harmonize columns, attach WSM scores/classes."""
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

    # Ensure green_score_value exists
    if "green_score_value" not in df.columns and "ecoscore_grade" in df.columns:
        df["green_score_value"] = (
            df["ecoscore_grade"].astype(str).str.upper().map(GREEN_GRADE_TO_VALUE)
        )

    # Attach WSM scores/classes
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
    if merged["wsm_score"].isna().any():
        raise ValueError("WSM merge failed; check join key or inputs.")

    return merged, merged["wsm_score"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    feature_cols = [
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


def rank_clusters_by_wsm(
    labels: np.ndarray, wsm_scores: pd.Series
) -> Dict[int, str]:
    """Order clusters by mean WSM score (high → A)."""
    means = (
        pd.DataFrame({"cluster": labels, "wsm_score": wsm_scores})
        .groupby("cluster")["wsm_score"]
        .mean()
        .sort_values(ascending=False)
    )
    mapping = {}
    for idx, (cluster_id, _) in enumerate(means.items()):
        if idx < len(CLASSES):
            mapping[cluster_id] = CLASSES[idx]
        else:
            mapping[cluster_id] = CLASSES[-1]
    return mapping


def compare_against(target: pd.Series, pred: pd.Series, name: str) -> None:
    if target.isna().all():
        print(f"\n{name} labels missing; skipping comparison.")
        return
    base = target.str.upper()
    cm = confusion_matrix(base, pred, labels=CLASSES)
    print(f"\nCluster labels vs {name} (rows={name}, cols=cluster label):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))


def save_confusion_plot(cm: np.ndarray, rows: List[str], cols: List[str], title: str, path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cols, yticklabels=rows, cbar=True)
    plt.title(title)
    plt.xlabel("Cluster label")
    plt.ylabel("Reference")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved plot: {path}")


def save_counts_plot(cluster_labels: pd.Series, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    cluster_labels.value_counts().sort_index().plot(kind="bar", color="#4C72B0")
    plt.title("Cluster label counts")
    plt.xlabel("Cluster label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved plot: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4.5 clustering solution")
    parser.add_argument("--data", default="food_database.csv", help="Input CSV")
    parser.add_argument("--wsm", default="weighted_sum_results.csv", help="WSM results CSV (computed if missing)")
    parser.add_argument("--output", default="clustering_results.csv", help="Output CSV with cluster labels")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument("--plots", action="store_true", help="Save plots (cluster counts + confusion heatmaps)")
    parser.add_argument("--shap", action="store_true", help="Compute SHAP explanations via a surrogate classifier")
    args = parser.parse_args()

    df, wsm_scores = load_and_prepare(Path(args.data), Path(args.wsm) if args.wsm else None)
    X, feature_cols = build_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=args.clusters, n_init=20, random_state=42)
    cluster_ids = km.fit_predict(X_scaled)

    cluster_to_label = rank_clusters_by_wsm(cluster_ids, wsm_scores)
    cluster_labels = pd.Series(cluster_ids).map(cluster_to_label)
    print("\nCluster → label mapping (by descending mean WSM score):")
    for cid, lbl in cluster_to_label.items():
        print(f"  cluster {cid} → {lbl}")

    # Save outputs
    out_df = df.copy()
    out_df["cluster_id"] = cluster_ids
    out_df["cluster_label"] = cluster_labels
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved clustering results to: {args.output}")

    # Summaries
    print("\nCluster size by label:")
    print(cluster_labels.value_counts().sort_index())

    # Comparisons
    if "nutri_score_label" in df.columns:
        compare_against(df["nutri_score_label"], cluster_labels, "Nutri-Score")
    if "wsm_class" in df.columns:
        compare_against(df["wsm_class"], cluster_labels, "WSM class")

    if args.plots:
        save_counts_plot(cluster_labels, Path(args.output).with_name("clustering_counts.png"))
        if "nutri_score_label" in df.columns:
            cm_n = confusion_matrix(df["nutri_score_label"].str.upper(), cluster_labels, labels=CLASSES)
            save_confusion_plot(
                cm_n,
                rows=CLASSES,
                cols=CLASSES,
                title="Cluster labels vs Nutri-Score",
                path=Path(args.output).with_name("clustering_vs_nutriscore.png"),
            )
        if "wsm_class" in df.columns:
            cm_w = confusion_matrix(df["wsm_class"].str.upper(), cluster_labels, labels=CLASSES)
            save_confusion_plot(
                cm_w,
                rows=CLASSES,
                cols=CLASSES,
                title="Cluster labels vs WSM class",
                path=Path(args.output).with_name("clustering_vs_wsm.png"),
            )

    if args.shap:
        try:
            import shap
        except ImportError:
            print("SHAP not installed; skipping SHAP explanations.")
        else:
            print("\nTraining surrogate classifier for SHAP explanations...")
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
            )
            clf.fit(X_scaled, cluster_ids)

            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_scaled)

            # For multi-class, aggregate across classes so all features appear
            if isinstance(shap_values, list):
                # Average absolute SHAP across classes to avoid cancellation
                shap_for_plot = np.mean(np.abs(np.stack(shap_values)), axis=0)
            else:
                shap_for_plot = shap_values

            feat_df = pd.DataFrame(X_scaled, columns=feature_cols)
            base_path = Path(args.output).with_name

            # Bar plot of mean |SHAP| across clusters
            shap.summary_plot(
                shap_for_plot,
                features=feat_df,
                feature_names=feature_cols,
                plot_type="bar",
                show=False,
                max_display=len(feature_cols),
            )
            plt.tight_layout()
            bar_path = base_path("shap_summary_bar.png")
            plt.savefig(bar_path, dpi=300)
            plt.close()
            print(f"Saved SHAP bar plot: {bar_path}")

            # Detailed summary plot
            shap.summary_plot(
                shap_for_plot,
                features=feat_df,
                feature_names=feature_cols,
                show=False,
                max_display=len(feature_cols),
            )
            plt.tight_layout()
            sum_path = base_path("shap_summary.png")
            plt.savefig(sum_path, dpi=300)
            plt.close()
            print(f"Saved SHAP summary plot: {sum_path}")


if __name__ == "__main__":
    main()
