"""
Customer segmentation using K-Means clustering.
"""

from typing import List, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline: StandardScaler for numeric columns,
    OneHotEncoder for categorical columns.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def find_optimal_k(X_processed, k_range=range(2, 9)) -> Tuple[List[float], List[float]]:
    """
    Compute inertia and silhouette score for a range of K values,
    to support the Elbow Method / Silhouette analysis.

    Returns
    -------
    (inertias, silhouette_scores) : tuple of lists, aligned with k_range
    """
    inertias, silhouette_scores = [], []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_processed)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_processed, labels))

        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

    return inertias, silhouette_scores


def fit_kmeans(X_processed, n_clusters: int = 3) -> KMeans:
    """
    Fit the final K-Means model with the chosen number of clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    return kmeans


def segment_customers(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Run the full segmentation step: preprocess, fit K-Means, and
    attach cluster labels to the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (the `Churn` target is excluded from the
        clustering features).
    n_clusters : int
        Number of clusters to fit (default 3, chosen via Elbow/Silhouette
        analysis - see `find_optimal_k`).

    Returns
    -------
    pd.DataFrame
        Original dataset with an added `Cluster` column.
    """
    df = df.copy()
    X_seg = df.drop(columns=["Churn"])

    preprocessor = build_preprocessor(X_seg)
    X_seg_processed = preprocessor.fit_transform(X_seg)

    kmeans = fit_kmeans(X_seg_processed, n_clusters=n_clusters)
    df["Cluster"] = kmeans.labels_

    print("Cluster counts:")
    print(df["Cluster"].value_counts().sort_index())

    return df
