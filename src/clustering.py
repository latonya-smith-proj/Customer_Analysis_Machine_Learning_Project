"""K-means cluster analysis of customers who accepted a promotion.

Ported from the R `cluster analysis.R` script (which used
`factoextra::fviz_nbclust` for the elbow method and a custom
`profile_plot` helper for cluster profiling).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

CLUSTER_FEATURES = [
    "Income", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds", "duration", "age", "children",
]


def plot_elbow(scaled_data, max_k: int = 10, save=True):
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=100, random_state=1234)
        km.fit(scaled_data)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Within-cluster sum of squares")
    plt.title("Elbow Method for Optimal k")
    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        plt.savefig(OUTPUT_DIR / "elbow_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    return inertias


def cluster_accepted_customers(df: pd.DataFrame, n_clusters: int = 3, target_col: str = "Accepted_Cmp"):
    accepted = df[df[target_col] == "Yes"].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(accepted[CLUSTER_FEATURES])

    km = KMeans(n_clusters=n_clusters, n_init=100, random_state=1234)
    accepted["cluster"] = km.fit_predict(scaled)

    return accepted, km, scaled


def profile_plot(clustered_df: pd.DataFrame, features=CLUSTER_FEATURES, save=True, filename="cluster_profile.png"):
    """Bar chart of each cluster's mean (standardized) value per feature —
    the Python equivalent of the R `profile_plot` helper."""
    scaler = StandardScaler()
    standardized = pd.DataFrame(
        scaler.fit_transform(clustered_df[features]), columns=features, index=clustered_df.index
    )
    standardized["cluster"] = clustered_df["cluster"].values

    profile = standardized.groupby("cluster").mean().T

    profile.plot(kind="bar", figsize=(12, 6), colormap="Set3")
    plt.title("Cluster Profiles (standardized feature means)")
    plt.ylabel("Standardized mean")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Cluster")
    plt.tight_layout()
    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    return profile


if __name__ == "__main__":
    from data_processing import load_and_clean_data

    df = load_and_clean_data()
    clustered, km, scaled = cluster_accepted_customers(df)

    print(f"Clustered {len(clustered)} customers who accepted a promotion into {km.n_clusters} groups.")
    print(clustered["cluster"].value_counts().sort_index())

    plot_elbow(scaled)
    profile = profile_plot(clustered)
    print("\nCluster profile (standardized means):\n", profile)
