
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --------------------------------
# CONFIGURATION
# --------------------------------

INPUT = Path("outputs/rfm_features.csv")
OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
K_VALUES = range(2, 9)
SELECTED_K = 4

# --------------------------------
# LOAD RFM DATA
# --------------------------------

rfm = pd.read_csv(INPUT)

features = ["recency", "frequency", "monetary"]
X = rfm[features].copy()

# --------------------------------
# SCALE FEATURES
# --------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------
# TEST K = 2 TO 8
# --------------------------------

print("\n========== K-MEANS EVALUATION ==========")

evaluation_results = []

for k in K_VALUES:
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    evaluation_results.append({
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette
    })

    print(
        f"K={k} | "
        f"Inertia={inertia:.1f} | "
        f"Silhouette={silhouette:.3f}"
    )

evaluation = pd.DataFrame(evaluation_results)
evaluation.to_csv(
    OUTPUT / "kmeans_evaluation.csv",
    index=False
)

# --------------------------------
# FINAL MODEL: K = 4
# --------------------------------

print("\n========== FINAL CLUSTERING ==========")

kmeans = KMeans(
    n_clusters=SELECTED_K,
    random_state=RANDOM_STATE,
    n_init=10
)

rfm["cluster"] = kmeans.fit_predict(X_scaled)

# --------------------------------
# CLUSTER PROFILING
# --------------------------------

profile = (
    rfm.groupby("cluster")
    .agg(
        customer_count=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean")
    )
    .round(2)
)

print("\nCluster profiles:")
print(profile.to_string())

# --------------------------------
# BUSINESS-FRIENDLY LABELS
# --------------------------------

cluster_ids = list(profile.index)

premium_cluster = profile["avg_monetary"].idxmax()

remaining = [
    cluster for cluster in cluster_ids
    if cluster != premium_cluster
]

loyal_cluster = profile.loc[remaining, "avg_frequency"].idxmax()

remaining = [
    cluster for cluster in remaining
    if cluster != loyal_cluster
]

at_risk_cluster = profile.loc[remaining, "avg_recency"].idxmax()

remaining = [
    cluster for cluster in remaining
    if cluster != at_risk_cluster
]

regular_cluster = remaining[0]

segment_mapping = {
    premium_cluster: "Premium",
    loyal_cluster: "Loyal",
    at_risk_cluster: "At Risk",
    regular_cluster: "Regular"
}

rfm["segment"] = rfm["cluster"].map(segment_mapping)

print("\nSegment mapping:")
for cluster, segment in segment_mapping.items():
    print(f"Cluster {cluster}: {segment}")

# --------------------------------
# PCA: TWO COMPONENTS
# --------------------------------

print("\n========== PCA ANALYSIS ==========")

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

rfm["pca_1"] = components[:, 0]
rfm["pca_2"] = components[:, 1]

explained_variance = pca.explained_variance_ratio_

print(
    "PC1 explained variance:",
    round(explained_variance[0], 3)
)

print(
    "PC2 explained variance:",
    round(explained_variance[1], 3)
)

print(
    "Total explained variance:",
    round(explained_variance.sum(), 3)
)

# --------------------------------
# SAVE FINAL CUSTOMER SEGMENTS
# --------------------------------

rfm.to_csv(
    OUTPUT / "customer_segments.csv",
    index=False
)

print("\nSegment counts:")
print(rfm["segment"].value_counts().to_string())

print("\nSaved: outputs/customer_segments.csv")
print("\nK-MEANS AND PCA COMPLETED")
