# Checkpoint: Understanding Evaluation Metrics
# Classification + Regression + Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes

plt.style.use('seaborn-v0_8')

print("=== EVALUATION METRICS CHECKPOINT ===\n")

# PART 3: CLUSTERING (Iris without labels)
print("PART 3 - CLUSTERING")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Using Iris features only (unsupervised)
X = iris.data
y_true = iris.target  # only for evaluation (Adjusted Rand Index)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
y_pred_cluster = kmeans.labels_

# Clustering Metrics
silhouette = silhouette_score(X, y_pred_cluster)
inertia = kmeans.inertia_
ari = adjusted_rand_score(y_true, y_pred_cluster)

print(f"Silhouette Score : {silhouette:.4f}  (closer to 1 = better)")
print(f"Inertia          : {inertia:.2f}   (lower = tighter clusters)")
print(f"Adjusted Rand Index: {ari:.4f}   (how well clusters match true labels)")

print("\nInterpretation (Clustering):")
print("- Silhouette Score of {:.3f} indicates reasonably well-separated clusters.".format(silhouette))
print("- K-Means recovered the structure of the 3 species quite well (high ARI).")
print("- Setosa forms a very distinct cluster, while Versicolor and Virginica overlap slightly.")
print("- To improve: Try different number of clusters, use DBSCAN, or scale features first.\n")

print("=== CHECKPOINT COMPLETE ===")
