
# 0. Adjust parameters
NUM_CLUSTERS = 2       # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER = 5           # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris)

# 1. Import any other required libraries (e.g., numpy, scikit-learn)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# 2. Load the Iris dataset using scikit-learn's load_iris() function
iris = load_iris(as_frame=True)
X = iris.data.values

# 3. Implement K-Means Clustering
    # 3.1. Import KMeans from scikit-learn
from sklearn.cluster import KMeans
    # 3.2. Create an instance of KMeans with the specified number of clusters and max_iter
kmeans = KMeans(n_clusters=NUM_CLUSTERS, max_iter=MAX_ITER, random_state=42)
    # 3.3. Fit the KMeans model to the data X
kmeans.fit(X)
    # 3.4. Obtain the cluster labels
labels = kmeans.labels_

# 4. Visualize the Results
    # 4.1. Extract the features for visualization
x_feature = X[:, FEATURE_X_INDEX]
y_feature = X[:, FEATURE_Y_INDEX]

    # 4.2. Create a scatter plot of x_feature vs y_feature, colored by the cluster labels
plt.figure(figsize=(10,6))

    # 4.3. Use different colors to represent different clusters
scatter = plt.scatter(x_feature, y_feature, c=labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, FEATURE_X_INDEX], centers[:, FEATURE_Y_INDEX], c='red', marker='X', s=150, edgecolors='black', linewidth=1, label='Centroids')

plt.xlabel(iris.feature_names[FEATURE_X_INDEX], fontsize=12)
plt.ylabel(iris.feature_names[FEATURE_Y_INDEX], fontsize=12)
plt.title(f'K-Means Clustering (k={NUM_CLUSTERS}, max_iter={MAX_ITER})', fontsize=14)
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
