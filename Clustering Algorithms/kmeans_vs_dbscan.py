import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN
# create data
X1, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
X2, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

############## 1 ##############
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X1[:, 0], X1[:, 1], s=30)
plt.title("X1 -  Moons Dataset")

plt.subplot(1, 2, 2)
plt.scatter(X2[:, 0], X2[:, 1], s=30)
plt.title("X2 - Blobs Dataset")

plt.show()

############## 2 ##############
print("The X1 dataset (from 'make_moons') is more suitable for the DBSCAN method,\
because it contains non-linear, curved shapes that DBSCAN can cluster well.\
DBSCAN groups points based on density, so it can detect moon-shaped clusters and is most suitable for them.")

print("\nThe X2 dataset (from 'make_blobs') is more suitable for the KMeans method,\
because it contains well-separated, circular clusters.\
KMeans works best with spherical shapes and equal-sized clusters around centroids.")

############## 3 ##############

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels1 = dbscan.fit_predict(X1)

kmeans = KMeans(n_clusters=3, random_state=0)
labels2 = kmeans.fit_predict(X2)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X1[:, 0], X1[:, 1], c=labels1, cmap='rainbow', s=30)
plt.title("DBSCAN on make_moons")

plt.subplot(1, 2, 2)
plt.scatter(X2[:, 0], X2[:, 1], c=labels2, cmap='plasma', s=30)
plt.title("KMeans on make_blobs")

plt.show()
