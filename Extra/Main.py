from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.metrics.cluster import adjusted_rand_score
import mglearn
import numpy as np
import matplotlib.pyplot as plt

# X, y = make_blobs(random_state=0, n_samples=12)
# linkage_array = ward(X)
# dendrogram(linkage_array)
# plt.show()

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler()
scaler = scaler.fit(X)
X_scaled = scaler.transform(X)
#
# dbscan = DBSCAN()
# cluster = dbscan.fit_predict(X_scaled)
#
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster, cmap=mglearn.cm2, s=60)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

random_state = np.random.RandomState(seed = 0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c = random_clusters, cmap = mglearn.cm3, s = 60)
axes[0].set_title("Random assignment - ARI: {:.2f}".format(adjusted_rand_score(y, random_clusters)))

for ax, algorithms in zip(axes[1:], algorithms):
    clusters = algorithms.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
    ax.set_title("{} - ARI: {:.2f}".format(algorithms.__class__.__name__, adjusted_rand_score(y, clusters)))

plt.show()


