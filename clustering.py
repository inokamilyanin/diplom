import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale


class KMeansPatternClustering:
    """
    Кластеризация z-векторов для одного pattern
    """

    def __init__(self, n_clusters=10, min_cluster_size=5):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size

        self.kmeans = None
        self.cluster_labels = None
        self.cluster_centers = None

        self.z_vectors = None
        self.targets = None

    def fit(self, z_vectors, targets):
        """
        z_vectors : shape (N, L)
        targets   : shape (N,)   — x_{t+h}
        """
        self.z_vectors = np.asarray(z_vectors)
        self.targets = np.asarray(targets)

        if len(z_vectors) < self.n_clusters:
            raise ValueError("Слишком мало данных для кластеризации")

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=42
        )
        self.cluster_labels = self.kmeans.fit_predict(self.z_vectors)
        self.cluster_centers = self.kmeans.cluster_centers_

    def predict_local(self, z_current, max_target_std=1.0):
        """
        Локальный прогноз для одного pattern

        Возвращает:
        - None, если точка non-predictable
        - float, если прогноз возможен
        """
        z_current = np.asarray(z_current).reshape(1, -1)
        cluster_id = self.kmeans.predict(z_current)[0]

        idx = np.where(self.cluster_labels == cluster_id)[0]

        # слишком маленький кластер → non-predictable
        if len(idx) < self.min_cluster_size:
            return None

        local_targets = self.targets[idx]

        # большой разброс будущего → non-predictable
        if np.std(local_targets) > max_target_std:
            return None

        return np.mean(local_targets)


class WishartClustering:
    """
    Модифицированный алгоритм кластеризации Уишарта
    (Wishart + Lapko–Chentsov)
    """

    def __init__(self, r=11, mu=0.2):
        self.r = r
        self.mu = mu

        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        X = np.asarray(X)
        n, dim = X.shape

        if n <= self.r:
            raise ValueError("Слишком мало точек для Wishart-кластеризации")

        # r-NN
        nn = NearestNeighbors(n_neighbors=self.r + 1).fit(X)
        distances, indices = nn.kneighbors(X)

        # d_r(x)
        dr = distances[:, -1]
        print(distances.shape)

        # сортировка по d_r
        order = np.argsort(dr)

        # плотность p(x) ~ r / (V_r * n), V_r ~ d_r^d
        density = self.r / (np.power(dr, dim) * n + 1e-12)

        labels = np.zeros(n, dtype=int)
        completed = {}
        current_cluster = 0

        def is_significant(cluster_id, new_point):
            pts = np.where(labels == cluster_id)[0]
            if len(pts) == 0:
                return False
            p_vals = density[pts]
            return np.max(np.abs(p_vals - density[new_point])) >= self.mu

        for q in order:
            neighbors = indices[q][1:]  # без самой точки
            connected = [
                labels[j] for j in neighbors
                if dr[j] >= np.linalg.norm(X[q] - X[j])
            ]
            connected = [c for c in connected if c != 0]

            if len(connected) == 0:
                current_cluster += 1
                labels[q] = current_cluster
                completed[current_cluster] = False
                continue

            unique_clusters = list(set(connected))

            if len(unique_clusters) == 1:
                c = unique_clusters[0]
                if completed[c]:
                    labels[q] = 0
                else:
                    labels[q] = c
                continue

            # несколько кластеров
            significant = [
                c for c in unique_clusters if is_significant(c, q)
            ]

            if len(significant) != 1:
                labels[q] = 0
                for c in significant:
                    completed[c] = True
                continue

            # объединение
            main = significant[0]
            for c in unique_clusters:
                if c != main:
                    labels[labels == c] = main
                    completed[c] = True

            labels[q] = main

        self.labels_ = labels

        # центры кластеров (без шума)
        self.cluster_centers_ = []
        self.cluster_ids_ = []

        for c in np.unique(labels):
            if c == 0:
                continue
            pts = X[labels == c]
            self.cluster_centers_.append(pts.mean(axis=0))
            self.cluster_ids_.append(c)

        self.cluster_centers_ = np.array(self.cluster_centers_)


class WishartPatternClustering:
    """
    Кластеризация z-векторов для одного pattern
    (модифицированный алгоритм Уишарта)
    """

    def __init__(self, r=11, mu=0.2, min_cluster_size=10):
        self.r = r
        self.mu = mu
        self.min_cluster_size = min_cluster_size

        self.clusterer = WishartClustering(r=r, mu=mu)

        self.cluster_labels = None
        self.cluster_centers = None

        self.z_vectors = None
        self.targets = None

    def fit(self, z_vectors, targets):
        """
        z_vectors : shape (N, L)
        targets   : shape (N,) — x_{t+h}
        """
        self.z_vectors = np.asarray(z_vectors)
        self.targets = np.asarray(targets)

        self.clusterer.fit(self.z_vectors)

        self.cluster_labels = self.clusterer.labels_
        self.cluster_centers = self.clusterer.cluster_centers_

    def predict_local(self, z_current, max_target_std=1.0):
        """
        Локальный прогноз для одного pattern

        Возвращает:
        - None, если точка non-predictable
        - float, если прогноз возможен
        """
        z_current = np.asarray(z_current)

        # расстояние до центров
        dists = np.linalg.norm(self.cluster_centers - z_current, axis=1)
        best = np.argmin(dists)
        cluster_id = self.clusterer.cluster_ids_[best]

        idx = np.where(self.cluster_labels == cluster_id)[0]

        # маленький кластер → non-predictable
        if len(idx) < self.min_cluster_size:
            return None

        local_targets = self.targets[idx]

        # большой разброс → non-predictable
        if np.std(local_targets) > max_target_std:
            return None

        return np.mean(local_targets)
