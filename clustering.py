import numpy as np
from scipy.special import gamma
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors


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


class DBSCANPatternClustering:
    """
    Кластеризация z-векторов для одного pattern (DBSCAN)
    """

    def __init__(self, eps=0.5, min_samples=2, min_cluster_size=5):
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size

        self.dbscan = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.cluster_ids = None

        self.z_vectors = None
        self.targets = None

    def fit(self, z_vectors, targets):
        """
        z_vectors : shape (N, L)
        targets   : shape (N,)   — x_{t+h}
        """
        self.z_vectors = np.asarray(z_vectors)
        self.targets = np.asarray(targets)

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.cluster_labels = self.dbscan.fit_predict(self.z_vectors)

        # центры кластеров (без шума, label=-1)
        self.cluster_centers = []
        self.cluster_ids = []

        unique_labels = np.unique(self.cluster_labels)
        for label in unique_labels:
            if label == -1:
                continue
            idx = np.where(self.cluster_labels == label)[0]
            if len(idx) >= self.min_cluster_size:
                self.cluster_centers.append(self.z_vectors[idx].mean(axis=0))
                self.cluster_ids.append(label)

        if len(self.cluster_centers) > 0:
            self.cluster_centers = np.array(self.cluster_centers)
        else:
            self.cluster_centers = np.array([])

    def predict_local(self, z_current, max_target_std=1.0):
        """
        Локальный прогноз для одного pattern

        Возвращает:
        - None, если точка non-predictable
        - float, если прогноз возможен
        """
        if len(self.cluster_centers) == 0:
            return None

        z_current = np.asarray(z_current)

        # расстояние до центров
        dists = np.linalg.norm(self.cluster_centers - z_current, axis=1)
        best = np.argmin(dists)
        cluster_id = self.cluster_ids[best]

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
        # print(X.shape, distances.shape, dr.shape)

        # сортировка по d_r
        order = np.argsort(dr)

        # коэффициент объёма d-мерного единичного шара
        Cd = (np.pi ** (dim / 2)) / gamma(dim / 2 + 1)

        # V_r(x)
        Vr = Cd * (dr ** dim)

        # p(x) = r / (V_r(x) * n)
        density = self.r / (Vr * n)

        processed = []
        labels = np.zeros(n, dtype=int)
        completed = {}
        current_cluster = 0

        for q in order:
            # print(dr[q]) ascending order
            processed.append(q)

            connected_clusters = set()
            connected_to_noise = False

            for j in processed:
                if j == q:
                    continue
                dist = np.linalg.norm(X[q] - X[j])
                if dist <= dr[j]:
                # if norm <= dr[q] or norm <= dr[j]:  # adds symmentry, just for try
                    if labels[j] != 0:
                        connected_clusters.add(labels[j])
                    else:
                        connected_to_noise = True

            # isolated vertex -> new cluster
            if len(connected_clusters) == 0:
                current_cluster += 1
                labels[q] = current_cluster
                completed[current_cluster] = False
                continue

            # connected to one cluster
            if len(connected_clusters) == 1:
                l = next(iter(connected_clusters))
                if completed[l]:
                    labels[q] = 0
                else:
                    labels[q] = l
                continue

            # connected to several clusters
            active = [c for c in connected_clusters if not completed[c]]

            if len(active) == 0:
                labels[q] = 0
                continue

            significant = []
            for c in active:
                pts = np.where(labels == c)[0]
                p_vals = density[pts]
                if np.max(p_vals) - np.min(p_vals) >= self.mu:
                    significant.append(c)

            # CHECK LOGIC
            if len(significant) == 0:
                labels[q] = 0
                for c in active:
                    completed[c] = True
                continue

            if len(significant) > 1 or connected_to_noise:
                labels[q] = 0
                for c in significant:
                    completed[c] = True
                for c in connected_clusters:
                    if c not in significant:
                        labels[labels == c] = 0
            else:
                if len(significant) == 0:
                    raise Exception("No significant clusters")
                centre = significant[0]
                labels[q] = centre
                for c in connected_clusters:
                    if c != centre:
                        labels[labels == c] = centre

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


class WishartClusteringFast:
    def __init__(self, r=11, mu=0.2, leaf_size=40, algorithm="ball_tree"):
        self.r = r
        self.mu = mu
        self.leaf_size = leaf_size
        self.algorithm = algorithm
        self.labels_ = None
        self.cluster_centers_ = None
        self.cluster_ids_ = None

    def fit(self, X):
        # import time 
        # start_time = time.time()
        
        X = np.asarray(X)
        n, dim = X.shape
        if n <= self.r:
            raise ValueError("too few points")

        nn = NearestNeighbors(
            n_neighbors=self.r + 1,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=-1,
        ).fit(X)

        distances, _ = nn.kneighbors(X)
        dr = distances[:, -1]
        order = np.argsort(dr)

        # print('NN time: ', time.time() - start_time)

        Cd = (np.pi ** (dim / 2)) / gamma(dim / 2 + 1)
        Vr = Cd * (dr ** dim)
        density = self.r / (Vr * n)

        # 1) Строим входящие ребра: кто "покрывает" вершину q
        in_edges = [[] for _ in range(n)]
        for j in range(n):
            # все q такие что ||x_q - x_j|| <= dr[j]
            idx = nn.radius_neighbors([X[j]], radius=float(dr[j]), return_distance=False)[0]
            # добавляем ребро j -> q (q может включать j, отфильтруем позже)
            for q in idx:
                if q != j:
                    in_edges[q].append(j)

        # print('edges time: ', time.time() - start_time)

        labels = np.zeros(n, dtype=int)
        completed = {}
        current_cluster = 0
        is_processed = np.zeros(n, dtype=bool)

        # для O(1) проверки значимости
        p_min = {}
        p_max = {}

        for q in order:
            # смотрим только уже обработанных "источников" j, которые ведут в q
            connected_clusters = set()
            connected_to_noise = False

            for j in in_edges[q]:
                if not is_processed[j]:
                    continue
                lj = labels[j]
                if lj == 0:
                    connected_to_noise = True
                else:
                    connected_clusters.add(lj)

            is_processed[q] = True

            if len(connected_clusters) == 0:
                current_cluster += 1
                labels[q] = current_cluster
                completed[current_cluster] = False
                p_min[current_cluster] = density[q]
                p_max[current_cluster] = density[q]
                continue

            if len(connected_clusters) == 1:
                l = next(iter(connected_clusters))
                labels[q] = 0 if completed[l] else l
                if labels[q] == l:
                    p_min[l] = min(p_min[l], density[q])
                    p_max[l] = max(p_max[l], density[q])
                continue

            active = [c for c in connected_clusters if not completed.get(c, False)]
            if len(active) == 0:
                labels[q] = 0
                continue

            # O(1) значимость
            significant = [c for c in active if (p_max[c] - p_min[c]) >= self.mu]

            if len(significant) == 0:
                labels[q] = 0
                for c in active:
                    completed[c] = True
                continue

            if len(significant) > 1 or connected_to_noise:
                labels[q] = 0
                for c in significant:
                    completed[c] = True
                # удалить незначимые подключенные
                for c in connected_clusters:
                    if c not in significant:
                        labels[labels == c] = 0
                continue

            # единственный значимый -> merge остальных в него
            centre = significant[0]
            labels[q] = centre
            p_min[centre] = min(p_min[centre], density[q])
            p_max[centre] = max(p_max[centre], density[q])

            for c in connected_clusters:
                if c == centre:
                    continue
                # relabel (это O(size_of_cluster), можно ускорять DSU, но часто ок)
                labels[labels == c] = centre
                # обновим p_min/p_max центра (грубовато, но корректно)
                p_min[centre] = min(p_min[centre], p_min.get(c, p_min[centre]))
                p_max[centre] = max(p_max[centre], p_max.get(c, p_max[centre]))

        # print('labels time: ', time.time() - start_time)

        self.labels_ = labels

        # центры кластеров (без шума)
        self.cluster_centers_ = []
        self.cluster_ids_ = []

        for c in np.unique(labels):
            if c == 0:
                continue
            pts = X[labels == c]
            if len(pts) == 0:
                continue
            self.cluster_centers_.append(pts.mean(axis=0))
            self.cluster_ids_.append(int(c))

        self.cluster_centers_ = np.array(self.cluster_centers_) if self.cluster_centers_ else np.empty((0, X.shape[1]))

        # print('centers time: ', time.time() - start_time)
        return self


class WishartPatternClustering:
    """
    Кластеризация z-векторов для одного pattern
    (модифицированный алгоритм Уишарта)
    """

    def __init__(self, r=11, mu=0.2, min_cluster_size=10, fast=False):
        self.r = r
        self.mu = mu
        self.min_cluster_size = min_cluster_size

        if fast:
            self.clusterer = WishartClusteringFast(r=r, mu=mu)
        else:
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
