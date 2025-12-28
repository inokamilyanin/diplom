import numpy as np
from clustering import KMeansPatternClustering
from clustering import DBSCANPatternClustering
from clustering import WishartPatternClustering
from utils import generate_z_vectors


class MultiPatternPredictor:
    """
    Partial multi-step prediction using multiple patterns
    """

    def __init__(
        self,
        patterns,
        horizon=1,
        n_clusters=10,
        min_cluster_size=5,
        max_target_std=1.0,
        eps=0.5,
        min_samples=2,
        r=11,
        mu=0.2,
        cluster_method='kmeans'
    ):
        self.patterns = patterns
        self.horizon = horizon
        self.max_target_std = max_target_std

        self.clusterings = []
        for _ in patterns:
            if cluster_method == 'kmeans':
                self.clusterings.append(
                    KMeansPatternClustering(
                        n_clusters=n_clusters,
                        min_cluster_size=min_cluster_size
                    )
                )
            elif cluster_method == 'dbscan':
                self.clusterings.append(
                    DBSCANPatternClustering(
                        eps=eps,
                        min_samples=min_samples,
                        min_cluster_size=min_cluster_size
                    )
                )
            elif cluster_method == 'wishart':
                self.clusterings.append(
                    WishartPatternClustering(
                        r=r,
                        mu=mu,
                        min_cluster_size=min_cluster_size
                    )
                )
            else:
                raise ValueError(f"Неизвестный метод кластеризации: {cluster_method}")
                

    def fit(self, series):
        """
        Обучение всех pattern-кластеризаций
        """
        series = np.asarray(series)

        for i, pattern in enumerate(self.patterns):
            max_lag = np.max(pattern)
            usable_length = len(series) - self.horizon

            z_vectors = []
            targets = []

            for t in range(max_lag, usable_length):
                indices = t - pattern
                z_vectors.append(series[indices])
                targets.append(series[t + self.horizon])

            z_vectors = np.asarray(z_vectors)
            targets = np.asarray(targets)

            self.clusterings[i].fit(z_vectors, targets)

    def predict(self, series):
        """
        Прогноз для текущего момента времени
        """
        series = np.asarray(series)

        predictions = []

        for pattern, clustering in zip(self.patterns, self.clusterings):
            max_lag = np.max(pattern)

            if len(series) <= max_lag:
                continue

            z_current = series[len(series) - 1 - pattern]
            pred = clustering.predict_local(
                z_current,
                max_target_std=self.max_target_std
            )

            if pred is not None:
                predictions.append(pred)

        if len(predictions) == 0:
            return None  # non-predictable point

        return float(np.mean(predictions))
