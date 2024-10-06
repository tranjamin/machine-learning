from Utils import GenericModel
import sklearn.mixture, sklearn.cluster
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

class GaussianMixtureModel(GenericModel):
    def __init__(self, num_classes: int, covariance_type: str):
        self.num_classes = num_classes
        self.model = sklearn.mixture.GaussianMixture(n_components=num_classes, covariance_type=covariance_type)
        self.covariance_type = covariance_type
    
    def get_model(self) -> sklearn.mixture.GaussianMixture:
        return self.model
    
    def fit(self) -> None:
        self.model.fit(self.x_train)
    
    def get_weights(self):
        return self.model.weights_

    def get_means(self):
        return self.model.means_

    def get_covariances(self):
        return self.model.covariances_
    
    def tune_with_bic(self, minima, maxima):
        bics = []
        ks = list(range(minima, maxima + 1))
        for i in ks:
            model = sklearn.mixture.GaussianMixture(n_components=i, covariance_type=self.covariance_type)
            model.fit(self.x_train)
            bic = model.bic(self.x_train)
            bics.append(bic)
        best_k = ks[bics.index(min(bics))]
        self.model = sklearn.mixture.GaussianMixture(n_components=best_k, covariance_type=self.covariance_type)
        self.num_classes = best_k
        return best_k, bics, list(ks),

    
    def get_bic(self):
        return self.model.bic(self.x_train)

        N = len(self.x_train)
        summands = np.zeros((N, self.num_classes))
        predictions = self.model.predict(self.x_train)
        pi_values = np.unique(predictions, return_counts=True)[1] / N
        for i in range(self.num_classes):
            pi_h = pi_values[i]
            p = self.x_train.shape[1]

            residuals = np.array(self.x_train) - self.get_means()[i]
            if covariance_type == "tied":
                exponent = -0.5 * (residuals @ np.linalg.inv(self.get_covariances()) @ residuals.T)[0]
                result = np.exp(exponent)
                result *= pi_h / (2*math.pi)**(p/2) / np.linalg.det(self.get_covariances())**0.5
            elif covariance_type == "full":
                exponent = -0.5 * (residuals @ np.linalg.inv(self.get_covariances()[i]) @ residuals.T)[0]
                result = np.exp(exponent)
                result *= pi_h / (2*math.pi)**(p/2) / np.linalg.det(self.get_covariances()[i])**0.5
            summands[:, i] = result
        
        logand = np.sum(summands, axis=1)
        sum = np.log(logand)
        total = np.sum(sum)
        if covariance_type == "tied":
            bic = -2*total + (self.num_classes + self.num_classes*p + p*(p+1)/2 - 1) * math.log(N)
        elif covariance_type == "full":
            bic = -2*total + (self.num_classes + self.num_classes*p + self.num_classes*p*(p+1)/2 - 1) * math.log(N)

        return bic
            
    def copy(self) -> GenericModel:
        return super().copy()
    
    def evaluate_testing_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        return super().evaluate_testing_performance(target, probabilistic)
    
    def evaluate_training_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        return super().evaluate_training_performance(target, probabilistic)
    
    def predict(self, x, probabilistic=False, target=None) -> list[float]:
        return self.model.predict(x)
    
    def augment_hyperparam(self, param: str, value):
        return super().augment_hyperparam(param, value)

class HierarchicalClustering(GenericModel):
    def __init__(self, num_classes: int, linkage="ward", compute_distances=True):
        self.num_classes = num_classes
        self.model = sklearn.cluster.AgglomerativeClustering(num_classes, linkage=linkage, compute_distances=compute_distances)
        self.linkage = linkage
    
    def get_model(self) -> sklearn.cluster.AgglomerativeClustering:
        return self.model
    
    def get_scattering(self):
        total_scatter = 0
        for c in range(self.num_classes):
            points = self.x_train[self.model.labels_ == c]
            cluster_centre = np.mean(points, axis=0)
            cluster_scatter = np.sum((points - cluster_centre)**2, axis=1)
            total_scatter += np.sum(cluster_scatter)
        return total_scatter
    
    def tune_with_gap(self, minima, maxima):
        gap_statistics = []
        ks = list(range(minima, maxima + 1))
        best_k: int = None

        sample_mins = self.x_train.min(axis=0)
        sample_maxs = self.x_train.max(axis=0)
        B = 10
        bootstrap_size = len(self.x_train)

        for i in ks:
            model = HierarchicalClustering(num_classes=i, linkage=self.linkage, compute_distances=False)
            model.add_training_data(self.x_train, None)
            model.fit()
            wc = model.get_scattering()

            wcbs = []
            for b in range(B):
                b_model = HierarchicalClustering(num_classes=i, linkage=self.linkage, compute_distances=False)
                bootstrap_sample = np.random.uniform(sample_mins, sample_maxs, size=(bootstrap_size, self.x_train.shape[1]))
                b_model.add_training_data(bootstrap_sample, None)
                b_model.fit()
                wc_b = b_model.get_scattering()
                wcbs.append(math.log(wc_b))
            
            average_wcb = sum(wcbs)/B
            gap_statistic = average_wcb - math.log(wc)
            gap_statistics.append(gap_statistic)

            sk = math.sqrt(1 + 1/B) * np.std(wcbs)
            if len(gap_statistics) > 1 and (gap_statistics[-1] - gap_statistics[-2] <= sk) and best_k is None:
                best_k = ks[len(gap_statistics) - 2]

        self.model = sklearn.cluster.AgglomerativeClustering(n_clusters=best_k, linkage=self.linkage, compute_distances=True)
        self.num_classes = best_k
        return best_k, gap_statistics, list(ks),
    
    def fit(self) -> None:
        self.model.fit(self.x_train)

    def copy(self) -> GenericModel:
        return super().copy()
    
    def evaluate_testing_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        return super().evaluate_testing_performance(target, probabilistic)
    
    def evaluate_training_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        return super().evaluate_training_performance(target, probabilistic)
    
    def predict(self, x, probabilistic=False, target=None) -> list[float]:
        return super().predict(x, probabilistic, target)
    
    def augment_hyperparam(self, param: str, value):
        return super().augment_hyperparam(param, value)

class KMeans(GenericModel):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.model = sklearn.cluster.KMeans(n_clusters=num_classes)
    
    def get_model(self) -> sklearn.cluster.KMeans:
        return self.model
    
    def fit(self) -> None:
        self.model.fit(self.x_train)
    
    def get_means(self):
        return self.model.cluster_centers_
    
    def get_score(self):
        return self.model.score(self.x_train)

    def tune_with_score(self, minima, maxima):
        scores = []
        ks = list(range(minima, maxima + 1))
        for i in ks:
            model = sklearn.cluster.KMeans(n_clusters=i)
            model.fit(self.x_train)
            score = model.score(self.x_train)
            scores.append(score)
        best_k = ks[scores.index(min(scores))]
        self.model = sklearn.cluster.KMeans(n_clusters=best_k)
        self.num_classes = best_k
        return best_k, scores, list(ks),

    def tune_with_gap(self, minima, maxima):
        gap_statistics = []
        ks = list(range(minima, maxima + 1))
        best_k: int = None

        sample_mins = self.x_train.min(axis=0)
        sample_maxs = self.x_train.max(axis=0)
        B = 10
        bootstrap_size = len(self.x_train)

        for i in ks:
            model = sklearn.cluster.KMeans(n_clusters=i)
            model.fit(self.x_train)
            wc = model.inertia_

            wcbs = []
            for b in range(B):
                b_model = sklearn.cluster.KMeans(n_clusters=i)
                bootstrap_sample = np.random.uniform(sample_mins, sample_maxs, size=(bootstrap_size, self.x_train.shape[1]))
                b_model.fit(bootstrap_sample)
                wc_b = b_model.inertia_
                wcbs.append(math.log(wc_b))
            
            average_wcb = sum(wcbs)/B
            gap_statistic = average_wcb - math.log(wc)
            gap_statistics.append(gap_statistic)

            sk = math.sqrt(1 + 1/B) * np.std(wcbs)
            if len(gap_statistics) > 1 and (gap_statistics[-1] - gap_statistics[-2] <= sk) and best_k is None:
                best_k = ks[len(gap_statistics) - 2]

        self.model = sklearn.cluster.KMeans(n_clusters=best_k)
        self.num_classes = best_k
        return best_k, gap_statistics, list(ks),
    
    
    def copy(self) -> GenericModel:
        return super().copy()
    
    def evaluate_testing_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        return super().evaluate_testing_performance(target, probabilistic)
    
    def evaluate_training_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        return super().evaluate_training_performance(target, probabilistic)
    
    def predict(self, x, probabilistic=False, target=None) -> list[float]:
        return self.model.predict(x)
    
    def augment_hyperparam(self, param: str, value):
        return super().augment_hyperparam(param, value)