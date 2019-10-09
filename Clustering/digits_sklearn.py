import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, \
    AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

import warnings
warnings.filterwarnings("ignore")

digits = load_digits()
data = scale(digits.data)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

print("n_digits: %d, \t n_samples: %d, \t n_features: %d"
      % (n_digits, n_samples, n_features))
print(50 * '_')
print('methods\t\t\tNMI\t\tHomo\tCompl')


def evaluation(estimator, name, data):
    estimator.fit(data)
    if name == "Gaussian mixtures":
        labels_pred = estimator.predict(data)
    else:
        labels_pred = estimator.labels_
    print('%-9s\t\t%.3f\t%.3f\t%.3f'
          % (name.replace(' ', '\n') if ' ' in name else name,
             metrics.normalized_mutual_info_score(labels, labels_pred),
             metrics.homogeneity_score(labels, labels_pred),
             metrics.completeness_score(labels, labels_pred),
             ))


def main():
    km = KMeans(init='k-means++', n_clusters=n_digits, n_init=1)
    af = AffinityPropagation()
    ms = MeanShift()
    sc = SpectralClustering(affinity='nearest_neighbors', n_clusters=n_digits)
    ward = AgglomerativeClustering(n_clusters=n_digits, linkage='ward')
    ac = AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=n_digits)
    db = DBSCAN(eps=0.3)
    gmm = GaussianMixture(n_components=n_digits, covariance_type='full')
    method = [km, af, ms, sc, ward, ac, db, gmm]
    name = ["k-means++", "Affinity Propagation", "Mean-Shift", "Spectral Clustering",
            "Ward hierarchical clustering", "Agglomerative clustering", "DBSCAN", "Gaussian mixtures"]
    for i in range(len(method)):
        print(50 * '_')
        evaluation(method[i], name=name[i], data=data)
    print(50 * '_')


if __name__ == '__main__':
    main()

