import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth, \
    SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
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
print(70 * '_')
print('methods\t\t\tNMI\t\tHomo\t\tCompl')


def evaluation(estimator, name, data):
    estimator.fit(data)
    if name == "Gaussian mixtures":
        labels_pred = estimator.predict(data)
    else:
        labels_pred = estimator.labels_
    print('%-9s\t\t%.3f\t\t%.3f\t\t%.3f'
          % (name.replace(' ', '\n') if ' ' in name else name,
             metrics.normalized_mutual_info_score(labels, labels_pred),
             metrics.homogeneity_score(labels, labels_pred),
             metrics.completeness_score(labels, labels_pred),
             ))


def main():
    # estimate bandwidth for mean shift
    bandwidth = estimate_bandwidth(data, quantile=0.2)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(data, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    km = KMeans(init='k-means++', n_clusters=n_digits, n_init=1)
    af = AffinityPropagation(preference=-50)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    sc = SpectralClustering(n_clusters=n_digits, eigen_solver='arpack', affinity="nearest_neighbors")
    ward = AgglomerativeClustering(n_clusters=n_digits, linkage='ward',connectivity=connectivity)
    ac = AgglomerativeClustering(linkage="average",  n_clusters=n_digits, connectivity=connectivity)
    db = DBSCAN(eps=4)
    gmm = GaussianMixture(n_components=n_digits, covariance_type='full')
    method = [km, af, ms, sc, ward, ac, db, gmm]
    name = ["k-means++", "Affinity Propagation", "Mean-Shift", "Spectral Clustering",
            "Ward hierarchical clustering", "Agglomerative clustering", "DBSCAN", "Gaussian mixtures"]
    for i in range(len(method)):
        print(70 * '_')
        evaluation(method[i], name=name[i], data=data)
    print(70 * '_')


if __name__ == '__main__':
    main()

