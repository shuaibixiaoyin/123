from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, \
    estimate_bandwidth, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
hasher = HashingVectorizer(n_features=100,
                           stop_words='english', alternate_sign=False,
                           norm=None, binary=False)
vectorizer = make_pipeline(hasher, TfidfTransformer())
X = vectorizer.fit_transform(dataset.data)
X = X.A
print("n_samples: %d, n_features: %d" % X.shape)
print(70 * '_')
print('methods\t\t\tNMI\t\tHomo\t\tCompl')


# #############################################################################
# Do the actual clustering
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
    bandwidth = estimate_bandwidth(X, quantile=0.2)
    af = AffinityPropagation()
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    sc = SpectralClustering(n_clusters=true_k, eigen_solver='arpack', affinity="nearest_neighbors")
    ward = AgglomerativeClustering(n_clusters=true_k, linkage='ward')
    ac = AgglomerativeClustering(linkage="average", n_clusters=true_k)
    db = DBSCAN()
    gmm = GaussianMixture(n_components=true_k, covariance_type='full')
    method = [km, af, ms, sc, ward, ac, db, gmm]
    name = ["k-means++", "Affinity Propagation", "Mean-Shift", "Spectral Clustering",
            "Ward hierarchical clustering", "Agglomerative clustering", "DBSCAN", "Gaussian mixtures"]
    for i in range(len(method)):
        print(70 * '_')
        evaluation(method[i], name=name[i], data=X)
    print(70 * '_')


if __name__ == '__main__':
    main()