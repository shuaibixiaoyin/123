from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, \
    AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
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
#categories = None
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

hasher = HashingVectorizer(n_features=10000,
                           stop_words='english', alternate_sign=False,
                           norm=None, binary=False)
vectorizer = make_pipeline(hasher, TfidfTransformer())
X = vectorizer.fit_transform(dataset.data)

# #############################################################################
# Do the actual clustering

print(50 * '_')
print('methods\t\tNMI\t\thomo\tcompl')


def evaluation(estimator, name, data):
    estimator.fit(data)
    if name == "Gaussian mixtures":
        labels_pred = estimator.predict(data)
    else:
        labels_pred = estimator.labels_
    print('%-9s\t%.3f\t%.3f\t%.3f'
          % (name.replace(' ', '\n') if ' ' in name else name,
             metrics.normalized_mutual_info_score(labels, labels_pred),
             metrics.homogeneity_score(labels, labels_pred),
             metrics.completeness_score(labels, labels_pred),
             ))



def main():
    km = KMeans(init='k-means++', n_clusters=true_k, n_init=1)
    af = AffinityPropagation()
    ms = MeanShift(bin_seeding=True)
    sc = SpectralClustering(affinity='nearest_neighbors', n_clusters=true_k)
    ward = AgglomerativeClustering(n_clusters=true_k, linkage='ward')
    ac = AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=true_k)
    db = DBSCAN(eps=0.3)
    gmm = GaussianMixture(n_components=true_k, covariance_type='full')
    method = [km, af, ms, sc, ward, ac, db, gmm]
    name = ["k-means++", "Affinity Propagation", "Mean-Shift", "Spectral Clustering",
            "Ward hierarchical clustering", "Agglomerative clustering", "DBSCAN", "Gaussian mixtures"]
    for i in range(len(method)):
        print(50 * '_')
        evaluation(method[i], name=name[i], data=X.toarray())
    print(50 * '_')


if __name__ == '__main__':
    main()