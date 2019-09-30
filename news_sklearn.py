from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline

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


hasher = HashingVectorizer(n_features=10000,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
vectorizer = make_pipeline(hasher, TfidfTransformer())
X = vectorizer.fit_transform(dataset.data)

# #############################################################################
# Do the actual clustering
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
print(82 * '_')
print('methods\t\tNMI\t\thomo\tcompl')

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%.3f\t%.3f\t%.3f'
          % (name,
             metrics.normalized_mutual_info_score(labels, estimator.labels_),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             ))
print()

bench_k_means(km, name="k-means++", data=X)
bench_k_means(MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000), name="MiniBatchKMeans", data=X)