# 第一次数据挖掘作业

## 一.作业要求

数据集：

- sklearn.datasets.load_digits

- sklearn.datasets.fetch_20newsgroups

测试sklearn中以下聚类算法在以上两个数据集上的聚类效果：

详见*聚类算法.png*

所使用的评价方法有：

- Normalized Mutual Information (NMI)
- Homogeneity: each cluster contains only members of a single class
- Completeness: all members of a given class are assigned to the same cluster

例子：

-  [A demo of K-Means clustering on the handwritten digits]( https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py)
- [Clustering text documents using k-means](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py)


## 二.实验过程

digits_sklearn.py为聚类算法在数据集digits上的效果

news_sklearn.py为聚类算法在数据集20newsgroups上的效果

分别改写例子，将聚类以及评价的结果封装在evaluation函数中，

```python
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
```

参数estimator是所需要选择的聚类算法，name 是算法名字，data是将数据集提取特征之后的numpy.ndarray类型的数据，结果以一种类表格形式输出。

聚类算法：

1. [`KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) 算法通过把样本分离成 n 个具有相同方差的类的方式来聚集数据，最小化称为惯量或 簇内平方和(within-cluster sum-of-squares)的标准。该算法需要指定簇的数量。它可以很好地扩展到大量样本，并已经被广泛应用于许多不同领域的应用领域。调用形式如下：

   `km = KMeans(init='k-means++', n_clusters=n_digits, n_init=1)`

2.  [`AffinityPropagation`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation) AP聚类是通过在样本对之间发送消息直到收敛的方式来创建聚类。然后使用少量模范样本作为聚类中心来描述数据集，而这些模范样本可以被认为是最能代表数据集中其它数据的样本。在样本对之间发送的消息表示一个样本作为另一个样本的模范样本的 适合程度，适合程度值在根据通信的反馈不断更新。更新迭代直到收敛，完成聚类中心的选取，因此也给出了最终聚类。 调用形式如下：

   `af = AffinityPropagation(preference=-50)`

3.  [`MeanShift`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift) 算法旨在于发现一个样本密度平滑的 *blobs* 。均值漂移(Mean Shift)算法是基于质心的算法，通过更新质心的候选位置，这些侯选位置通常是所选定区域内点的均值。然后，这些候选位置在后处理阶段被过滤以消除近似重复，从而形成最终质心集合。调用形式如下：

   `ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)`

4.  [`SpectralClustering(谱聚类)`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering) 是在样本之间进行关联矩阵的低维度嵌入，然后在低维空间中使用 KMeans 算法。谱聚类 需要指定簇的数量。这个算法适用于簇数量少时，在簇数量多时是不建议使用。 调用形式如下：

   `sc = SpectralClustering(n_clusters=n_digits, eigen_solver='arpack', affinity="nearest_neighbors")`

5.  [`AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering) 使用自下而上的方法进行层次聚类:开始是每一个对象是一个聚类， 并且聚类别相继合并在一起。 连接标准决定用于合并策略的度量: 

   - **Ward** 最小化所有聚类内的平方差总和。这是一种方差最小化(variance-minimizing )的优化方向， 这是与k-means 的目标函数相似的优化方法，但是用 凝聚分层（agglomerative hierarchical）的方法处理。调用形式：

     `ward = AgglomerativeClustering(n_clusters=n_digits, linkage='ward',connectivity=connectivity)`

   - **Average linkage** 最小化成对聚类间平均样本距离值。调用形式：

     `ac = AgglomerativeClustering(linkage="average",  n_clusters=n_digits, connectivity=connectivity)`

6.  [`DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN) 算法将簇视为被低密度区域分隔的高密度区域。由于这个相当普遍的观点， DBSCAN发现的簇可以是任何形状的，与假设簇是凸的 K-means 相反。 DBSCAN 的核心概念是 *core samples*, 是指位于高密度区域的样本。因此一个簇是一组核心样本，每个核心样本彼此靠近（通过某个距离度量测量） 和一组接近核心样本的非核心样本。算法中的两个参数`min_samples` 和 `eps`正式定义了*稠密性*。较高的 `min_samples` 或者较低的 `eps` 都表示形成簇所需的较高密度。 调用形式：

   `db = DBSCAN(eps=4)`

7.  [`GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) 对象实现了用来拟合高斯混合模型的 [期望最大化](https://sklearn.apachecn.org/docs/0.21.3/20.html#expectation-maximization) (EM) 算法。它还可以为多变量模型绘制置信椭圆体，同时计算 BIC（Bayesian Information Criterion，贝叶斯信息准则）来评估数据中聚类的数量。调用形式：

   `gmm = GaussianMixture(n_components=n_digits, covariance_type='full')`

## 三.实验结果

digits_sklearn.py：详见*digits.png*

news_sklearn.py：详见*news0.png*；*news1.png*

在news_sklearn.py中的feature在进行mean-shift等算法聚类时，默认为10000个，需要花费许多的时间和其他开销去处理，因此通过减少feature的数量到100个，但是所有聚类算法的聚类效果都因为特征值的减少明显下降了许多。