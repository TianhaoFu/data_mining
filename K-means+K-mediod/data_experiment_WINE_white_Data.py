import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

def minmax_Norm(dataSet):
    minVals = dataSet.min(0) # 取每一列的最小值
    maxVals = dataSet.max(0) # 取每一列的最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet

def ZScore_Norm(dataSet):
    mu = dataSet.average()
    sigma = dataSet.std()
    x = (x - mu) / sigma
    return datsSet

def sigmoid_Norm(dataSet,useStatus):
    if useStatus:
        return 1.0 / (1 + np.exp(-float(dataSet)))
    else:
        return float(dataSet)

if __name__ == '__main__':
    # 导入数据
    df1 = pd.read_csv('C:\\Users\\Tianh\\Desktop\\DMLab\\data\\wineequality\\winequality-white.csv', delimiter=';')
    print(df1.head())

    # 数据预处理
    true_label = df1['quality']
    true_label = np.array(true_label)
    fea = df1.drop('quality', axis=1)
    fea = np.array(fea)
    res = np.unique(fea)
    print(res)

    norm_fea = minmax_Norm(fea)
    norm_fea

    # k-means 聚类
    cluster_num = 6
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(fea)
    predict_label = kmeans.labels_
    print(kmeans.cluster_centers)

    # 聚类评价
    # Mutual Information based scores 互信息
    print(metrics.adjusted_mutual_info_score(true_label, predict_label))

    # Homogeneity 同质性 每个群集只包含单个类的成员
    print(metrics.homogeneity_score(true_label, predict_label))

    # completeness 完整性 类的所有成员都分配给同一个群集
    print(metrics.completeness_score(true_label, predict_label))

    # 两者的调和平均V-measure：
    print(metrics.v_measure_score(true_label, predict_label))

    #  Fowlkes-Mallows scores
    # The Fowlkes-Mallows score FMI is defined as the geometric mean
    # of the pairwise precision and recall:
    print(metrics.fowlkes_mallows_score(true_label, predict_label))

    # Silhouette Coefficient 轮廓系数
    print(metrics.silhouette_score(fea, predict_label, metric='euclidean'))

    # Calinski-Harabaz Index 分数值ss越大则聚类效果越好
    # 类别内部数据的协方差越小越好，
    # 类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。
    print(metrics.calinski_harabaz_score(fea, predict_label))
    
