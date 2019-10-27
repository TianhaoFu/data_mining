import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import csv

from sklearn.cluster import DBSCAN
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

if __name__ == '__main__':
    # 1、数据集导入
    data1 = scipy.io.loadmat('C:\\Users\\Tianh\\Desktop\\DMLab\\data\\face\\ORL_32x32.mat')
    df2 = pd.read_csv('C:\\Users\\Tianh\\Desktop\\DMLab\\data\\wineequality\\winequality-white.csv', delimiter=';')
    df3 = pd.read_csv('C:\\Users\\Tianh\\Desktop\\DMLab\\data\\wineequality\\winequality-red.csv', delimiter=';')
    data4 = scipy.io.loadmat('C:\\Users\\Tianh\\Desktop\\DMLab\\data\\face\\Yale_32x32.mat')
    data5 = pd.read_csv(r"C:\Users\\Tianh\Desktop\DMLab\data\Iris\iris_data.csv", header=None)

    # 2、数据集预处理得到数据集样本特征及标签
    fea1 = data['fea'].astype(np.float64)
    fea1 = data['fea'].astype(np.float64)
    true_label1 = data['gnd'].ravel()

    fea2 = df2.drop('quality', axis=1)
    fea2 = np.array(fea2)
    true_label2 = df2['quality']
    true_label2 = np.array(true_label2)

    fea3 = df3.drop('quality', axis=1)
    fea3 = np.array(fea3)
    true_label3 = df3['quality']
    true_label3 = np.array(true_label3)

    fea4 = data4['fea'].astype(np.float64)
    true_label4 = data4['gnd'].ravel()

    fea5 = np.array(data5.iloc[:, 0:4])
    true_label5 = data5[4]

    # 3、DBSCAN聚类
    predict_label = DBSCAN(eps=0.5, min_samples=1).fit_predict(fea5)
    plt.scatter(fea1[:, 0], fea2[:, 1], c=predict_label)
    plt.show()
    # eps min_samples 初始时值取得过大导致结果皆为一类
    # 如果数据量比较大的话，eps 就大一些，min_samples 就小一些，先跑一下看情况再继续调

    # 4、K-means聚类（与DBSCAN结果相比较）
    predict_label = KMeans(n_clusters=3, random_state=9).fit_predict(fea1)
    plt.scatter(fea[:, 0], fea[:, 1], c=predict_label)
    plt.show()

    # 5、结果分析 计算评价指标
    true_label = true_label5

    # Mutual Information based scores 互信息
    x1 = metrics.adjusted_mutual_info_score(true_label, predict_label)

    # Homogeneity 同质性 每个群集只包含单个类的成员
    x2 = metrics.homogeneity_score(true_label, predict_label)

    # completeness 完整性 类的所有成员都分配给同一个群集
    x3 = metrics.completeness_score(true_label, predict_label)

    # 两者的调和平均V-measure：
    x4 = metrics.v_measure_score(true_label, predict_label)

    #  Fowlkes-Mallows scores
    # The Fowlkes-Mallows score FMI is defined as the geometric mean
    # of the pairwise precision and recall:
    x5 = metrics.fowlkes_mallows_score(true_label, predict_label)

    # Silhouette Coefficient 轮廓系数
    # x6 = metrics.silhouette_score(fea1, predict_label, metric='euclidean')

    # Calinski-Harabaz Index 分数值ss越大则聚类效果越好
    # 类别内部数据的协方差越小越好，
    # 类别之间的协方差越大越好，这样的Calinski-Harabasz分数会高。
    # x7 = metrics.calinski_harabaz_score(fea1, predict_label)

    # 6、将评价指标导出为csv文件
    rows = [['Mutual Information based scores', x1]
        , ['Homogeneity', x2]
        , ['completeness', x3]
        , ['V-measure', x4]
        , ['Fowlkes-Mallows scores', x5]
            #         ['Silhouette Coefficient',x6]
            #         ['Calinski-Harabaz Index',x7]
            ]

    with open(r"C:\Users\Tianh\Desktop\ans1.csv", 'w', newline='') as csv_file:
        # 获取一个csv对象进行内容写入
        writer = csv.writer(csv_file)
        for row in rows:
            # writerow 写入一行数据
            writer.writerow(row)
        # 写入多行
        writer.writerows(rows)

