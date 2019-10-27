import numpy as np
import pandas as pd
import matplotlib as plt

import scipy.io
import scipy.cluster.hierarchy as sch

import seaborn as sns
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

    # 3、层次聚类
    disMat = sch.distance.pdist(fea5, 'euclidean')
    Z = sch.linkage(disMat, method='average')
    P = sch.dendrogram(Z)
    # plt.show()
    # plt.savefig('plot_dendrogram.png')
    # cluster= sch.fcluster(Z, t=1, 'inconsistent')
    # print("Original cluster by hierarchy clustering:\n",cluster)
