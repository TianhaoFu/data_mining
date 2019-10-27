import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
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

def pearson_distance(vector1, vector2):
    from scipy.spatial.distance import pdist
    X = vstack([vector1, vector2])
    d2 = pdist(X)
    return d2

def totalcost(blogwords, costf, medoids_idx):
    size = len(blogwords)
    total_cost = 0.0
    medoids = {}
    for idx in medoids_idx:
        medoids[idx] = []
    for i in range(size):
        choice = None
        min_cost = inf
        for m in medoids:
            tmp = distances_cache.get((m, i), None)
            if tmp == None:
                tmp = pearson_distance(blogwords[m], blogwords[i])
                distances_cache[(m, i)] = tmp
            if tmp < min_cost:
                choice = m
                min_cost = tmp
        medoids[choice].append(i)
        total_cost += min_cost
    return total_cost, medoids

def kmedoids(blogwords, k):
    import random
    size = len(blogwords)
    medoids_idx = random.sample([i for i in range(size)], k)
    pre_cost, medoids = totalcost(blogwords, pearson_distance, medoids_idx)
#     print pre_cost
    current_cost = inf  # maxmum of pearson_distances is 2.
    best_choice = []
    best_res = {}
    iter_count = 0
    while 1:
        for m in medoids:
            for item in medoids[m]:
                if item != m:
                    idx = medoids_idx.index(m)
                    swap_temp = medoids_idx[idx]
                    medoids_idx[idx] = item
                    tmp, medoids_ = totalcost(blogwords, pearson_distance, medoids_idx)
                    # print tmp,'-------->',medoids_.keys()
                    if tmp < current_cost:
                        best_choice = list(medoids_idx)
                        best_res = dict(medoids_)
                        current_cost = tmp
                    medoids_idx[idx] = swap_temp
        iter_count += 1
#         print current_cost, iter_count
        if best_choice == medoids_idx: break
        if current_cost <= pre_cost:
            pre_cost = current_cost
            medoids = best_res
            medoids_idx = best_choice

    return current_cost, best_choice, best_res

if __name__ == '__main__':
    # 导入数据
    data = scipy.io.loadmat('C:\\Users\\Tianh\\Desktop\\DMLab\\data\\face\\ORL_32x32.mat')

    # 对数据进行预处理
    fea = data['fea']
    print(fea)

    norm_fea = minmax_Norm(fea)
    norm_fea

    # k-means 聚类
    cluster_num = 40
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(norm_fea)
    predict_label = kmeans.labels_
    print(kmeans.cluster_centers_)

    k-medoid聚类
    distances_cache = {}
    best_cost, best_choice, best_medoids = kmedoids(data, 15)
    data = mat(data)



