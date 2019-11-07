import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve  # 导入指标库
import prettytable  # 导入表格库

# 参考代码：
# http://www.dataivy.cn/blog/classification_with_skelarn_tree/
# https://www.cnblogs.com/pinard/p/6056319.html

if __name__ == '__main__':
    # 数据集预处理
    X = pd.read_csv(r"C:\Users\Tianh\Desktop\DMLab\data\car_data.csv", header=None)
    X = np.array(X)

    le = preprocessing.LabelEncoder()
    le.fit(["vhigh", "high", "med", "low"])
    le1 = preprocessing.LabelEncoder()
    le1.fit(["big", "med", "small"])
    le2 = preprocessing.LabelEncoder()
    le2.fit(["vgood", "good", "acc", "unacc"])

    X[:, 0] = le.transform(X[:, 0])
    X[:, 1] = le.transform(X[:, 1])
    X[:, 4] = le1.transform(X[:, 4])
    X[:, 5] = le.transform(X[:, 5])
    X[:, 6] = le2.transform(X[:, 6])

    for i in range(len(X)):
        if X[i, 2] == '5more':
            X[i, 2] = 5
        if X[i, 3] == 'more':
            X[i, 3] = 5

    # X_label = X.iloc[:,6]
    x = X[:, 0:6].astype(np.int32)
    y = X[:, 6].astype(np.int32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # 决策树训练
    clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0.0, random_state=None)

    clf.fit(x_train, y_train)

    # 决策树评价
    pre_y = clf.predict(x_test)

    # 混淆矩阵
    confusion_m = confusion_matrix(y_test, pre_y)  # 获得混淆矩阵
    confusion_matrix_table = prettytable.PrettyTable()  # 创建表格实例
    confusion_matrix_table.add_row(confusion_m[0, :])  # 增加第一行数据
    confusion_matrix_table.add_row(confusion_m[1, :])  # 增加第二行数据
    print ('confusion matrix')
    print (confusion_matrix_table)  # 打印输出混淆矩阵

    # 核心评估指标
    y_score = clf.predict_proba(x_test)  # 获得决策树的预测概率
    # print(y_score)
    # fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # ROC
    # # 错误 因为ROC曲线仅适用于二进制分类任务？
    # auc_s = auc(fpr, tpr)  # AUC
    accuracy_s = accuracy_score(y_test, pre_y)  # 准确率
    precision_s = precision_score(y_test, pre_y, average=None)  # 精确度
    recall_s = recall_score(y_test, pre_y, average=None)  # 召回率
    f1_s = f1_score(y_test, pre_y, average=None)  # F1得分
    # core_metrics = prettytable.PrettyTable()  # 创建表格实例
    # core_metrics.field_names = [ 'accuracy', 'precision', 'recall', 'f1']  # 定义表格列名
    # core_metrics.add_row([accuracy_s, precision_s, recall_s, f1_s])  # 增加数据
    # print ('core metrics')
    # print (core_metrics)  # 打印输出核心评估指标
    print("accuracy")
    print(accuracy_s)
    print('-----------')
    print("precision")
    print(precision_s)
    print('-----------')
    print("recall")
    print(recall_s)

    print('-----------')
    print("f1")
    print(f1_s)
    print('-----------')
    # 返回各类的准确率

    # 决策树可视化
    from IPython.display import Image
    from sklearn import tree
    import pydotplus
    import os

    os.environ["PATH"] += os.pathsep + "D:\\Google\\graphviz-2.38\\release\\bin\\"
    feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    target_names = ['unacc', 'acc', 'good', 'vgood']
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())