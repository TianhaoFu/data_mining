import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
if __name__ == '__main__':
    X = pd.read_csv(r"C:\Users\Tianh\Desktop\DMLab\data\car_data.csv", header=None)
    print(X.shape)
#标签数值化
    X = np.array(X)
    le = preprocessing.LabelEncoder()
    le.fit(["vhigh", "high", "med", "low"])
    le1 = preprocessing.LabelEncoder()
    le1.fit(["big", "med", "small"])
    le2 = preprocessing.LabelEncoder()
    le2.fit(["vgood", "good", "acc", "unacc"])

    print(X[:,4])

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
#分割数据集为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print(y_train.shape)
    print(y_test.shape)

    y_train = y_train.reshape(1382, 1)
    y_test = y_test.reshape(346, 1)
    dataSet_train = np.hstack((x_train, y_train))
    dataSet_train = list(dataSet_train)
    dataSet_test = np.hstack((x_test, y_test))
    dataSet_test = list(dataSet_test)
# 保存数据文件为txt格式方便后续处理
    np.savetxt("./dataSet_train.txt", dataSet_train, fmt="%d", delimiter=',')
    np.savetxt("./dataSet_test.txt", dataSet_test, fmt="%d", delimiter=',')