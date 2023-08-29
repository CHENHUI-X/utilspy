import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

def proximityMatrix(model, X, normalize=True):
    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    # 所有的样本在第一棵树上的结果,叶子索引
    a = terminals[:, 0]

    # 在第一棵树上,样本结果索引,两两判断是否相等,相等,ij位置就是1
    proxMat = 1 * np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:, i]  # 把第i棵树的结果取到
        proxMat += 1 * np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat


train = load_breast_cancer()

model = RandomForestClassifier(n_estimators=10, max_features=2, min_samples_leaf=40)
model.fit(train.data, train.target)
proximityMatrix(model, train.data, normalize=True)
