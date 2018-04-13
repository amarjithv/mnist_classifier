"""
@author : amarjith
"""

import pandas as pd

dataset = pd.read_csv('train.csv')
test_dataset =pd.read_csv('test.csv')
x = dataset.iloc[:,1:785].values
y = dataset.iloc[:,0].values
x_test = dataset.iloc[:,1:785].values
y_test = test_dataset.iloc[:,0].values


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(x, y)

