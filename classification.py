#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: amazingamar
"""
import pandas as pd

import numpy as np



dataset = pd.read_csv('train.csv')
test_dataset =pd.read_csv('test.csv')
x = dataset.iloc[:,1:785].values
y = dataset.iloc[:,0].values
x_test = test_dataset.iloc[:,0:784].values
y_test = test_dataset.iloc[:,0].values









from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=20)
forest_clf.fit(x, y)
u=forest_clf.predict(x)

from sklearn.model_selection import cross_val_score
z=cross_val_score(forest_clf, x, y, cv=3, scoring="accuracy")

v= forest_clf.predict(x_test)
print(z)

submission = pd.DataFrame(v, columns=['Label']).to_csv('submission.csv', index_label="ImageID")

df = pd.read_csv('submission.csv')

df.index=df.index+1


