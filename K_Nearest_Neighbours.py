import numpy as np
from sklearn import preprocessing, cross_validation, neighbors

import pandas as pd

df = pd.read_csv('data.csv')
df.drop(['id'], 1, inplace=True)

print(df.head())
df.replace('?', -999, inplace=True)  # replacing garbage with a value for calculation purpose
df = df.fillna(df.median(axis=0))

# Making features and labels

X = np.array(df.drop(['diagnosis'], 1))
y = np.array(df['diagnosis'])


# Creating testing and training samples

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Defining classifier

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

