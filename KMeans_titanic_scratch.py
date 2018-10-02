# In this implement, I am going to write the k Means algorithm from scratch
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from sklearn import preprocessing
import numpy as np
style.use('ggplot')

colors = 10*["g","r","c","b","k"]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):

            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featurest in data:
                distance = [np.linalg.norm(featurest - self.centroids[centroid]) for centroid in self.centroids]
                classification = distance.index(min(distance))
                self.classifications[classification].append(featurest)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distance = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distance.index(min(distance))
        return classification


df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())


def handle_non_numeric_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_val = {}

        def convert_int(val):
            return text_digit_val[val]

        if df[column].dtype != np.float64 and df[column].dtype != np.int64:
            column_content = df[column].values.tolist()
            unique_column = set(column_content)
            x = 0
            for unique in unique_column:
                if unique not in text_digit_val:
                    text_digit_val[unique] = x
                    x += 1

            df[column] = list(map(convert_int, df[column]))

    return df


df = handle_non_numeric_data(df)
print(df.head())

df.drop(['ticket','home.dest'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['survived'])

clf = K_Means()
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == Y[i]:
        correct += 1

print(correct/len(X))