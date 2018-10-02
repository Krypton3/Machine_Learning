# In this implement, I am going to write the k Means algorithm from scratch
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

'''
What I am going to do:

Choose value for K
Randomly select K feature sets to start as your centroids
Calculate distance of all other feature sets to centroids
Classify other feature sets as same as closest centroid
Take mean of each class (mean of all feature sets by class), making that mean the new centroid
Repeat steps 3-5 until optimized (centroids no longer moving)

'''

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9,11]])

plt.scatter(X[:, 0], X[:, 1], s=100)
# plt.show()

'''
algorithm optimized if the centroid is not moving more than the tolerance value.
max_iter denotes the maximum number of tie we are going to run the KMeans
'''

colors = 10*["g","r","c","b","k"]


class KMeans:
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


clf = KMeans()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="X", color=color, s=150, linewidths=5)

test = np.array([[1,3],
                     [8,9],
                     [0,3],
                     [5,4],
                     [6,4],])
for t in test:
    classification = clf.predict(t)
    plt.scatter(t[0], t[1], color=colors[classification], s=150, marker="*", linewidths=5)

plt.show()