import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.cluster import MeanShift
import numpy as np
from sklearn import preprocessing
style.use('ggplot')

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name'], 1, inplace=True)
df.fillna(0, inplace=True)


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

df.drop(['ticket', 'home.dest'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_  = len(np.unique(labels))
survival_rates = {}

for i in range(n_clusters_):
    team_df = original_df[ (original_df['cluster_group'] == float(i))]

    survival_cluster = team_df[ (team_df['survived'] == 1)]
    survival_rate = len(survival_cluster)/len(team_df)

    survival_rates[i] = survival_rate

print(survival_rates)