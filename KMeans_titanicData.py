# This clustering example is based on titanic dataset
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn import preprocessing, cross_validation
style.use('ggplot')


'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())

# handling non numeric data and convert them to numeric data


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

# Setting features
df.drop(['sex', 'boat'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)

# Setting labels
Y = np.array(df['survived'])

# Initializing the algorithm to fit the dataset
clf = KMeans(n_clusters=2)
clf.fit(X)

# print(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    # print(predict_me)
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1

print(correct/len(X))