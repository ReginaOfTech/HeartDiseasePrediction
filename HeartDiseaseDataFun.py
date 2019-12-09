# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:58:23 2019

@author: AdminUser
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

dataSet = pd.read_csv("heart.csv")
print(dataSet.head(10))
print(dataSet.dtypes)
print("Any Null Values in DataSet?")
print(dataSet.isnull().sum())

sns.countplot(x='cp', data=dataSet)
plt.xlabel("CP (0: typical angina; 1: atypical angina; 2: non-anginal pain; 3: asymptomatic)")
plt.show()

sns.countplot(x='thal', data=dataSet)
plt.xlabel("thal (0 = no info; 1 = normal; 2 = fixed defect; 3 = reversable defect)")
plt.show()

sns.countplot(x='slope', data=dataSet)
plt.xlabel("slope (0: upsloping; 1: flat; 2: downsloping)")
plt.show()

#due to cp,thal, and slope being categorical these will become one hot encoded
cp_Encoded = pd.get_dummies(dataSet['cp'], prefix = "cp")
thal_Encoded = pd.get_dummies(dataSet['thal'], prefix = "thal")
slope_Encoded = pd.get_dummies(dataSet['slope'], prefix = "slope")
#Bring everything together now!
dataSet = pd.concat([dataSet, cp_Encoded, thal_Encoded, slope_Encoded], axis=1)
#need to drop the old columns since its one hot encoded now
dataSet = dataSet.drop(columns = ['cp','thal','slope'])

#Visualize any correlations
ax = sns.heatmap(
    dataSet.corr(), 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
print(dataSet.corr())

#based on correlationd data can remove these columns
dataSet = dataSet.drop(columns = ['ca','exang','age','chol'])

#Normalize the data
dataSet = (dataSet - np.min(dataSet)) / (np.max(dataSet) - np.min(dataSet)).values
print(dataSet.head())

#Create the target and main dataframes
data_x = dataSet.drop('target', axis=1)
data_y = dataSet['target']

#Create the test and training data split
X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.20, random_state=42)


#Create, train, and predict Support Vector Classifier
clf = svm.LinearSVC().fit(X_train, Y_train)
pred = clf.predict(X_test)

'''clf = RandomForestClassifier(random_state=0, n_jobs=-1)
model = clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
importances = model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_test.columns[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X_train.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X_train.shape[1]), names, rotation=90)

# Show plot
plt.show()'''

print(pred)
print(Y_test)
print(clf.score(X_test, Y_test))
print("DONE")


