# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 15:56:51 2022

@author: USER
"""
#part 1

# confirm if you can read the first ten records from the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from sklearn import datasets
#from sklearn.metrics import silhouette_samples

iris_data=pd.read_csv("iris.csv")
print(iris_data.head(10))

#part 2
#if you take a look at the data on column 'species', you will notice that there
# some kind of extra string (Iris) before the species name. Let us delete the 
#extra string
#print()


iris_data["Species"]=iris_data.Species.str.replace("Iris-", "")
print(iris_data.head(10))

#part 3
# Since the problem we are trying to solve is a clustering problem which is an
#example of unsupervised learning we don't need a target variable
#because we don't need a target variable therefore we need to select the features
# we need based on the use of iloc[]

x=iris_data.iloc[:,[1,2,3,4]]
#The statement above means select all rows based on the : and columns 1,2,3,4

print(x.head(10))

#part 4

#we want the data to be display in an array form which will exclude the header
# of each columns
x=np.array(x)

print(x)

#part 5
#find the optimal number of clusters which means 
#determine the maximum value of k without guessing

#sse means sum square error

sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(x)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6,6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.title("Elbow method")
plt.show()





