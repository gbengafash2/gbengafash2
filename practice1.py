# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("clustering_dataset.csv")
print(data.head(10))
#after typing the code to this point you can run the program
#To be sure it is reading from the dataset
# once it is running remove the comment sign against the import statement matplotlib.pyplot

x=data[["LoanAmount", "ApplicantIncome"]]
plt.scatter(x["ApplicantIncome"],x["LoanAmount"],c='black')
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.show()

# implement step 1 and 2 in the K-means clustering algorithm
K=3

# Select random observation as centroids
Centroids = (x.sample(n=K))
plt.scatter(x["ApplicantIncome"],x["LoanAmount"],c='black')
plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='red')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()


#step 3, 4 and 5
diff = 1
j=0

while(diff!=0):
    XD=x
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["ApplicantIncome"]-row_d["ApplicantIncome"])**2
            d2=(row_c["LoanAmount"]-row_d["LoanAmount"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        x[i]=ED
        i=i+1

    C=[]
    for index,row in x.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    x["Cluster"]=C
    Centroids_new = x.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['LoanAmount'] - Centroids['LoanAmount']).sum() + (Centroids_new['ApplicantIncome'] - Centroids['ApplicantIncome']).sum()
        print(diff.sum())
    Centroids = x.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]


#when the difference is zero then we stop the training
color=['blue','green','yellow']
for k in range(K):
    data=x[x["Cluster"]==k+1]
    plt.scatter(data["ApplicantIncome"],data["LoanAmount"],c=color[k])
plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='red')
plt.xlabel('Income')
plt.ylabel('Loan Amount')
plt.show()
    
#Note that the red dot represent the centroid of each cluster
    
    



