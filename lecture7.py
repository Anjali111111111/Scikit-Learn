#k means algorithm
# K-Means Clustering is an unsupervised machine learning algorithm used to group data into clusters based on similarity.
# 
# Key Concepts:
# "K" stands for the number of clusters you want to form.
# step 1 : is to choose k vale . we can use Elbow method for this.
# step 2 : choose random points from data as starting centers for each cluster
# Each cluster is represented by its centroid (mean of the points in that cluster).
# step 3: measure the distance between the centroids and all other points in data.
# step 4 : assign data to the cluster whose center is closest to it.
# step 5: find middle point of each cluster , which becomes the new centeroids.
# step 6: repeat steps 3 to 5 ,adjusting centers and group assignments until they stop changing much accuarate.
# The algorithm tries to minimize the distance between data points and their respective cluster centroids.
#It is used for clustering, not classification or regression.
#It finds structure in unlabeled data.
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from sklearn import datasets
data=datasets.load_wine()
x=data.data
y=data.target
 # as we know we dont use labelfor unsupervised algorithms so here we dont use target in whole program except at last for calculating the accuracy of our k-n algo.
df=pd.DataFrame(x,columns=data.feature_names)
df['Wine Class']=y
print(df)
print(df.isnull().sum())# we can see there are no null values
print(df.describe())# this function is used to get more info about mathematical content of dataset 
#mean = mean value of differet features 
#Standard deviation = how the data is distributed w.r.t the mean value 
from sklearn.preprocessing import StandardScaler# this method will scale down mean for every feature to zero
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.cluster import KMeans
# we studied above different ways to take k . here we are using elbow method that is wss.
wss=[]#creates an empty list for storing wss values
for i in range(1,11):# clusters possible tsken from 1 to 10 , min 1 cluster is present and max can be 10 . we take this range for almost all datasets unless the dataset is very big and contain subgroup. so it is not compulsory to take this range but it is most used range  
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmeans.fit(x)
    wss.append(kmeans.inertia_)# add wss value to list 
f3,ax=plt.subplots(figsize=(8,6))#Creates a figure and one or more subplots (axes) in one step.f3 is an object for figure and ax for axes
plt.plot(range(1,11),wss)
plt.title("the elbow technique")
plt.xlabel("Number of clusters")
plt.ylabel("wss")
plt.show()
# we can see that the elbow is getting formed at k=3, so again making k means model with 3 clusters
N=3
#here we see that accuracy comes tout to be very bad so we do some modification in our mode
# k_means=KMeans(init ="k-means++",n_clusters=N)
k_means=KMeans(init ="k-means++",n_clusters=N,n_init=20,max_iter=360)# now we are getting a 96 percent accuracy 
#init="k-means++" Determines how initial cluster centroids are chosen. alternative we can use "random" but "k-means" is much better
# n_init=10(by default)How many times the algorithm will run with different initial centroid seeds.
#max_iter= 300(by default)no of times iteration occur per k-means
k_means.fit(x)
labels=k_means.labels_
print(labels)
from sklearn.metrics import accuracy_score
print(accuracy_score(labels,y))
