#KNN (k-nearest neighbour ) ALGORITHM
#KNN (K-Nearest Neighbors) is a machine learning algorithm used for classification and regression, but it's more commonly used for classification.

#It works like this:
#"To decide what this new thing is, I’ll look at the K most similar things I already know about — and I’ll choose the most common label among them."
# For example, K = 3 means it will look at the 3 nearest neighbors.

#Measure distance – It calculates the distance (usually Euclidean distance) between the new point and all the training points.
# Simple Example.Let’s say we want to classify fruits based on weight and color:
#Apple,Banana,Orange
#Now, a new fruit comes in. KNN checks the 3 closest fruits in the training data.If 2 of them are Apples and 1 is an Orange →
#It labels the new fruit as Apple.
#Euclidean distance:Distance=in rootsquare((𝑥2−𝑥1)^2+(𝑦2−𝑦1)^2
#Summary
#kNN is simple and intuitive.
# It’s a lazy learner: it doesn’t build a model beforehand.
#t’s great for small datasets, but can be slow with big data

#the process of choosing k value is known as parameter tuning and is a very imp aspect to determine the accuracy of our model.
#methods to choose k:
#1.hit and trial
#2.sqrt(n),where n stands for total no of datasedata samples in dataset
#3.odd value of k is selected to avoid confusion between two classes of data  

#when do we use KNN 
#1. when the data is properly labeled(eg. either the animal wil be cat or lion)
#2. data shoulde be noise free
#works well on small scale data set and KNN is better when you want to create model with higher accuracy on cost of computational resources.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 
# for this particular model we'll be using WINE DATASET 
data=datasets.load_wine(as_frame=True)
print(data )
x=data.data #it store the feature part 
y=data.target
name = data.target_names
# we have three targets ['class_0','class_1','class_2']
print(name)
df =pd.DataFrame(x,columns=data.feature_names)
df['wine classes']= data.target
df['wine classes']=df['wine classes'].replace(to_replace=[0,1,2],value=['class_0','class_1','class_2'])
print(df)
sns.pairplot(df,hue='wine classes',palette='Set2')# from these multiple graphs we can see that dataset seems linearly difficult to  seperate 
plt.show()
# now we check is ther any null values to impute in the datset 
print(df.isnull().sum())#we can see that ther is no value to be imputed
#next we segregrate pur data into training and testing sample
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
import math
math.sqrt(len(y_test))
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,pred))# accuracy comes out to be about 64 percent which is very low to resolve this we can scale our data with the help of  standard scalar 
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()# to covert object values to numeric
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
knn1=KNeighborsClassifier(n_neighbors=7,metric='euclidean')
knn1.fit(x_train,y_train)
pred2=knn1.predict(x_test)
print(metrics.accuracy_score(y_test,pred2))# accuracy scaled up to 98 percent successfully.

