#RANDOM FOREST CLASSIFIER
# Random Forest is an ensemble learning method used for classification and regression.
# It builds multiple decision trees and combines their results to make more accurate and stable predictions.

# 🔧 How It Works
# Bootstrap Sampling:
# Random subsets of the training data are created with replacement.
# Each subset trains a decision tree (this is called bagging).

# RANDOM SAMPLING:
# the process of creating rows for subsets .

# Feature selction- the process of selecting features for bootstrap sample :
#  only a random subset of features is considered.
# for classification it is =underroot(total no of features)(by default)
# for regression it is= total no of features /3(by default)
# This makes trees more diverse and less correlated.

# Multiple Trees:
# Dozens or hundreds of decision trees are trained on different data samples and feature subsets.

# Voting/Averaging:
# For classification → Majority vote from all trees.
# For regression → Average of all tree outputs.

# 📦 Summary of Steps
# Take n random samples from dataset (with replacement).
# Build a decision tree on each sample.
# At each split in the tree, use a random subset of features.
# Combine predictions from all trees:
# Classification: vote.
# Regression: average.

# 💡 Why It’s Powerful
# Reduces overfitting (compared to a single decision tree).
# Handles missing values well.
# Works for both classification and regression.
# Good performance even without heavy tuning.

# 💡 Disadvantages
#  complexity
# computationally expensive
# black box model
# unable to handle imbalanced data
# biased towards features with many levels.
import pandas as pd
df=pd.read_csv("car.data")
#print(df)
print(df.head())# as the dataset is too large we are printing only starting 5 rows.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
# as we can see we dont have columns name in dataset 
# so we are giving acc to info provided about column name in uci dataset .
# class values in which we have to classify car are unacc(unacceptable),acc,good,vgood
col_names=['buying','maint','doors','persons','lug_boot','safety','class']
df.columns=col_names
print(df.head())
print(df.info())
print(df.isnull().sum())#we can see their no null values 
print(df.describe(include='all').T)
for col in col_names:
    print(df[col].value_counts())# this will give the detailed info about each column values with their frequency.
print(df[df.duplicated()])
# we can see that mostly columns are of object datatype so we need to convert them to numeric datatype
from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()
df['buying']=oe.fit_transform(df[['buying']])
df['maint']=oe.fit_transform(df[['maint']])
df['doors']=oe.fit_transform(df[['doors']])
df['persons']=oe.fit_transform(df[['persons']])
df['lug_boot']=oe.fit_transform(df[['lug_boot']])
df['safety']=oe.fit_transform(df[['safety']])
df['class']=oe.fit_transform(df[['class']])
print(df.head())
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))




