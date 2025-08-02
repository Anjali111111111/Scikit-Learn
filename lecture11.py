#ADABOOST ALGORITHM
# unlike in random forest tree where the different subparts can have different length of tree . 
# here in adaboost algo we have one parent node and two leaves . this structure is called stump thus adaboost is a forest of stump.
# ada boost combines weak learners to land at the decision.that is in RF we use every feature in tree to get a result , but here we can use only one feature in one stump so the result is weaker . But the AF take advantage of this.
# In RF each tree has one vote but in AF different stumps get varied amount of say towards final classification , we can say larger the stump more the amount of say towards final decision
# In RF the order of subtrees are not important but here in AF the order of stumps is quite important . the error that the first stump makes will influence the second stump and so on..
#ADV - handels complex data , good accuracy, versatile , handles imbalanced data
#DIS- computational complexity , lack of transparency, susceptibility of biased data

#IMPLEMENTATION
import pandas as pd
test=pd.read_csv("test.csv")
train=pd.read_csv("train.csv")
print(test.head())
print(train.head())
#concatinating train and test in one dataframe 
df=pd.concat([train,test],sort=False)
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
print(df.info())
print(df.isnull().sum())
# we will drop useless feature 'cabin' having null values and no need to do imputation for this
df=df.drop('Cabin',axis=1)
# using imputer for value imputation of null values of other column
from sklearn.impute import SimpleImputer
print(f'number of null values in Age column before imputataion is:{df.Age.isnull().sum()}')
si=SimpleImputer(strategy='mean')
df['Age']=si.fit_transform(df[['Age']])
print(f'number of null values in Age column after imputataion is:{df.Age.isnull().sum()}')

print(f'number of null values in Survived column before imputataion is:{df.Survived.isnull().sum()}')
df['Survived']=si.fit_transform(df[['Survived']])
print(f'number of null values in Survived column after imputataion is:{df.Survived.isnull().sum()}')

# the data in Embarked is not of numeric type so we change the strategy to most_frequent
si1=SimpleImputer(strategy='most_frequent')
print(f'number of null values in Embarked column before imputataion is:{df.Embarked.isnull().sum()}')
df['Embarked']=si1.fit_transform(df[['Embarked']]).ravel()
print(f'number of null values in Embarked column after imputataion is:{df.Embarked.isnull().sum()}')
print(df.isnull().sum())
df['Fare']=si1.fit_transform(df[['Fare']]).ravel()
print(df.isnull().sum())
# dropping columns that are of no use for our result
df=df.drop(['Name','Ticket'],axis=1)
# Adaboost is a sensitive algo so now we need to remove outliers from here 
# to check outliers we'll use z_score
#In z-score ±3 is a common threshold for detecting outliers .however it is not universal value it may change accordingly if required
threshold=3
from scipy.stats import zscore
# here we only have two numeric features age anda fare so we apply to them only 
numerical_features=['Age','Fare']
z_scores=np.abs(zscore(df[numerical_features]))
print(z_scores)
outliers= np.where(z_scores>threshold)
print(outliers)# here we can see the outliers in our data
# here we use winsorize : Instead of deleting outliers, it replaces the most extreme values with less extreme values — specifically, the values at a given percentile threshold.
from scipy.stats.mstats import winsorize
df['Age']= winsorize(df['Age'],limits=[0.15,0.15])# 15 percent of data acc to z_score normal distribution graph is considered as outlier part
df['Fare']= winsorize(df['Fare'],limits=[0.15,0.15])
print("lets check outliers now ")
z_scores=np.abs(zscore(df[numerical_features]))# as we see nothing prints that means our data are now free of outliers
outliers= np.where(z_scores>threshold)
print(outliers)
# now converting the columns object value to numeric values
from sklearn.preprocessing import OrdinalEncoder
oe= OrdinalEncoder()
df['Sex']=oe.fit_transform(df[['Sex']])
df['Embarked']=oe.fit_transform(df[['Embarked']])
df['Survived']=oe.fit_transform(df[['Survived']])
print(df.head())
# splitting data to x and y
x= df.drop('Survived',axis=1)
y=df['Survived']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
# fitting our data to ml algo ADABOOST
from sklearn.ensemble import AdaBoostClassifier
ab = AdaBoostClassifier(n_estimators=45,learning_rate=1,random_state=0)
ab.fit(x_train,y_train)
y_pred=ab.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))


