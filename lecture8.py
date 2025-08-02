#Gauassian Naive Bayes algorithm
#Naive Bayes is a classification algorithm based on Bayes' Theorem
# Guassian distrobution also known as normal distribution is a bell shaped curve and it is assumed that during any 
# any measurement values will follow a normal distribution  with an equal nummber of measurements above and below the eman value 
# data has to be continuos to attain guassian distribution 
# say we have give some data like height , weight and footsize to check whether the person belong to male category or female 
# we have some sample and data of male features and female features 
# we are assuming that the give data is of normal distribution type
# so firt we calculate prior probabilities for mae and female . it epends on the training data 
# here we have 8  training data samples out of which 4 are male and 4 are female . 
# p(M)=P(F)=4/4+4 =0.5
# now we will calculate mean and standard deviation for male features ie, height , weight and footsize and similarly for women
# now we calculate posterior probability
# posterior(M)= (P(M) *P(H\M)*P(W\M)*P(FS\M))/evidence . h\m means height given the person is male similarly for all
#posterior(F)= (P(F) *P(H\F)*P(W\F)*P(FS\F))/evidence
# to calculate these we will use probability density function
# whose posterior probabilty is bigger the give person belong to that category

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
df=pd.read_csv('heart.csv')
print (df.head)
#print(df.isnull().sum())
print(df.info())#like the above commented line used to know null values this command also do the same work and give other info also about data set like datatype of every column etc.
print(df.duplicated())#The .duplicated() function in pandas is used to identify duplicate rows in a DataFrame (or Series).
#it Returns a boolean Series: True if the row is a duplicate, False otherwise
#syntax = DataFrame.duplicated(subset=None, keep='first')
#subset: Column label(s) to check for duplicates. Default is all columns.
#keep:
#'first': Mark duplicates except for the first occurrence (default).
#'last': Mark duplicates except for the last occurrence.
#False: Mark all duplicates as True.
print(df[df.duplicated()])#this will only that row which is duplicated
# we can see 164 is the only duplicated row
df.drop_duplicates(keep='first',inplace=True)# this will drop the duplicated row .
# keep ='first' will keep only the one value that first appear and delete rest and inplace will make the changes to our original data frame
# now lets do some data visualisation
#over sex
x=(df.sex.value_counts())
print(f'number of people having sex as 1 are {x[1]} and number of people has sex as 0 are {x[0]}')
p= sns.countplot(data = df,x='sex')
plt.show()
#over chestpain
x=(df.cp.value_counts())
print(x)
p= sns.countplot(data = df,x='cp') 
plt.show()
# over age distribution
plt.figure(figsize=(10,10))
sns.displot(df.age,color='red',label='age',kde=True)#KDE stands for Kernel Density Estimate. It’s a smoothed curve that estimates the probability density function (PDF) of a continuous variable. It gives a better sense of the distribution shape than a binned histogram.
plt.legend()
plt.show()
# over resting blood pressure
plt.figure(figsize=(10,10))
sns.displot(df.trtbps,color='green',label='resting blood pressure',kde=True)
plt.show()#
# complex visualization between features and target vector 
# age vs heartattack
plt.figure(figsize=(10,10))
sns.distplot(df[df['output']==0]['age'],color='green',kde=True,)# this particular person will not get the heart attack
sns.distplot(df[df['output']==1]['age'],color='red',kde=True)# this particular person will get the heart attack
plt.title('ATTACK VS AGE')
plt.show()
# cholestrol vs heartattack
plt.figure(figsize=(10,10))
sns.distplot(df[df['output']==0]['chol'],color='green',kde=True,)# this particular person will not get the heart attack
sns.distplot(df[df['output']==1]['chol'],color='red',kde=True)# this particular person will get the heart attack
plt.title('ATTACK VS CHOLESTROL')
plt.show()
# splitting data into input and output 
y=df.iloc[:,-1].values
x=df.iloc[:,1:-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y , random_state=42, test_size =0.25)
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test= sc.transform(x_test)
from sklearn.naive_bayes import GaussianNB
gnb= GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))

