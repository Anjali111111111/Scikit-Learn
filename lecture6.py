#logistic regression
#in case of linear regression we saw that the predicted values are continuous but here in logistic regression they are discrete 
# Logistic Regression in scikit-learn (sklearn) is a supervised machine learning algorithm used for binary or multiclass classification tasks. Despite its name, it is actually a classification algorithm, not a regression one.
#It predicts the probability of a target variable belonging to a particular class using the logistic (sigmoid) function, which outputs values between 0 and 1.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
#fetch_openml function is used to fetch dataset.
#we do as_frame = True because it will convert the data into a dataframe
titanic_data=fetch_openml("titanic",version=1,as_frame=True)
df=titanic_data['data']
df['survived']=titanic_data['target']
print(df.head())# to check first five entries and features we have in dataframe
#to visualize how many are survived and how many not 
sns.countplot(x="survived",data=df)
plt.show()# the bigger block show people died and other survived
sns.countplot(x="survived",hue="sex",data=df)# now this will show how many women survived how many man survived and how many men and women died
plt.show()# from this we can see that many women had survived than males
#In the Titanic dataset, the passenger classes are represented by the pclass column, which indicates which class a passenger was traveling in
#There are 3 passenger classes:
#pclass	Description
#1	1st class (Upper)
#2	2nd class (Middle)
#3	3rd class (Lower)
sns.countplot(x="survived",hue="pclass",data=df)#pclass stands for passenger class.
plt.show()#this shows 1st class people survived most
#now to know which age group maximum people belong too.
df['age'].plot.hist()
plt.show()#max people belong to the age group pf 20-30.
print(df.info())#show the type of values dataset has
#now to check how mauny null values are in different columns
print(df.isnull().sum())
#now to convert this sum into percentage and represent using a bar graph
miss_vals=pd.DataFrame(df.isnull().sum()/len(df)*100)
miss_vals.plot(kind="bar",title="missing values in percentage",ylabel="percentage")
plt.show()
#the body column has more than 80% missing values and also in cabin,boat. so we will drop these columns eventually
#there are two column i.e, sibsp and parch in dataframe which shows with how many (sibling and spouse) and (parents) a person is travel for so using this we can create a new column to check wether a person is travelling alone or not.
# lets create a new column/feature family
df['family']=df['sibsp']+df['parch']
df.loc[df['family']>0,'travelled_alone']=0#family !=0 that is person travelling with family i.e 0
df.loc[df['family']==0,'travelled_alone']=1#family==0 person is travelling alone i.e 1
print(df['family'].head())
# now as we have family feature we dont need sibsp and parch so we ca drop them
#sns.countplot(x="survived",data=df)
#plt.show()
#print(df['fanily'].head())
df.drop(['sibsp','parch'],axis=1,inplace=True)
sns.countplot(x='travelled_alone',data=df)
plt.title("number of passengers travelling alone")
plt.show()# we can see there about 800 people who travelled alone
# now as we have drop sibsp and parch from our dataframe and add family and travelled_alone . we can see it
print(df.head())
# now we have some other features in our dataset like name , ticket, home.dest which don't contribute to main agenda i.e how many people survived so we can drop them
df.drop(['name','ticket','home.dest'],axis=1,inplace=True)# axis=1 means remove the column
#now we can again check our df , these features are removed.
print(df.head())
df.drop(['cabin','body','boat'],axis=1,inplace=True)#also irrelevant
#now there are some feature in our dataset which are uncomputable like sex(female/male) we cannot compute it .
#for this to understand by system we can either use get_dummies values or 1 hot encoding
sex=pd.get_dummies(df['sex'])
print(sex)
#we see that it has two colum female and male when male is 0 then female is 1 and vice versa. so we dont need two column we can drop one . as we get to know from other column whether its female or male
sex=pd.get_dummies(df['sex'],drop_first=True)# drop female column
df['sex']=sex
print(df.isnull().sum())
#now we can see that we have some null values in age,fare, and embark column
from sklearn.impute import SimpleImputer
imp_mean=SimpleImputer(strategy='mean')
df['age']=imp_mean.fit_transform(df[['age']])
df['fare']=imp_mean.fit_transform(df[['fare']])
print(df.isnull().sum())#now only embark has 2 null values
# we see above like sex (female and male), embark has characteristic values like (S,C,Q)which cannot be impute but before that we need to find the two values null . and beacuse of this we cant use strategy like mean , median,mode.
#we have to use mostfrequent strategy
imp_freq=SimpleImputer(strategy='most_frequent')
df['embarked']=imp_freq.fit_transform(df[['embarked']]).ravel()# here we hwt a vale error 2 if we didnt use ravel() this because you're trying to assign a 2D array (which SimpleImputer.fit_transform() returns) into a single pandas Series (1D) column, which causes a mismatch.
print(df.isnull().sum())#now no null values left
print(df.head())# as embark has characteristic data so we use get_dummy method
embark=pd.get_dummies(df['embarked'])
print(embark) 
# we have three colum here C Q S . when one of three is 0 the other two is 1, i.e, we dont need any one column out of three
embark=pd.get_dummies(df['embarked'],drop_first=True)
# we drop the embark column and concatenate the new feature embark to our dataset
df.drop(['embarked'],axis=1,inplace =True)
df=pd.concat([df,embark],axis=1)
print(df.head())# now here we see that embark column has removed and Q S is added
# now we before pass our data to algo lets divide it into x and y label
x=df.drop(['survived'],axis=1)
print(x.head())# x will act as input
y=df['survived']
print(y.head())# y act as output or label data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)# text size is 0.3 so that we can have 30 percent data for testing purpose
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
from sklearn.linear_model import LogisticRegression
mod=LogisticRegression() #object to store functionalities of this algorithm
mod.fit(x_train,y_train)
pred=mod.predict(x_test)# model to predict for the texting sample
# to check accuracy of my model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))#pass the true result and the predicted value
# to check how many right prediction our model made
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred))
# we can see our model made 204 true positve pred and 109 true negatiive and 48 and 32 that are false positive and false negative are wrong prediction made by our model 