#  Decision Tree algorithm is a supervised machine learning method used for classification and regression tasks. It works by recursively splitting the dataset into subsets based on feature values, 
# creating a tree-like model of decisions.
#A decision tree consists of:
#Root Node: The top decision node based on the most significant feature.
#Internal Nodes: Test conditions (like feature < value) that split the data.
#Leaf Nodes: Final predictions (classes or values).
#It splits data to reduce impurity using criteria like Gini index(default), Entropy(classification), or Mean Squared Error(regression).
# its one of the advantage isthat it doesnt make any assumption about data .This makes it suitable for both sontinuous and dis continuous data .
# it is also called CART (classification and regression Tree)
# not suitable for large and complex datasets.
# important terms:
#1.ENTROPY - it is the measure of randomness or unpredictibility in the dataset
# we have to calculate entropy for every level of tree. Generally the entropy lies between 0 and 1. but its totally depend on the dataset. if its more than 1 or 2 then we can say that this algo is not suitavle for classification of this dataset
#2. INFORMATION GAIN - it is how much entropy was removed during splitting at a node 
#3. GINI IMPURITY - it is the purity of the split at nodes of decision tree.
# its value varies from 0 to 0.5
# a node is pure when the gini attribute value is zero.
# formulas are on video scikit intellipat
import pandas as pd
df= pd.read_csv("drug200.csv")
print(df)
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
print(df.isnull().sum()) # df.isna().sum() this will alsp same 
print(df.duplicated())
print(df[df.duplicated()])
print(df.info())
x= df.Sex.value_counts()# this wil show how many type value like male or female
print(x)
p=sns.countplot(data=df,x='Sex')
plt.show()
x= df.Drug.value_counts()# this wil show how many type value like male or female
print(x)
p=sns.countplot(data=df,x='Drug')
plt.show()
df['Drug'].unique()
plt.figure(figsize=(10,10))
# sns.distplot() is deprecated and may behave inconsistently.
# A deprecated function is a function or feature that is still available in the current version of a library or language, but it is no longer recommended for use.
# we can use other function like histplot kdeplot for this
# fill in kde is used to fill the area under the curve.
sns.kdeplot(data=df[df['Drug']=='drugY'], x='Age', color='red', label='drugY', fill=True)
sns.kdeplot(data=df[df['Drug']=='drugX'], x='Age', color='green', label='drugX', fill=True)
sns.kdeplot(data=df[df['Drug']=='drugA'], x='Age', color='black', label='drugA', fill=True)
sns.kdeplot(data=df[df['Drug']=='drugB'], x='Age', color='orange', label='drugB', fill=True)
sns.kdeplot(data=df[df['Drug']=='drugC'], x='Age', color='blue', label='drugC', fill=True)
plt.title('AGE VS DRUG CLASS')
plt.legend()
plt.show()
# we can see that we dont get line of drugY in the graph. this is because the number of rows in drugY is either 0 or very less . we can also check it 
print(df[df['Drug'] == 'drugY'].shape[0])  # Number of rows with drugY
print(df[df['Drug'] == 'drugX'].shape[0])  # Number of rows with drugY
# dealing with non numeric value
from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder()# to covert object values to numeric
df['BP']=oe.fit_transform(df[['BP']])
df['Sex']=oe.fit_transform(df[['Sex']])
df['Cholesterol']=oe.fit_transform(df[['Cholesterol']])
df['Drug']=oe.fit_transform(df[['Drug']])
print(df)
#splitting data
x= df.iloc[:, 0:-1]# all aree input ecept the last column
y= df.iloc[:,-1]# this will be the label (only the last column)
print(x)
print(y)
# train-test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
print(x_train)
print(y_train)
from sklearn.tree import DecisionTreeClassifier
clf_gini= DecisionTreeClassifier(criterion='gini',random_state=0)
clf_gini.fit(x_train,y_train)
y_pred_gini=clf_gini.predict(x_test)
print(y_pred_gini)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_gini,y_test))
# the best advantage which Decision tree provides is that we can check the background implementation of how our algo implement this particular problem
from sklearn import tree
plt.figure(figsize=(10,10))
tree.plot_tree(clf_gini.fit(x_train,y_train))
plt.show()
# we can also apply other function or features of Decision Classifier.
clf_entropy=DecisionTreeClassifier(criterion='entropy',random_state=0,max_depth=3)
clf_entropy.fit(x_train,y_train)
y_pred_entropy=clf_entropy.predict(x_test)
print(accuracy_score(y_pred_entropy,y_test))
# its accuracy is low because of tree depth limit 
#lets see its implementation too
plt.figure(figsize=(10,10))
tree.plot_tree(clf_entropy.fit(x_train,y_train))
plt.show()