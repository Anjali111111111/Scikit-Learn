#SVM algorithm
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
iris= load_iris()
x=iris.data # features
y=iris.target
names=iris.target_names 
print(x.shape)
print(y.shape)
df=pd.DataFrame(x,columns=iris.feature_names)
df['species']=iris.target
print(df)# in load_iris the flowers are divided into three species denoted by 0,1,2
df['species']=df['species'].replace(to_replace=[0,1,2],value=['septsa','versicolor','virginica'])# here we are replacing the species number with their name 
print(df)
import seaborn as sns
sns.pairplot(data =df,hue='species',palette='Set2')
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=1)
print(x_train.shape,y_train.shape)
print(x_test.shape , y_test.shape)
from sklearn.svm import SVC
svm=SVC(kernel='linear',random_state=0)
svm.fit(x_train,y_train)
pred =svm.predict(x_test)
print(pred)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
rbf=SVC(kernel='rbf',random_state=0)
rbf.fit(x_train,y_train)
pred=rbf.predict(x_test)
print(accuracy_score(y_test,pred))