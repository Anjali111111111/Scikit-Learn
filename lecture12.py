# gradient boost algo
#A Gradient Boosting Algorithm is a powerful machine learning technique used for both regression and classification tasks. It belongs to the family of ensemble methods, which combine the predictions of multiple models to produce a more accurate final result.
# here the trr start with leaf called base leaf
# Start with an initial prediction.

#For regression, it might be the mean of the target variable.

#For classification, it could be log-odds or class probability.

#Calculate the residuals (errors) between the actual values and the predicted values.

#Train a new decision tree to predict these residuals.

#Update the model by adding the new tree's predictions, scaled by a learning rate.

#Repeat steps 2–4 for a predefined number of trees or until the error stops improving.
   
# implementation 
from sklearn import datasets
data =  datasets.load_diabetes()
#print(data)# now we will convert this dictionary like data into dataframe
import pandas as pd 
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=pd.Series(data.target)
print(df.head())
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
print(df.info())# we can all features of float datatype so we dont need to do value encoding
print(df.isnull().sum())# we can see there are no null values 
print(df[df.duplicated()])# we can see we are not getting any record so our dataframe do not consis any duplicated value
# now divivding data in to x and y
x=df.drop('target',axis=1)
y=df['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=100)
gbr.fit(x_train,y_train)
y_pred=gbr.predict(x_test)
print(y_pred)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_pred,y_test)
print(mae)# as we can see that it coes out to be 51.75 that is the difference between observed and predicted value is around 52 which is huge error and need to be minimized
# feature_score tells about which feature contribution towards our model
feature_scores=pd.Series(gbr.feature_importances_,index=x_train.columns).sort_values(ascending=False)
print(feature_scores)
sns.barplot(x=feature_scores,y=feature_scores.index)
plt.xlabel("feature importance score")
plt.show()
# devance is defined as the difference between our model and saturated model(ideal model)
# lower the deviance better the model is while higher eviance means more room for improvement in model
# by checking deviance of different models we can check which fits with the data more nicely.
test_score=np.zeros((100,),dtype=np.float64)
from sklearn.metrics import mean_squared_error
for i , y_pred in enumerate(gbr.staged_predict(x_test)):
    test_score[i]=mean_squared_error(y_test,y_pred)# loss_ calculate deviance # we can use mean_absolute_error where we more concerned about small errors and less sensitivity to outliers otherwise for large errors we use mean_squared_error
fig=plt.figure(figsize=(10,10))
plt.subplot(1,1,1)# only one subplot 1 row, 1 column ,1 position
plt.title('Deviance')
plt.plot(np.arange(100)+1,gbr.train_score_,'b-',label='training set deviance')
plt.plot(np.arange(100)+1,test_score,'r-',label='test set deviance')
plt.legend(loc='upper right')
plt.xlabel('boosting iterations')
plt.ylabel('deviance')
fig.tight_layout()
plt.show()# the graph shows that our model needs improvement 
# we can use hyperparameter tunning for improving the working of our model 
# we are using grid search for hyperparameter tunning here
# 
from sklearn.model_selection import GridSearchCV
param=[{'max_depth': list(range(10,15)),'max_features': list(range(1,11)) }]
grid= GridSearchCV(gbr,param,cv=10,scoring = 'neg_mean_absolute_error',n_jobs=-1)
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
y_pred1=grid.predict(x_test)
mae1=mean_absolute_error(y_pred1,y_test)
print(mae1)
# now  its printing 48 which shows better performance than before.
