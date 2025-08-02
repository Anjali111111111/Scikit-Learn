#linear regression
# Linear regression is a way to find the relationship between two things — one that you want to predict (like someone's height) and one or more things you already know (like their age).

#It draws a straight line through your data to help make predictions.

#Beginner Example:
#Let’s say you want to predict a student’s exam score based on how many hours they studied.

#You collect this data:


#Hours Studied	Exam Score
#1	50
#2	55
#3	65
#4	70
#5	80
#Now, you can draw a straight line that best fits these points. This line can be used to predict future scores.

#So if someone studies for 6 hours, linear regression might predict:

#Score ≈ 90

#That’s linear regression! 
#It found the pattern between hours studied and exam score, then used that pattern to make predictions.

#difference between regrssion and classification 
#    REGRESSION                                                                  CLASSIFICATION
# What it predicts	A continuous value (e.g., price, age)	    |      A category/label (e.g., cat or dog, pass/fail)
# Example	Predicting house prices, temperature	            |      Predicting if an email is spam or not
# Output	Any number	                                        |      A class/label (like 0 or 1, "yes" or "no")
# Algorithms used	Linear Regression, Ridge, Lasso, etc.       |      Logistic Regression, Decision Tree, SVM, etc.
# Metric examples	MSE, RMSE, R² Score	                        |      Accuracy, Precision, Recall, F1 Score
#  Real-world Example:
#  Regression:
# You're trying to predict house prices based on size, location, and number of bedrooms.
# Output: $354,000, $520,000, etc.
#
# Classification:
# You're building an email filter that tells whether an email is spam or not.
# Output: Spam or Not Spam (or 1/0)









from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
data=fetch_california_housing(as_frame=True)
print(data)
x=data.data
y=data.target
print(x.shape)
print(y.shape)
#plt.scatter(x.iloc[:,0],y)
#plt.show()
#plt.scatter(x.iloc[:,1],y)
#plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
l_reg=linear_model.LinearRegression()
model=l_reg.fit(x_train,y_train)
prediction=model.predict(x_test)
print(prediction)
plt.scatter(prediction,y_test)
plt.show()
print("Accuracy R^2",l_reg.score(x,y))
#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,prediction))
#we dont use this method because it is made for classification preocess and not regression
print("Regression coefficints",l_reg.coef_)
print("y intercept",l_reg.intercept_)

