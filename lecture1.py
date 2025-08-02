import sklearn
# load_iris is a dataset in sklearn
from sklearn.datasets import load_iris
# load_iris is a dataset in sklearn
#x,y are the data sample
data = load_iris()
x = data.data
y = data.target

print(data)      # To see the full dataset details
print(x)         # Features
print(y)         # Target labels

from sklearn.linear_model import LinearRegression
model = LinearRegression()
#here i have created an object named model on which i have assigned my model linear 
#regression. the next thing we have to do before predicting our model is that we have to 
#fit our data into the model
model.fit(x,y)
print(model.predict(x))
from sklearn.neighbors import KNeighborsRegressor
model2= KNeighborsRegressor()
model2.fit(x,y)
print(model2.predict(x))
import matplotlib.pyplot as plt
pred=model2.predict(x)
plt.scatter(pred,y)
plt.show()
import pandas as pd
from sklearn.datasets import fetch_openml
#The fetch_openml function from sklearn.datasets is used to download datasets from the OpenML 
# repository—a popular online platform for sharing datasets, especially for machine learning tasks
#a repository is a place where things are stored and organized, especially data, code
#fetch_openml() lets you access a wide variety of datasets without having to manually download 
#or preprocess them. Once fetched, the datasets can be used directly in your ML models.EXAMPLE:
#from sklearn.datasets import fetch_openml

# Load a well-known dataset like MNIST
#mnist = fetch_openml('mnist_784', version=1)

# Access the features and labels
#X, y = mnist.data, mnist.target

df =fetch_openml('titanic',version=1,as_frame=True)['data']
#downloading the Titanic dataset from OpenML and storing its features (data) in a variable called df.
# Select only the data (features), excluding the target/label
#as_frame: If True, returns a pandas DataFrame instead of NumPy arrays (default is False)
print(df.info)
print(df.isnull().sum())
import seaborn as sns
sns.set()
#missing values percentage find 
miss_value_per = pd.DataFrame((df.isnull().sum()//len(df))*100)
miss_value_per.plot(kind="bar",title="missing values in percentage",ylabel="percentage")
plt.show()
print(f'size of datset:{df.shape}')
# so basically we see that body has many nan values but we have to provide appropriate dat for 
# our model to learn properly , model can make blunt predecition thus we have to find resolution to add something in place of 
# these nan values or have to drop the entire feature so it wouldnt affect the learning adversely 1. one way 
# for this is dropping
df.drop(['body'],axis=1,inplace = True)
print(f'after dropping a column body from dataset,size is:{df.shape}')
# however dropping is never favoured since by eliminating a column we are losing one factor that 
# produces the outcome or contribute to the output. this is will eventually lead to generalization
# error 
#value imputation 
# Simpleimputer is a very convenient strategy for missing data imputation , it replaces all missing 
# values with statistic created from other values in a column. Used statistic,mean,median,mode.
from sklearn.impute import SimpleImputer
print(f"the number of nan values before imputing :{df.age.isnull().sum()}")
#Argument | Description(in simpleimputer )
#{{  top 3 are :
#1.missing_values | What to consider as missing. Default is np.nan, but you can set it to something like 0 or 'NA'.
#2. strategy | How to fill in the missing values. Options: 'mean', 'median', 'most_frequent', or 'constant'.
#3.fill_value | Used when strategy='constant'. This value will replace missing entries.
#others are :
# verbose | Deprecated. Previously controlled verbosity level.
#copy | Whether to make a copy of the data (True) or change it in-place (False).
#keep_empty_features | If True, features with all missing values are kept. Default is False.
#add_indicator | If True, it adds extra columns to indicate which values were missing. Useful for keeping track of missingness as a feature.
imp=SimpleImputer(strategy='mean')
df['age']=imp.fit_transform(df[['age']])# fit_transform will fit oyr data to imputer and will generate the values to replace the nan values
print(f'number of nan values in age after imputation:{df.age.isnull().sum()} ')# we can see it become zero i.e, the nan values are imputed 
# our dataset can contain diffrent data tupe values . some even are strings and computations like mean , mode, media do not work upon strings . hence we need to program a separate function which allows us to understand what kind of datatypes do we have inside our dataset 
def get_parameters(df):
    parameters={}
    #now we'll run a for loop over the column of data set 
    for col in df.columns [df.isnull().any()]:
        if df[col].dtype== 'float64' or df[col].dtype=='int64' or df[col].dtype=='int32':
            strategy='mean'
        else:
            strategy='most_frequent'
# when we work with mix data type the columns turne dinto nd arrays in broadast upcasting that why we create a separate variable for missing values
        missing_values=df[col][df[col].isnull()].values[0]
        parameters[col]={"missing_values":missing_values,'strategy':strategy}
    return parameters
get_parameters(df)#return a dicitionary with missing value type like nan, none and stratety for corresponding missing value like mean, most_frequents
parameters=get_parameters(df)#we again create an variable named parameters it doesn't show any error because it outside the local scope of earlier one 
for col, param in parameters.items():
    missing_values=param['missing_values']
    strategy=param['strategy']
    imp=SimpleImputer(missing_values=missing_values,strategy=strategy)
    df[col]=imp.fit_transform(df[[col]]).ravel()#here a error is occuring if i dont use ravel() or [:, 0] here because the array we're trying to assign is a 2D array (fit_transform returns a 2D array) to a single column in a pandas DataFrame, which expects a 1D array or Series.
#You need to flatten the 2D result to 1D using .ravel() or [:, 0].
print(df.isnull().sum())
# now we can see the in the output that there is no nan values anymore in dataset.

# next we have feature engineering
df['family']=df['sibsp']+df['parch']
df.loc[df['family']>0,'travelled_alone']=0
df.loc[df['family']==0,'travelled_alone']=0
df['travelled_alone'].value_counts().plot(title='passenger travelled alone?',kind='bar')
plt.show()

