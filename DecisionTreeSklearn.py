import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn import metrics
import matplotlib as pyplot
#---------------------DecisionTreeClassifier-------------------------#
#importing dataset

dataset=pd.read_csv("/home/shruti/Downloads/data_banknote_authentication.csv")

#data analysis

print(dataset.shape)
print(dataset.head())

#preparing the data

x=dataset.drop('class',axis=1)
print(x)
y=dataset['class']
print(y)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20)
print(X_train)
print(Y_train)

#training and making predictions

classifier=DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
print(Y_pred)

#evaluating the algorithm

print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


#----------DecisionTreeRegression-----------------#
#importing dataset
data=pd.read_csv('/home/shruti/Downloads/petrol_consumption.csv')
print(data)

#data analysis
print(data.shape)
print(data.head())
print(data.describe)

#data preparation
X=data.drop('Petrol_Consumption',axis=1)
print(x)
Y=data['Petrol_Consumption']
print(y)
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.20,random_state=0)
print(train_X)
print(train_Y)

#training and making predictions
regressor=DecisionTreeRegressor()
print(regressor.fit(train_X,train_Y))
y_pred=regressor.predict(test_X)
print(y_pred)

#compare predicted value with actual value
df=pd.DataFrame({'Actual':test_Y,'Predicted':y_pred})
print(df)

#Evaluating algorithm
print('Mean absolute error:',metrics.mean_absolute_error(test_Y,y_pred))
print('Mean squared error:',metrics.mean_squared_error(test_Y,y_pred))
print('Root mean squared error:',np.sqrt(metrics.mean_squared_error(test_Y,y_pred)))

