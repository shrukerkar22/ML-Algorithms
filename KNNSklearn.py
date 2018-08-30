import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

#Importing the dataset
names=['SepalLength','SepalWidth','PetalLength','PetalWidth','Class']
dataset=pd.read_csv("iris.data",names=names)
print(dataset)
print(dataset.head())

#Data preprocessing

x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,4].values
print(y)

#Train test split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20)
print(X_train)
print(Y_train)

#Feature Scaling

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#Training and making predictions

classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)
print(y_pred)

#Evaluating the algorithm

print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))

#Compare error rate with K-value
error=[]

#calculating K values between 1 to 40
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    pred= knn.predict(X_test)
    error.append(np.mean(pred!=Y_test))

#plot error values
plt.figure(figsize=(12,6))
plt.plot(range(1,40),error,color='red',linestyle='--',marker='o',markerfacecolor='blue',markersize=10)
plt.title('Error Rate K value')
plt.xlabel('K value')
plt.ylabel('Mean Error')
plt.show()