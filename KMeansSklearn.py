import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster,preprocessing


dataset=pd.read_excel('/home/shruti/Downloads/titanic.xls')
dataset.drop(['body','name'],1,inplace=True)
dataset.convert_objects(convert_numeric=True)

#repalce nan with 0
dataset.fillna(0,inplace=True)

print(dataset.head())

#Convert non-numerical values to numerical
columns=dataset.columns.values

for i in columns:
    text_value_int={}
    def text_to_val(val):
        return text_value_int[val]
    if dataset[i].dtype != np.int64  and dataset[i].dtype != np.float64:
        unique_elements=set(dataset[i].values.tolist())

        x=0
        for unique in unique_elements:
            if unique not in text_value_int:
                text_value_int[unique]=x
                x+=1
        text=map(text_to_val,dataset[i])
        dataset[i]=list(text)
dataset.head()

x=np.array(dataset.drop('survived',1))
print(x)
y=np.array(dataset['survived'])
print(y)

clf=cluster.KMeans(n_clusters=2)
clf.fit(x)

found=0
for i in range(len(x)):
    pred=np.array(x[i].astype(float))
    pred=pred.reshape(-1,len(pred))
    prediction=clf.predict(pred)
    if prediction[0]==y[i]:
        found=+1

accuracy =(found/len(x))*100
print(accuracy)

x=preprocessing.scale(x)
clf.fit(x)
found=0
for i in range(len(x)):
    pred=np.array(x[i].astype(float))
    pred=pred.reshape(-1,len(pred))
    prediction=clf.predict(pred)
    if prediction[0]==y[i]:
        found=+1
accuracy =(found/len(x))*100
print(accuracy)