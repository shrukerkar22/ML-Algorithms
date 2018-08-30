import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#load dataset
dataset=pd.read_table('/home/shruti/Downloads/SMSSpamCollection',sep='\t',header=None,names=['label','message'])
print(dataset)

#data preprocessing
dataset['label']=dataset.label.map({'ham':0,'spam':1})
dataset['message']=dataset.message.map(lambda x:x.lower())
dataset['message']=dataset.message.str.replace('[^\w\s]','')

nltk.download()
dataset['message']=dataset['message'].apply(nltk.word_tokenize)
stemmer = PorterStemmer()
dataset['message']=dataset['message'].apply(lambda x:[stemmer.stem(y) for y in x])
dataset ['message']=dataset['message'].apply(lambda x:''.join(x))
cv=CountVectorizer()
counts=cv.fit_transform(dataset['message'])

transformer=TfidfTransformer.fit(counts)
counts=transformer.transform(counts)

X_train,X_test,Y_train,Y_test= train_test_split(counts,dataset['label'],test_size=0.2,random_state=69)
model=MultinomialNB().fit(X_train,Y_train)

#Evaluating model
pred=model.predict(X_test)
print(np.mean(pred ==Y_test))

print(confusion_matrix(Y_test,pred))




