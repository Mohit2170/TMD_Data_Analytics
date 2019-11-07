# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#dataset importing
dataset=pd.read_csv("dataset_5.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,66].values

#mising data
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
imp=imp.fit(x[:,0:66])
x[:,0:66] = imp.transform(x[:,0:66]) 

#catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_y=LabelEncoder()
y =le_y.fit_transform(y)

#dataset splitting train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2 , random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


#classifier to predict train set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(x_train,y_train)

#yprediction on the y test values
y_pred = classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred) 

#accuracy
from sklearn import metrics
print (metrics.accuracy_score(y_test,y_pred))





