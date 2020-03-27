# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:14:52 2020

@author: ugeure
"""
#importing lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data importing
data=pd.read_csv("knn4.csv")
Normal_person=data[data.class1=="Normal"]
Sick_person=data[data.class1=="Abnormal"]

#info about data
#data.info()

#plotting data features
#plt.scatter(Normal_person.pelvic_tilt_numeric,Normal_person.degree_spondylolisthesis,color="green",label="Normal")
#plt.scatter(Sick_person.pelvic_tilt_numeric,Sick_person.degree_spondylolisthesis,color="red",label="Sick")

data.class1=[0 if each=="Normal" else 1 for each in data.class1]

#takeing data out
y=data.class1
x_data=data.drop(["class1"],axis=1)

#normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

#data split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)

# modelling data fimport KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
score=[] #♦ to keep the best K value
for each in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=each)
    knn.fit(x_train,y_train)
    prediction = knn.predict(x_test)
    score.append(knn.score(x_test,y_test))

#from the graph we can easily see our best K value
plt.plot(range(1,40),score,color="pink")
plt.xlabel("K Value")
plt.ylabel("Score")
plt.legend()
plt.show()