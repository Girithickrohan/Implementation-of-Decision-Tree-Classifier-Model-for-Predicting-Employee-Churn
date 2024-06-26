# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas and read the csv file.
2.Import Decision tree classifier.
3.Fit the data in the model
4.Find the accuracy score. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: GIRITHICK ROHAN N
RegisterNumber:  212223230063
*/

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
<b>Initial data set:</b>

![image](https://github.com/prasannavenkat01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150702500/6a63f780-73eb-40c8-aa0e-ace8d7347028)

<b>Data info </b>

![image](https://github.com/prasannavenkat01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150702500/e0f67af6-a6f7-4f0b-a431-b04241ea3507)

<b>Optimization of null values</b>

![image](https://github.com/prasannavenkat01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150702500/e2361c73-3939-44d0-a556-4b4111e3cb78)

<b>Assignment of x and y values</b>

![image](https://github.com/prasannavenkat01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150702500/bf2f93ad-6767-420a-8289-a092b02a0369)

![image](https://github.com/prasannavenkat01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150702500/fa0f0dd0-0b84-4996-9e6f-aea7863075af)


<b>Converting string literals to numerical values using label encoder </b>

![image](https://github.com/prasannavenkat01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150702500/b5d31c97-80c6-4bf6-a16b-94e9fbebe49a)

<b>Accuracy</b>

![image](https://github.com/prasannavenkat01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150702500/22aec650-d46c-4653-b7c5-1b9a33403fca)

<b>Prediction</b>

![image](https://github.com/prasannavenkat01/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150702500/11b978cb-24db-441e-b3d6-fb33403bb1ef)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
