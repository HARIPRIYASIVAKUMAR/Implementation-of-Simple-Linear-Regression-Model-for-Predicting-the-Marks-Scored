# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.




## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HARIPRIYA S
RegisterNumber:  212223220029
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

## df.head()

![image](https://github.com/user-attachments/assets/f0058a9d-703d-4b5d-bb1b-aa30aefba437)

## df.tail()

![image](https://github.com/user-attachments/assets/9c84b117-51c4-4664-8203-2d6f6b081ed9)

## Array value of X 

![image](https://github.com/user-attachments/assets/d54027f0-86d8-411b-8b06-44535f5245ec)

## Array value of Y

![image](https://github.com/user-attachments/assets/6959feb4-71de-4f9f-a8d2-2bfba81d72f8)

## Values of Y prediction

![image](https://github.com/user-attachments/assets/82e588e3-f65e-4681-a24a-1ca63d915ce5)

## Array values of Y test

![image](https://github.com/user-attachments/assets/800229c3-d7cf-4b74-a9bb-5b475e443c98)

## Training set graph

![image](https://github.com/user-attachments/assets/38ebf752-653f-4a0c-b7da-2071cc713b9c)

## Test set graph

![image](https://github.com/user-attachments/assets/d31abd5f-5208-40a9-91e0-80530aba5f5b)

## Values of MSE,MAE and RMSE

![image](https://github.com/user-attachments/assets/bc17af65-63de-445f-b1b0-d623b31afeb0)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
