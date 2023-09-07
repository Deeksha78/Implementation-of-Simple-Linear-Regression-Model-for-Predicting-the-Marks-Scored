# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Deeksha P
RegisterNumber:  212222040031
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="darkseagreen")
plt.plot(x_train,regressor.predict(x_train),color="plum")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="darkblue")
plt.plot(x_test,regressor.predict(x_test),color="plum")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:

df.head()


![exp2 img1](https://github.com/Deeksha78/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116204/24c310fb-96bc-4f1a-a17a-992f8c3c0b5b)

df.tail()


![exp2 img2](https://github.com/Deeksha78/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116204/fa868322-408f-4371-ab82-681034fa82d1)


Array value of X

![exp2 img3](https://github.com/Deeksha78/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116204/add1e776-6061-4cd2-a176-3521d86f9715)



Array value of Y

![exp2 img4](https://github.com/Deeksha78/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116204/246f41fb-9dcb-4c54-87f4-b3ed5ee1800e)


Array values of Y test

![exp2 img5](https://github.com/Deeksha78/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116204/f18ea894-fcc6-48d8-8e00-da04a65e1b69)


Training Set Graph


![exp2 img6](https://github.com/Deeksha78/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116204/400e7f15-b52b-40e4-9a02-dc08d81434ff)


Test Set Graph


![exp2 img7](https://github.com/Deeksha78/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116204/f4e4910f-1e88-4de8-b988-9304f96d5327)

Values of MSE, MAE and RMSE

![exp2 img8](https://github.com/Deeksha78/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128116204/afa7fd20-2356-4b38-bc9c-0293c545e92c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
