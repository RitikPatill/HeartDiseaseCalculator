import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv')

df.head()


print(df.head())

print('hello')

print(df.tail())

print(df.shape)


print(df.describe())

# 0 for people who will not get heart attack and vice versa 

print(df['target'].value_counts())\
    
x=df.drop(columns='target',axis=1)
y=df['target']

#splitting train test data 

x_train,x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,stratify=y,random_state=23)

print(x_test.shape, x_train.shape)
print(y_test.shape,y_train.shape)

# model training  we are using logistic regression

model =LogisticRegression()

# Training the model with training data

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_train)

print(accuracy_score(y_test,y_pred))

# Bulding a predictive system 

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# changing the input data into numpy array

arr = np.array(input_data)

prediction =model.predict(arr.reshape(1,-1)) 


# if 0 then you are healthy 1 you are not
print(prediction)