#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv("D:\DVT\heart_disease_data.csv")
x=df.drop(columns='target',axis=1)
y=df['target']
#splitting the data into training data and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=3)
print(x.shape,x_train.shape,x_test.shape)
#model training(logistic regression)
model=LogisticRegression()
model.fit(x_train,y_train)
#accuracy on training dataset
x_train_prediction=model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
print("Accuracy on training data:",training_data_accuracy)
# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)
print("Accuracy on test data:",test_data_accuracy)
#predictive system
input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)
#change the input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("the person does not have a heart disease")
else:
    print("the person have a heart disease")
