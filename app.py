import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



data = pd.read_csv('AirQualityUCI.csv', sep=";")

data.head()

data.isnull().sum()


num_cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

print("Null values replaced with median")


cat_cols = data.select_dtypes(include=['object']).columns

for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])



data.isnull().sum()

data = data.drop(columns=(['Unnamed: 16','Unnamed: 15']))


data.head()


le = LabelEncoder()

categorical_columns = [
    "Date",
    "Time",
    "CO(GT)",
    "C6H6(GT)",
    "T",
    "RH",
    "AH"

]

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])


data.head()


data.hist(bins=50, figsize=(15,8))
plt.show()

X = data.drop(['CO(GT)','PT08.S1(CO)', 'NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)'], axis=1)
Y = data['CO(GT)']




print(X)
print(Y)

X_train, X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

SVC = SVC()
SVC.fit(X_train,Y_train)

train_data = SVC.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Accuracy on training data:", training_data_accuracy)

test_data = SVC.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Accuracy on testing data:", testing_data_accuracy)


RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

train_data = RandomForestClassifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Accuracy on training data:", training_data_accuracy)


test_data = RandomForestClassifier.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Accuracy on testing data:", testing_data_accuracy)

Lg = LogisticRegression()
Lg.fit(X_train, Y_train)


train_data = Lg.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Accuracy on training data:", training_data_accuracy)


test_data  = Lg.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Accuracy on testing data:", testing_data_accuracy)


Dd = DecisionTreeClassifier()
Dd.fit(X_train, Y_train)

train_data = Dd.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, train_data)
print("Accuracy on training data:", training_data_accuracy)

test_data = Dd.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, test_data)
print("Accuracy on testing data:", testing_data_accuracy)

