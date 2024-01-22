import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data = pd.read_csv('/content/creditcard.csv')

credit_card_data.head()

credit_card_data.tail()

credit_card_data.info()

#checking the number of missing values in each column
credit_card_data.isnull().sum()

credit_card_data['Class'].value_counts()

legit = credit_card_data[credit_card_data.Class ==0]
fraud = credit_card_data[credit_card_data.Class ==1]

legit.shape #printing the shape of legit transactions

fraud.shape #printing the shape of fraud transactions

# statistical measure of the data
legit.Amount.describe()

fraud.Amount.describe()

# compare the values for both transactions

credit_card_data.groupby('Class').mean()

# undersampling
# Build a sample dataset containing similar distribution of normal transactions

legit_sample = legit.sample(n=1000)

# concatenating two dataframes

new_dataset = pd.concat([legit_sample, fraud], axis=0)

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

# splitting the data into features and targets (0 and 1)

X = new_dataset.drop(columns = 'Class', axis =1)
Y = new_dataset['Class']

# Split the data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# MOdel training by logistic regressioin

model = LogisticRegression()

# training the logistic regression model wiht training data

model.fit(X_train, Y_train)

# Model evaluation by accuracy score

# accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy on training data : ", training_data_accuracy)

X_test_predictions = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predictions,Y_test)

print("Accuracy on testing data : ", test_data_accuracy)
