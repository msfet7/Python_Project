from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from ownNaive import useOwnModel


data = pd.read_csv('C:\\Users\\mateu\\Desktop\\Python_Project\\Python_Project\\Diabetespred.csv')
data.head()

X = data.drop('Outcome', axis = 1)
Y = data['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 125)

model = GaussianNB()

model.fit(X_train, Y_train)

prediction = model.predict(X_test)

secondPrediction = useOwnModel(X_train, Y_train, X_test)

accuracy = accuracy_score(Y_test, prediction)

secondAccuracy = accuracy_score(Y_test, secondPrediction)

print(f"Dokładność domyślnego modelu: {accuracy * 100:.2f}%")

print(f"Dokładność naszego modelu: {secondAccuracy * 100:.2f}%")
