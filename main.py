from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv('Python_Project\\Diabetespred.csv')
data.head()

X = data.drop('Outcome', axis = 1)
Y = data['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state= 125)

model = GaussianNB()

model.fit(X_train, Y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(Y_test, prediction)

print(f"Dokładność modelu: {accuracy * 100:.2f}%")
