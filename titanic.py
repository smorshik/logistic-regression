import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('titanic.csv', index_col=0, parse_dates=True)
df.head()

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

def age_category(Age):
    if Age < 18:
        return 0**2
    elif Age < 55:
        return 1**2
    else:
        return 2**2
df['Age'] = df['Age'].apply(age_category)
df['Pclass'] = df['Pclass']**2 

df['Fare_Cut'] = pd.cut(df['Fare'], 
                        bins=[-1, 50, 100, 1000], 
                        labels=[0, 1, 2])

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare_Cut']

xtrain, xtest, ytrain, ytest = train_test_split(df[features], df["Survived"], test_size=0.2, random_state=42)



scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model = LogisticRegression()
model.fit(xtrain, ytrain)


pclass = int(input("Класс (1, 2, 3): "))**2
sex = int(input("Пол (0 - мужской, 1 - женский): "))
age = int(input("Возраст: "))
sibsp = int(input("Количество братьев/сестер: "))
parch = int(input("Количество родителей/детей: "))
Fare = float(input("Стоимость билета от 0 до 100: "))

jack = [[pclass, sex, age_category(age), sibsp, parch, Fare]]
jack_scaled = scaler.transform(jack)

prediction = model.predict(jack_scaled)

print("выжил ли?:", prediction)

coeff = pd.DataFrame(model.coef_.T, index=features, columns=['Вес (Weight)'])
print(coeff)
print("Базовый уровень (Intercept):", model.intercept_)
print("Точность модели на тестовых данных:", model.score(xtest, ytest))

ypred = model.predict(xtest)
cm = confusion_matrix(ytest, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Погиб (0)", "Выжил (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.show()