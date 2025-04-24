import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("heart.csv")
categorical_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
le = LabelEncoder()
data[categorical_columns] = data[categorical_columns].apply(le.fit_transform)


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.theta = None  # Вектор весов

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros((n, 1))  # Инициализируем веса нулями

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.theta)  # Предсказание
            gradient = (1 / m) * np.dot(X.T, (y_pred - y))  # Градиент
            self.theta -= self.lr * gradient  # Обновление весов

    def predict(self, X):
        return np.dot(X, self.theta)


X = data.drop(columns=["Cholesterol"])  # Все столбцы кроме Cholesterol
y = data["Cholesterol"].values.reshape(-1, 1)

X = X.values

# Добавляем столбец единиц для учета свободного коэффициента (b0)
X = np.c_[np.ones(X.shape[0]), X]

# Разделяем на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Делаем предсказания
y_pred = model.predict(X_test)

# Оцениваем модель (Mean Squared Error)
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse}")