import random

import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def predict(x, weight):
    return x * weight

def mse_loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def optimizer(x, y, y_pred):
    return (-2 / len(x)) * sum(x * (y - y_pred))


def train(x, y, iterations=5, learning_rate=0.013):
    weight = random.random()
    for i in range(iterations):
        y_pred = predict(x, weight)
        mse_loss_ = mse_loss(y, y_pred)
        gradient = optimizer(x, y, y_pred)
        weight -= learning_rate * gradient
        print(f"Iteration {i}: Loss = {mse_loss_:.4f}, Weight = {weight:.4f}")
    return round(weight, 4)



w = train(x, y)
x_test = np.array([2, 5, 10, 11, 15, 20, 50, 100])
y_predicted = predict(x_test, w)

print("\nFinal Weight:", w)
print("Predictions:\n", y_predicted)