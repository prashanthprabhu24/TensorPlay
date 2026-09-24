import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])


def loss(y, y_pred):
    return sum(y - y_pred)


def predict(x_test, weight):
    return x_test * weight

def train(x, y):
    weight1 = 1
    for i in range(100):
        y_pred = x * weight1
        loss_score = loss(y, y_pred)
        weight1 += loss_score * 0.01
    return weight1

w = train(x, y)
x_test = np.array([2, 5, 10, 11, 15, 20, 50, 100])
y_predicted = predict(x_test, w)
print(y_predicted)

