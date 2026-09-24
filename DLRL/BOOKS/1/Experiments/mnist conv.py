from keras.datasets import mnist
from keras import layers, Sequential

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(10000, 28, 28, 1).astype("float32") / 255.0

model = Sequential(
    [
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")