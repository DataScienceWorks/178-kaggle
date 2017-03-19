import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

from keras.models import Sequential
from keras.layers import Dense, Activation

# Load the data
X = np.genfromtxt("data/X_train.txt", delimiter=None)
Y = np.genfromtxt("data/Y_train.txt", delimiter=None)
Xte = np.genfromtxt("data/X_test.txt", delimiter=None)

X, Y = ml.shuffleData(X, Y)
Xtr = X[:10000]
Ytr = Y[:10000]

Xva = X[10000:20000]
Yva = Y[10000:20000]


# Simple feed-forward architecture
model = Sequential()
model.add(Dense(units=10000, input_dim=14))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# Optimize with SGD
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Fit model in batches
model.fit(Xtr, Ytr, epochs=5, batch_size=32)

# Evaluate model
classes = model.predict(Xte, batch_size=128)

print(classes)
