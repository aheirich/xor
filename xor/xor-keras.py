import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense

# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, nb_epoch=500, verbose=2)

print model.predict(training_data).round()

w01 = model.layers[0].get_weights()[0]
b01 = model.layers[0].get_weights()[1]
w12 = model.layers[1].get_weights()[0]
b12 = model.layers[1].get_weights()[1]
print "input weights to hidden layer"
print w01
print "hidden layer biases"
print b01
print "hidden weights to output layer"
print w12
print "output layer biases"
print b12
