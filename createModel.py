import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

count_truth = 139
count_lie = 122
max_timespan = 1265

X = np.zeros(shape = (count_truth + count_lie, max_timespan, 37)) # input matrix: samples X timesteps X total features
y = np.zeros(shape = (count_truth + count_lie)) # expected output matrix: samples

# loading input matrix and expected output
for x in range(0, count_truth, 1):
    fname = "Truth_" + str(x)
    arr = np.genfromtxt(f'/content/drive/MyDrive/Processed/{fname}.csv', delimiter = ',')
    arr = np.concatenate((np.zeros(shape = (max_timespan - arr.shape[0], 37)), arr), axis = 0)
    X[x - 1] = arr
    y[x - 1] = 1

for x in range(0, count_lie, 1):
    fname = "Lie_" + str(x)
    arr = np.genfromtxt(f'/content/drive/MyDrive/Processed/{fname}.csv', delimiter = ',')
    arr = np.concatenate((np.zeros(shape = (max_timespan - arr.shape[0], 37)), arr), axis = 0)
    X[x + 59] = arr
    y[x + 59] = 0

X = np.array(X, dtype = "float64")
y = np.array(y, dtype = "float64")

# splitting data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# building the model
model = Sequential()
model.add(LSTM(units=64, input_shape=(None, 37)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 30
batch_size = 16

# training the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
model.save('lstm_model.h5')

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')