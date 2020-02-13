import pandas
import numpy
from tensorflow.python.keras import models, layers
from matplotlib import pyplot


def train(training_data, targets, epochs, batch_size, validation_split, hidden_neurons=5):
    # Resize data matrix to have 3 dimensions
    training_data = training_data[:, numpy.newaxis, :]
    targets = targets.transpose()

    training_model = models.Sequential()
    training_model.add(layers.LSTM(hidden_neurons, return_sequences=True))
    training_model.add(layers.LSTM(hidden_neurons, return_sequences=True))
    training_model.add(layers.LSTM(hidden_neurons, return_sequences=True))
    training_model.add(layers.LSTM(hidden_neurons, return_sequences=True))
    training_model.add(layers.LSTM(hidden_neurons))
    training_model.add(layers.Dense(1, activation='sigmoid'))

    training_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    training_history = training_model.fit(training_data, targets, epochs=epochs, batch_size=batch_size,
                                          validation_split=validation_split)

    return training_model, training_history

def test(training_model, test_matrix):
    test_matrix = test_matrix[:, numpy.newaxis, :]

    return training_model.predict(test_matrix)

def plot_training_diagnostics(history):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()