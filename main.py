import numpy
import pandas
from sklearn import feature_extraction, model_selection, preprocessing
from keras import models, layers, optimizers
from matplotlib import pyplot

import preprocess

import string
import re

TRAINING_DATA = "data/train.csv"
TESTING_DATA = "data/test.csv"

MODEL_NAME = "models/model.h5"

EPOCHS = 100
BATCH_SIZE = 512
VALIDATION_SPLIT = 0.1

def load_data():
    train = pandas.read_csv(TRAINING_DATA)
    test = pandas.read_csv(TESTING_DATA)

    return train, test

def train(data, targets):
    #Resize data matrix to have 3 dimensions 
    #([Samples, No. Character, Features])
    data = data[:, numpy.newaxis, :]

    targets = targets.transpose()

    print(data.shape)
    print(targets.shape)

    optimiser = optimizers.Adam(lr = 0.001)
    training_model = models.Sequential()

    training_model.add(layers.LSTM(1, return_sequences = True))
    training_model.add(layers.LSTM(1, return_sequences = True))
    training_model.add(layers.LSTM(1, return_sequences = True))
    training_model.add(layers.LSTM(1, return_sequences = True))
    training_model.add(layers.LSTM(1))
    training_model.add(layers.Dense(1, activation = 'sigmoid'))
    training_model.compile(loss = 'binary_crossentropy', optimizer = optimiser, 
        metrics = ['accuracy'])

    training_history = training_model.fit(data, targets, epochs = EPOCHS, 
            validation_split = VALIDATION_SPLIT, batch_size = BATCH_SIZE)

    training_model.save(MODEL_NAME)

    return training_model, training_history

def test(training_model, test_matrix):
    test_matrix = test_matrix[:, numpy.newaxis, :]

    return training_model.predict(test_matrix)

def plot_diagnostics(history):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

train_dataframe, test_dataframe = load_data()

print(train_dataframe)
print(test_dataframe)

train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)

vectorizer = feature_extraction.text.TfidfVectorizer()
scaler = preprocessing.MinMaxScaler(feature_range = (0,1))

train_matrix = preprocess.preprocess(train_dataframe, vectorizer, scaler)
test_matrix = preprocess.preprocess(test_dataframe, vectorizer, scaler, reset_word_bag = True)

model, training_history = train(train_matrix, numpy.asmatrix(train_dataframe['target']))
plot_diagnostics(training_history)

results = test(model, test_matrix)
results = results.round(0)
results = results.astype(int)

sample = pandas.read_csv("sample_submission.csv")

sample['target'] = results
sample.to_csv("submission.csv", index = False)






