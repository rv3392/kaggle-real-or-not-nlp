import numpy
import pandas
from sklearn import feature_extraction, model_selection, preprocessing
from keras import models, layers, optimizers
from matplotlib import pyplot

import string
import re

TRAINING_DATA = "train.csv"
TESTING_DATA = "test.csv"

def load_data():
    train = pandas.read_csv(TRAINING_DATA)
    test = pandas.read_csv(TESTING_DATA)

    return train, test

def preprocess(train_dataframe, test_dataframe):
    vectorizer = feature_extraction.text.CountVectorizer()
    scaler = preprocessing.MinMaxScaler(feature_range = (0,1))

    train_dataframe['text'] = train_dataframe['text'].apply(lambda text: str(text).lower())
    test_dataframe['text'] = test_dataframe['text'].apply(lambda text: str(text).lower())

    print(train_dataframe['location'])

    train_dataframe['location'] = train_dataframe['location'].apply(lambda text: str(text).lower())
    test_dataframe['location'] = test_dataframe['location'].apply(lambda text: str(text).lower())

    print(train_dataframe)

    train_dataframe['text'] = train_dataframe['text'].str.replace(
            '[{}]'.format(string.punctuation),'')

    test_dataframe['text'] = test_dataframe['text'].str.replace(
            '[{}]'.format(string.punctuation),'')

    train_dataframe['location'] = train_dataframe['location'].str.replace(
            '[{}]'.format(string.punctuation),'')

    test_dataframe['location'] = test_dataframe['location'].str.replace(
            '[{}]'.format(string.punctuation),'')

    print(train_dataframe)

    #Convert text to a word count vector.
    train_matrix_text = vectorizer.fit_transform(train_dataframe['text'])
    test_matrix_text = vectorizer.transform(test_dataframe['text'])

    train_matrix_location = vectorizer.fit_transform(train_dataframe['location'])
    test_matrix_location = vectorizer.transform(test_dataframe['location'])

    train_matrix = numpy.concatenate((train_matrix_text.todense(), train_matrix_location.todense()), axis = 1)
    test_matrix = numpy.concatenate((test_matrix_text.todense(), test_matrix_location.todense()), axis = 1)

    print(train_matrix.shape)
    print(test_matrix.shape)

    #Normalise word counts to between 0 and 1 where 0 means none of the words
    #in the tweet are the given word and 1 means all words in the tweet are
    train_matrix = numpy.asmatrix(scaler.fit_transform(train_matrix))
    test_matrix = numpy.asmatrix(scaler.fit_transform(test_matrix))

    return train_matrix, test_matrix

def train(data, targets):
    #Resize data matrix to have 3 dimensions 
    #([Samples, No. Character, Features])
    data = data[:, numpy.newaxis, :]

    targets = targets.transpose()

    print(data.shape)
    print(targets.shape)

    optimiser = optimizers.Adam(lr = 0.00001)
    training_model = models.Sequential()

    training_model.add(layers.LSTM(128, input_shape = (1, data.shape[2]), 
            return_sequences = True))
    training_model.add(layers.LSTM(64))
    training_model.add(layers.Dense(1, activation = 'linear'))
    training_model.compile(loss = 'mean_squared_error', optimizer = optimiser, metrics = ['accuracy'])

    training_history = training_model.fit(data, targets, epochs = 10, 
            validation_split = 0.1, batch_size = 10)

    return training_model, training_history

def test():
    pass

def plot_diagnostics(history):
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

train_dataframe, test_dataframe = load_data()

train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)
test_dataframe = test_dataframe.sample(frac=1).reset_index(drop=True)

train_matrix, test_matrix = preprocess(train_dataframe, test_dataframe)

model, training_history = train(train_matrix, numpy.asmatrix(train_dataframe['target']))
plot_diagnostics(training_history)






