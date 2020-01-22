import numpy
import pandas
from sklearn import feature_extraction, model_selection, preprocessing
from keras import models, layers, optimizers
from matplotlib import pyplot

TRAINING_DATA = "train.csv"
TESTING_DATA = "test.csv"

def load_data():
    train = pandas.read_csv(TRAINING_DATA)
    test = pandas.read_csv(TESTING_DATA)

    return train, test

def preprocess(train_dataframe, test_dataframe):
    vectorizer = feature_extraction.text.CountVectorizer()
    scaler = preprocessing.MinMaxScaler(feature_range = (0,1))

    #Convert text to a word count vector.
    train_matrix = vectorizer.fit_transform(train_dataframe['text'])
    test_matrix = vectorizer.transform(test_dataframe['text'])

    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()

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

    training_model.add(layers.LSTM(16, input_shape = (1, data.shape[2])))
    training_model.add(layers.Dense(1, activation = 'linear'))
    training_model.compile(loss = 'mean_squared_error', optimizer = optimiser, metrics = ['accuracy'])

    training_history = training_model.fit(data, targets, epochs = 50, 
            validation_split = 0.1, batch_size = 5)

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






