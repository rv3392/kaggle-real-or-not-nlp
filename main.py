import numpy
from sklearn import feature_extraction, preprocessing

import preprocess
import model
import utils

TRAINING_DATA = "data/train.csv"
TESTING_DATA = "data/test.csv"

MODEL_NAME = "models/model.h5"

EPOCHS = 25
BATCH_SIZE = 600
VALIDATION_SPLIT = 0.1

train_dataframe, test_dataframe = utils.load_data(TRAINING_DATA, TESTING_DATA)

print(train_dataframe)
print(test_dataframe)

#Shuffle training data
train_dataframe = train_dataframe.sample(frac=1).reset_index(drop=True)

#Preprocessing
vectorizer = feature_extraction.text.TfidfVectorizer()
scaler = preprocessing.MinMaxScaler(feature_range = (0,1))

training_data = preprocess.preprocess(train_dataframe, vectorizer, scaler)
testing_data = preprocess.preprocess(test_dataframe, vectorizer, scaler, reset_word_bag=True)

#Training
targets = numpy.asmatrix(train_dataframe['target'])
trained_model, training_history = model.train(training_data=training_data, targets=targets, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, hidden_neurons=5)
model.plot_training_diagnostics(training_history)

#Testing
results = model.test(trained_model, testing_data)
results = results.round(0)
results = results.astype(int)

#Output
utils.save_results(results)






