import pandas

def load_data(training_data, testing_data):
    train = pandas.read_csv(training_data)
    test = pandas.read_csv(testing_data)

    return train, test