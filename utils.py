import re
import pandas

def remove_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return re.sub(url_pattern, '', text)

def load_data(training_data_file, testing_data_file):
    train = pandas.read_csv(training_data_file)
    test = pandas.read_csv(testing_data_file)

    return train, test

def save_results(results):
    sample = pandas.read_csv("sample_submission.csv")
    sample['target'] = results
    sample.to_csv("submission.csv", index=False)