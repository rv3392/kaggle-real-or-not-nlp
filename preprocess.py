import re
import numpy
import string
import utils
from sklearn import feature_extraction, preprocessing

def preprocess(data, vectorizer, scaler, reset_word_bag=False):
    data = clean_text(data)
    data = vectorize_text(vectorizer, data, reset_word_bag)
    data = data.todense()
    data = scale_data(scaler, data)
    data = numpy.array(data)

    return data

def clean_text(data):
    data['text'] = data['text'].apply(lambda text: str(text).lower())
    data['text'] = data['text'].apply(lambda text: utils.remove_url(text))
    data['location'] = data['location'].apply(lambda text: str(text).lower())
    data['text'] = data['text'].str.replace(
        '[{}]'.format(string.punctuation), '')
    data['location'] = data['location'].str.replace(
        '[{}]'.format(string.punctuation), '')

    return data

def vectorize_text(vectorizer, data, reset_word_bag):
    if reset_word_bag == False:
        data = vectorizer.fit_transform(data['text'])
    else:
        data = vectorizer.transform(data['text'])
    return data

def scale_data(scaler, data):
    return scaler.fit_transform(data)