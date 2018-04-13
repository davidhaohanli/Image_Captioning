from os import listdir
from pickle import load
from keras.preprocessing.text import Tokenizer
import random

trainSize=6000
testSize=100

#train test split
def train_test_split(descriptions):
    ids = list(descriptions.keys())
    ids = sorted(ids)
    random.Random(4).shuffle(ids)
    #print(ids)
    train_id = ids[:trainSize]
    test_id = ids[trainSize:trainSize+testSize]
    return train_id,test_id

#train test descriptions container
def train_test_desc(descriptions,train_id,test_id):
    train_descriptions = dict()
    test_descriptions = dict()
    for img,desc in descriptions.items():
        if img in train_id:
            train_descriptions[img] = desc
        if img in test_id:
            test_descriptions[img] = desc
    return train_descriptions,test_descriptions

#train test features container
def train_test_features(whichModel, filename, train_id,test_id):
    # load all features
    with open(filename,'rb') as f:
        all_features = load(f)
    # split into train test
    train_features = {k: all_features[whichModel][k] for k in train_id}
    test_features = {k: all_features[whichModel][k] for k in test_id}
    return train_features, test_features

# fit a tokenizer and generate a dictionary for index word mapping
def tokenize(descriptions):
    lines = list(descriptions.values())
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    idx2word = {val:key for key,val in tokenizer.word_index.items()}
    return tokenizer, idx2word