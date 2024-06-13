import pandas as pd
import nltk
import random
nltk.download('punkt')

"""This script augments a (low-resource) training set with sentences from a chosen file.
Each pair consist of two equivalent sentences with a relatedness score of 1.
The number of new pairs is equal to the number of pairs in the original training set."""

# choose the language: afr/esp/hin/mar/pan, and the file for new sentences
language = input('Language: ')
directory = language + '/'
file = directory + input('Name of the file with new sentences: ')

# read the original training set (and remove ids)
original_training_set = pd.read_csv(directory + language + '_train.csv')
original_training_set = original_training_set.drop(columns=['PairID'])

# get the number of samples in the training set to be used for the number of new samples
num_samples = len(original_training_set)

# open the file with new sentences
with open(file, encoding='utf-8') as f:
    # get sentences from 10 to 20 words
    usable_sentences = []
    for line in f:
        print(line)


print(num_samples)
print(original_training_set)
