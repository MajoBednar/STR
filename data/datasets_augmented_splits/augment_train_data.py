import pandas as pd
import nltk
import random
from sklearn.utils import shuffle

"""This script augments a (low-resource) training set with sentences from a chosen file.
Each new pair consist of two equivalent sentences with a relatedness score of 1.
The number of new pairs is equal to the number of pairs in the original training set."""

try:
    # try to use the punkt tokenizer
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # if not found, download the punkt resource
    nltk.download('punkt')

# choose the language: afr/esp/hin/mar/pan, and the file for new sentences
language = input('Language: ')
directory = language + '/'
file = directory + input('Name of the file with new sentences: ')

# read the original training set (and remove ids)
original_training_set = pd.read_csv(directory + language + '_train.csv')

# get the number of samples in the training set to be used for the number of new samples
num_samples = len(original_training_set)

# find the shortest and longest sentence from the original training set (to limit the length of usable sentences)
sentence_lengths = []
for pair in original_training_set['Text']:
    sentence1, sentence2 = pair.split('\n')
    sentence_lengths.append(len(nltk.word_tokenize(sentence1)))
    sentence_lengths.append(len(nltk.word_tokenize(sentence2)))
min_words, max_words = min(sentence_lengths), max(sentence_lengths)

# open the file with new sentences and filter the ones that fall in the range of appropriate length
with open(file, encoding='utf-8') as f:
    usable_sentences = []
    for line in f:
        line = line.strip()
        if min_words <= len(nltk.word_tokenize(line)) <= max_words:
            usable_sentences.append(line)

# select the appropriate number of sentences from usable sentences
selected_sentences = random.sample(usable_sentences, num_samples)

# duplicate the sentences and separate them with newline
new_sentence_pairs = [sentence + '\n' + sentence for sentence in selected_sentences]

# create a new DataFrame from the selected sentences with a score of 1
new_data = {
    'PairID': [f'{language.upper()}-aug-{i}' for i in range(len(new_sentence_pairs))],
    'Text': new_sentence_pairs,
    'Score': [1] * len(new_sentence_pairs)
}
new_df = pd.DataFrame(new_data)

# concatenate the new DataFrame with the original DataFrame
combined_df = pd.concat([original_training_set, new_df], ignore_index=True)

# shuffle the new training set
combined_df = shuffle(combined_df)

# replace the new training set
combined_df.to_csv(directory + language + '_train.csv', index=False)
