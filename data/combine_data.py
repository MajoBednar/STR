import pandas as pd
import glob
import os
from sklearn.utils import shuffle

"""This script is used to combine the original data splits into one dataset.
It can be done for each language, but also the option 'all' to combine all languages.
The data are also shuffled to ensure randomness."""

# choose the language: afr/eng/esp/hin/mar/pan/all
language = input('Language: ')

if language != 'all':
    # List all CSV files in the directory
    file_paths = glob.glob('datasets_original_splits/' + language + '/*.csv')
else:
    file_paths = glob.glob('datasets_original_splits/*/*.csv')

# Initialize an empty list to store data from all files
combined_data = []

# Read data from each file and append to the combined_data list
for file_path in file_paths:
    data = pd.read_csv(file_path)
    combined_data.append(data)

# Concatenate data from all files into a single dataframe
combined_df = pd.concat(combined_data, ignore_index=True)

# Shuffle the combined dataframe
combined_df_shuffled = shuffle(combined_df)

# Write the shuffled dataframe to a new CSV file
directory = 'datasets_custom_splits/' + language + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

combined_df_shuffled.to_csv(directory + language + '_combined.csv', index=False)
