import pandas as pd

"""This script is used to generate train/dev/test splits based on the combined (and shuffled) data."""

# choose the language: afr/eng/esp/hin/mar/pan/all
language = input('Language: ')

# Read the combined and shuffled data from the previously created file
directory = 'datasets_custom_splits/' + language + '/'
combined_df = pd.read_csv(directory + language + '_combined.csv')

# Define proportions for train, development, and test sets
train_split = 0.7  # 70% of data for training
dev_split = 0.1    # 10% of data for development (validation)
test_split = 0.2   # 20% of data for testing

# Calculate the sizes of each split
num_samples = len(combined_df)
num_train = int(train_split * num_samples)
num_dev = int(dev_split * num_samples)

# Split the data into train, dev, and test sets
train_data = combined_df.iloc[:num_train]
dev_data = combined_df.iloc[num_train:num_train + num_dev]
test_data = combined_df.iloc[num_train + num_dev:]

# Save each split into separate files
train_data.to_csv(directory + language + '_train.csv', index=False)
dev_data.to_csv(directory + language + '_dev.csv', index=False)
test_data.to_csv(directory + language + '_test.csv', index=False)
