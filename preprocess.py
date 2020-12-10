import tensorflow as tf
import numpy
import pandas
from sklearn.model_selection import train_test_split

# Gutenberg Dataset from: https://www.kaggle.com/terminate9298/gutenberg-poetry-dataset
GUTENBERG_DATASET_FILE = 'data/Gutenberg-Poetry.csv'  # Poetry from Gutenberg Project containing 2703086 Rows of Sentences

# Poetry Foundation Dataset from: https://www.kaggle.com/johnhallman/complete-poetryfoundationorg-dataset
POETRY_FOUNDATION_DATASET_FILE = 'data/kaggle_poem_dataset.csv'  # Dataset containing most poems found on poetryfoundation.org


# QUESTION: Should we stick to a single dataset since there are some inconsistencies?

def split_gutenberg(shuffle=False, test_size=0.2):
    """
    Poetry from Gutenberg projects contains 2703086 Rows of Sentences
    :param shuffle:
    :param test_size:
    :return: 2162468 Training sentences & 540618 Testing sentences
    """
    gutenberg_data = pandas.read_csv(GUTENBERG_DATASET_FILE)
    gutenberg_data = gutenberg_data[['s']].drop_duplicates(subset=['s'])
    if shuffle:
        gutenberg_data = gutenberg_data.sample(frac=1).reset_index(drop=True)
    gutenberg_train, gutenberg_test = train_test_split(gutenberg_data, test_size=test_size)
    # print(gutenberg_data.head())
    # print(gutenberg_train.shape)
    # print(gutenberg_test.shape)
    return gutenberg_train, gutenberg_test


def split_poetry_foundation(shuffle=False, test_size=0.2):
    """
    Entire poems (15638) from poetryfoundation.org
    :param shuffle:
    :param test_size:
    :return: 12510 whole Training poems, 3128 whole Testing poems
    """
    poetry_foundation_data = pandas.read_csv(POETRY_FOUNDATION_DATASET_FILE)
    poetry_foundation_data = poetry_foundation_data[['Content']].drop_duplicates(subset='Content')
    if shuffle:
        poetry_foundation_data = poetry_foundation_data.sample(frac=1).reset_index(drop=True)
    poetry_foundation_train, poetry_foundation_test = train_test_split(poetry_foundation_data, test_size=test_size)
    # print(poetry_foundation_data.head())
    # print(poetry_foundation_train.shape)
    # print(poetry_foundation_test.shape)
    return poetry_foundation_train, poetry_foundation_test


def get_data(train_file, test_file):
    """
    Read and parse train & test file line by line
    Tokenize sentences to build train & test data individually
    Construct Vocab Dictionary - uniques tokens from data as keys to unique integer values
    Vectorize train & test data based on Vocab Dictionary
    :param train_file: Training file path
    :param test_file: Testing file path
    :return:
    (1) Training Words in vectorized ID form
    (2) Testing Words in vectorized ID form
    (3) Map of indices & words "Dictionary"
    """
    train_ids, test_ids, vocabulary_dict = [], [], {}
    gutenberg_train, gutenberg_test = split_gutenberg()
    poetry_foundation_train, poetry_foundation_test = split_poetry_foundation()
    # train_data = pandas.concat([gutenberg_train, poetry_foundation_train])
    train_data = gutenberg_train.append(poetry_foundation_test)
    test_data = pandas.concat([gutenberg_test, poetry_foundation_test])
    print(train_data)
    print(test_data)
    return train_ids, test_ids, vocabulary_dict


get_data(None, None)
