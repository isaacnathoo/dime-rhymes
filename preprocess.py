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
    gutenberg_data = pandas.read_csv(GUTENBERG_DATASET_FILE)
    gutenberg_data = gutenberg_data[['s']].drop_duplicates(subset=['s'])
    if shuffle:
        gutenberg_data = gutenberg_data.sample(frac=1).reset_index(drop=True)
    gutenberg_train, gutenberg_test = train_test_split(gutenberg_data, test_size=test_size)
    print(gutenberg_data.head())
    print(gutenberg_train.shape)
    print(gutenberg_test.shape)


def split_poetry_foundation(shuffle=False, test_size=0.2):
    poetry_foundation_data = pandas.read_csv(POETRY_FOUNDATION_DATASET_FILE)
    poetry_foundation_data = poetry_foundation_data[['Content']].drop_duplicates(subset='Content')
    if shuffle:
        poetry_foundation_data = poetry_foundation_data.sample(frac=1).reset_index(drop=True)
    poetry_foundation_train, poetry_foundation_test = train_test_split(poetry_foundation_data, test_size=test_size)
    print(poetry_foundation_data.head())
    print(poetry_foundation_train.shape)
    print(poetry_foundation_test.shape)
    # TODO: Parse through content to clean and match gutenberg's consistency

split_gutenberg()
split_poetry_foundation()


def get_data(train_file, test_file):
    """

    :param train_file: Training file path
    :param test_file: Testing file path
    :return:
    (1) Training Words in vectorized ID form
    (2) Testing Words in vectorized ID form
    (3) Map of indices & words "Dictionary"
    """
    train_ids, test_ids, vocabulary_dict = [], [], {}

    return train_ids, test_ids, vocabulary_dict
