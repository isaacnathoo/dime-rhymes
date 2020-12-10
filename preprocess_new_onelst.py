import tensorflow as tf
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import nltk
import string

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
    poetry_foundation_data = np.asarray(poetry_foundation_data)[1:]
    if shuffle:
        poetry_foundation_data = poetry_foundation_data.sample(frac=1).reset_index(drop=True)
    words_lst = []
    for idx,line in enumerate(poetry_foundation_data):
        one_line = line[0].replace('\r', '').replace(u'\xa0', u'').lower()
        one_line = "".join([char for char in one_line if char not in string.punctuation])
        line_lst = one_line.split('\n')
        ss_line = [line1+' *' for line1 in line_lst]
        line_str = ''
        for e in ss_line:
            if e != ' *':
                line_str += e + ' '
        line_str = line_str.replace('  ', ' ')
        split_line = line_str.split(' ')
        split_line.remove('')
        words_lst += split_line
        if idx == 0:
            print(line[0])
            print(line_lst)
            print(ss_line)
            print(line_str)
            print(one_line)
            print(split_line)

    print(words_lst[:13])

    unique_words = list(set(words_lst))
    words_dict = dict((key, i) for i,key in enumerate(unique_words))

    poetry_foundation_train, poetry_foundation_test = train_test_split(words_lst, test_size=test_size)

    training_tokens = [words_dict.get(k) for k in poetry_foundation_train]
    testing_tokens = [words_dict.get(k) for k in poetry_foundation_test]

    print(training_tokens[:13])

    return training_tokens, testing_tokens, words_dict


def get_data():
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
    # train_ids, test_ids, vocabulary_dict = [], [], {}
    # gutenberg_train, gutenberg_test = split_gutenberg()
    poetry_foundation_train, poetry_foundation_test, vocabulary_dict = split_poetry_foundation()
    # print(np.asarray(poetry_foundation_train)[1])
    # train_data = pandas.concat([gutenberg_train, poetry_foundation_train])
    # train_data = gutenberg_train.append(poetry_foundation_test)
    # test_data = pandas.concat([gutenberg_test, poetry_foundation_test])
    # print(train_data)
    # print(test_data)
    # return train_ids, test_ids, vocabulary_dict
    return poetry_foundation_train, poetry_foundation_test, vocabulary_dict


# get_data()