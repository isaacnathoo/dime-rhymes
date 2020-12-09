import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size, phenome_size):

        # Model that generates poetry text
        super(Model, self).__init__()

        # initialize vocab_size, emnbedding_size

        self.vocab_size = vocab_size
        self.phenome_size = phenome_size
        self.window_size = 20 
        self.embedding_size_words = 64
        self.embedding_size_phenomes = 64
        self.batch_size = 64 

        # initialize embeddings and forward pass weights (weights, biases)
        self.E_words = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size_words], stddev=.1, dtype=tf.float32))
        self.E_phenomes = tf.Variable(tf.random.normal([self.phenome_size, self.embedding_size_phenomes], stddev=.1, dtype=tf.float32))
        self.rnn_size1 = 150
        self.lstm_layer1 = tf.keras.layers.LSTM(self.rnn_size1, return_sequences=True, return_state=True)
        self.rnn_size2 = 100
        self.lstm_layer2 = tf.keras.layers.LSTM(self.rnn_size2, return_sequences=True, return_state=True)
        self.dense_layer = tf.keras.layers.Dense(self.vocab_size)
        self.learning_rate = 1e-2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, input_words, input_phenomes, initial_state=None):
        embedding_phenomes = tf.nn.embedding_lookup(self.E_phenomes, input_phenomes)
        wholeseq_phenomes, memory_phenomes, carry_phenomes = self.lstm_layer1(embedding_phenomes, initial_state) 
        embedding_words = tf.nn.embedding_lookup(self.E_words, input_words)
        wholeseq_words, memory_words, carry_words = self.lstm_layer2(embedding_words, initial_state=(memory_phenomes, carry_phenomes)) 
        logits = self.dense_layer(wholeseq_words)
        probs = tf.nn.softmax(logits)
        return probs,(memory_words,carry_words)

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

def generate_line():
    pass

def test(model):
    pass

def train(model):
    pass

def main():
    train_poems, test_poems, vocab_dict, phenome_dict, padding_index = get_data('/data/kaggle_poems.txt','/data/poetry_foundation.txt')
    vocab_size = len(vocab_dict)
    phenome_size = len(phenome_dict)
    model = Model(vocab_size=vocab_size, phenome_size=phenome_size)

if __name__ == '__main__':
    main()
