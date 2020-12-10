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

    def accuracy_function(self, prbs, labels, mask):
		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


def generate_line():
    pass

def train(model, input_sentences):
    num_sentences,sentence_len = np.shape(input_sentences)
	english_labels = input_sentences
    input_sentences = input_sentences[:,:(sentence_len-1)]
	batch_size = model.batch_size
	for i in range(0, num_sentences, batch_size):
		if batch_size + i > num_sentences:
			break
		batch_encoder_inputs = input_sentences[i:i+batch_size,:]
		batch_decoder_inputs = input_sentences[i:i+batch_size,:]
		batch_labels = english_labels[i:i+batch_size,1:]
		mask_tensor = np.where(batch_labels == eng_padding_index, 0, 1)
		with tf.GradientTape() as tape:
			predictions = model.call(batch_encoder_inputs, batch_decoder_inputs)
			loss = model.loss_function(predictions, batch_labels, mask_tensor)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_poems, padding_index):
    num_sentences,sentence_len = np.shape(test_poems)
	english_labels = test_poems
	test_poems = test_poems[:,:(sentence_len-1)]
	batch_size = model.batch_size
	acc_lst = []
	loss_lst = []	
	total_words = 0
	for i in range(0, num_sentences, batch_size):
		if batch_size + i > num_sentences:
			break
		batch_encoder_inputs = test_poems[i:i+batch_size,:]
		batch_decoder_inputs = test_poems[i:i+batch_size,:]
		batch_labels = english_labels[i:i+batch_size,1:]
		mask_tensor = np.where(batch_labels == eng_padding_index, 0, 1)
		num_words = np.count_nonzero(mask_tensor == 1)
		total_words += num_words
		probabilities = model.call(batch_encoder_inputs, batch_decoder_inputs)
		loss = model.loss_function(probabilities, batch_labels, mask_tensor)
		# loss = loss / num_words
		loss_lst.append(loss)
		accuracy = model.accuracy_function(probabilities, batch_labels, mask_tensor)
		accuracy = accuracy * num_words
		acc_lst.append(accuracy)
	perplexity = np.exp(np.sum(loss_lst)/total_words)
	avg_accuracy = tf.reduce_sum(acc_lst)/total_words
	return perplexity,avg_accuracy 

def main():
    train_poems, test_poems, vocab_dict, phenome_dict, padding_index = get_data('/data/kaggle_poems.txt','/data/poetry_foundation.txt')
    vocab_size = len(vocab_dict)
    phenome_size = len(phenome_dict)
    model = Model(vocab_size=vocab_size, phenome_size=phenome_size)

if __name__ == '__main__':
    main()
