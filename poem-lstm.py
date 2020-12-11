import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = 20 # DO NOT CHANGE!
        self.embedding_size = 64
        self.batch_size = 512
        self.lstm_size = 32
        self.dense_1_size = 16384

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        self.embedding = tf.Variable(tf.random.truncated_normal(
            [self.vocab_size, self.embedding_size], stddev=0.1))

        self.lstm = tf.keras.layers.LSTM(
                self.lstm_size, return_sequences=True, return_state=True)

        self.dense_1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, initial_state):
        embeds = tf.nn.embedding_lookup(self.embedding, inputs)
        seqs, final_mem, final_carry = self.lstm(embeds, initial_state=initial_state)
        logits = self.dense_2(self.dense_1(seqs))
        probs = tf.nn.softmax(logits)
        return probs, (final_mem, final_carry)

    def loss(self, probs, labels):
        return self._loss(labels, probs)


def train(model, train_inputs, train_labels):
    num_ids = train_inputs.shape[0]
    window_groups = [model.window_size] * (num_ids // model.window_size) + [num_ids % model.window_size]

    input_windows =     tf.convert_to_tensor(tf.split(train_inputs, window_groups)[:-1])
    label_windows =     tf.convert_to_tensor(tf.split(train_labels, window_groups)[:-1])

    num_inputs = input_windows.shape[0]
    batch_sizes = [model.batch_size] * (num_inputs // model.batch_size) + [num_inputs % model.batch_size]

    input_batches =     tf.split(input_windows, batch_sizes)[:-1]
    label_batches =     tf.split(label_windows, batch_sizes)[:-1]
    all_batches =       list(zip(input_batches, label_batches))

    np.random.shuffle(all_batches)


    for (inputs, labels) in all_batches:
        with tf.GradientTape() as tape:
            out, _ = model.call(inputs, None)
            loss = model.loss(out, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return


def test(model, test_inputs, test_labels):
    num_ids = test_inputs.shape[0]
    window_groups =     [model.window_size] * (num_ids // model.window_size) + [num_ids % model.window_size]
    input_windows =     tf.convert_to_tensor(tf.split(test_inputs, window_groups)[:-1])
    label_windows =     tf.convert_to_tensor(tf.split(test_labels, window_groups)[:-1])

    num_inputs = input_windows.shape[0]
    batch_sizes = [model.batch_size] * (num_inputs // model.batch_size) + [num_inputs % model.batch_size]

    input_batches =     tf.split(input_windows, batch_sizes)[:-1]
    label_batches =     tf.split(label_windows, batch_sizes)[:-1]
    all_batches =       list(zip(input_batches, label_batches))

    total_loss = 0

    for inp, label in all_batches:
        out, _ = model.call(inp, None)
        total_loss += model.loss(out, label)

    return tf.math.exp(total_loss / len(all_batches))

def generate_sentence(word1, length, vocab, model, sample_n=10):
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    train_ids, test_ids, vocab =    get_data("../../data/train.txt", "../../data/test.txt")
    model = Model(len(vocab))
    print("model initialized")

    train_inputs = train_ids[:-1]
    train_labels = train_ids[1:]

    test_inputs = test_ids[:-1]
    test_labels = test_ids[1:]

    for epoch in range(1):
        train(model, train_inputs, train_labels)

    print("training complete")

    checkpoint = tf.train.Checkpoint(
            optimizer=model.optimizer,
            embedding=model.embedding,
            lstm=model.lstm,
            dense_1=model.dense_1,
            dense_2=model.dense_2)
    manager = tf.train.CheckpointManager(checkpoint, './ckpts', max_to_keep=10)
    manager.save()

    print("Train perplexity: " + str(test(model, train_inputs, train_labels)))
    print("Test perplexity: " + str(test(model, test_inputs, test_labels)))

    return
    
if __name__ == '__main__':
    main() 
