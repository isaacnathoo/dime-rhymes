import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess_new_onelst_final import get_data
from matplotlib import pyplot as plt
from keras.preprocessing.sequence import pad_sequences

class Model(tf.keras.Model):
    def __init__(self, vocab_size, phonvoc_size):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.phonvoc_size = phonvoc_size
        self.window_size = 20 # DO NOT CHANGE!
        self.embedding_size_words = 250
        self.embedding_size_phonemes = 50
        self.batch_size = 64
        self.lstm_size = 250
        self.lstm_size2 = 250
        self.dense_1_size = 16384

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self._loss = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        self.embedding = tf.Variable(tf.random.truncated_normal(
            [self.vocab_size, self.embedding_size_words], stddev=0.1))

        self.embedding_phonemes = tf.Variable(tf.random.truncated_normal(
            [self.phonvoc_size, self.embedding_size_phonemes], stddev=0.1))

        self.lstm = tf.keras.layers.LSTM(
                self.lstm_size, return_sequences=True, return_state=True)

        self.lstm2 = tf.keras.layers.LSTM(
                self.lstm_size2, return_sequences=True, return_state=True)

        self.dense_1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, initial_state, input_phonemes):
        if input_phonemes is not None:
            phon_embeds = tf.nn.embedding_lookup(self.embedding_phonemes, input_phonemes)
            seqsp, final_memp, final_carryp = self.lstm2(phon_embeds, initial_state=initial_state)
            embeds = tf.nn.embedding_lookup(self.embedding, inputs)
            seqs, final_mem, final_carry = self.lstm(embeds, initial_state=(final_memp, final_carryp))
            logits = self.dense_2(self.dense_1(seqs))
            probs = tf.nn.softmax(logits)
        else:
            embeds = tf.nn.embedding_lookup(self.embedding, inputs)
            seqs, final_mem, final_carry = self.lstm(embeds, initial_state=initial_state)
            logits = self.dense_2(self.dense_1(seqs))
            probs = tf.nn.softmax(logits)
        return probs, (final_mem, final_carry)

    def loss(self, probs, labels):
        # return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
        return self._loss(labels, probs)


def train(model, train_inputs, train_labels, train_phonemes, train_phonemes_labels, phenome_dict):
    loss_lst = []
    train_inputs = np.asarray(train_inputs)
    num_ids = train_inputs.shape[0]
    window_groups = [model.window_size] * (num_ids // model.window_size) + [num_ids % model.window_size]

    input_windows =     tf.convert_to_tensor(tf.split(train_inputs, window_groups)[:-1])
    label_windows =     tf.convert_to_tensor(tf.split(train_labels, window_groups)[:-1])

    train_labels = np.asarray(train_labels)
    num_inputs = input_windows.shape[0]
    batch_sizes = [model.batch_size] * (num_inputs // model.batch_size) + [num_inputs % model.batch_size]

    input_batches =     tf.split(input_windows, batch_sizes)[:-1]
    label_batches =     tf.split(label_windows, batch_sizes)[:-1]
    all_batches =       list(zip(input_batches, label_batches))

    np.random.shuffle(all_batches)

    for (inputs, labels) in all_batches:
        input_phonemes = []
        for wind in inputs.numpy():
            temp = []
            for wor in wind:
                temp += phenome_dict.get(wor)
            input_phonemes.append(temp)
        max_len_lst = 0
        for lst in input_phonemes:
            if len(lst) > max_len_lst:
                max_len_lst = len(lst)
        input_phonemes = pad_sequences(input_phonemes, maxlen=max_len_lst, padding="post")
        with tf.GradientTape() as tape:
            out, _ = model.call(inputs, None, input_phonemes)
            loss = model.loss(out, labels)
            loss_lst.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_lst

def test(model, test_inputs, test_labels):
    num_examples = len(test_inputs)
    batch_size = model.batch_size
    window_size = model.window_size
    jumps = batch_size*window_size
    loss_lst = []
    for i in range(0, num_examples, jumps):
        if jumps + i > num_examples:
            break
        batch_inputs = test_inputs[i:i+jumps]
        batch_inputs = np.reshape(batch_inputs, (batch_size,window_size))
        true_vals = test_labels[i:i+jumps]
        true_vals = np.reshape(true_vals, (batch_size,window_size))
        predictions, _ = model.call(batch_inputs, None, None)
        loss = model.loss(predictions, true_vals)
        loss_lst.append(loss)
    perplexity = np.exp(np.mean(loss_lst))
    return perplexity  

def generate_sentence(word1, length, vocab, model, sample_n=10):
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state, None)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]
    
    for idx,word in enumerate(text):
        if word == '*':
            text[idx] = '\n'
    print(" ".join(text))


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list field 

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss Per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_perplexities(perplexities): 
    """
    Uses Matplotlib to visualize the perplexities of our model.
    :param losses: list of perplexities data stored from test.

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(perplexities))]
    plt.plot(x, perplexities)
    plt.title('Perplexities Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.show()


def main():
    train_ids, test_ids, vocab, phonemes_train, phonemes_test, phenome_dict =    get_data()
    model = Model(len(vocab), len(phenome_dict))
    print("model initialized")

    train_phonemes = phonemes_train[:-1]
    train_phonemes_labels = phonemes_train[1:]
    train_inputs = train_ids[:-1]
    train_labels = train_ids[1:]

    test_phonemes = phonemes_test[:-1]
    test_phonemes_labels = phonemes_test[1:]
    test_inputs = test_ids[:-1]
    test_labels = test_ids[1:]
    print("Untrained Sentences:")
    generate_sentence("love",5,vocab,model)
    generate_sentence("love",10,vocab,model)

    loss_lst = []
    perplexity_lst = []
    for epoch in range(10):
        losses = train(model, train_inputs, train_labels, train_phonemes, train_phonemes_labels, phenome_dict)
        loss_lst += losses
        perp = test(model, train_inputs, train_labels)
        perplexity_lst.append(perp)
    visualize_loss(loss_lst)
    visualize_perplexities(perplexity_lst)

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

    print("Poem 1:")
    generate_sentence("love",5,vocab,model)

    print("Poem 2:")
    generate_sentence("love",10,vocab,model)

    print("Poem 3:")
    generate_sentence("love",15,vocab,model)

    print("Poem 4:")
    generate_sentence("love",20,vocab,model)

    print("Poem 5:")
    generate_sentence("love",20,vocab,model)

    print("Poem 6:")
    generate_sentence("love",40,vocab,model)

    print("Poem 7:")
    generate_sentence("love",40,vocab,model)
    
    print("Poem 8:")
    generate_sentence("love",40,vocab,model)
    
    print("Poem 9:")
    generate_sentence("love",40,vocab,model)

    print("Poem 10:")
    generate_sentence("love",100,vocab,model)

    print("Poem 11:")
    generate_sentence("love",150,vocab,model)
    return

if __name__ == '__main__':
    main()
