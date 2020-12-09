from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self):
        # Model that generates poetry text
        super(Model, self).__init__()
    def call(self):
        pass

    def loss(self):
        pass

def generate_line():
    pass

def test(model):
    pass

def train(model):
    pass

def main():
    model = Model()
    train_inputs, test_input, dictionary = get_data(None, None)
