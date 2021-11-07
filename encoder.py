import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dimension, hidden_size, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        ## -- Embedding Layer -- ##
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dimension)

        ## -- LSTM Layer -- ##
        # TODO: consider using a BiLSTM
        # return sequences return the hidden state output for each input time step.
        # return state returns the hidden state output and cell state for the last input time step.
        self.lstm_layer = tf.keras.layers.LSTM(
            self.hidden_size,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        # initial c, h
        return [tf.zeros((self.batch_size, self.hidden_size)), tf.zeros((self.batch_size, self.hidden_size))]

if __name__ == "__main__":
    encoder = Encoder(vocab_size=16731, embedding_dimension=192, batch_size=64, hidden_size=192)
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_h, sample_c = encoder()