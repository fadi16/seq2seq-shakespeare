import tensorflow as tf
import tensorflow_addons as tfa

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dimension, hidden_size, batch_size, max_length_input, max_length_output):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dimension)

        # final dense layer for softmax to be applied on
        self.fc = tf.keras.layers.Dense(vocab_size)

        # fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.hidden_size)

        # sampler, todo: what is this?
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # todo: what difference does it make to have memory?
        # create attention mechanism with memory = None
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)
        self.attention_mechanism = tfa.seq2seq.LuongAttention(units=self.hidden_size,
                                                              memory=None,
                                                              memory_sequence_length=self.batch_size * [self.max_length_input])

        # wrap attention with the fundamental rnn cell of decoder
        self.rnn_cell = tfa.seq2seq.AttentionWrapper(cell=self.decoder_rnn_cell,
                                                attention_mechanism=self.attention_mechanism,
                                                attention_layer_size=self.hidden_size)

        # define the decoder with respect to the fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(cell=self.rnn_cell,
                                                sampler=self.sampler,
                                                output_layer=self.fc)

    def build_initial_state(self, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=self.batch_size, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        # todo where do these inputs come from?
        outputs, _, _ = self.decoder(x,
                                     initial_state=initial_state,
                                     sequence_length=self.batch_size * [self.max_length_output - 1])
        return outputs