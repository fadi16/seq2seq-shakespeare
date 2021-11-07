import os.path
import time

import tensorflow as tf
from dataset import Dataset
from encoder import Encoder
from decoder import Decoder

BUFFER_SIZE = 32000
BATCH_SIZE = 64
EMBEDDING_DIMENSION = 192
HIDDEN_SiZE = 192
EPOCHS = 10

CHECKPOINTS_DIR_PATH = "./checkpoints"


def loss_function(real, pred):
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    # set from_logits to True because the prediction is expected to be a logits tensor
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = cross_entropy(y_true=real, y_pred=pred)
    # todo: I have no idea what this is
    mask = tf.logical_not(tf.math.equal(real, 0))  # output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss


# todo: what does this mean?
@tf.function
def train_step(inp, out):
    with tf.GradientTape() as tape:
        encoder_initial_state = encoder.initialize_hidden_state()
        encoder_output, encoder_h, encoder_c = encoder(inp, encoder_initial_state)

        # todo: why do this? maybe for max input and max output length?
        # ignore <end> token
        decoder_input = out[:, :-1]
        # ignore <start> token
        real = out[:, 1:]

        # set attention mechanism with encoder outputs
        decoder.attention_mechanism.setup_memory(encoder_output)

        # set decoder initial state based on "h" and "c" from encoder
        decoder_initial_state = decoder.build_initial_state([encoder_h, encoder_c], tf.float32)

        prediction = decoder(decoder_input, decoder_initial_state)
        logits = prediction.rnn_output
        loss = loss_function(real=real, pred=logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    # todo: what is tape here? what does this really do?
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


if __name__ == "__main__":
    dataset = Dataset(path_to_input_style_file="./data/train_valid_modern.txt",
                      path_to_output_style_file="./data/train_valid_original.txt")

    train_dataset, val_dataset = dataset.call(BUFFER_SIZE, BATCH_SIZE)

    # remember train_dataset and val_dataset are divided into batches
    no_of_samples = len(train_dataset) * BATCH_SIZE + len(val_dataset) * BATCH_SIZE
    print("no of samples = " + str(no_of_samples))

    steps_per_epoch = no_of_samples // EPOCHS

    example_input_batch, example_output_batch = next(iter(train_dataset))

    print(example_input_batch.shape, example_output_batch.shape)

    vocab_size = len(dataset.tokenizer.word_index) + 1
    max_length_input = example_input_batch.shape[1]
    max_length_output = example_output_batch.shape[1]

    print("vocab size, max input (modern) sentence length, max output (shakespearian) sentence length")
    print(vocab_size, max_length_input, max_length_output)

    # test encoder layer
    encoder = Encoder(vocab_size=vocab_size,
                      embedding_dimension=EMBEDDING_DIMENSION,
                      batch_size=BATCH_SIZE,
                      hidden_size=HIDDEN_SiZE)
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
    print('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))

    # test decoder layer
    decoder = Decoder(vocab_size=vocab_size,
                      embedding_dimension=EMBEDDING_DIMENSION,
                      hidden_size=HIDDEN_SiZE,
                      batch_size=BATCH_SIZE,
                      max_length_input=max_length_input,
                      max_length_output=max_length_output)

    # dummy to simulate previous decoder output
    sample_x = tf.random.uniform((BATCH_SIZE, max_length_output))
    # memory for attention based on encoder outputs
    decoder.attention_mechanism.setup_memory(sample_output)
    initial_state = decoder.build_initial_state([sample_h, sample_c], tf.float32)
    sample_decoder_outputs = decoder(sample_x, initial_state)
    print("Decoder outputs shape: ", sample_decoder_outputs.rnn_output.shape)

    ####################################
    ##  REAL DEAL STARTS HERE
    ####################################

    encoder = Encoder(vocab_size=vocab_size,
                      embedding_dimension=EMBEDDING_DIMENSION,
                      batch_size=BATCH_SIZE,
                      hidden_size=HIDDEN_SiZE)

    decoder = Decoder(vocab_size=vocab_size,
                      embedding_dimension=EMBEDDING_DIMENSION,
                      hidden_size=HIDDEN_SiZE,
                      batch_size=BATCH_SIZE,
                      max_length_input=max_length_input,
                      max_length_output=max_length_output)

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_prefix = os.path.join(CHECKPOINTS_DIR_PATH, "model")
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=encoder,
        decoder=decoder)

    for epoch in range(EPOCHS):
        start = time.time()

        total_loss = 0

        for (batch, (training_input, training_output)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(training_input, training_output)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix="model-{0}".format(str(epoch)))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))




