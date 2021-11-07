import re
import tensorflow as tf
from sklearn.model_selection import train_test_split

START = "<start>"
END = "<end>"
UNK = "<unk>"


class Dataset:
    def __init__(self, path_to_input_style_file, path_to_output_style_file):
        self.path_to_input_style_file = path_to_input_style_file
        self.path_to_output_style_file = path_to_output_style_file
        self.tokenizer = None

    # remove special characters and add start/end tokens
    def preprocess_sentence(self, sent):
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        sent = re.sub(r"([?.!,¿])", r" \1 ", sent)
        sent = re.sub(r'[" "]+', " ", sent)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sent = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sent)

        # add start and end tokens
        sent = START + " " + sent + " " + END
        return sent

    def create_dataset(self):
        # return 2 lists, one containing preprocessed sents from input and one with preprocessed sents from output
        # input_style_sents, output_style_sents
        input_sents = open(self.path_to_input_style_file, "r").readlines()
        preprocessed_input_sents = [self.preprocess_sentence(input_sent) for input_sent in input_sents]

        output_sents = open(self.path_to_output_style_file, "r").readlines()
        preprocessed_output_sents = [self.preprocess_sentence(output_sent) for output_sent in output_sents]

        assert len(preprocessed_input_sents) == len(preprocessed_output_sents)
        return preprocessed_input_sents, preprocessed_output_sents

    # creates w2id and id2w dicts
    # pads sents to len of max sent
    def tokenize(self, input_sents, output_sents):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=UNK)
        # updates internal vocab in the tokenizer
        # use both input/output to create the vocab, we deal with them as both being part of the same vocab
        sentences = input_sents + output_sents
        self.tokenizer.fit_on_texts(sentences)

        # convert sents from seq of words to seq of ids
        input_tensor = self.tokenizer.texts_to_sequences(input_sents)
        output_tensor = self.tokenizer.texts_to_sequences(output_sents)
        # pad to max sent len (after end of seq)
        input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, padding='post')
        output_tensor = tf.keras.preprocessing.sequence.pad_sequences(output_tensor, padding='post')

        return input_tensor, output_tensor

    # note buffer size must be >= to the size of the dataset
    def call(self, BUFFER_SIZE, BATCH_SIZE):
        input_sents, output_sents = self.create_dataset()
        input_tensor, output_tensor = self.tokenize(input_sents, output_sents)

        input_tensor_train, input_tensor_val, output_tensor_train, output_tensor_val = train_test_split(input_tensor,
                                                                                                        output_tensor,
                                                                                                        test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, output_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, output_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset
