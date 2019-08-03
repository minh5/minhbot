import lopgging
import pickle

import click
import click_log
import keras
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model
import numpy as np


# load assets
with open('corpus.pckl', 'w') as fd:
    CORPUS = pickle.load(fd)

LOGGER = logging.getLogger(__name__)
MODEL = keras.models.load_model('chatbot.h5')

# Model Variables
LATENT_DIM = 256
NUM_ENCODER_TOKENS, NUM_DECODER_TOKENS = len(CORPUS)
MAX_ENCODER_LENGTH = len(str("{0:b}".format(NUM_ENCODER_TOKENS + 1)))
MAX_DECODER_LENGTH = len(str("{0:b}".format(NUM_DECODER_TOKENS + 1)))


@click.group()
def main():
    logging.basicConfig(level=logging.INFO)
 

@main.group(name="chatbot")
def chat_command():
    """Chatbot CLI utility"""


def define_models():
    """Define the encoder and decoder model"""
    encoder_inputs = Input(shape=(None, MAX_ENCODER_LENGTH))
    encoder = LSTM(LATENT_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, MAX_DECODER_LENGTH))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    decoder_dense = Dense(MAX_DECODER_LENGTH, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(LATENT_DIM,))
    decoder_state_input_c = Input(shape=(LATENT_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return decoder_model, encoder_model


def decode_sequence(input_sequence):
    """Decode a sequence using the trained model"""
    decoder_model, encoder_model = define_models()
    # Encode the input as state vectors.
    states_value = MODEL.predict(input_sequence)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, NUM_DECODER_TOKENS))
    # Populate the first character of target sequence with the start character.
    start_binary = "{0:b}".format(CORPUS['\t']).zfill(MAX_ENCODER_LENGTH)
    target_seq[0, 0, :] = [i for i in start_binary]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = CORPUS[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > MAX_DECODER_LENGTH):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, NUM_DECODER_TOKENS))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


@click.argument("sentence")
@click.option(
    "-s",
    "--sentence",
    help="Sentence to send to the Minhbot",
)
@bq_command.command(name="send")
def send_sentence(sentence):
    splitted = sentence.split()
    sequence_input = np.zeros(
        (0, len(splitted), MAX_ENCODER_LENGTH), dtype="float32")

    for i, word in enumerate(splitted):
        encode_binary = "{0:b}".format(CORPUS[word]).zfill(MAX_ENCODER_LENGTH)
        sequence_input[0, i, :] = [i for i in encode_binary]

    result = decode_sequence(sequence_input)
    LOGGER.info(f'input sentence is: {sentence}')
    LOGGER.info(f'output setence is: {result}')


if __name__ == "__main__":
    main()