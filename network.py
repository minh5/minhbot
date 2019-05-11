import pathlib

from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model
import numpy as np

from preprocess import *


DATA_PATH = os.path.join(pathlib.Path.home(), "minhbot/messages/inbox")

# preprocessing
inputs, outputs = create_inputs_outputs(DATA_PATH)
inputs, outputs = truncate_words(inputs, outputs, 300)
input_corpus = create_corpus(inputs)
output_corpus = create_corpus(outputs)
vec_inputs = vectorize_words(input_corpus, inputs)
vec_outputs = vectorize_words(output_corpus, outputs, "output")

# Model Variables
MAX_ENCODER_SEQ_LEN = max([len(txt.split()) for txt in inputs])
MAX_DECODER_SEQ_LEN = max([len(txt.split()) for txt in outputs])
NUM_ENCODER_TOKENS = len(input_corpus)
NUM_DECODER_TOKENS = len(output_corpus)
BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 100  # Number of epochs to train for.
LATENT_DIM = 256  # Latent dimensionality of the encoding space.


# initializing training data
encoder_input_data = np.zeros(
    (len(inputs), MAX_ENCODER_SEQ_LEN, NUM_ENCODER_TOKENS), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(inputs), MAX_DECODER_SEQ_LEN, NUM_DECODER_TOKENS), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(inputs), MAX_DECODER_SEQ_LEN, NUM_DECODER_TOKENS), dtype="float32"
)


for i, (input_sentence, output_sentence) in enumerate(zip(inputs, outputs)):
    for t, word in enumerate(input_sentence.split()):
        encoder_input_data[i, t, input_corpus[word]] = 1.0
    for t, word in enumerate(output_sentence.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, output_corpus[word]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, output_corpus[word]] = 1.0


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, NUM_ENCODER_TOKENS))
encoder = LSTM(LATENT_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, NUM_DECODER_TOKENS))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(NUM_DECODER_TOKENS, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)
