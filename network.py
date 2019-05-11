
import pathlib

from keras.layers import  Dense
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import LSTM
from keras.models import Model
import numpy as np

from preprocess import *


DATA_PATH  = os.path.join(pathlib.Path.home(), 'minhbot/messages/inbox')

# preprocessing
inputs, outputs = create_inputs_outputs(DATA_PATH)
inputs, outputs = truncate_words(inputs, outputs, 300)
input_corpus = create_corpus(inputs)
output_corpus = create_corpus(outputs)
vec_inputs = vectorize_words(input_corpus, inputs)
vec_outputs = vectorize_words(output_corpus, outputs, 'output')

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
    (len(inputs), MAX_ENCODER_SEQ_LEN, NUM_ENCODER_TOKENS),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(inputs), MAX_DECODER_SEQ_LEN, NUM_DECODER_TOKENS),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(inputs), MAX_DECODER_SEQ_LEN, NUM_DECODER_TOKENS),
    dtype='float32')


for i, (input_sentence, output_sentence) in enumerate(zip(inputs, outputs)):
    for t, word in enumerate(input_sentence.split()):
        encoder_input_data[i, t, input_corpus[word]] = 1.
    for t, word in enumerate(output_sentence.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, output_corpus[word]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, output_corpus[word]] = 1.


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(NUM_ENCODER_TOKENS, LATENT_DIM)(encoder_inputs)
x, state_h, state_c = LSTM(LATENT_DIM,
                           return_state=True)(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
x = Embedding(NUM_DECODER_TOKENS, LATENT_DIM)(decoder_inputs)
x = LSTM(LATENT_DIM, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(NUM_DECODER_TOKENS, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)
# Save model
model.save('s2s.h5')
