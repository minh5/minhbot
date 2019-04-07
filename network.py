import numpy as np
import keras.models as kem
import keras.layers as kel
import keras.callbacks as kec
import sklearn.preprocessing as skprep

from preprocess import create_words_list
from preprocess import create_corpus
from preprocess import words_to_idx

max_features =  16

num_mem_units = 64
size_batch = 1
num_timesteps = 1
num_features = 1
num_targets = 1
num_epochs = 10


def train_network(data,
                  max_features=16,
                  num_mem_units=64, 
                  size_batch=1, 
                  num_timesteps=1, 
                  num_features=1, 
                  num_targets=1, 
                  num_epochs=10):
    model = kem.Sequential()
    model.add(
        kel.LSTM(num_mem_units,
                 stateful=True,
                 batch_input_shape=(size_batch, num_timesteps, num_features),
        return_sequences=True))
    model.add(kel.Dense(num_targets, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam')

    range_act = (0, 1) # sigmoid
    range_features = np.array([0, max_features]).reshape(-1, 1)
    normalizer = skprep.MinMaxScaler(feature_range=range_act)
    normalizer.fit(range_features)

    reset_state = kec.LambdaCallback(on_epoch_end=lambda *_ : model.reset_states())

    # training
    for seq in data:
        X = seq[:-1]
        y = seq[1:] # predict next element
        X_norm = normalizer.transform(
            np.array(X).reshape(-1, 1)).reshape(-1, num_timesteps,num_features)
        y_norm = normalizer.transform(
            np.array(y).reshape(-1, 1)).reshape(-1, num_timesteps, num_targets)
        model.fit(X_norm,
                  y_norm,
                  epochs=num_epochs, 
                  batch_size=size_batch, 
                  shuffle=False,
                  callbacks=[reset_state])
        return model


def predict(data, model):
    for seq in data:
        model.reset_states() 
        for istep in range(len(seq)-1): # input up to not incl last
            val = seq[istep]
            X = np.array([val]).reshape(-1, 1)
            X_norm = normalizer.transform(X).reshape(-1, num_timesteps, num_features)
            y_norm = model.predict(X_norm)
        yhat = int(normalizer.inverse_transform(y_norm[0])[0, 0])
        y = seq[-1] # last
        put = '{0} predicts {1:d}, expecting {2:d}'.format(
            ', '.join(str(val) for val in seq[:-1]),
            yhat,
            y)
        print(put)


if __name__ == '__main__':
    words = create_words_list('/messages/inbox/')
    corpus = create_corpus(words)
    data = words_to_idx(words, corpus)
    training = data[:80000]
    test = data[80000:]
    model = train_network(training)
    predict(test, model)