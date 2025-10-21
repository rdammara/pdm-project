
import tensorflow as tf
from tensorflow import keras

def make_lstm(input_shape, out_dim=1, task='regression', hidden=64, layers=1):
    model = keras.Sequential([keras.layers.Input(shape=input_shape)])
    for i in range(layers-1):
        model.add(keras.layers.LSTM(hidden, return_sequences=True))
    model.add(keras.layers.LSTM(hidden, return_sequences=False))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(out_dim, activation=('sigmoid' if task=='classification' else None)))
    loss = 'binary_crossentropy' if task=='classification' else 'mse'
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    return model
