
import tensorflow as tf
from tensorflow import keras

def make_cnn(input_shape, out_dim=1, task='regression'):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(out_dim, activation=('sigmoid' if task=='classification' else None))
    ])
    loss = 'binary_crossentropy' if task=='classification' else 'mse'
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    return model
