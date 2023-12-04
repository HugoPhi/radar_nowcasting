import keras
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Conv2D, Conv3D, LayerNormalization, GlobalAveragePooling2D, Dense, Attention, LeakyReLU, BatchNormalization

def get_model(input_shape, filters=64):
    model = keras.Sequential()
    model.add(ConvLSTM2D(filters=filters, kernel_size=(7, 7), input_shape=input_shape, padding='same',activation=LeakyReLU(alpha=0.005), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=filters, kernel_size=(5, 5), padding='same',activation=LeakyReLU(alpha=0.005), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
    padding='same',activation=LeakyReLU(alpha=0.005), return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=filters, kernel_size=(1, 1),
    padding='same',activation=LeakyReLU(alpha=0.005), return_sequences=True))
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))

    return model
 
