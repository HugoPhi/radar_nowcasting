import dataload.dataloadv4 as dl
import matplotlib.pyplot as plt
import dataprocess as dp
import keras
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Conv2D, Conv3D, LayerNormalization, GlobalAveragePooling2D, Dense, Attention, LeakyReLU, BatchNormalization



# Load data
dir = '/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1'
altitude = '1.0km'
datasets = dl.load_data(dir, altitude)

# Set window size and stride
window_size = 10
overlap = 0.5

# Preprocess data
X_train, X_test, y_train, y_test = dp.load_xy(datasets, range(len(datasets)), window_size, overlap, dp.norm_param)


# Build model
def build_model(input_shape, filters=64):
    # model = keras.Sequential()
    # model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, input_shape=input_shape, padding='same', return_sequences=True))
    # model.add(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


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

# Set input shape
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])

# Build the model
model = build_model(input_shape, filters=32)
model.summary()

# Train the model
X_train_convlstm = X_train[:, :, :, :, :]
y_train_target = y_train[:, :, :, :, 0:1]
X_test_convlstm = X_test[:, :, :, :, :]
y_test_target = y_test[:, :, :, :, 0:1]

input_shape = (10, 256, 256, 3)
combined_model = model

# train
epochs = 25
batch_size = 4


history = combined_model.fit(
    X_train_convlstm,  
    y_train_target,  
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test_convlstm, y_test_target),
    verbose='1'
)

print(history.history)

# draw pictures 
# import matplotlib.pyplot as plt
#
# history = combined_model.fit(
#     [X_train_convlstm, X_train_transformer],
#     y_train_target,
#     epochs=10,
#     batch_size=32,
#     validation_data=([X_test_convlstm, X_test_transformer], y_test_target)
# )
#
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(train_loss) + 1)
#
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
# plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#

# Save the model
model.save('.model/convlstm_model.h5')

