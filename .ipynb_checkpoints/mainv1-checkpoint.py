from keras.api._v2 import keras
import dataload.dataloadv4 as dl
import matplotlib.pyplot as plt
import dataprocess as dp
from keras.models import Model
from keras.layers import Input, ConvLSTM2D, Conv2D, LayerNormalization, GlobalAveragePooling2D, Dense, Attention



# Load data
dir = '/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1'
altitude = '1.0km'
datasets = dl.load_data(dir, altitude)

# Set window size and stride
window_size = 10
overlap = 0.88

# Preprocess data
X_train, X_test, y_train, y_test = dp.load_xy(datasets, range(len(datasets)), window_size, overlap, dp.norm_param)


# Build model
def build_model(input_shape, filters=64, kernel_size=(3, 3), transformer_units=1):
    """
    构建ConvLSTM和Transformer的联合模型

    Parameters:
    - input_shape: 输入数据的形状
    - filters: ConvLSTM中的卷积滤波器数量
    - kernel_size: ConvLSTM中的卷积核大小
    - transformer_units: Transformer中的单元数

    Returns:
    - model: 构建好的模型
    """
    # # 定义ConvLSTM模型
    # convlstm_input = Input(shape=input_shape)
    # convlstm = ConvLSTM2D(filters=filters, kernel_size=kernel_size, padding='same', return_sequences=True)(convlstm_input)
    # convlstm_output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(convlstm)
    #
    # # 定义Transformer模型
    # transformer_input = Input(shape=input_shape)
    # transformer = Attention(use_scale=True)([transformer_input, transformer_input])
    # for _ in range(transformer_units):
    #     transformer = LayerNormalization(epsilon=1e-6)(transformer)
    # 
    # # 在这里添加 GlobalAveragePooling2D
    # transformer_output = GlobalAveragePooling2D()(transformer)
    # transformer_output = Dense(1, activation='sigmoid')(transformer_output)
    #
    # # 将ConvLSTM和Transformer连接
    # merged = tf.keras.layers.Concatenate(axis=-1)([convlstm_output, transformer_output])
    #
    # # 最终模型
    # model = Model(inputs=[convlstm_input, transformer_input], outputs=merged)
    #
    # # 编译模型
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    model = keras.Sequential()
    model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, input_shape=input_shape, padding='same', return_sequences=True))
    model.add(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Set input shape
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])

# Build the model
model = build_model(input_shape, filters=32, kernel_size=(3, 3), transformer_units=1)
model.summary()

# Train the model
X_train_convlstm = X_train[:, :, :, :, :]
X_train_transformer = X_train[:, :, :, :, :]
y_train_target = y_train[:, :, :, :, 0:1]
X_test_convlstm = X_test[:, :, :, :, :]
X_test_transformer = X_test[:, :, :, :, :]
y_test_target = y_test[:, :, :, :, 0:1]

input_shape = (10, 256, 256, 3)
combined_model = model

# train
epochs = 1
batch_size = 1


history = combined_model.fit(
    [X_train_convlstm, X_train_transformer],  
    y_train_target,  
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([X_test_convlstm, X_test_transformer], y_test_target)  
)

print(history.history)

# draw pictures 
import matplotlib.pyplot as plt

history = combined_model.fit(
    [X_train_convlstm, X_train_transformer],
    y_train_target,
    epochs=10,
    batch_size=32,
    validation_data=([X_test_convlstm, X_test_transformer], y_test_target)
)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Save the model
model.save('.model/convlstm_model.h5')

