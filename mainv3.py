import dataload.dataloadv4 as dl
import os
import datetime
import json
import matplotlib.pyplot as plt
import dataprocess as dp
import models.model_convlstm_v1 as mol


# parameter list
param = {
    'dir': '/root/CodeHub/py/radar-pol-wforcast/.data/new_2308_1',
    'altitude': '1.0km',
    'sample_dir_size': 1,
    'window_size': 10,
    'stride': 5,
    'split_ratio': 0.8,
    'features': ('dBZ', 'ZDR', 'KDP'),
    'normalparam': ([0, 65], [-1, 5], [-1, 6]),  # dBZ: [0, 65], ZDR: [-1, 5], KDP: [-1, 6]
    'filters': 64,
    'epochs': 100,
    'batch_size': 32,
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
}



# Load data
datasets = dl.load_data(param['dir'], param['altitude'], param['sample_dir_size'], features=param['features'])

# Preprocess data
X_train, X_test, y_train, y_test = dp.load_xy(datasets, param['sample_dir_size'], param['normalparam'], split_ratio=param['split_ratio'], window_size=param['window_size'], stride=param['stride'])


# Build model
def build_model(input_shape, filters=64):
    # model = keras.Sequential()
    # model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, input_shape=input_shape, padding='same', return_sequences=True))
    # model.add(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # model = keras.Sequential()
    # model.add(ConvLSTM2D(filters=filters, kernel_size=(7, 7), input_shape=input_shape, padding='same',activation=LeakyReLU(alpha=0.005), return_sequences=True))
    # model.add(BatchNormalization())
    # model.add(ConvLSTM2D(filters=filters, kernel_size=(5, 5), padding='same',activation=LeakyReLU(alpha=0.005), return_sequences=True))
    # model.add(BatchNormalization())
    # model.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
    # padding='same',activation=LeakyReLU(alpha=0.005), return_sequences=True))
    # model.add(BatchNormalization())
    # model.add(ConvLSTM2D(filters=filters, kernel_size=(1, 1),
    # padding='same',activation=LeakyReLU(alpha=0.005), return_sequences=True))
    # model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
    
    model = mol.get_model(input_shape, filters=filters)
    model.compile(optimizer=param['optimizer'], loss=param['loss'], metrics=param['metrics'])
    
    return model

# Set input shape
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])

# Build the model
model = build_model(input_shape, filters=param['filters'])
model.summary()

# Train the model
X_train_convlstm = X_train[:, :, :, :, :]
y_train_target = y_train[:, :, :, :, 0:1]
X_test_convlstm = X_test[:, :, :, :, :]
y_test_target = y_test[:, :, :, :, 0:1]

combined_model = model

# train
history = combined_model.fit(
    X_train_convlstm,  
    y_train_target,  
    epochs=param['epochs'],
    batch_size=param['batch_size'],
    validation_data=(X_test_convlstm, y_test_target),
)

print(history.history)
model.evaluate(X_test_convlstm, y_test_target)


# Save the model and its parameters and train info
os.makedirs(f'.model/model_{datetime.datetime}', exist_ok=True)

## record train history 
with open(f'.model/model_{datetime.datetime}/history.txt', 'w') as f:
    f.write(str(history.history))

with open(f'.model/model_{datetime.datetime}/history.json', 'w') as f:
    json.dump(history.history, f)

## draw pictures 
epochs = param['epochs']
train_loss = history.history['loss']
test_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']


### create directory 
if not os.path.exists(f'.model/model_{datetime.datetime}/figures'):
    os.makedirs(f'.model/model_{datetime.datetime}/figures')

### loss vs epoch 
plt.figure(1)
plt.plot(epochs, train_loss, 'orange', label='Train Loss')
plt.plot(epochs, test_loss, 'green', label='Test Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'.model/model_{datetime.datetime}/figures/accuracy_vs_epoch.png')

### accuracy vs epoch
plt.figure(2)
plt.plot(epochs, train_accuracy, 'orange', label='Train Accuracy')
plt.plot(epochs, test_accuracy, 'green', label='Test Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f'.model/model_{datetime.datetime}/figures/accuracy_vs_epoch.png')

### show plot
plt.show()


## Save the model according to the parameters and date
model.save(f'.model/model_{datetime.datetime}/model.h5')

## write parameters to json file under the model
with open(f'.model/model_{datetime.datetime}/param.json', 'w') as f:
    json.dump(param, f)




