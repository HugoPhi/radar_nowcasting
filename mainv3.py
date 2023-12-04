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
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(1, param['epochs'] + 1):
    print(f"Epoch {epoch}/{param['epochs']}")
    
    # Training
    train_metrics = model.fit(X_train_convlstm, y_train_target, batch_size=param['batch_size'], epochs=1, verbose='0')
    train_loss_list.extend(train_metrics.history['loss'])
    train_acc_list.extend(train_metrics.history['accuracy'])

    # Validation
    val_metrics = model.evaluate(X_test_convlstm, y_test_target, verbose='0')
    val_loss_list.append(val_metrics[0])
    val_acc_list.append(val_metrics[1])

    # Plot metrics after each batch
    plt.figure(figsize=(12, 6))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, 'bo-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, 'bo-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save the figure for each batch
    save_dir = 'path_to_save_figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, f'batch_metrics_epoch_{epoch}.png'))

### Plot overall metrics after each epoch
plt.figure(figsize=(12, 6))

### Plot training loss
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, 'bo-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()

### Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, 'bo-', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

# Save the figure for overall metrics
plt.savefig(os.path.join(f'.model/model_{datetime.datetime}', 'overall_metrics.png'))

## Save the model according to the parameters and date
model.save(f'.model/model_{datetime.datetime}/model.h5')

## write parameters to json file under the model
with open(f'.model/model_{datetime.datetime}/param.json', 'w') as f:
    json.dump(param, f)




