
import glob
import math
import numpy as np
import os

import tensorflow as tf
from tensorflow_core.python.keras.optimizers import Adam
from tensorflow_core.python.layers.convolutional import Conv2D
from tensorflow_core.python.training import momentum

from modelrequirements import ModelRequirements
from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam

import numpy
import math
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt


def _split_dataset(image_dataset):
    image_dataset = tf.data.TFRecordDataset(file_list)
    split_index =  4000
    iterator = image_dataset.make_one_shot_iterator()
    count = 0
    try:
      while True:
          if count < split_index:
             next_element = iterator.get_next()
             image_source, upscaled_image = next_element['original_image'].numpy(), next_element['label_image'].numpy()
          else:
             image_source_validation, upscaled_image_validation = next_element['original_image'].numpy(), next_element['label_image'].numpy()
          count = count + 1
    except: tf.errors.OutOfRangeError
    pass

    return [image_source, upscaled_image, image_source_validation, upscaled_image_validation]


def plot_fig(string, history):
    fig = plt.figure()
    plt.plot(range(1, ModelRequirements.EPOCH_AMOUNT + 1), history.history['val_acc'], label='validation')
    plt.plot(range(1, ModelRequirements.EPOCH_AMOUNT + 1), history.history['acc'], label='training')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xlim([1, ModelRequirements.EPOCH_AMOUNT])
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()
    fig.savefig(string + '_accuracy.jpg')
    plt.close(fig)


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))


class LossHistory_(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay_start_init(len(self.losses)))
        print('lr:', exp_decay_start_init(len(self.losses)))


def step_decay_start_init(epoch):
    epoch_border = 5
    step_lrate = 0.1 / epoch_border
    initial_lrate = 0.1
    drop = 0.5;
    epochs_drop = 2.0
    if epoch <= epoch_border + 1:
        lrate = 0.001 + (epoch - 1) * step_lrate
    else:
        lrate = (initial_lrate + 0.001) * math.pow(drop, math.floor((epoch) / epochs_drop))
    return lrate


def exp_decay_start_init(epoch):
    epoch_border = 5
    step_lrate = 0.1 / epoch_border
    initial_lrate = 0.1;
    k = 0.1
    if epoch <= epoch_border + 1:
        lrate = 0.001 + (epoch - 1) * step_lrate
    else:
        lrate = (initial_lrate + 0.001) * np.exp(-k * epoch)
    return lrate


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5;
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch) / epochs_drop))
    return lrate


def exp_decay(epoch):
    initial_lrate = 0.1;
    k = 0.1
    lrate = initial_lrate * np.exp(-k * epoch)
    return lrate

def model():
    # lrelu = LeakyReLU(alpha=0.1)
    # SRCNN - SUPER RESOLUTION CONVOLUTION NEURAL NETWORK
    decay_rate = 0.0
    SRCNN = Sequential()
    optimizer = SGD(lr=ModelRequirements.LEARNING_RATE, momentum=momentum, decay=decay_rate, nesterov=False)
    #optimizer = tf.keras.optimizers.SGD(custom_learning_rate = LearningRateScheduler(step_decay(ModelRequirements.EPOCH_AMOUNT)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (9, 9), activation='relu', padding = 'same',
                     input_shape=(ModelRequirements.CROP_HEIGHT, ModelRequirements.CROP_WIGHTS, 3)))
    SRCNN.add(Conv2D(filters=32,  kernel_size = (5, 5), activation='relu', border_mode='same', bias=True))
    # SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(filters=8, kernel_size = (3, 3),  padding = 'same', activation='relu', bias=True))

    SRCNN.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return SRCNN
if __name__ == "__main__":

    record_name = 'record_folder'
    image_shape = (200, 300, 3)

    """
    batch_size = 1;
    epochs = 20
    learning_rate = 0.01;
    decay_rate = 0.0;
    momentum = 0.0

    model = Sequential()
    _, height, width, channels = X_image_train.shape
    optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.add(Convolution2D(filters=8, kernel_size=(9, 9), padding='same', input_shape=(height, width, channels),
                            activation='relu'))
    model.add(Convolution2D(filters=4, kernel_size=(1, 1), activation='relu'))
    model.add(Convolution2D(filters=channels, padding='same', kernel_size=(5, 5)))
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    #    history = model.fit(X_image_train, Y_label_train,
    #                        validation_data=(X_image_val, Y_label_val),
    #                        batch_size=batch_size, epochs=epochs, verbose=2)
    #    plot_fig('const_lr', history)
    #    model.save(model_path)

    #    loss_history = LossHistory()
    #    lrate = LearningRateScheduler(step_decay)
    #    callbacks_list = [loss_history, lrate]
    #
    #    # fit the model
    #    history = model.fit(X_image_train, Y_label_train,
    #                        validation_data=(X_image_val, Y_label_val),
    #                        epochs=epochs, batch_size=batch_size,
    #                        callbacks=callbacks_list, verbose=2)
    #    plot_fig('step_lr', history)
"""
    loss_history_ = LossHistory_()
    lrate_ = LearningRateScheduler(exp_decay)
    callbacks_list_ = [loss_history_, lrate_]
    file_list = glob.glob(ModelRequirements.TFRECORDS_PATH + '/*.tfrecord')
    file_number = len(file_list)
    #input_shape = (200, 300, 3)
    image_dataset = tf.data.TFRecordDataset(file_list)
    #images, labels = iterator.get_next()
    #iterator = image_dataset.make_one_shot_iterator()
    print("RTRFRTRTRFGBVNM<>NMBVCBNM<>MNBVNM<>?<MNBVCBNM<>?" + str(_split_dataset(image_dataset)[0]))
    history = model().fit(_split_dataset(image_dataset, image_shape)[0] , _split_dataset(image_dataset, image_shape)[1],
                        validation_data=(_split_dataset(image_dataset, image_shape)[2],_split_dataset(image_dataset, image_shape)[3]),
                        epochs=ModelRequirements.EPOCH_AMOUNT, batch_size=ModelRequirements.BATCH_SIZE_,
                        callbacks=callbacks_list_, verbose=2)
    plot_fig('exp_lr', history)