# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:44:06 2019

@author: valdis
"""
import glob
from PIL import Image

import io

import tensorflow as tf
import os, cv2, numpy as np

from modelrequirements import ModelRequirements



def _int64_feature(value):
    int_list = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list=int_list)


def _bytes_feature(value):
    byte_list = tf.train.BytesList(value=[value])
    return tf.train.Feature(bytes_list=byte_list)


def _create_features(width, height, image, label):
    return {'image_width': _int64_feature(width),
            'image_height': _int64_feature(height),
            'original_image': _bytes_feature(image),
            'label_image': _bytes_feature(label)}


def _serialize_example(width, height, image, label):
    features = _create_features(width, height, image, label)
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def _parse_features(example_item):
    features = {'image_width': tf.io.FixedLenFeature([], tf.int64),
                'image_height': tf.io.FixedLenFeature([], tf.int64),
                'label_image': tf.io.FixedLenFeature([], tf.string),
                'original_image': tf.io.FixedLenFeature([], tf.string)}

    return tf.io.parse_single_example(example_item, features)


def _split_dataset(image_dataset, file_number, image_shape):
    split_index = file_number // 3
    X_image_train = np.zeros(shape=(file_number - split_index,) + image_shape, dtype='uint8')
    Y_label_train = np.zeros(shape=(file_number - split_index,) + image_shape, dtype='uint8')

    X_image_val = np.zeros(shape=(split_index,) + image_shape, dtype='uint8')
    Y_label_val = np.zeros(shape=(split_index,) + image_shape, dtype='uint8')

    for index, image_features in enumerate(image_dataset):
        image_value = tf.image.decode_jpeg(image_features['label_image']).numpy()
        label_value = tf.image.decode_jpeg(image_features['original_image']).numpy()
        if index < split_index:
            Y_label_val[index] = label_value
            X_image_val[index] = image_value
        else:
            X_image_train[index - split_index] = label_value
            Y_label_train[index - split_index] = image_value

    return (X_image_train, Y_label_train), (X_image_val, Y_label_val)


def crop_image(current_image, crop_shape):
    #current_image = wrap_bytes(current_image)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " + str(current_image))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " + str(type(current_image)))
    current_image = tf.image.decode_jpeg(current_image).numpy()
    #current_image = tf.compat.v2.make_ndarray(current_image)
    #current_image = tf.make_ndarray(current_image)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" + str(type(current_image)))
    height, width, _ = current_image.shape
    crop_height, crop_width, _ = crop_shape
    diff_height = (height - crop_height) // 2
    diff_width = (width - crop_width) // 2
    image_cropped = current_image[diff_height: crop_height + diff_height, diff_width: crop_width + diff_width]
    #image_cropped = tf.image.encode_jpeg(i)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&" + str(type(current_image)))
    print("******************************"  + str(type(tf.image.encode_jpeg(image_cropped))))
    #encode_image = cv2.imencode(".jpg", current_image)
   # current_byte_image = list(encode_image).tobytes()
    return image_cropped.tobytes()





def build_data():
    height, width, = ModelRequirements.CROP_HEIGHT, ModelRequirements.CROP_WIGHTS
    file_list = glob.glob(ModelRequirements.UPSCALED_IMAGE_PATH + '/*.jpg')
    for file in file_list:
        # current_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        with tf.io.gfile.GFile(file, 'rb') as f:
            image_data = f.read()
            #'Hello world'.replace('world', 'Guido')
        new_file = file.replace('output', 'input').split('_c_')[0] + "_c.jpg"
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" + str(new_file))
        with tf.io.gfile.GFile(new_file, 'rb') as f:
            image_data_source = f.read()

        image_data_source = crop_image(image_data_source,(height,width,3))
        image_data = crop_image(image_data, (height, width, 3))

        serialized_example = _serialize_example(width, height, image_data_source, image_data)
        record_file = os.path.join(ModelRequirements.TFRECORDS_PATH,
                                   os.path.basename(file).replace(".jpg", '.tfrecord'))
        #for creating batch of img in single record
        """
        ranges = []
        while len(ranges) < ModelRequirements.RECORDS_BATCH_SIZE:
            ranges.append(record_file)
       """
        writer = tf.io.TFRecordWriter(record_file)
        writer.write(serialized_example)
        """
        writer = tf.io.TFRecordWriter(ranges)
        writer.write(serialized_example)
        ranges.clear()
        """
class CreateSetOfRecordsIntoOne():
    def create_global_record(self): \
            # Create dataset from multiple .tfrecord files
        list_of_tfrecord_files = glob.glob(ModelRequirements.TFRECORDS_PATH + '/*.tfrecord')
        dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)

        # Save dataset to .tfrecord file
        filename = 'test.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(dataset)


class ExponentDecay():
    def __init__(self, initAlpha=0.01, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        alpha = self.initAlpha * np.exp(-self.power * epoch)
        # return the new learning rate
        return float(alpha)


class StepDecay():
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        # return the learning rate
        return float(alpha)

if __name__ == "__main__":
    build_data()