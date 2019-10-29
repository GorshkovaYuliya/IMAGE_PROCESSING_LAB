import os
import glob
import logging
import pathlib

import tensorflow as tf


PATH = "C:\\Users\\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\output\\downscale\\"
TF_RECORDS_PATH = "C:\\Users\\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\output\\upscale\\"


data_root = pathlib.Path(PATH)



print(data_root)
for item in data_root.iterdir():
  print(item)
  import random

all_image_paths = list(data_root.glob("*"))
all_image_paths = [str(path) for path in all_image_paths]
tf.random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)
print(all_image_paths[0])
dataset = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
print(dataset)

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

it = iter(dataset)

print(next(it).numpy())