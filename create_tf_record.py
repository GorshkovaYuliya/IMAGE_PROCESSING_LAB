from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

from modelrequirements import ModelRequirements


def define_flags():
    tf.app.flags.DEFINE_string('data_directory', '/tmp/', 'Data directory')
    tf.app.flags.DEFINE_float('validation_ratio', '0.1', 'Amount of validation data')
    tf.app.flags.DEFINE_string('output_directory', '/tmp/', 'Output data directory')
    tf.app.flags.DEFINE_integer('train_shards', 10, 'Number of shards in training TFRecord files.')
    tf.app.flags.DEFINE_integer('validation_shards', 2, 'Number of shards in validation TFRecord files.')
    tf.app.flags.DEFINE_integer('num_threads', 2, 'Number of threads to preprocess the images.')
    tf.app.flags.DEFINE_string('labels_file', 'labels', 'Labels file')
    return  tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
    """Build an Example proto for an example.

    """



    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(ModelRequirements.colorspace)),
        'image/channels': _int64_feature(ModelRequirements.channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
        'image/format': _bytes_feature(tf.compat.as_bytes(ModelRequirements.image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example



def _is_png(filename):
    """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
    return filename.endswith('.png')


def _process_image(filename, coder):
    """Process a single image file"""
    # Read the image file.
    with tf.gfile.GFil(filename, 'rb') as f:
        image_data = f.read()

    # Convert any PNG to JPEG's for consistency.
    if _is_png(filename):
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)
    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width, filename


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
    """
    Processes and saves list of images as TFRecord in 1 thread.
  """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    #int result
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(define_flags().output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected error while decoding %s.' % filename)
                continue

            example = _convert_to_example(filename, image_buffer, label,
                                          text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()




def _process_image_files(name, filenames, texts, labels, usage_index, num_shards):
    """Process and save list of images as TFRecord of Example protos.
  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    filenames = [filenames[i] for i in range(len(usage_index)) if usage_index[i]]
    texts = [texts[i] for i in range(len(usage_index)) if usage_index[i]]
    labels = [labels[i] for i in range(len(usage_index)) if usage_index[i]]

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), define_flags().num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (define_flags().num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file, val_ratio):
    """Build a list of all images files and labels in the data set.
    Args:
      data_dir: string, path to the root directory of images.
        Assumes that the image data set resides in JPEG files located in
        the following directory structure.
        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg
        where 'dog' is the label associated with these images.
      labels_file: string, path to the labels file.
        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          dog
          cat
          flower
        where each line corresponds to a label. We map each label contained in
        the file to an integer starting with the integer 0 corresponding to the
        label contained in the first line.
    Returns:
      filenames: list of strings; each string is a path to an image file.
      texts: list of strings; each string is the class, e.g. 'dog'
      labels: list of integer; each integer identifies the ground truth.
  """
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.GFile(labels_file, 'r').readlines()]

    labels = []
    filenames = []
    texts = []
    train_split = np.empty((0,), dtype=np.bool)

    # Leave label index 0 empty as a background class.
    label_index = 1

    # Construct the list of JPEG files and labels.
    for text in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)
        train_indexes = np.ones(len(matching_files), dtype=bool)
        train_indexes[:int(len(matching_files) * val_ratio)] = 0
        np.random.shuffle(train_indexes)
        train_split = np.append(train_split, train_indexes)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (label_index, len(labels)))
        label_index += 1

    print('Found %d JPEG files across %d labels inside %s.' % (len(filenames), len(unique_labels), data_dir))
    return filenames, texts, labels, train_split


def _shuffle(filenames, texts, labels, train_split):
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    return [filenames[i] for i in shuffled_index], \
           [texts[i] for i in shuffled_index], \
           [labels[i] for i in shuffled_index], \
           [train_split[i] for i in shuffled_index]


def main(unused_argv):
    assert not define_flags().train_shards % define_flags().num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not define_flags().validation_shards % define_flags().num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.validation_shards')
    print('Saving results to %s' % define_flags().output_directory)

    # Get all files and split it to validation and training data
    filenames, texts, labels, train_split = _find_image_files(
        define_flags().data_directory, define_flags().labels_file, define_flags().validation_ratio
    )

    for file in os.listdir(ModelRequirements.SOURCE_IMAGE_PATH):
        compare_images(file)
    # Run it!
    _process_image_files('train', filenames, texts, labels, train_split, define_flags().train_shards)
    _process_image_files('validation', filenames, texts, labels, np.logical_not(train_split), define_flags().validation_shards)

def compare_images(file):
    filenames = np.empty(5, dtype=str)
    for file_upscaled in os.listdir(ModelRequirements.UPSCALED_IMAGE_PATH):
        if file[:-3] == file_upscaled[:-15] or file[:-3] == file_upscaled[:-14]:
           filenames.append(file_upscaled)
    return filenames, file



if __name__ == '__main__':
    tf.app.run()
