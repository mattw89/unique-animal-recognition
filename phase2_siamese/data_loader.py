from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Parse the input file name
def _read_label_file(file, delimiter):
  f = open(file, "r")
  filepaths1 = []
  filepaths2 = []
  labels = []
  for line in f:
    tokens = line.rstrip().split(delimiter)
    filepaths1.append(tokens[0])
    filepaths2.append(tokens[1])
    
    labels.append(float(tokens[2]))
  return filepaths1,filepaths2, labels

def read_inputs(is_training, args):
  filepaths1, filepaths2, labels = _read_label_file(args.data_info, args.delimiter)

  filenames1 = [os.path.join(args.path_prefix,i) for i in filepaths1]
  filenames2 = [os.path.join(args.path_prefix,i) for i in filepaths2]

  # Create a queue that produces the filenames to read.
  if is_training:
    filename_queue = tf.train.slice_input_producer([filenames1, filenames2, labels], shuffle= args.shuffle, capacity= 1024)
  else:
    filename_queue = tf.train.slice_input_producer([filenames1, filenames2, labels], shuffle= False,  capacity= 1024, num_epochs =1)

  # Read examples from files in the filename queue.
  file_content1 = tf.read_file(filename_queue[0])
  file_content2 = tf.read_file(filename_queue[1])
  # Read JPEG or PNG or GIF image from file
  reshaped_image1 = tf.to_float(tf.image.decode_jpeg(file_content1, channels=args.num_channels))
  reshaped_image2 = tf.to_float(tf.image.decode_jpeg(file_content2, channels=args.num_channels))
  # Resize image to 256*256
  reshaped_image1 = tf.image.resize_images(reshaped_image1, args.load_size)
  reshaped_image2 = tf.image.resize_images(reshaped_image2, args.load_size)

  label = tf.cast(filename_queue[2], tf.float32)
  img_info1 = filename_queue[0]
  img_info2 = filename_queue[1]

  if is_training:
    reshaped_image1 = _train_preprocess(reshaped_image1, args)
    reshaped_image2 = _train_preprocess(reshaped_image2, args)
  else:
    reshaped_image1 = _test_preprocess(reshaped_image1, args)
    reshaped_image2 = _test_preprocess(reshaped_image2, args)
   # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(5000*
                           min_fraction_of_examples_in_queue)
  #print(batch_size)
  print ('Filling queue with %d images before starting to train. '
         'This may take some times.' % min_queue_examples)
  batch_size = args.chunked_batch_size if is_training else args.batch_size

  # Load images and labels with additional info 
  if hasattr(args, 'save_predictions') and args.save_predictions is not None:
    images1, images2, label_batch, info1, info2 = tf.train.batch(
        [reshaped_image1, reshaped_image2, label, img_info1, img_info2],
        batch_size= batch_size,
        num_threads=args.num_threads,
        capacity=min_queue_examples+3 * batch_size,
        allow_smaller_final_batch=True if not is_training else False)
    return images1, images2, label_batch, info1, info2
  else:
    images1, images2, label_batch = tf.train.batch(
        [reshaped_image1, reshaped_image2, label],
        batch_size= batch_size,
        allow_smaller_final_batch= True if not is_training else False,
        num_threads=args.num_threads,
        capacity=min_queue_examples+3 * batch_size)
    return images1, images2, label_batch


def _train_preprocess(reshaped_image, args):
  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  reshaped_image = tf.random_crop(reshaped_image, [args.crop_size[0], args.crop_size[1], args.num_channels])

  # Randomly flip the image horizontally.
  #reshaped_image = tf.image.random_flip_left_right(reshaped_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  reshaped_image = tf.image.random_brightness(reshaped_image,
                                               max_delta=63)
  # Randomly changing contrast of the image
  reshaped_image = tf.image.random_contrast(reshaped_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  reshaped_image = tf.image.per_image_standardization(reshaped_image)

  # Set the shapes of tensors.
  reshaped_image.set_shape([args.crop_size[0], args.crop_size[1], args.num_channels])
  #read_input.label.set_shape([1])
  return reshaped_image


def _test_preprocess(reshaped_image, args):

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         args.crop_size[0], args.crop_size[1])

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([args.crop_size[0], args.crop_size[1], args.num_channels])

  return float_image

