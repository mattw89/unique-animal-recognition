"""Evaluating a trained model on the test data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf
import argparse
import arch
import data_loader
import sys


def evaluate(args):

  # Building the graph
  with tf.Graph().as_default() as g, tf.device('/cpu:0'):
    # Get images and labels
    images1, images2, labels, urls1, urls2 = data_loader.read_inputs(False, args)

    # Performing computations on a GPU
    with tf.device('/gpu:0'):
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = arch.get_model(images1, 0.0, False, args)

        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(args.log_dir, g)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

      ckpt = tf.train.get_checkpoint_state(args.log_dir)

      # Load the latest model
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)

      else:
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()

      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      step = 0
      predictions_format_str = ('%s,%s\n')
      batch_format_str = ('Batch Number: %d')
      
      out_file = open(args.save_predictions,'w')
      while step < args.num_batches and not coord.should_stop():
        features, urls_values, label_values = sess.run([logits, urls1, labels])
        for i in range(0,urls_values.shape[0]):
          out_file.write(predictions_format_str%(urls_values[i].decode("utf-8") , ','.join(map(str,features[i]))))
          out_file.flush()
        print(batch_format_str%(step,))
        sys.stdout.flush()
        step += 1

      out_file.close()
 
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      coord.request_stop()
      coord.join(threads)

def main():
  parser = argparse.ArgumentParser(description='Process Command-line Arguments')
  parser.add_argument('--load_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width and height of images for loading from disk')
  parser.add_argument('--crop_size', nargs= 2, default= [224,224], type= int, action= 'store', help= 'The width and height of images after random cropping')
  parser.add_argument('--batch_size', default= 64, type= int, action= 'store', help= 'The testing batch size')
  parser.add_argument('--num_classes', default= 100, type= int, action= 'store', help= 'The number of classes')
  parser.add_argument('--top_n', default= 2, type= int, action= 'store', help= 'Top N accuracy')
  parser.add_argument('--num_channels', default= 3, type= int, action= 'store', help= 'The number of channels in input images')
  parser.add_argument('--num_batches' , default=-1 , type= int, action= 'store', help= 'The number of batches of data')
  parser.add_argument('--path_prefix' , default='./../../images/', action= 'store', help= 'The prefix address for images')
  parser.add_argument('--delimiter' , default=',', action = 'store', help= 'Delimiter for the input files')
  parser.add_argument('--data_info'   , default= 'train_labels_siamese_mf.csv', action= 'store', help= 'File containing the addresses and labels of testing images')
  parser.add_argument('--num_threads', default= 4, type= int, action= 'store', help= 'The number of threads for loading data')
  parser.add_argument('--architecture', default= 'resnet', help='The DNN architecture')
  parser.add_argument('--depth', default= 50, type= int, help= 'The depth of ResNet architecture')
  parser.add_argument('--log_dir', default='./log/', action= 'store', help='Path for saving Tensorboard info and checkpoints')
  parser.add_argument('--save_predictions', default='./output.csv', action= 'store', help= 'Save top-5 predictions of the networks along with their confidence in the specified file')

  args = parser.parse_args()
  args.num_samples = sum(1 for line in open(args.data_info))
  if args.num_batches==-1:
    if(args.num_samples%args.batch_size==0):
      args.num_batches= int(args.num_samples/args.batch_size)
    else:
      args.num_batches= int(args.num_samples/args.batch_size)+1

  print(args)

  evaluate(args)


if __name__ == '__main__':
  main()
