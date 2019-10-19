# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
from yolo.train import train_fn
from yolo.config import ConfigParser


import os
from ctypes import *

TF_GPU_SETUP = os.getenv("TF_GPU_SETUP", "local")
if(TF_GPU_SETUP == "ifi"):
  lib1 = cdll.LoadLibrary('/software-shared/cudnn10-1/lib64/libcudnn.so.7')
  lib2 = cdll.LoadLibrary('/software-shared/lib-cuda10.0/libcublas.so.10.0')
  lib3 = cdll.LoadLibrary('/software-shared/lib-cuda10.0/libcufft.so.10.0')
  lib4 = cdll.LoadLibrary('/software-shared/lib-cuda10.0/libcurand.so.10.0')
  lib5 = cdll.LoadLibrary('/software-shared/lib-cuda10.0/libcusolver.so.10.0')
  lib6 = cdll.LoadLibrary('/software-shared/lib-cuda10.0/libcusparse.so.10.0')
  lib7 = cdll.LoadLibrary('/software-shared/lib-cuda10.0/libcudart.so.10.0')

if(TF_GPU_SETUP == "iis"):
  lib1 = cdll.LoadLibrary('/home/david.peer/bin/cuda/lib64/libcudnn.so.7')

  
argparser = argparse.ArgumentParser(
    description='train yolo-v3 network')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/svhn.json",
    help='config file')


if __name__ == '__main__':
    args = argparser.parse_args()
    config_parser = ConfigParser(args.config)

    # Select device and log placement
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    device = "/CPU:0" if len(gpus) == 0 else "/GPU:0"
    device = "/GPU:1" if TF_GPU_SETUP == "iis" else device
    
    # 1. create generator
    with tf.device(device):
      train_generator, valid_generator = config_parser.create_generator()
      
      # 2. create model
      model = config_parser.create_model()

      # 3. training
      learning_rate, save_dname, n_epoches = config_parser.get_train_params()
      train_fn(model,
              train_generator,
              valid_generator,
              learning_rate=learning_rate,
              save_dname=save_dname,
              num_epoches=n_epoches)

