# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt

from yolo.utils.box import visualize_boxes
from yolo.config import ConfigParser
from yolo.dataset.augment import resize_image


argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/predict_coco.json",
    help='config file')

argparser.add_argument(
    '-i',
    '--image',
    default='tests/dataset/iis/test/imgs/1.png',
    help='path to image file')


if __name__ == '__main__':
    args = argparser.parse_args()
    image_path   = args.image
    
    # Create model
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)
    
    # Load image
    image = cv2.imread(image_path)
    image, _ = resize_image(image, None, config_parser.get_net_size(), keep_ratio=True)
    image = image[:,:,::-1]
   
    # Predict
    boxes, labels, probs = detector.detect(image, 0.9)
    visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())
    plt.imshow(image)
    plt.show()

 


