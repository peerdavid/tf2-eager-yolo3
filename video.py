import time
import argparse
import tensorflow as tf
import cv2

from yolo.utils.box import visualize_boxes
from yolo.config import ConfigParser



argparser = argparse.ArgumentParser(
    description='Detect objects with YoloV3 from a video stream')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/predict_coco.json",
    help='config file')

argparser.add_argument(
    '-k',
    '--camera',
    default=0,
    help='camera that should be used for stream')



def main(args):
    """ Show detected objects with boxes, lables and prediction scores in a vide stream
    """
    # 1. create yolo model & load weights
    print("Create YoloV3 model")
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)
    
    cap = cv2.VideoCapture(args.camera)
    times = []
    while True:
        _, image = cap.read()

        t1 = time.time()
        image = image[:,:,::-1]
        boxes, labels, probs = detector.detect(image, 0.5)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())
        image = cv2.putText(image, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
     args = argparser.parse_args()
     main(args)