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
    # Load yolo model with pretrained weights
    print("Create YoloV3 model")
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)
    
    # Open video stream
    cap = cv2.VideoCapture(args.camera)
    if (cap.isOpened()== False): 
        print("(Error) Could not open video stream")
        exit()

    # Detect objects in stream
    times = []
    detect = 0
    while True:

        # Capture every nth frame only because we are too slow 
        # to capture every frame...
        ret, image = cap.read()
        if not ret:
            print("(Error) Lost connection to video stream")
            break

        # Detect objects and measure timing
        if detect <= 0:
            t1 = time.time()
            boxes, labels, probs = detector.detect(image, 0.5)
            t2 = time.time()
            times.append(t2-t1)
            times = times[-20:]
            detect = 50
        detect -= 1

        # Display detected objects
        visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())
        image = cv2.putText(image, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('Frame', image)

        # Exit with 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break           

    cap.release()
    cv2.destroyAllWindows()


#
# M A I N
#
if __name__ == '__main__':
     args = argparser.parse_args()
     main(args)