#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

import cv2
import numpy as np
from PIL import Image as Img
from yolov4.tf import YOLOv4


# ROS related imports
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from message_filters import TimeSynchronizer, Subscriber
from sign_msg_output_ros import SignMessageOutput

MODEL_ROOT =  os.path.join(os.path.dirname(sys.path[0]),'models') + "/"
DETECT_THRESHOLD = 0.15


# creates model.
yolo = YOLOv4()
yolo.config.parse_names(MODEL_ROOT + "classes.names")
yolo.config.parse_cfg(MODEL_ROOT + "v4tiny-lisats.cfg")
yolo.make_model()
yolo.load_weights(MODEL_ROOT+"v4tiny-lisats_best.weights", weights_type="yolo")
"""
    Alternatives from drive, based on suffix:
    (name)_(number, multiple of 10,000).weights -> weight saved at that number of iterations.
    (name)_last.weights -> the last or most recent set of weights. May not be the best!
    (name)_ema.weights -> exponential moving average of all training weights.
    (name)_best.weights -> the best set of weights thus far, before training is done.
    (name)_final.weights -> the best set of weights, once training is done.
"""



# names can be accessed: converts int to string label
label_dict = yolo.config._names


def darknet2ROS(bboxes, dims):
    out_boxes = []

    # [x_center, y_center, width, height, label index, confidence]
    for bbox in bboxes:
        label = label_dict[int(bbox[4])]
        conf = bbox[5]
        width = int(dims[1] * bbox[2])
        height = int(dims[0] * bbox[3])

        # ROS uses top-left rather than center
        # calculates center x, then subtracts off half of width
        x = int((dims[1] * bbox[0]) - (width/2))
        # calculates center y, then adds on half of height
        y = int((dims[0] * bbox[1]) + (height/2))

        out_boxes.append([x, y, width, height, label, conf])
    return out_boxes



class Detector:

    def __init__(self):
        
        self.bridge = CvBridge()
        # inputting info, image
        self.vis_pub = rospy.Publisher("visualize_image",Image, queue_size=1)
        self.image_sub = rospy.Subscriber("/front_camera/image_raw", Image, self.image_yolo, queue_size=1, buff_size=2**24)
        
        # outputting sign info management
        self.signMessageOutput = SignMessageOutput(label_dict)

    def image_yolo(self, data):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        frame = cv2.cvtColor(cv_frame,cv2.COLOR_BGR2RGB) # cv2 uses bgr, everything else is rgb

        time0 = rospy.get_rostime()
        # fomat for bboxes_darknet from yolo is: [x_center, y_center, width, height, label index, probability.]
        # x, y, w, h are all a percentage of total image size [0, 1]
        bboxes_darknet = yolo.predict(frame, prob_thresh=DETECT_THRESHOLD)
        time1 = rospy.get_rostime()
        self.signMessageOutput.setTimeToNow()
        if(bboxes_darknet[0][5] > 0):
            # bboxes = darknet2ROS(bboxes_darknet, frame.shape)
            cv_frame = yolo.draw_bboxes(cv_frame, bboxes_darknet)
            [ self.signMessageOutput.addbboxToMessage(box) for box in bboxes_darknet ]

        try:
            image_out = self.bridge.cv2_to_imgmsg(cv_frame,"bgr8")
        except CvBridgeError as e:
            print(e)
        self.vis_pub.publish(image_out)
        self.signMessageOutput.publishMessages()
        
    

def main(args):
    rospy.init_node('detector_node')
    
    obj=Detector()
    print("Started working!")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ShutDown")
    cv2.destroyAllWindows()

if __name__=='__main__':
    main(sys.argv)
