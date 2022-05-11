#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

import cv2
import numpy as np
from PIL import Image as Img


# ROS related imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from message_filters import TimeSynchronizer, Subscriber
from sign_msg_output_ros import SignMessageOutput
from speed_limit_svm import SpeedLimitSVM



# model import
from darknet import darknet

MODEL_ROOT =  os.path.join(os.path.dirname(sys.path[0]),'models') + "/"
THRESHOLD = 0.30

# creates model.
network, class_names, class_colors = darknet.load_network(
    MODEL_ROOT+"v4csp-lisats.cfg", # .cfg file
    MODEL_ROOT+"obj.data",          # .data file
    MODEL_ROOT+"v4csp-lisats_best.weights",    # .weights file           
    batch_size=1        #batch size
)

class Detector:

    def __init__(self):
        
        self.bridge = CvBridge()
        # inputting info, image
        self.vis_pub = rospy.Publisher("visualize_image",Image, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_yolo, queue_size=1, buff_size=2**24)
        
        # outputting sign info management
        self.signMessageOutput = SignMessageOutput(class_names)
        self.speedLimitSVM = SpeedLimitSVM()


    def image_detection(self, image, network, class_names, class_colors, thresh):
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        crop_window_y1, crop_window_y2 = 0, 2000
        crop_window_x1, crop_window_x2 = int(1000), int(3000)

        width = darknet.network_width(network)
        height = darknet.network_height(network)
        darknet_image = darknet.make_image(width, height, 3)


        # img_copy = image_np
        # img_copy = cv2.rectangle(img_copy, (crop_window_x1,crop_window_y1), (crop_window_x2,crop_window_y2), (255, 0, 255), thickness=2)
        image_np = image[crop_window_y1:crop_window_y2, crop_window_x1:crop_window_x2]

        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)

        darknet.free_image(darknet_image)
        image = darknet.draw_boxes(detections, image_resized, class_colors)
        for i in range(len(detections)):

            detection = detections[i]
            value = 0
            if detection[0] == 'speedLimit':
                X_coord, Y_coord, signWidth, signHeight = detection[2]
                # image sized down to 512, 512
                # original height would be 
                originalWidth = crop_window_x2 - crop_window_x1
                originalHeight = crop_window_y2 - crop_window_y1

                croppedImg = image_rgb[int( (Y_coord - signHeight/2) * originalHeight/height):int( (Y_coord + signHeight/2)* originalHeight/height), int( (X_coord-signWidth/2) * originalWidth/width):int( (X_coord + signWidth/2) * originalWidth/width)]
                # cv2.imwrite('/home/autodrive/noetic_ws/src/traffic_detection/test.png',croppedImg)
                value = self.speedLimitSVM.predictImg(croppedImg)

            detections[i] = [*detection, value]
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


    def image_yolo(self, data):
        try:
            cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        time0 = rospy.get_rostime()
        # [ label, prob, (X_coord, Y_coord, width, heigh), value] - Darknet bounding box format
        # -- x,y,w,h in pixel count. unsure if x & y are top left or center
        image_darknet, detections = self.image_detection(
            cv_frame, network, class_names, class_colors, THRESHOLD
            )
        time1 = rospy.get_rostime()
        scaledWidth, scaledHeight, _ = image_darknet.shape
        if(len(detections) > 0):
            self.signMessageOutput.setTimeToNow()
            [ self.signMessageOutput.addDarknetbboxToMessage(detection, scaledWidth, scaledHeight) for detection in detections ]

        try:
            image_out = self.bridge.cv2_to_imgmsg(image_darknet,"bgr8")
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
