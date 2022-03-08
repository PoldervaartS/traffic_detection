#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

import cv2
import numpy as np
from PIL import Image as Img

#ROS related imports
import rospy
from std_msgs.msg import String , Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from message_filters import TimeSynchronizer, Subscriber
from vision_msgs.msg import sign_array_msg,sign_detection_msg

#model import
MODEL_ROOT =  os.path.join(os.path.dirname(sys.path[0]),'models') + "/"
CONFIDENCE_THRESHOLD = 0.15
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
FRAME_WIDTH = 512
FRAME_HEIGHT = 512
SCALE = 1/255.0

class_names = []
with open(MODEL_ROOT+"classes.names", "r") as f:
    class_names = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNet(MODEL_ROOT+"v4tiny-lisats_best.weights", MODEL_ROOT+"v4tiny-lisats.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
outNames = net.getUnconnectedOutLayersNames()

def process_frame(frame, outs):
    #puts bounding boxes on detected objects
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPrediction(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if class_names:
            assert(classId < len(class_names))
            label = '%s: %s' % (class_names[classId], label)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'DetectionOutput':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for out in outs:
            for detection in out[0, 0]:
                confidence = detection[2]
                if confidence > CONFIDENCE_THRESHOLD:
                    left = int(detection[3])
                    top = int(detection[4])
                    right = int(detection[5])
                    bottom = int(detection[6])
                    width = right - left + 1
                    height = bottom - top + 1
                    if width * height <= 1:
                        left = int(detection[3] * frameWidth)
                        top = int(detection[4] * frameHeight)
                        right = int(detection[5] * frameWidth)
                        bottom = int(detection[6] * frameHeight)
                        width = right - left + 1
                        height = bottom - top + 1
                    classIds.append(int(detection[1]) - 1)  # Skip background label
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'Region':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        #non mamximum suppression to get the box with the highest confidence
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPrediction(classIds[i], confidences[i], left, top, left + width, top + height)
        #send resulting bounding box to be drawn


class Detector:

  def __init__(self):
    
    self.bridge = CvBridge()

    self.vis_pub = rospy.Publisher("visualize_image",Image, queue_size=1)
    self.image_sub = rospy.Subscriber("/front_camera/image_raw",Image,self.image_yolo,queue_size=1,buff_size=2**24)

    self.object_pub = rospy.Publisher("objects", sign_array_msg, queue_size=1)

  def image_yolo(self,data):
    try:
      cv_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    time0 = rospy.get_rostime()

    height = cv_frame.shape[0]
    width = cv_frame.shape[1]

    blob = cv2.dnn.blobFromImage(image=cv_frame, size = (FRAME_WIDTH,FRAME_HEIGHT), swapRB=True,crop=False,ddepth=cv2.CV_32F)
    #ddepth - cv2.CV_32F OR cv2.CV_8U

    net.setInput(blob,scalefactor=SCALE)
    if net.getLayer(0).outputNameToIndex('im_info') != -1:
        resized_frame = cv2.resize(cv_frame, (width, height))
        cv_frame = resized_frame
        net.setInput(np.array([[height,width,1.6]], dtype = np.float32), 'im_info')
    outs = net.forward(outNames)

    process_frame(cv_frame, outs)

    time1 = rospy.get_rostime()

    try:
      image_out = self.bridge.cv2_to_imgmsg(cv_frame, "bgr8")
    except CvBridgeError as e:
      print(e)

    self.vis_pub.publish(image_out)

def main(args):
  rospy.init_node('detector_node', anonymous=True)
  obj = Detector()
  print("Started working!")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("ShutDown")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)