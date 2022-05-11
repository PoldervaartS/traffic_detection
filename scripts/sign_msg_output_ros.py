import rospy
from std_msgs.msg import String , Header
from vision_msgs.msg import sign_array_msg,sign_detection_msg

class SignMessageOutput:
    # sign size in inches
    signHeights = {
        'stop' :  24,
        'stopAhead' : 30,
        'speedLimit' : 30,
        'pedestrianCrossing' : 24,
        'turnLeft' : 24,
        'turnRight' : 24,
        'slow' : 24,
        'rightLaneMustTurn' : 30,
        'signalAhead' : 24,
        'keepRight' : 48,
        'laneEnds' : 36,
        'school' : 36,
        'schoolSpeedLimit' : 30,
        'merge' : 36,
        'curveRight' : 24,
        'curveLeft' : 24,
        'yield' : 36,
        'yieldAhead' : 36,
        'thruMergeLeft' : 24,
        'thruMergeRight' : 24,
        'noLeftTurn' : 24,
        'noRightTurn' : 24,
        'doNotEnter' : 24,
        'doNotPass' : 48,
        'roundabout' : 24,
        'intersection' : 36,
    }
# stop
# yield
# pedestrianCrossing
# speedLimit
# turnRight
# turnLeft
# noRightTurn
# noLeftTurn
# doNotEnter
# rightLaneMustTurn not 100% sure on this

    signLabelsToInts = {
        # 0: 'Unknown', 
        'speedLimit': 1, 
        # 2: 'Speed Limit (KPH)', We don't have this in training model.
        'stop': 3,# 3: 'Stop',
        'yield': 4, 
        'noLeftTurn': 5, 
        'noRightTurn': 6,
        'doNotEnter': 7, 
        'turnLeft': 8,
        'turnRight': 9,
        #  10: 'Railroad Crossing',
        'pedestrianCrossing': 11,
        #  12: 'No Turn On Red',
        #  13: 'No U-Turn', 
        # 14: 'One Way (Left)',
        #  15: 'One Way (Right)',
        #  16: 'No Parking',
        #  17: 'Handicap Parking'
    }

    INCHTOMETERS = 0.0254

    def __init__(self, label_dict, cameraWidth=2048, cameraHeight=2048, focalLength=1508.48021):
        self.label_dict = label_dict
        self.cameraWidth = cameraWidth
        self.cameraHeight = cameraHeight
        self.detectedSigns = sign_array_msg()
        self.detectedSigns.s = []
        header_camera = Header()
        header_camera.frame_id = 'front_camera'
        self.detectedSigns.header = header_camera

        # FOCAL LENGTH OF camera according to camera matrix
        self.focalLength = focalLength
        
        # outputting info
        self.object_pub = rospy.Publisher("objects", sign_array_msg, queue_size=1)

    # USAGE: yolo bounding box is: [x_center, y_center, width, height, label index, probability.]
    # x, y, w, h are all a percentage of total image size [0, 1]
    def addbboxToMessage(self, box):
        if( self.detectedSigns.header.stamp is None):
            raise TimeStampError('The rosmessage does not have a timestamp\t use setTimeToNow() before formulating message')
        msg = sign_detection_msg()
        msg.what = self.label_dict[int(box[4])]

        pixelWidth = box[2] * self.cameraWidth
        msg.size_x = int(pixelWidth)
        x = box[0] * self.cameraWidth - pixelWidth/2
        msg.coor_x = int(x)

        pixelHeight = box[3] * self.cameraHeight
        msg.size_y = int(pixelHeight)
        y = box[1] * self.cameraHeight - pixelHeight/2
        msg.coor_y = int(y)
        inchDistance = self.focalLength * SignMessageOutput.signHeights[msg.what] / pixelHeight
        msg.z = inchDistance * SignMessageOutput.INCHTOMETERS
        msg.header = self.detectedSigns.header
        self.detectedSigns.s.append(msg)

    # [ label, prob, (X_coord, Y_coord, width, height), value] - Darknet bounding box format
    # -- x,y,w,h in pixel count. unsure if x & y are top left or center
    # scaledWidth & scaledHeight are due to having to remake & scale the image
    def addDarknetbboxToMessage(self, box, scaledWidth, scaledHeight, xCrop0, yCrop0):
        if( self.detectedSigns.header.stamp is None):
            raise TimeStampError('The rosmessage does not have a timestamp\t use setTimeToNow() before formulating message')
        msg = sign_detection_msg()
        # Do the map here. If not in set then just skip it?
        try:
            msg.type = self.signLabelsToInts[box[0]]
        except:
            msg.type = 0
        msg.value = box[3]

        msg.size_x = int(box[2][2] * self.cameraWidth/scaledWidth)
        msg.coor_x = int(box[2][0] * self.cameraWidth/scaledWidth) + xCrop0


        pixelHeight = box[2][3] * self.cameraHeight/scaledHeight
        msg.size_y = int(pixelHeight)
        msg.coor_y = int(box[2][1] * self.cameraHeight/scaledHeight) + yCrop0 - int(msg.size_y/2)
        inchDistance = self.focalLength * SignMessageOutput.signHeights[box[0]] / pixelHeight
        msg.z = inchDistance * SignMessageOutput.INCHTOMETERS
        msg.header = self.detectedSigns.header
        self.detectedSigns.s.append(msg)

    def publishMessages(self):
        if(len(self.detectedSigns.s) > 0):
            if( self.detectedSigns.header.stamp is None):
                raise TimeStampError('The rosmessage does not have a timestamp\t use setTimeToNow() before formulating message')
            self.object_pub.publish(self.detectedSigns)
            # clears the detectedSigns
            self.detectedSigns.s = []
    
    def setTimeToNow(self):
        self.detectedSigns.header.stamp = rospy.get_rostime()

    def setCameraDimensions(self, cameraWidth, cameraHeight, focalLength):
        self.cameraHeight = cameraHeight
        self.cameraWidth = cameraWidth
        self.focalLength = focalLength