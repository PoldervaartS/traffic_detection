import cv2
from sklearn.svm import SVC
import pickle
import os
import sys

class SpeedLimitSVM:

    def __init__(self) -> None:
        MODEL_ROOT =  os.path.join(os.path.dirname(sys.path[0]),'models') + "/"
        filename = MODEL_ROOT + "best_svm.sav"
        self.svm = pickle.load(open(filename, 'rb'))
        # self.test()

    def predictImg(self, coloredTrafficSign):
        image = cv2.cvtColor(coloredTrafficSign, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (80,100), interpolation=cv2.INTER_AREA).ravel()
        prediction = self.svm.predict(image.reshape(1,-1))[0]
        return prediction

    def test(self):
        # for sanity testing purposes, can download image here https://github.tamu.edu/AutoDrive-II-Common/speed_limit_classification_mirror/blob/master/SVM/speedLimit25_1398988309.avi_image3.png
        # Should predict 25
        image = cv2.imread("speedLimit25_1398988309.avi_image3.png")
        prediction = self.predictImg(image)
        print(f'Predicted Speed: {prediction}')

