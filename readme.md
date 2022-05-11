## About

GM Autodrive Challenge is a competition inviting various research focused schools in competition to develop the best autonomous vehicle. This repository represents the ROS package for Traffic Sign and Signal recognition 2021 utilizing Yolov4 architecture to identify relevant visual information such as speed limit signs and traffic lights for autonomous driving. Worked alongside Angelo Yang https://www.linkedin.com/in/ryang42, Nicholas Vacek https://www.linkedin.com/in/jung-hoon-seo-5ab813178, and Jung Hoon Seo https://www.linkedin.com/in/nicholas-vacek-314a23187
We ultimately managed to place 3rd in great success and more information can be found here! http://autodrive.tamu.edu/ 

## Demo

[![Link to Youtube with Video Demonstration of project](https://img.youtube.com/vi/pdphwxuSitw/0.jpg)](https://www.youtube.com/watch?v=pdphwxuSitw)


## Get Darknet weight and SVM files here:
- https://drive.google.com/drive/folders/1JYdu6nVTAw-i4xpqJ-Zrzcw2Wiwc3alU?usp=sharing
	- use the best weights zip
	- when unzipped into /models change obj.data to have the right file path
- https://drive.google.com/file/d/14GAAv7weqv6pUgdmdOUirZx_qOCrAiqp/view?usp=sharing 
	- only 1 file
	- Integrated the SVM module of [Speed Limit Classification SVM from Capstone](https://github.tamu.edu/AutoDrive-II-Common/speed_limit_classification_mirror) into message output
### Both go into traffic_detection/models folder

### Weight suffix descriptions:
- (name)_**(number, multiple of 10,000)**.weights
	- weight saved at that number of iterations.
- (name)_**last**.weights
	- the last or most recent set of weights. May not be the best!
- (name)_**ema**.weights
	- exponential moving average of all training weights.
- (name)_**best**.weights
	- the best set of weights thus far, before training is done.
- (name)_**final**.weights
	- the best set of weights, once training is done.


put selected `.weight`, `classes.names`, and `<model>.cfg` files in the `models/` folder.
Make sure to update the references in `scripts/detect_ros.py` to point to these files!

## To test:
- bunch of packages: `todo` just install whatever is missing for now
- CFG files need tweaking to work correctly for inference. THis is just the nature of how it works.
- If using darknet rather than tensorflow, it imports an obj.data file, which can be set.
	- See: [this link](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects), scroll to step 3.
	- Train, valid, and backup files don't matter.
	- classes & names need to be set correctly!

## Requirements:
- `sudo apt install libopencv-dev`
### Specific versions required to install tensorflow 2.1 on python 2.7 
- `python -m pip install --upgrade pip==19.9.1`
- `python -m pip install keras==2.2.4`
- `python -m pip install tensorflow`

# tensorflow-based prediction with yolov4-csp

Using this: https://github.com/hhk7734/tensorflow-yolov4 branch to import the model into tensorflow.

## To change version of OpenCV found
- Command to remove old version of opencv from python path without deleting it:
- `sudo mv /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so /opt/ros/kinetic/lib/python2.7/cv2.so`
- [Follow these install instructions for OpenCV](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)
- If the build doesn't work:
	- `cd ~/catkin_ws/src/traffic_detection/src/opencv/build`
	- `cmake ../opencv-3.4.11`
	- `cmake --build .`
- `cd ~/catkin_ws/src/traffic_detection/src/opencv/build && sudo make install`
- Need to Catkin Make with the local OpenCV install specified:
- `catkin_make -DOpenCV_DIR=~/catkin_ws/src/traffic_detection/src/opencv/build`



### Running inference inside ROS with ROSBag:
- `cd ~/catkin_ws/src && git clone <this repo>` to clone it into the correct location.
- `source ~/catkin_ws/devel/setup.bash` to start environment
- `cd ~/catkin_ws/src/traffic_detection && python bind.py build` 	-> should no longer be needed (handled by setup.py)
- `cd ~/catkin_ws/src/traffic_detection/src/darknet && make`		-> should no longer be needed (handled by setup.py)
- `cd ~/catkin_ws && catkin_make`
- edit `traffic_detect.launch` to point to where the bag file is. Make sure 
- edit `darknet_detect_ros.py` line 43 if rosbag has different topics for the camera image
- `roslaunch traffic_detection darknet_detect.launch`
