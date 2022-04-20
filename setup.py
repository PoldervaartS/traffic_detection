from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

"""
selectable build system

"""
import os

# choose package to build here
packages_to_build = {
    "yolov4tf": False,
    "darknet": True,
    "opencv" : False,
    "tensorflow": False, # TODO
}

__version__ = "0.0.1"
# ----------------------------------------



# stores current working directory for later recovery
cwd = os.getcwd()
home = os.path.expanduser("~")
output_packages = []

# -- builds YOLOv4, adds output packages.
if packages_to_build["yolov4tf"]:
    print("################## Binding YOLOv4 Files ########################")
    os.chdir(cwd)
    os.system("python3 bind.py build")

    output_packages.extend([
        'yolov4',
        'yolov4.common',
        'yolov4.common.metalayer',
        'yolov4.tf',
        'yolov4.tf.layers',
        'yolov4.tf.utils'
    ])

# -- Builds darknet, adds output packages
if packages_to_build["darknet"]:
    print("################## Running make in darknet #####################")
    os.chdir(cwd + "/src/darknet")
    os.system("make")

    output_packages.extend([
        "darknet"
    ])

# -- Builds opencv, adds output packages
if packages_to_build["opencv"]:
    #   TODO, perhaps further build is needed?
    #   TODO: current opencv build also causes conflicts with already installed version. 
    #               Perhaps rename this so that you can use specifically opencv34.cv2?
    output_packages.extend(["opencv.build"])


# returns to original directory
os.chdir(cwd)




setup_args = generate_distutils_setup(
    packages = output_packages,
    package_dir={'': 'src'},
    version=__version__,
)

setup(**setup_args)