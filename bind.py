from setuptools import setup
from pybind11.setup_helpers import ParallelCompile, Pybind11Extension, build_ext

root_dir = "src/yolov4/"

__version__ = "0.1"

ParallelCompile("NPY_NUM_BUILD_JOBS").install()

ext_modules = [
    Pybind11Extension(
        # Ref: distutils.extension.Extension
        name="yolov4.common._common",
        sources=sorted(glob(root_dir+"c_src/**/*.cpp", recursive=True)),
        include_dirs=[root_dir+"c_src/"],  # -I
        define_macros=[(("VERSION_INFO", __version__))],  # -D<string>=<string>
        undef_macros=[],  # [string] -D<string>
        library_dirs=[],  # [string] -L<string>
        libraries=[],  # [string] -l<string>
        runtime_library_dirs=[],  # [string] -rpath=<string>
        extra_objects=[],  # [string]
        extra_compile_args=[],  # [string]
        extra_link_args=[],  # [string]
    ),
]


setup(
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

# move compiled file over.
import glob
import shutil

# .so for linux, .pyd for windows
src = glob.glob("build/lib.*/yolov4/common/*.so")[0]

dest = "src/yolov4/common"
shutil.copy(src, dest)
shutil.rmtree("build")