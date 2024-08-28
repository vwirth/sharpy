#! /bin/bash
# use -D option to manally set cmake variables such as:
# <package_name>_DIR or
# <module-name>_MODULE
cmake -B build -S . \
-DWITH_CUDA=True \
-DOPENCV_MODULE=True \
-DCAMERA_MODULE=False \
-DOPENGL_MODULE=True \
-DDNN_MODULE=True
