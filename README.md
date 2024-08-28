![sharpy](assets/sharpy.png)

### [Project Page](https://vwirth.github.io/sharpy) | [Paper](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Wirth_ShaRPy_Shape_Reconstruction_and_Hand_Pose_Estimation_from_RGB-D_with_ICCVW_2023_paper.html) |  [Arxiv](https://arxiv.org/abs/2303.10042)

# ShaRPy Documentation

This is the official code repository of the paper [ShaRPy: Shape Reconstruction and Hand Pose Estimation from RGB-D with Uncertainty](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Wirth_ShaRPy_Shape_Reconstruction_and_Hand_Pose_Estimation_from_RGB-D_with_ICCVW_2023_paper.html).      
ShaRPy is a RGB-D Shape Reconstruction and hand Pose tracking system, which provides uncertainty estimates of the computed pose, e.g., when a finger is hidden or its estimate is inconsistent with the observations in the input.

![sharpy_viewer](assets/sharpy_viewer.gif)



# Structure
This project is structured as follows:
* `data`: Contains example data
* `assets`: Contains images for this documentation
* `code/build`: Location of the compiled files, including the executable binaries
* `code/src`: Location of source files, e.g. `.cpp` and `.cu` files
* `code/include`: Location of header files (without external libraries!)
* `code/external`: External libraries
* `code/examples`: Example code
* `code/cmake` (optional): Additional cmake files, e.g. to search external libraries on the host

# Modules
This repository contains different modules, which are disabled by default in the `CMakeLists.txt` file.    
If you want to enable a Module without modifying the `CMakeLists.txt` file, either pass those Cmake options in a manual Cmake command (see `cmake-template.sh`) or if you are using **VSCode** you can pass the Cmake options to the internal Cmake system via modifying `settings.json` file. You can check the `vscode` directory (in the root directory of this repository) for more details.
It is recommended that you copy the provided `vscode/settings.json` to your `.vscode/settings.json`.

The following modules are available:
* `OPENCV_MODULE` (required): Use the OpenCV and OpenCV contrib libraries
* `OPENGL_MODULE` (required): Use OpenGL
* `DNN_MODULE` (required): Use deep learning libraries, e.g. Torch
* `NVDIFFRAST_MODULE` (required): Use Nvdiffrast for optimization
* `CAMERA_MODULE` (optional): Use APIs for controlling Kinect Cameras (e.g. V2 and Azure V4)


# Dependencies/Requirements

Install cmake and git:
```
sudo apt-get install cmake git
```
and some more development dependencies:
```
sudo apt-get install python3-dev
sudo apt-get install python3-numpy
sudo apt-get install patchelf
# Ubuntu 20.04
sudo apt-get install clang-10
# Ubuntu 22.04
sudo apt-get install clang-14
```

## OpenGL
For OpenGL, the following additional packages are required:
```
libgl1-mesa-dev
libfreeimage3
libfreeimage-dev
libtiff-dev
libxrandr-dev
libxinerama-dev
libxcursor-dev
libxi-dev
libglu1-mesa-dev
```

## Submodules
This repository uses other sub-repositories (a so-called git **submodule**), which are located in the `external` directory inside of the `code` directory. You can initialize the submodules with the following call:
``` 
git submodule init
```
You then need to actually clone the submodules in to the `external` directory. You can do that with:
```
git submodule update --remote --recursive
```
In case this command fails, it is very likely that the provided submodule URL is wrong. In this case, you need to set the URL correctly inside your `.gitmodules` file and the `.git/config` file.

## Cuda
CUDA >= 10.2, <=11.8 is required.
Make sure that you have the right version of Cuda and the respective Compiler. Also make sure that your OS is listed.
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements

Either install it system wide by following these instructions: https://developer.nvidia.com/cuda-11-4-4-download-archive

Or install it locally via conda or a similar package manager:
```
conda install -c nvidia cuda
# alternatively, with specific version:
#conda install -c "nvidia/label/cuda-11.4.4" cuda
conda install -c nvidia cudatoolkit
conda install -c nvidia cudnn
```
If your CUDA install location is not `/usr/local/cuda`, specify it in `code/CMakeLists.txt`

Note that you need to rebuild OpenCV everytime you change your CUDA version.

## CuDNN (Manual Installation)
With root:
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb

Installing via conda is also possible.

## TensorRT
Follow the instructions at https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian

If root access is not available, do the tar installation.
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
Make sure the TensorRT version matches *exactly* the version the yolact_weights were built with (currently 8.4.2.4, TensorRT 8.4 GA Update 1)
Change the download url of the .tar file if necessary.
Untar and set TensorRT_BASE_PATH in Dependencies_dnn.cmake

If you did the local installation, you will need to add the required libs to LD_LIBRARY_PATH (e.g. in .zshrc)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/cuda/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<reconstruction-lib>/code/external/thirdparty/Torch/source/lib
```

## Done

**All other necessary requirements should be installed automatically as soon as you run the Cmake configure command.**
See below for information on how to manually install them in case something goes wrong.
The requirements are located in the `cmake` directory and can be adjusted in case of errors.

# Compiling

Build with cmake either from VSCode or from the commandline (see `cmake-template.sh`).

If all of the additional dependencies have a Cmake configuration provided, you only need to adjust the Cmake variables accordingly before configuration:

For example, if you compiled the `Eigen3` library yourself in `/usr/local/lib/Eigen3` and in this path there is a `CMakeLists.txt` file located,
you will only need to configure the library path accordingly in cmake by adjusting the `Eigen3_DIR` variable:
```
cmake -B build -S . -DEigen3_DIR=/usr/local/lib/Eigen3/additional/subpaths/to/CmakeLists/file;
cd build;
make -j4;
```

If one of the dependencies does **not have CMake** support, you will need to adjust the `CMakeLists.txt` file accordingly and search for the header/library paths yourself.

## Improve Build Time
For faster build and link time, install
```
ccache
lld
```

If your home directory is on a slow storage (e.g. a network drive), edit or create `~/.config/ccache/ccache.config` to contain:
```
cache_dir = /fast/folder/.cache/ccache
```

To increase link time significantly, tell VSCode to use lld as a linker. In `.vscode/settings.json`, append to `"cmake.configureArgs"`:
```
"-DCMAKE_LINKER=/usr/bin/lld-14",
"-DCMAKE_EXE_LINKER_FLAGS_INIT=-fuse-ld=lld",
"-DCMAKE_MODULE_LINKER_FLAGS_INIT=-fuse-ld=lld",
"-DCMAKE_SHARED_LINKER_FLAGS_INIT=-fuse-ld=lld",
```

Using tmpfs did not result in any consistent link-time improvement. However, using it has no disadvantage and might prove useful for some applications.
To use it, link the directory containing all object files into a tmpfs after a cmake configuration.
```
mkdir /dev/shm/reclib.dir
cd build/src/CMakeFiles
rm -rf reclib.dir
ln -s /dev/shm/reclib.dir .
```


# Data
* Example RGB-D data is given in `data/h2o` such that you only need to adjust the absolute paths in the `configs/sharpy.yaml`, e.g. `Dataset/path` and `Dataset/sequence_file`.
* The OpenPose model that is used for comparison in our paper can be downloaded here:
    * The model description is provided [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/hand/pose_deploy.prototxt). Download it and adjust the respective `OpenPose/protofile` option in `configs/sharpy.yaml`
    * The model weights are provided [here](http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel), alternatively you can use the scripts provided by the [OpenPose Repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/getModels.sh). Download it and adjust the respective `OpenPose/weights` option in `configs/sharpy.yaml`
* The **TensorRT** model weights of the adapted [Yolact](https://github.com/dbolya/yolact) model are given in `data/yolact_weights.zip`. Make sure to unzip it and adjust the respective `Network/weights` option in `configs/sharpy.yaml`. **Please note that they can only be successfully loaded if you install the exact TensorRT version (see documentation above)**
* The MANO model can be downloaded from [here](https://mano.is.tuebingen.mpg.de). Extract the `.zip` file such that you should have a `models/MANO_LEFT.pkl` and `models/MANO_RIGHT.pkl` file. **To load them in the program, they must be converted to `.npz`**. You can use the script provided in `scripts/pkl2npz.py` to do so.

# Running the program
Make sure you update the paths in `configs/sharpy.yaml`.
Run the binary, preferrably from the `code` directory such that the default `imgui.ini` file for the GUI is loaded:
```
cd code && ./build/examples/OFF/bin/sharpy
```

# Controlling the Viewer
* You can move around with your mouse and by using the keys `W` `A` `S` `D`.
* The GUI debug interface can be enabled with `Tab`
* Depending on your settings in `configs/sharpy.yaml` the optimization stages are split into several debug stages, which you can trigger by pressing the numbers `1` `2` `3` `4` `5` in their ascending order.

# Optional Dependencies
If you want to use the Camera module, install
* libfreenect2
* k4a
See below for a guide.

## OpenCV (Manual Installation)
**Follow this case in case the automatic installation of OpenCV in `cmake/Dependencies_opencv.cmake` did not work.**
Retry the automatic installation by removing `opencv` and `opencv_contrib` in `code/external/thirdparty`, setting `FORCE_MANUAL_INSTALLATION` in `Dependencies_opencv.cmake` to `ON` and then reconfiguring.

Clone [opencv](https://github.com/opencv/opencv) and [opencv-contrib](https://github.com/opencv/opencv_contrib) at the version of your choice. The project is known to work with `4.5.0`.
You can find all the available versions with `git tag`. You can switch to the selected version with with `git checkout <tag>`.

Afterwards, you first need to configure the repository with `cmake`. Make sure you have cmake installed. You can configure the library with `cmake -S . -B build`. On top of that, Cmake needs some additional options to be specified:

Cmake Options:    
| Option               | Value                                                                                                                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CMAKE_INSTALL_PREFIX | Set this to the installation directory of your choice, e.g. `/usr/local/lib/opencv`. Afterwards you need to specify this directory in the `OpenCV_DIR` option of this repository.  |
| OPENCV_EXTRA_MODULES | Path to the `opencv_contrib/modules` directory, e.g. `/home/virth/opencv_contrib/modules`                                                                                          |
| WITH_EIGEN           | ON                                                                                                                                                                                 |
| WITH_CUDA            | ON                                                                                                                                                                                 |
| CUDA_GENERATION      | Set this to your GPU Generation, e.g. "Pascal", you can find your Generation at [this link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) |
| BUILD_opencv_python3 | OFF                                                                                                                                                                                |
| BUILD_opencv_python2 | OFF                                                                                                                                                                                |

Alternatively, you can run the following command in the terminal: 
```
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<your-installation-path> -DOPENCV_EXTRA_MODULES_PATH=<modules-path> -DWITH_EIGEN=ON -DWITH_CUDA=ON -DCUDA_GENERATION=<your-gpu-generation> -DBUILD_opencv_python3=OFF -DBUILD_opencv_python2=OFF
```

Afterwards, you can build the library with `cmake --build <build-directory>` and install it to your selected directory with `cmake --build <build-directory> --target install`.

## Eigen3 (Manual Installation)
**Follow this case in case the automatic installation of Eigen in `cmake/Dependencies_core.cmake` did not work.**

Just clone the repository at [gitlab](https://gitlab.com/libeigen/eigen). Checkout to the specific version you want.
You can find all the available versions with `git tag`. You can switch to the selected version with with `git checkout <tag>`.

Afterwards, you first need to configure the repository with `cmake`. Make sure you have cmake installed. You can configure the library with `cmake -S . -B build`. On top of that, Cmake needs some additional options to be specified:

Cmake Options:    
| Option               | Value                                                                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CMAKE_INSTALL_PREFIX | Set this to the installation directory of your choice, e.g. `/usr/local/lib/eigen3`. Afterwards you need to specify this directory in the `Eigen3_DIR` option of this repository. |

Afterwards, you can build the library with `cmake --build <build-directory>` and install it to your selected directory with `cmake --build <build-directory> --target install`.

**Note**: Eigen is a header-only library, so by executing `cmake --build <build-directory>` it will not actually build something. Instead, by running `cmake --build <build-directory> --target install` it will only copy the necessary header files into the specified directory.

## Freenect2 - API for Kinect V2 (Optional)
Clone the repository at [https://github.com/OpenKinect/libfreenect2](https://github.com/OpenKinect/libfreenect2). For now it seems that there are no specific versions as there are no useful tags so just use the `master` branch.

### Dependencies
* `libusb-1.0-dev`
* `libturbojpeg0-dev`
* `libglfw3-dev`
* `libopenni2-dev`

```shell
sudo apt-get install libusb-1.0-dev libturbojpeg0-dev libglfw3-dev libopenni2-dev
```

In case you have a **Windows** operating system, make sure to check out the `README` file in the freenect2 repository.

### Configure & Install 
Afterwards, you first need to configure the repository with `cmake`. Make sure you have cmake installed. You can configure the library with `cmake -S . -B build`. On top of that, Cmake needs some additional options to be specified:

Cmake Options:    
| Option               | Value                                                                                                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CMAKE_INSTALL_PREFIX | Set this to the installation directory of your choice, e.g. `/usr/local/lib/freenect2`. Afterwards you need to specify this directory in the `freenect2_DIR` option of this repository. |

Afterwards, you can build the library with `cmake --build <build-directory>` and install it to your selected directory with `cmake --build <build-directory> --target install`.

### Copy the Udev Rules (Linux)
In case you are on Linux, you need to copy the `udev` rules that are inside the `build` directory of Cmake.    
```
sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
```
Then, replug the Kinect. Test if it works by running the test program:

```shell
./bin/Protonect
```

## K4a - API for Kinect V4 Azure (Optional)
**Beware: This guide is for only Linux systems**

### Dependencies
* ninja-build
* libsoundio-dev
* uuid-dev
* libudev-dev
* libusb-1.0-0-dev
* libssl-dev

### Configure & Install 

Unfortunately the Microsoft repository for Ubuntu 20.04 does not contain the correct binaries. Therefore, we have to download the software manually.

Download **version 1.3** (Version 1.4 does not work currently) of k4a-tools at [https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/](https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/)

Download **version 1.3** of libk4a and libk4a-dev at [https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/](https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/)

You can install the `.deb` packages with `sudo dpkg -i <deb-file>`

After that, follow the tutorial at [https://vinesmsuic.github.io/2020/10/31/azure-kinect-setup/#invoke-k4aviewer-in-bin](https://vinesmsuic.github.io/2020/10/31/azure-kinect-setup/#invoke-k4aviewer-in-bin):


```
git clone https://github.com/microsoft/Azure-Kinect-Sensor-SDK.git
cd Azure-Kinect-Sensor-SDK
mkdir build && cd build
cmake .. -GNinja
ninja
sudo ninja install
```

Eventually you could get the following error:   
```
<libusb.h> not found
```
If this is the case, make sure to check the cmake include path with `grep -rn "libusb-1.0` and verify the directory name.
If it points to `/usr/include/libusb-1.0` then this is simply a naming issue. Create a symlink in that directory with the name `libusb` as a workaround.

Test your setup with `sudo k4aviewer` first.

### Copy the Udev Rules (Linux)

Define additional `udev` rules in order to be able to run `k4aviewer` without `sudo` privilege.
Therefore create the file `/etc/udev/rules.d/99-k4a.rules` with the following content:
```
# Bus 002 Device 116: ID 045e:097a Microsoft Corp.  - Generic Superspeed USB Hub
# Bus 001 Device 015: ID 045e:097b Microsoft Corp.  - Generic USB Hub
# Bus 002 Device 118: ID 045e:097c Microsoft Corp.  - Azure Kinect Depth Camera
# Bus 002 Device 117: ID 045e:097d Microsoft Corp.  - Azure Kinect 4K Camera
# Bus 001 Device 016: ID 045e:097e Microsoft Corp.  - Azure Kinect Microphone Array

BUS!="usb", ACTION!="add", SUBSYSTEM!=="usb_device", GOTO="k4a_logic_rules_end"

ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097a", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097b", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097c", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097d", MODE="0666", GROUP="plugdev"
ATTRS{idVendor}=="045e", ATTRS{idProduct}=="097e", MODE="0666", GROUP="plugdev"

LABEL="k4a_logic_rules_end"
```

# Known Issues
**Please note that this repository is forked from an internal research framework such that not all issues may apply.**

* OpenCV libraries not being linked appropriately, e.g. `libopencv_imgproc.so.4.5 not found: No such file or directory`
  * *(Likely) caused by*: OpenCV 4.5.2, cmake file seems to be missing library paths
  * *Solution*: Add OpenCV library path to `LD_LIBRARY_PATH` environment variable
* Program crashes with Exception in OpenCV `cv::GpuMat::create` function 
  * *(Likely) caused by*: OpenCV Cuda version mismatch, wrong architecture in `CMakeLists.txt`
  * *Solution*: Rebuild OpenCV, remove GPU architecture constraints in `CMakeLists.txt` and set to `auto` instead
  * *Detailed Solution*: 
* Eigen operations in CUDA kernel code cause `illegal memory error`
  * *(Likely) caused by*: Buggy Eigen3 library within OpenCV **OR** Eigen3 library below version `3.4.0`
  * *Solution*: use own, pre-compiled Eigen **OR** Use Eigen3 library with version `3.4.0`
* Segmentation faults in `free()` when using Eigen Structures in `.cu` files and pass them to functions of `.cpp` files
  * *(Likely) caused by*: Alignment issues, since nvcc and gcc align Eigen Structures differently in memory
  * *Solution*: Disable alignment in `.cpp` and `.cu` files or execute only functions that are compiled by nvcc
* Ceres does not build on Ubuntu 22
  * *(Likely) caused by*: A new version of libtbb, which is not compatible with Ceres 2.0 ([Github issue](https://github.com/ceres-solver/ceres-solver/issues/612))
  * *Solution*: Use Ceres 2.1
* Cuda version does not match compiler version
  * Solution: check installed cuda version: `nvcc --version` and check for compatibility in corresponding docu, e.g. in https://docs.nvidia.com/cuda/archive/11.5.2/cuda-installation-guide-linux/index.html#system-requirements
* Segfault when using TensorRT
  * *(Likely) Caused by*: Using weights generated on a different GPU. (Error message: "Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.")
  * Solution: delete all .trt files, they will be regenerated from the .ts files during the next execution.
* `no member named 'SearchParams' in namespace 'nanoflann'`
  * *Caused by*: Update in nanoflann.
  * Solution: check out a previous version of nanoflann: `cd code/external/thirdparty/include/nanoflann && git checkout v1.4.3`
* `/usr/lib/llvm-14/lib/clang/14.0.0/include/__clang_cuda_builtin_vars.h(53): error: identifier "property" is undefined`
  * *(Likely) Caused by*: for some reason, clang added an include of `__clang_cuda_builtin_vars.h` to one of your cuda files.
  * Solution: delete the include line
* "\<Program\> is not responding dialog" often blocks the window and prevents interaction
  * *Caused by*: slow frame updates
  * Solution: increase alive timeout: `gsettings set org.gnome.mutter check-alive-timeout 60000`
* System hangs during build process
  * *(Likely) Caused by*: Not enough RAM.
  * Either upgrade your hardware or increase your swap space. 16GB necessary, 32GB recommended.  
  * You can also try decreasing the amount of parallel jobs. In VS Code in Settings > Extensions > CMake Tools > Cmake: Parallel Jobs
* `/usr/bin/../lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/ext/numeric_traits.h(70): error: qualified name is not allowed`
  * *Caused by*: you are compiling cuda code with clang
  * Solution: Use gcc/g++ for cuda code. Check your CUDAHOSTCC and CUDAHOSTCXX environment variables.
* `fatal error: 'torch/torch.h' file not found`
  * *(Likely) Caused by*: Old build files after Torch version change
  * Solution: Clean Project and rebuild
* "OpenCV static library was compiled with CUDA x.y support. Please, use the same version or rebuild OpenCV with CUDA x.y"
  * *Caused by*: OpenCV was built with a different CUDA version than it is used with
  * Solution: Make sure -DCUDA_TOOLKIT_ROOT_DIR is passed to OpenCV with the correct value during build


# Acknowledgements
 This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – SFB 1483 – Project-ID 442419336, EmpkinS.

This work used the German Research Foundation (DFG) funded major instrument (reference number INST90 / 985- 1 FUGG) at the Institute of Applied Dynamics (Sigrid Leyendecker), Friedrich-Alexander Universität Erlangen- Nürnberg Germany.

The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center of the Friedrich-Alexander-Universität Erlangen-Nürnberg. 

Furthermore, the authors would like to thank the following people:
* Jasmin Hacker for her highly valuable contributions to the quality of this framework and its documentation
* [Nikolai Hofmann](https://github.com/nihofm), [Laura Fink](https://lorafib.github.io/) and [Linus Franke](https://lfranke.github.io/) for providing the CppGL framework that our code is based on
* [Darius Rückert](https://github.com/darglein/personal), who gave us advice on the implementation details of [Saiga](https://github.com/darglein/saiga), of which we use the datatype interface between Eigen and GLM
* The authors and contributors of [Yolact](https://github.com/dbolya/yolact), of which we use the code for training the network parameters
* The authors and contributors of [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), of which we use the provided models and weights for evaluation
* The authors and contributors of [smplxpp](https://github.com/sxyu/smplxpp), who provided a C++ implementation of the SMPL-X models, which we adjusted to fit our MANO model
* The authors and contributors of [Nvdiffrast](https://github.com/NVlabs/nvdiffrast), which we use in our optimization
* The authors and contributors of [Avatar](https://github.com/sxyu/avatar), of which we used the implementation details to interact with depth cameras

# Citation
If you use Sharpy or its code base in your work, please cite:
```
@InProceedings{Wirth_2023_ICCV,
          author    = {Wirth, Vanessa and Liphardt, Anna-Maria and Coppers, Birte and Br\"aunig, Johanna and Heinrich, Simon and Leyendecker, Sigrid and Kleyer, Arnd and Schett, Georg and Vossiek, Martin and Egger, Bernhard and Stamminger, Marc},
          title     = {ShaRPy: Shape Reconstruction and Hand Pose Estimation from RGB-D with Uncertainty},
          booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
          month     = {October},
          year      = {2023},
          pages     = {2625-2633}
      }
```
