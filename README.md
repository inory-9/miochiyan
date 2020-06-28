# miochiyan
This repository has referred freely to "talking-head-anime-demo"(https://github.com/pkhungurn/talking-head-anime-demo).It makes by python, so I move it to c++, and use its models.
## Hardware Requirements
* a webcam 
* Nvidia GPU that support CUDA.
## build
On Microsoft Winsows ,make sure that you have install cmake. UI creat by QT5, so you must install QT5 and add library path to operate system.Complier version should be visual c++ 2017 or greater.Moreover,we need some third-party libs:
* libtorch(1.5.0 by cuda 10.2)
* openCV(4.1.1 or greater)
* dlib(19.20.0 or greater)
You can prepare these libs by yourself ,or use libs that prepared by me, click this link to download a zip file, uncompress it to project root path. 
## run
make sure that dynamic link library files are complete in your executable path,you can find them in my lib file. at last,you should click this to download a zip file,uncompress it to your executable path.This file contains models and some pictures that can be used to test.
