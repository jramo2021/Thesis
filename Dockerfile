# FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# ENV OS=ubuntu2004
# ENV cudnn_version=8.6.0.163
# ENV cuda_version=cuda11.8

# RUN apt update -y
# RUN apt install gcc-9 g++-9 -y

# RUN apt-get update -y
# RUN apt upgrade -y
# RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
# RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}

# RUN apt install python3 -y
# RUN apt install python3-pip -y
# RUN pip install --upgrade pip
# RUN pip install tensorflow==2.12.0 numpy scipy matplotlib tf-nightly ipython jupyter pandas sympy nose tensorflow-datasets opencv-python

FROM tensorflow/tensorflow:latest-gpu
RUN pip install --upgrade pip
RUN pip install numpy scipy matplotlib tf-nightly ipython jupyter pandas sympy nose tensorflow-datasets opencv-python torchstain
