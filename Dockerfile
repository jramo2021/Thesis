FROM tensorflow/tensorflow:latest-gpu
RUN pip install --upgrade pip
RUN pip install numpy scipy matplotlib tf-nightly ipython jupyter pandas sympy nose tensorflow-datasets opencv-python