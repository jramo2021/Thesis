#!/bin/bash
docker run -v /home/rdalke/Thesis:/home/Thesis -u $(id -u):$(id -g) --runtime=nvidia -it thesis_dalke:latest #python3 home/Thesis/segment.py #python3 home/Thesis/breast_cancer_eda.py
