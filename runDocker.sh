#!/bin/bash
docker run -v /home/jramos60/Thesis:/home/Thesis -u $(id -u):$(id -g) --runtime=nvidia -it thesis_dalke:latest #python3 home/Thesis/main.py
