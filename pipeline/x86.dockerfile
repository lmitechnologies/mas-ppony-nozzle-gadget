# last version running on ubuntu 20.04, require CUDA 12.1 
FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG DEBIAN_FRONTEND=noninteractive

ARG PACKAGE_VER
ARG PYPI_SERVER

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user  opencv-python
RUN pip install ultralytics -U

# clone LMI AI Solutions repository
WORKDIR /home/gadget/workspace
RUN git clone https://github.com/lmitechnologies/LMI_AI_Solutions.git

RUN python3 -m pip install gadget_pipeline_server==$PACKAGE_VER   --extra-index-url $PYPI_SERVER

CMD gadget_pipeline_server

