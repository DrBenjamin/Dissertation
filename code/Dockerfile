# syntax=docker/dockerfile:1.6
FROM --platform=linux/amd64 python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installing system dependencies
USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libegl1 \
        libgles2 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Setting the working directory inside the container
WORKDIR /workspace/project

# Adding requirements file for offline install
ADD requirements.txt /workspace/project/requirements.txt 

# Installing Python dependencies
RUN python -m pip install --upgrade pip setuptools \
    && python -m pip install --no-cache-dir -r /workspace/project/requirements.txt \
    && python -m pip install --no-cache-dir ipykernel \
    && python -m ipykernel install --user --name=python3

# Copying the build context (contents of the local code/ directory)
COPY . /workspace/project/

# Starting Streamlit app on default port
CMD ["python", "-m", "streamlit", "run", "code/mediapipe_pose.py", "--server.port=8502", "--server.address=0.0.0.0"]
