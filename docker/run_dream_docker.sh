#!/bin/bash

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

CONTAINER_NAME=$1
if [[ -z "${CONTAINER_NAME}" ]]; then
    CONTAINER_NAME=nvidia-dream-v1
fi

# This specifies a mapping between a host directory and a directory in the
# docker container. This mapping should be changed if you wish to have access to
# a different directory
HOST_DIR=$2
if [[ -z "${HOST_DIR}" ]]; then
    HOST_DIR=`realpath ${PWD}/..`
fi

CONTAINER_DIR=$3
if [[ -z "${CONTAINER_DIR}" ]]; then
    CONTAINER_DIR=/root/catkin_ws/src/dream
fi

echo "Container name     : ${CONTAINER_NAME}"
echo "Host directory     : ${HOST_DIR}"
echo "Container directory: ${CONTAINER_DIR}"
DREAM_ID=`docker ps -aqf "name=^/${CONTAINER_NAME}$"`
if [ -z "${DREAM_ID}" ]; then
    echo "Creating new DREAM docker container."
    xhost +
    docker run --gpus all -it --privileged --network=host -v ${HOST_DIR}:${CONTAINER_DIR}:rw -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix${DISPLAY} --name=${CONTAINER_NAME} nvidia-dream:kinetic-v1 bash
else
    echo "Found DREAM docker container: ${DREAM_ID}."
    # Check if the container is already running and start if necessary.
    if [ -z `docker ps -qf "name=^/${CONTAINER_NAME}$"` ]; then
        xhost +local:${DREAM_ID}
        echo "Starting and attaching to ${CONTAINER_NAME} container..."
        docker start ${DREAM_ID}
        docker attach ${DREAM_ID}
    else
        echo "Found running ${CONTAINER_NAME} container, attaching bash..."
        docker exec -it ${DREAM_ID} bash
    fi
fi
