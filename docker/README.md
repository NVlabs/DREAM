## DREAM in a Docker Container

Running ROS inside of [Docker](https://www.docker.com/) is an excellent way to
experiment with DREAM, as it allows the user to completely isolate all software
and configuration changes from the host system.  This document describes how to
create and run a Docker image that contains a complete ROS environment that
supports DREAM, including all required components, such as ROS Kinetic, OpenCV,
CUDA, and other packages.

The current configuration assumes all components are installed on an x86 host
platform running Ubuntu 16.04 or later.  Further, use of the DREAM Docker container
requires an NVIDIA GPU.


### Steps

1. **Create the container**
   ```
   $ cd deep-arm-cal/docker
   $ docker build -t nvidia-dream:kinetic-v1 -f Dockerfile.kinetic ..
   ```
   This will take several minutes and requires an internet connection.

2. **Run the container**
   ```
   $ ./run_dream_docker.sh [name] [host dir] [container dir]
   ```
   Parameters:
   - `name` is an optional field that specifies the name of this image. By default, it is `nvidia-dream-v1`.  By using different names, you can create multiple containers from the same image.
   - `host dir` and `container dir` are a pair of optional fields that allow you to specify a mapping between a directory on your host machine and a location inside the container.  This is useful for sharing code and data between the two systems.  By default, it maps the directory containing DREAM to `/root/catkin_ws/src/dream` in the container.

      Only the first invocation of this script with a given name will create a container. Subsequent executions will attach to the running container allowing you -- in effect -- to have multiple terminal sessions into a single container.

3. **Install DREAM**

   Inside the running Docker container:
   ```
   $ cd /root/catkin_ws/src/dream
   $ pip install -e .
   ```

4. **Run the tests**

   Inside the running Docker container:
   ```
   $ cd /root/catkin_ws/src/dream
   $ pytest test/
   ```

See the README.md in the top-level directory of this repository for information about
running DREAM.