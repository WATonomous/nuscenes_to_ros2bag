FROM ros:humble-ros-core

RUN apt-get update
RUN apt-get install -y git python3-pip python3-tf2-ros libgl1 libgeos-dev ros-humble-ros2bag ros-humble-rosbag2* ros-humble-foxglove-msgs  ros-humble-tf2-tools ros-humble-tf-transformations
RUN apt install -y python3-colcon-common-extensions

RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install shapely numpy==1.26.4 nuscenes-devkit mcap tqdm requests
RUN pip3 install git+https://github.com/DanielPollithy/pypcd.git

COPY . /work

WORKDIR /work
