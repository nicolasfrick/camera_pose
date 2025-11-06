# POSE DETECTION 
## using a Realsense camera and Apriltags

Some Apriltags with a known transformation from the world (zero) pose are used to estimate the camera's extrinsics. Another set of markers applied to an object can be used to estimate the object's pose with respect to the world frame.

## Installation

The installation is tested for Ubuntu 22.04 and ROS2 Humble.

We use a conda environment for the package installation and the build: [miniconda install](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
```
mkdir -p $HOME/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda3/miniconda.sh
bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3
rm $HOME/miniconda3/miniconda.sh
```

The robostack conda environment is huge (~4 GB) for this package but it builds flawless: [instructions](https://robostack.github.io/GettingStarted.html#__tabbed_3_2).

**Do not source the system ROS environment**

When there is an installation available of ROS on the system, in non-conda environments, there will be interference with the environments as the PYTHONPATH set in the setup script conflicts with the conda environment.

```
$ conda create -n ros_env -c conda-forge -c robostack-humble ros-humble-desktop
$ conda activate ros_env
$ conda install -c conda-forge compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
# fix unknown build flag using --symlink-install with colcon 
# and setup.py by downgrading setuptools
$ conda install -n ros_env -c conda-forge setuptools=79.0.1
```

Clone the repository and install package dependencies.
```
$ mkdir -p $HOME/camera_ws/src
$ cd $HOME/camera_ws/src
$ git clone https://github.com/nicolasfrick/camera_pose.git
$ conda activate ros_env
$ pip install opencv-python dt-apriltags scipy PyYAML pandas
$ cd $HOME/camera_ws
$ colcon build --symlink-install
```

Verify buildtool running `which colcon`.
Expected output: `$HOME/miniconda3/envs/ros_env/bin/colcon`
If `/usr/bin/colcon` appears, try the build from a new terminal to remove unwanted environment variables.

## Run
Place some markers at a known pose and a realsense camera with the markers visible in its field of view.

```
$ cd $HOME/camera_ws
$ source install/setup.bash
$ ros2 launch camera_pose cam_pose.launch.py
```

## ToDo

EKF filter implementation
RANSAC implementation 
