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
### fix unknown build flag using --symlink-install with colcon and setup.py by downgrading setuptools ###
$ conda install -n ros_env -c conda-forge setuptools=79.0.1
```

Clone the repository and install package dependencies.
```
$ mkdir -p $HOME/camera_ws/src
$ cd $HOME/camera_ws/src
$ git clone https://github.com/nicolasfrick/camera_pose.git

# install conda packages
$ conda activate ros_env
$ conda install -c conda-forge -c robostack-staging -c robostack-humble ros-humble-librealsense2 ros-humble-realsense2-camera ros-humble-realsense2-camera-msgs ros-humble-realsense2-description

# install python packages
$ pip install opencv-python dt-apriltags scipy PyYAML pandas
$ rosdep init && rosdep update
$ rosdep install --from-paths src --ignore-src -r -y --skip-keys="ament_python opencv-python dt-apriltags"

# build
$ cd $HOME/camera_ws
$ colcon build --symlink-install
```

Verify buildtool running `which colcon`.
Expected output: `$HOME/miniconda3/envs/ros_env/bin/colcon`
If `/usr/bin/colcon` appears, try the build from a new terminal to remove unwanted environment variables.

## Run

![Initial View](https://raw.githubusercontent.com/nicolasfrick/camera_pose/main/images/initial_view.jpg)

### (1) Camera Pose estimation - Ground Truth Marker Set

In the first step, a camera pose is estimated from a set of known marker poses and their observations in the image using a least squares algorithm and a solution to the perspective-n-point problem from the Apriltag library.

Attach one or more markers to an object and enter their translation and rotation with respect to the center of the world coordinate system in the file `config/camera_marker_poses.yaml` under the `camera_pose_marker`key. Translation is required in meter. Rotation is required as radians (extrinsic `xyz` Euler angles). Check the example in the image for the coordinate system representation of an Apriltag marker to determine the marker poses. The Apriltag marker id has to be the key for the relevant marker pose as shown in the example below. 


```
camera_pose_marker:
  0:                                # Marker id
    rpy: &euler [*N_PI_2, 0, *PI_2] # roration in rad (extrinsic `xyz` Euler angles)
    xyz: [0.550, 0.020, 0.327]      # translation in meter
  1:
    ...
```

### (2) Marker Pose Computation - Target Marker Set

Second, the poses of the target markers are computed from their detections. 

Attach one or more markers to an object and enter their ids in the file `config/camera_marker_poses.yaml` under the `target-i_marker`key. `i` refers to the i'th object. The `xyz` and `rpy` keys are supposed to be `null`. Only those marker ids will be respected for the computation.


```
# 1st target
target-1_marker:
  10:
    rpy: null
    xyz: null

  11:
    rpy: null
    xyz: null

# 2nd target
target-2_marker:
  ...
```

Optionally, you may enter the pose of a target frame wrt. to a target marker's coordinate system instead of `null` to compute the pose of the frame wrt. the world frame.


```
# 1st target
target-1_marker:
  10:
    rpy: &euler_target1 [*N_PI_2, *PI_2, 0]
    xyz: [0.010, 0.033, 0.0325]

  11:
    rpy: *euler_target1
    xyz: [-0.010, 0.033, 0.0325]

# 2nd target
target-2_marker:
  ...
```


### Run the Script

Run the pose estimation on the test image:

```
$ cd $HOME/camera_ws
$ source install/setup.bash
$ ros2 launch camera_pose cam_pose.launch.py cam_pose_marker_length:=0.03 test:=true 
```

Required launch arguments: 

    - cam_pose_marker_length: Value (float) of the marker length in meter. 
                              Marker length refers to the distance between the detection corners as shown in the image. 

Optional launch arguments: 

    - marker_family: Default `tag16h5`.

### Results

The results of the pose estimations are written to the file `results/results.yaml` and contains the camera pose estimate in different representations and the target marker poses with respect to the world coordinate system's center. If a target pose has been specified, the entry `filtered_target_pose` shows an estimate of the target frame's pose.

During the estimation process, the results are visualized under the ROS topic `/camera_pose_detections`.

![Camera Pose](https://raw.githubusercontent.com/nicolasfrick/camera_pose/main/images/camera_pose_estimation.jpg)


![Target Pose](https://raw.githubusercontent.com/nicolasfrick/camera_pose/main/images/target-2_marker_estimation.jpg)


## ToDo

EKF filter implementation

RANSAC implementation 

Migrate internal rotation representation
