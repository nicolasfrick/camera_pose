from setuptools import setup, find_packages

package_name = 'camera_pose'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/cam_pose.launch.py', 'launch/realsense.launch.py']),
        ('share/' + package_name + '/config', []),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nicolas Frick',
    maintainer_email='nicolas.frick@studium.uni-hamburg.de',
    description='ROS 2 port of camera pose estimation using AprilTags.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_pose_node = camera_pose.detect_node:main',
        ],
    },
)
