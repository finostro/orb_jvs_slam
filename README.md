# orb_jvs_slam

This package uses the Joint Vector-Set Approach to SLAM using ORB visual fetures for stereo cameras. 

In order to run this code 
- compile with colcon (ROS 2): `colcon build --packages-select hglmbslam`
- set the path to the euroc_dataset folder in  cfg/vglmbslam6d.yaml
- set the pathto the results folder in  cfg/vglmbslam6d.yaml
- run code using `ros2 run hglmbslam vglmbslam6d -c cfg/vglmbslam6d.yaml`
- visualize with `ros2 launch hglmbslam rviz.launch.py` (MarkerArray on `/hglmbslam/marker`) 


