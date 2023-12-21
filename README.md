# GND-LO
Ground Decoupled 3D Lidar Odometry based on Planar Patches. See paper [here](https://ieeexplore.ieee.org/abstract/document/10243099).
Cite as:
```
@ARTICLE{gndlo,
  author={Galeote-Luque, Andres and Ruiz-Sarmiento, Jose-Raul and Gonzalez-Jimenez, Javier},
  journal={IEEE Robotics and Automation Letters}, 
  title={GND-LO: Ground Decoupled 3D Lidar Odometry Based on Planar Patches}, 
  year={2023},
  volume={8},
  number={11},
  pages={6923-6930},
  doi={10.1109/LRA.2023.3313057}}
```

## Introduction
This repository includes a ROS2 node that subscribes to a pair of message topics (image and sensor information) and publishes the resulting odometry. 
[Ceres](http://ceres-solver.org/) must be installed for GND-LO to work.

To use GND-LO, clone into your ROS2 workspace and build as usual. 
```
cd ros2_ws/src
git clone https://github.com/MAPIRlab/GND-LO
cd ..
colcon build --packages-select gndlo
```
The included launch and rviz file should provide enough information to make it run. 
```
ros2 launch gndlo gndlo_launch.xml
```

## Parameters
The launch file allows adjusting the parameters without needing to rebuild. Now for a brief explanation of the available parameters:
* subs_topic: where the node should subscribe for the input images and the sensor information. It should have this structure: /subs_topic/range/image, /subs_topic/range/sensor_info, where the first topic is sensor_msgs/Image and the second is sensor_msgs/LaserScan.
* num_threads: number of threads used by Ceres solver.
* valid_ratio: ratio (from 0 to 1) of valid pixels in a neighborhood to consider it a valid group of points.
* flag_verbose: true to output in terminal some information about timing.
* flag_flat_blur: wether or not to blur the flatness image (recommended: true).
* flag_solve_backforth: wether or not to use patches from both input scans. More accurate, but needs more time.
* flag_filter: wether or not to filter the resulting motion estimation using previous instances of the movement.
* select_radius: radius used in kernels to calculate the curvature and smooth the flatness image. radius = 1 means 3x3 blocks.
* gaussian_sigma: sigma used in gaussian kernels to blur flatness and obtain points. Set to -1 to make it based on the radius.
* quadtrees_avg: threshold on the average of the quadtree blocks to select them as a patch. Higher means more (less flat) patches are selected).
* quadtrees_std: threshold on the standard deviation of the quadtree blocks to divide them. Higher means blocks are divided less often.
* quadtrees_min_lvl: minimum level when performing quadtrees = 2^quadtrees_min_lvl. 1 means 2x2 blocks are also tested for selection.
* quadtrees_max_lvl: maximum level when performing quadtrees = 2^quadtrees_max_lvl. 5 means testing starts at 32x32 blocks.
* count_goal: goal of number of patches contributing to each principal direction. Higher means more patches are included for each direction.
* starting_size: starting block size when culling. 4 means blocks of 4x4 are included as the initial set, then smaller blocks are tested.
* ground_threshold_deg: threshold on the angular difference (between patch and previous ground plane) to consider patch as ground, in degrees.
* wall_threshold_deg: threshold on the angular difference (between patch and previous ground plane) to consider patch as wall, in degrees.
* iterations: maximum iterations on the update of the correspondences. It averages around 7 iterations per estimation.
* huber_loss: starting value of the huber loss function parameter. Gets substituted with MAD of residuals later.
* trans_bound: maximum translation in any direction. Set to very high to ignore it.
* pix_threshold: pixel difference threshold when updating correspondences to check convergence. 
* trans_threshold: convergence threshold on the norm of the difference in translation.
* rot_threshold: convergence threshold on the angle between rotations.
* filter_kd, filter_pd, filter_kf, filter_pf: filter parameters. Dynamic component is cd = kd\*covar.sum()^pd, fixed component is cf = kf\*covar.sum()^pf.
* flag_save_results: wether the resulting poses should be saved in a Freiburg style .txt file. 
* results_file_name: file name of the results if the flag_save_results is set to true.

## More information on the input topics
Download the ROS2 bags: https://uma365-my.sharepoint.com/:f:/g/personal/0619166273_uma_es/EoQ89QoqmdJAu3pm4E4W-zABHmV9lO0FPr2iRR8UQzpkCQ?e=r2KD8R

Images are obtained from the KITTI dataset using a spherical projection with a 1000x64 resolution, vertical angle from 2º to -24.8º, and horizontal from 180º to -180º. 
The sensor information is stored in a sensor_msgs/LaserScan in the following manner:
* LaserScan.range_min = rows of the image.
* LaserScan.range_max = columns of the image.
* LaserScan.angle_min = horizontal angle on the first (left) pixel.
* LaserScan.angle_max = horizontal angle on the last (right) pixel.
* LaserScan.ranges\[] = vector of the vertical angles. Expected to have length = LaserScan.range_min (rows).
