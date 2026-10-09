# Demetra Robot Package for Field Robot Event 2025

`demetra_robot`: meta-package that include all the packages in this repository

`demetra_description`: package that contains URDFs, ros2_control configurations and meshes

`demetra_msgs`: package that implement additional interfaces for the robot

`demetra_serial_driver`: package that contains the serial driver for the robot

**UNUSED** `demetra_hardware_interface`: package that contains the hardware interface for the robot to work with ros2_control

## Installation

### From Source

1. Clone the repository into your ROS 2 workspace:
   ```bash
   git clone https://github.com/FRE-Polito/demetra_robot.git
    ```

2. Install dependencies using `rosdep`:
   ```bash
   rosdep update --rosdistro $ROS_DISTRO
   rosdep install -i --from-path src --ignore-src -y
   ```

3. Build the packages:
   ```bash
   colcon build --symlink-install --packages-up-to demetra_robot
   ```

## Robot bringup

To bring up the robot, you can use the provided launch files. For example, to launch the robot with the default configuration, run:

```bash
ros2 launch demetra_robot bringup.launch.py
```

### Launch arguments
- `use_sim_time`: For future use. (default: `False`)
- `use_ros2_control`: For future use. (default: `False`)
- `driver_params`: Path to the YAML file with the parameters for the serial driver. (default: `demetra_serial_driver/config/driver_params.yaml`)

## Sensors setup

- Common
   - [ ] **AHRS/IMU**: Microstrain 3DM-GX5
   - [ ] **LiDAR**: Livox MID-360

- Task 1

- Task 2
   - [ ] **Camera 0**: Intel RealSense D435 Rigth
   - [ ] **Camera 1**: Intel RealSense D435 Left

- Task 3
   - [ ] **Camera 0**: Intel RealSense D435 Tower

- Task 4
   - [ ] **Camera 0**: Intel RealSense D435 Front
   - [ ] **Camera 1**: Intel RealSense D435 Right (?)
   - [ ] **Camera 2**: Intel RealSense D435 Left (?)