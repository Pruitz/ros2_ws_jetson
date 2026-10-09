# demetra_serial_driver

This package provides a ROS2 driver for the Demetra serial communication protocol.

## Installation

See main [README](../README.md) for installation instructions.

## Executables

### `driver_node`
This is the main node that handles the serial communication with the Demetra robot.

#### **Subscribers**
- `cmd_vel [geometry_msgs/msg/Twist]`: This topic is used to send velocity commands to the robot. Only `linear.x` and `angular.z` components are used.
- `right_light [std_msgs/msg/Bool]`: This topic is used to control the right light of the robot. Send `true` to turn on the light on for 500ms
- `left_light [std_msgs/msg/Bool]`: see `right_light`.

#### **Publishers**
- `odom [nav_msgs/msg/Odometry]`: This topic publishes the odometry of the robot.
- `/tf [tf2_msgs/msg/TFMessage]`: This topic publishes the transform between the robot's base frame and the odometry frame that can be configured through parameters.

#### **Services**
- `reset_odometry [std_srvs/srv/Empty]`: This service resets the odometry of the robot. It sets the position to (0, 0) and the orientation to 0 radians. 

#### **Actions**
- `turn_in_place [demetra_robot_msgs/action/TurnInPlace]`: This action allows the robot to turn in place. It takes as goal a delta angle in radians.

#### Parameters
```yaml
demetra_node:
  ros__parameters:
    odom_period_ms: 20 # 50 Hz
    serial_period_ms: 10 # 100 Hz
    cmd_vel_timeout_ms: 500

    # Serial port configuration
    port_name: "/dev/ttyUSB0" 
    baudrate: 115200

    # Geometry configuration
    wheel_base: 0.4  # Distance between the front and rear wheels
    wheel_diameter: 0.192
    max_steering_angle_deg: 30.0
    max_linear_velocity_m_s: 1.5
    max_angular_velocity_rad_s: 6.0
    odom_frame_id: "odom"
    base_frame_id: "base_footprint"
    publish_tf: true
```
    

## Libraries
See header files for more details.

- `driver_node_component`: The node component that is loaded by the `driver_node`. It can be composed into a larger ROS2 application.

- `serial_driver`: This library provides the low-level serial communication functionality for the Demetra robot.

- `kinematics`: This library provides the kinematics calculations for the Demetra robot, i.e bycicle model with both front and rear steering.