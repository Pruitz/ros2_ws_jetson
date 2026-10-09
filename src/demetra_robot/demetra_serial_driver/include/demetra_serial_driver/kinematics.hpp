#ifndef DEMETRA_SERIAL_DRIVER__KINEMATISCS_HPP
#define DEMETRA_SERIAL_DRIVER__KINEMATISCS_HPP

#include <cmath>
#include <limits>
#include <algorithm>

namespace demetra_serial_driver
{

class Kinematics
{
public:
  Kinematics();

  /**
   * \brief Constructor for Kinematics class
   * \param wheel_base Distance between front and rear axle [m]
   * \param wheel_diameter Diameter of the wheels [m]
   * \param max_steering_angle Maximum steering angle [rad]
   * \param max_linear_velocity Maximum linear velocity [m/s]
   * \param max_angular_velocity Maximum angular velocity [rad/s]
   * \param min_steering_angle Minimum steering angle [rad] (optional, default to - max_steering_angle)
   * \param min_linear_velocity Minimum linear velocity [m/s] (optional, default to - max_linear_velocity)
   * \param min_angular_velocity Minimum angular velocity [rad/s] (optional, default to - max_angular_velocity)
   */
  Kinematics(
    double wheel_base, double wheel_diameter, double max_steering_angle, double max_linear_velocity,
    double max_angular_velocity, double min_steering_angle = std::nan(""),
    double min_linear_velocity = std::nan(""), double min_angular_velocity = std::nan(""));

  // Getters for odometry state
  double getX() const {return x_;}
  double getY() const {return y_;}
  double getYaw() const {return yaw_;}
  double getVbx() const {return v_bx_;}
  double getOmegabz() const {return omega_bz_;}

  /**
   * \brief Get commands for the platform from the desired twist
   * \param v_bx          Linear velocity of the robot in x_b-axis direction [m/s]
   * \param omega_bz      Angular velocity of the robot around x_z-axis [rad/s]
   * \param steering_angle Output steering angle [rad]
   */
  void twist_to_commands(
    const double v_bx, const double omega_bz, double & steering_angle);

  /**
   * \brief Compute odometry based on platform velocity and steering angle
   * \param platform_speed  Velocity of the robot base [m/s]
   * \param steering_angle Steering angle [rad]
   * \param dt      time difference to last call
   */
  void update_odometry(double platform_speed, double steering_angle, double dt);

  // Reset pose
  void reset_odometry(
    double x = 0.0, double y = 0.0, double yaw = 0.0,
    double v_bx = 0.0, double omega_bz = 0.0);

private:
  // Odometry state
  double x_;        // Position X (meters)
  double y_;        // Position Y (meters)
  double yaw_;      // Orientation (radians)+
  double v_bx_;     // Linear velocity in body frame
  double omega_bz_; // Angular velocity in body frame

  // Kinematic parameters
  double wheel_base_;   // Distance between front and rear axle (meters)
  double wheel_radius_;   // Radius of the wheels (meters)
  double max_steering_angle_; // Maximum steering angle (radians)
  double min_steering_angle_; // Minimum steering angle (radians)
  double max_linear_velocity_; // Maximum linear velocity (m/s)
  double min_linear_velocity_; // Minimum linear velocity (m/s)
  double max_angular_velocity_; // Maximum angular velocity (rad/s)
  double min_angular_velocity_; // Minimum angular velocity (rad/s)

  /**
   * \brief Calculates steering angle from the desired twist
   * \param v_bx     Linear velocity of the robot in x_b-axis direction
   * \param omega_bz Angular velocity of the robot around x_z-axis
   */
  double convert_twist_to_steering_angle(const double v_bx, const double omega_bz);
};

} // namespace demetra_serial_driver
#endif // DEMETRA_SERIAL_DRIVER__KINEMATISCS_HPP
