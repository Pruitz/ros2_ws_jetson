#include "demetra_serial_driver/kinematics.hpp"
#include "angles/angles.h"

namespace demetra_serial_driver
{
Kinematics::Kinematics(){}

Kinematics::Kinematics(
  double wheel_base, double wheel_diameter,
  double max_steering_angle, double max_linear_velocity, double max_angular_velocity,
  double min_steering_angle, double min_linear_velocity, double min_angular_velocity)
{
  wheel_base_ = wheel_base;
  wheel_radius_ = wheel_diameter / 2.0;

  max_steering_angle_ = max_steering_angle;
  min_steering_angle_ = std::isnan(min_steering_angle) ? -max_steering_angle : min_steering_angle;

  max_linear_velocity_ = max_linear_velocity;
  min_linear_velocity_ =
    std::isnan(min_linear_velocity) ? -max_linear_velocity : min_linear_velocity;

  max_angular_velocity_ = max_angular_velocity;
  min_angular_velocity_ =
    std::isnan(min_angular_velocity) ? -max_angular_velocity : min_angular_velocity;

  // Initialize odometry
  x_ = 0.0;
  y_ = 0.0;
  yaw_ = 0.0;
  v_bx_ = 0.0;
  omega_bz_ = 0.0;
  last_steering_angle_ = 0.0;
}

void Kinematics::update_odometry(double platform_speed, double steering_angle, double dt)
{
  v_bx_ = platform_speed;
  omega_bz_ = 2 * v_bx_ * std::tan(steering_angle) / wheel_base_;

  x_ += v_bx_ * std::cos(yaw_) * dt;
  y_ += v_bx_ * std::sin(yaw_) * dt;
  yaw_ += omega_bz_ * dt;
  yaw_ = angles::normalize_angle(yaw_);
}

double Kinematics::convert_twist_to_steering_angle(double v_bx, double omega_bz)
{
  if (std::abs(v_bx) < std::numeric_limits<double>::epsilon())
  {
    return last_steering_angle_;
  }

  if (std::abs(omega_bz) < std::numeric_limits<double>::epsilon())
  {
    last_steering_angle_ = 0.0;
    return 0.0;
  }

  double phi = std::atan2(wheel_base_ * omega_bz, 2 * v_bx);
  phi = std::isfinite(phi) ? phi : 0.0;

  if (v_bx < 0.0 && omega_bz < 0.0) {
    phi += M_PI; // Adjust for reverse motion
  } else if (v_bx < 0.0 && omega_bz > 0.0) {
    phi -= M_PI; // Adjust for reverse motion
  }
  // Ensure the steering angle is within the limits
  phi = std::clamp(phi, min_steering_angle_, max_steering_angle_);

  last_steering_angle_ = phi;
  return phi;
}

void Kinematics::twist_to_commands(
  const double v_bx, const double omega_bz, double & steering_angle)
{
  steering_angle = convert_twist_to_steering_angle(v_bx, omega_bz);
}

void Kinematics::reset_odometry(double x, double y, double yaw, double v_bx, double omega_bz)
{
  x_ = x;
  y_ = y;
  yaw_ = yaw;
  v_bx_ = v_bx;
  omega_bz_ = omega_bz;
  last_steering_angle_ = 0.0;
}

} // namespace demetra_serial_driver
