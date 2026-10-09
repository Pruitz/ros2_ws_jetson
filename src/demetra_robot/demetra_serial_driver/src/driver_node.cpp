#include <sstream>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_srvs/srv/empty.hpp>
#include <demetra_msgs/action/turn_in_place.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <angles/angles.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include "demetra_serial_driver/serial_driver.hpp"
#include "demetra_serial_driver/kinematics.hpp"

namespace demetra_serial_driver
{

class DriverNodeComponent : public rclcpp::Node
{
public:
  using EmptySrv = std_srvs::srv::Empty;
  using TurnInPlace = demetra_msgs::action::TurnInPlace;
  using GoalHandleTurnInPlace = rclcpp_action::ServerGoalHandle<TurnInPlace>;

  explicit DriverNodeComponent(const rclcpp::NodeOptions & options)
  : rclcpp::Node("demetra_node", options),
    serial_driver_("", 0) // temporary values, will be set below
  {
    // declare parameters
    odom_period_ms_ = this->declare_parameter<int>("odom_period_ms", 10);
    serial_period_ms_ = this->declare_parameter<int>("serial_period_ms", 10);
    cmd_vel_timeout_ms_ = this->declare_parameter<int>("cmd_vel_timeout_ms", 500);
    port_name_ = this->declare_parameter<std::string>("port_name", "/dev/ttyUSB0");
    baudrate_ = this->declare_parameter<int>("baudrate", 115200);
    wheel_base_ = this->declare_parameter<double>("wheel_base", 0.5);
    wheel_diameter_ = this->declare_parameter<double>("wheel_diameter", 0.2);
    max_steering_angle_ =
      angles::from_degrees(this->declare_parameter<double>("max_steering_angle_deg", 30.0));
    max_linear_velocity_ = this->declare_parameter<double>("max_linear_velocity_m_s", 1.0);
    max_angular_velocity_ = this->declare_parameter<double>("max_angular_velocity_rad_s", 1.0);
    odom_frame_id_ = this->declare_parameter<std::string>("odom_frame_id", "odom");
    base_frame_id_ = this->declare_parameter<std::string>("base_frame_id", "base_link");
    publish_tf_ = this->declare_parameter<bool>("publish_tf", true);

    // Re-initialize serial_driver_ with correct parameters using placement new
    new (&serial_driver_) SerialDriver(port_name_, baudrate_);

    kinematics_ = Kinematics(
      wheel_base_, wheel_diameter_,
      max_steering_angle_, max_linear_velocity_, max_angular_velocity_);

    // define subscribers, publishers, services
    using namespace std::placeholders;

    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 10,
      std::bind(&DriverNodeComponent::cmd_vel_callback, this, _1));
    // right_light_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    //   "right_light", 10,
    //   [this](const std_msgs::msg::Bool::SharedPtr msg) {
    //     this->light_callback(msg, "right");
    //   });
    // left_light_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    //   "left_light", 10,
    //   [this](const std_msgs::msg::Bool::SharedPtr msg) {
    //     this->light_callback(msg, "left");
    //   });
    light_sub_ = this->create_subscription<std_msgs::msg::Int32>(
      "/light", 10,
      [this](const std_msgs::msg::Int32::SharedPtr msg) {
        this->light_callback(msg);
      });

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }

    reset_odometry_srv_ = this->create_service<EmptySrv>(
      "reset_odometry",
      std::bind(&DriverNodeComponent::handle_reset_odometry, this, _1, _2));

    turn_in_place_action_server_ = rclcpp_action::create_server<TurnInPlace>(
      this->get_node_base_interface(),
      this->get_node_clock_interface(),
      this->get_node_logging_interface(),
      this->get_node_waitables_interface(),
      "turn_in_place",
      std::bind(&DriverNodeComponent::handle_turn_in_place_goal, this, _1, _2),
      std::bind(&DriverNodeComponent::handle_turn_in_place_cancel, this, _1),
      std::bind(&DriverNodeComponent::handle_turn_in_place_accepted, this, _1)
    );

    // start timers
    open_serial_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(5000),
      [this]() {
        if (!serial_driver_.isOpen()) {
          RCLCPP_ERROR(this->get_logger(), "Serial port is not open, trying to reopen...");
          open_serial_port();
        }
      });
    odom_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(odom_period_ms_),
      std::bind(&DriverNodeComponent::publish_odometry, this));
    serial_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(serial_period_ms_),
      std::bind(&DriverNodeComponent::poll_serial_data, this));
    cmd_vel_watchdog_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      [this]() {
        if (this->now() - last_cmd_vel_time_ >
        rclcpp::Duration::from_seconds(cmd_vel_timeout_ms_ / 1000.0f) &&
        !serial_driver_.areCommandsZero())
        {
          serial_driver_.resetPlatformSpeed();
        }
      });
  }

private:
  void open_serial_port()
  {
    if (!serial_driver_.open()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open serial port");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Serial port opened successfully: %s", port_name_.c_str());
  }

  /* brief Timer callback to handle cmd_vel messages. It sets the velocities on the serial driver. */
  void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    last_cmd_vel_time_ = this->now();
    double linear_x = msg->linear.x;
    double angular_z = msg->angular.z;

    linear_x = std::clamp(linear_x, -max_linear_velocity_, max_linear_velocity_);
    angular_z = std::clamp(angular_z, -max_angular_velocity_, max_angular_velocity_);

    double steering_angle;
    kinematics_.twist_to_commands(linear_x, angular_z, steering_angle);

    serial_driver_.setPlatformSpeed(linear_x);
    serial_driver_.setSteeringAngle(steering_angle);
  }

  /* brief Callback to handle light commands. It sets the light state on the serial driver. */
  void light_callback(const std_msgs::msg::Int32::SharedPtr msg)//, const std::string & side)
  {
    if (!serial_driver_.setLights(msg->data)) {
      RCLCPP_ERROR(
        this->get_logger(), "Failed to set light state to %d",
        msg->data);
      return;
    }
  }

  /* brief Publish odometry message and broadcast transform using the latest kinematics state. */
  void publish_odometry()
  {
    static nav_msgs::msg::Odometry odom_msg_;
    static geometry_msgs::msg::TransformStamped transform;
    static rclcpp::Time stamp;

    stamp = this->now();

    // Update odometry message with current state
    odom_msg_.header.stamp = stamp;
    odom_msg_.header.frame_id = odom_frame_id_;
    odom_msg_.child_frame_id = base_frame_id_;
    odom_msg_.pose.pose.position.x = kinematics_.getX();
    odom_msg_.pose.pose.position.y = kinematics_.getY();

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, kinematics_.getYaw());
    odom_msg_.pose.pose.orientation = tf2::toMsg(q);

    odom_msg_.twist.twist.linear.x = kinematics_.getVbx();
    odom_msg_.twist.twist.angular.z = kinematics_.getOmegabz();

    odom_pub_->publish(odom_msg_);

    if (publish_tf_) {
      transform.header.stamp = stamp;
      transform.header.frame_id = odom_frame_id_;
      transform.child_frame_id = base_frame_id_;
      transform.transform.translation.x = kinematics_.getX();
      transform.transform.translation.y = kinematics_.getY();
      transform.transform.rotation = odom_msg_.pose.pose.orientation;
      tf_broadcaster_->sendTransform(transform);
    }
  }

  /* brief Poll serial data by sending the command packet.
   * Wait for the feedback packet using a bloching read, then update the kinematic state of the robot.
   */
  void poll_serial_data()
  { 
    if (!serial_driver_.isOpen()) return;

    static rclcpp::Time last_read_time = this->now();
    ssize_t n_bytes;

    auto commands = serial_driver_.getCommands();
    RCLCPP_DEBUG_STREAM(
      this->get_logger(),
      "Sending commands: " <<[&]() {
        std::ostringstream oss;
        for (const auto & value : commands) {
          oss << std::fixed << std::setprecision(3) << value << " ";
        }
        return oss.str();
        } ()
    );
    serial_driver_.write();

    if (serial_driver_.read(n_bytes)) {
      auto now = this->now();

      kinematics_.update_odometry(
        serial_driver_.getPlatformSpeed(), serial_driver_.getSteeringAngle(),
        (now - last_read_time).seconds());

      RCLCPP_DEBUG_STREAM(
        this->get_logger(),
        "Received feedback: " <<[&]() {
          std::ostringstream oss;
          for (const auto & value : serial_driver_.getFeedback()) {
            oss << std::fixed << std::setprecision(3) << value << " ";
          }
          oss << (now - last_read_time).seconds();
          return oss.str();
          } ()
      );
      last_read_time = now;
    } else if (n_bytes > 0) {
      RCLCPP_WARN(
        this->get_logger(), "Failed to read data from serial port, read %zu bytes", n_bytes);
    } else {
      RCLCPP_WARN(this->get_logger(), "No data read from serial port");
    }
  }

  void handle_reset_odometry(
    const std::shared_ptr<EmptySrv::Request> request,
    const std::shared_ptr<EmptySrv::Response> response)
  {
    (void)request;  // Unused parameter
    (void)response; // Unused parameter
    kinematics_.reset_odometry();
    RCLCPP_INFO(this->get_logger(), "Odometry reset to zero.");
  }

  rclcpp_action::GoalResponse handle_turn_in_place_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const TurnInPlace::Goal> goal)
  {
    (void)uuid;  // Unused parameter
    serial_driver_.setFlagInPlace(true);
    serial_driver_.setAngleOfRotation(goal->radians);
    RCLCPP_INFO(
      this->get_logger(), "Received TurnInPlace goal, starting rotation with angle: %.2f degrees",
      angles::to_degrees(goal->radians));

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_turn_in_place_cancel(
    const std::shared_ptr<GoalHandleTurnInPlace> & goal_handle)
  {
    (void)goal_handle;  // Unused parameter
    serial_driver_.resetFlagInPlace();
    serial_driver_.resetAngleOfRotation();
    RCLCPP_INFO(
      this->get_logger(),
      "Received request to cancel TurnInPlace goal. Action is being canceled, but the robot will continue to turn in place until the angle is reached.");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void execute_turn_in_place(
    const std::shared_ptr<GoalHandleTurnInPlace> & goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing TurnInPlace action...");
    rclcpp::Rate rate(10); // 10 Hz loop
    auto feedback = std::make_shared<TurnInPlace::Feedback>();
    auto result = std::make_shared<TurnInPlace::Result>();
    float end_flag = -1.0;

    while (rclcpp::ok() && (goal_handle->is_active() || goal_handle->is_canceling())) {

      if (end_flag < 0.0f && !(serial_driver_.getIsTurning() > 0.0f)) {
        rate.sleep();
        RCLCPP_DEBUG(this->get_logger(), "Waiting for ack...");
        continue; // wait for the robot to start turning
      }

      end_flag = serial_driver_.getIsTurning();

      if (end_flag > 0.0f) {
        // turn still in progress
        serial_driver_.resetFlagInPlace();
        // serial_driver_.resetAngleOfRotation();
        feedback->turn_progress = 0.0f;
        goal_handle->publish_feedback(feedback);
        RCLCPP_DEBUG(this->get_logger(), "Turning");
      } else if (end_flag == 0.0f && !goal_handle->is_canceling()) {
        // turn completed
        // serial_driver_.resetFlagInPlace();
        serial_driver_.resetAngleOfRotation();

        result->success = true;
        goal_handle->succeed(result);
        RCLCPP_INFO(this->get_logger(), "TurnInPlace action completed successfully.");
        return;
      } else if (end_flag == 0.0f) {
        // turn canceled
        result->success = true;
        goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "TurnInPlace action canceled.");
        return;
      }

      if (goal_handle->is_canceling()) {
        RCLCPP_WARN(this->get_logger(), "TurnInPlace still in progress, but canceling requested.");
      }

      rate.sleep();

    }
  }

  void handle_turn_in_place_accepted(
    const std::shared_ptr<GoalHandleTurnInPlace> & goal_handle)
  {
    using namespace std::placeholders;
    std::thread(
      std::bind(&DriverNodeComponent::execute_turn_in_place, this, _1), goal_handle).detach();
  }


  SerialDriver serial_driver_;
  Kinematics kinematics_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  // rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr right_light_sub_;
  // rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr left_light_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr light_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Service<EmptySrv>::SharedPtr reset_odometry_srv_;
  rclcpp_action::Server<TurnInPlace>::SharedPtr turn_in_place_action_server_;

  rclcpp::TimerBase::SharedPtr open_serial_timer_;
  rclcpp::TimerBase::SharedPtr odom_timer_;
  rclcpp::TimerBase::SharedPtr serial_timer_;
  rclcpp::TimerBase::SharedPtr cmd_vel_watchdog_timer_;

  //parameters
  int serial_period_ms_;
  int odom_period_ms_;
  int cmd_vel_timeout_ms_;
  std::string port_name_;
  int baudrate_;
  double wheel_base_;
  double wheel_diameter_;
  double max_steering_angle_;
  double max_linear_velocity_;
  double max_angular_velocity_;
  // double min_steering_angle_;
  // double min_linear_velocity_;
  // double min_angular_velocity_;
  std::string odom_frame_id_;
  std::string base_frame_id_;
  bool publish_tf_;

  rclcpp::Time last_cmd_vel_time_ = this->now();
};  // class DriverNodeComponent

}  // namespace demetra_serial_driver

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(demetra_serial_driver::DriverNodeComponent)
