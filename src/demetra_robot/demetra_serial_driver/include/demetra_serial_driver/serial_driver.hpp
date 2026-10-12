#ifndef DEMETRA_SERIAL_DRIVER__SERIAL_DRIVER_HPP
#define DEMETRA_SERIAL_DRIVER__SERIAL_DRIVER_HPP

#include <string>
#include <cstdint>
#include <cstddef>
#include <sys/types.h>
#include <termios.h>
#include <vector>
#include <mutex>
#include <chrono>
#include <cmath>

namespace demetra_serial_driver
{
class SerialDriver
{
public:
  SerialDriver();
  SerialDriver(const std::string & port, int baudrate);
  ~SerialDriver();

  void init(const std::string & port, int baudrate);

  bool open();
  void close();

  bool read();
  bool read(ssize_t & n_bytes_read);
  bool write();
  bool write(ssize_t & n_bytes_written);

  bool isOpen() const;

  bool setPlatformSpeed(double platform_speed);
  bool resetPlatformSpeed() {return setPlatformSpeed(0.0);}
  bool setSteeringAngle(double steering_angle);
  bool resetSteeringAngle() {return setSteeringAngle(0.0);}
  bool setFlagInPlace(bool flag_on_spot);
  bool resetFlagInPlace() {return setFlagInPlace(false);}
  bool setAngleOfRotation(double angle_of_rotation);
  bool resetAngleOfRotation() {return setAngleOfRotation(0.0);}
  bool setLights(int label_id);
  bool resetLights() {return setLights(0);}
  bool resetCommands();
  bool areCommandsZero();

  float getPlatformSpeed() const;
  float getSteeringAngle() const;
  float getIsTurning() const;
  float getMode() const;
  float getFeedback5() const;
  std::vector<float> getFeedback() const;
  std::vector<float> getCommands() const;

  static constexpr ssize_t COMMANDS_PACKET_SIZE = 6;
  static constexpr ssize_t COMMANDS_BINARY_PACKET_SIZE = COMMANDS_PACKET_SIZE * sizeof(float);
  static constexpr ssize_t FEEDBACK_PACKET_SIZE = 6;
  static constexpr ssize_t FEEDBACK_BINARY_PACKET_SIZE = FEEDBACK_PACKET_SIZE * sizeof(float);

private:
  bool configurePort();
  speed_t baudrateToTermios(int baudrate);

  using Clock = std::chrono::steady_clock;
  static constexpr std::chrono::milliseconds lights_hold_delay{3500};
  Clock::time_point last_light_time_ = Clock::now();

  void encodePacket();
  void decodePacket();

  void resetLightsIfExpired();

  std::string port_name_;
  int baudrate_;
  int timeout_ms_;
  int fd_;
  mutable std::mutex commands_mutex_;
  std::vector<float> commands_;
  uint8_t commands_binary_[COMMANDS_BINARY_PACKET_SIZE];
  mutable std::mutex feedback_mutex_;
  std::vector<float> feedback_;
  uint8_t feedback_binary_[FEEDBACK_BINARY_PACKET_SIZE];
};
} // namespace demetra_serial_driver
#endif // DEMETRA_SERIAL_DRIVER__SERIAL_DRIVER_HPP
