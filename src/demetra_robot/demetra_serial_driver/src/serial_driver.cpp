#include <iostream>
#include <vector>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <algorithm>
#include <errno.h>
#include <thread>
#include <chrono>

#include "demetra_serial_driver/serial_driver.hpp"

namespace demetra_serial_driver
{
SerialDriver::SerialDriver() {}
SerialDriver::SerialDriver(const std::string & port, int baudrate)
: port_name_(port), baudrate_(baudrate), timeout_ms_(100), fd_(-1), commands_(
    COMMANDS_PACKET_SIZE, 0.0), feedback_(
    FEEDBACK_PACKET_SIZE, 0.0) {}

SerialDriver::~SerialDriver()
{
  close();
}

bool SerialDriver::open()
{
  fd_ = ::open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (fd_ < 0) {
    perror("Error opening serial port");
    return false;
  }
  if (!configurePort()) {
    ::close(fd_);
    fd_ = -1;
    return false;
  }
  return true;
}

void SerialDriver::close()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool SerialDriver::setPlatformSpeed(double platform_speed)
{
  std::lock_guard<std::mutex> lock(commands_mutex_);
  commands_[0] = static_cast<float>(platform_speed);
  return true;
}
bool SerialDriver::setSteeringAngle(double steering_angle)
{
  std::lock_guard<std::mutex> lock(commands_mutex_);
  commands_[1] = static_cast<float>(steering_angle);
  return true;
}
bool SerialDriver::setFlagInPlace(bool flag_in_place)
{
  std::lock_guard<std::mutex> lock(commands_mutex_);
  commands_[2] = static_cast<float>(flag_in_place ? 1.0 : 0.0);
  return true;
}
bool SerialDriver::setAngleOfRotation(double angle_of_rotation)
{
  std::lock_guard<std::mutex> lock(commands_mutex_);
  commands_[3] = static_cast<float>(angle_of_rotation);
  return true;
}
void SerialDriver::resetLightsIfExpired()
{
  {
    std::lock_guard<std::mutex> lock(commands_mutex_);
    if (commands_[4] == 0.0f ||
      Clock::now() - last_light_time_ <= lights_hold_delay)
    {
      return; 
    }
  }
  resetLights();
}

bool SerialDriver::setLights(int label_id)
{
  std::lock_guard<std::mutex> lock(commands_mutex_);
  // 0 -> OFF, 1 -> WHITE, 2 -> RED, 3 -> YELLOW, 4 -> GREEN
  // commands_[4] = static_cast<float>(label_id);
  // if (label_id > 0) {
  //   last_light_time_ = Clock::now();
  // }
  // return true;
  if (label_id > 0) {
    if (commands_[4] != 0.0f && Clock::now() - last_light_time_ <= lights_hold_delay) {
      return false;
    }
    commands_[4] = static_cast<float>(label_id);
    last_light_time_ = Clock::now();
    return true;
  }
  commands_[4] = 0.0f;
  return true;
}

bool SerialDriver::areCommandsZero()
{
  std::lock_guard<std::mutex> lock(commands_mutex_);
  for (size_t i = 0; i < COMMANDS_PACKET_SIZE; i++) {
    if (commands_[i] != 0.0f) {
      return false;
    }
  }
  return true;
}

bool SerialDriver::resetCommands()
{
  std::lock_guard<std::mutex> lock(commands_mutex_);
  for (size_t i = 0; i < COMMANDS_PACKET_SIZE; i++) {
    commands_[i] = 0.0f;
  }
  return true;
}

float SerialDriver::getPlatformSpeed() const
{
  std::lock_guard<std::mutex> lock(feedback_mutex_);
  return feedback_[0];
}
float SerialDriver::getSteeringAngle() const
{
  std::lock_guard<std::mutex> lock(feedback_mutex_);
  return feedback_[1];
}
float SerialDriver::getIsTurning() const
{
  std::lock_guard<std::mutex> lock(feedback_mutex_);
  return feedback_[2];
}
float SerialDriver::getMode() const
{
  std::lock_guard<std::mutex> lock(feedback_mutex_);
  return feedback_[3];
}
float SerialDriver::getFeedback5() const
{
  std::lock_guard<std::mutex> lock(feedback_mutex_);
  return feedback_[4];
}
std::vector<float> SerialDriver::getFeedback() const
{
  std::lock_guard<std::mutex> lock(feedback_mutex_);
  return feedback_;
}

std::vector<float> SerialDriver::getCommands() const
{
  std::lock_guard<std::mutex> lock(commands_mutex_);
  return commands_;
}

void SerialDriver::encodePacket()
{
  std::lock_guard<std::mutex> lock(commands_mutex_);

  commands_[COMMANDS_PACKET_SIZE - 1] = 0.0f; // Reset checksum
  for (size_t i = 0; i < COMMANDS_PACKET_SIZE - 1; i++) {
    commands_[COMMANDS_PACKET_SIZE - 1] += commands_[i];
  }
  for (size_t i = 0; i < COMMANDS_PACKET_SIZE; i++) {
    memcpy(&commands_binary_[i * sizeof(float)], &commands_[i], sizeof(float));
  }
}

void SerialDriver::decodePacket()
{
  std::lock_guard<std::mutex> lock(feedback_mutex_);
  for (size_t i = 0; i < FEEDBACK_PACKET_SIZE; i++) {
    memcpy(&feedback_[i], &feedback_binary_[i * sizeof(float)], sizeof(float));
  }
}

bool SerialDriver::read()
{
  ssize_t n_bytes_read;
  return read(n_bytes_read);
}

bool SerialDriver::read(ssize_t & n_bytes_read)
{

  auto start_time = std::chrono::steady_clock::now();
  n_bytes_read = 0;
  auto elapsed_time_ms = 0;

  if (fd_ < 0) {return false;}
  

  while (n_bytes_read < FEEDBACK_BINARY_PACKET_SIZE && elapsed_time_ms < timeout_ms_) {
    elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_time).count();
    n_bytes_read += ::read(
      fd_, &feedback_binary_[n_bytes_read],
      FEEDBACK_BINARY_PACKET_SIZE - n_bytes_read);
    std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    
  }

  if (n_bytes_read == FEEDBACK_BINARY_PACKET_SIZE) {
    // memcpy(feedback_binary_, tmp_buffer, FEEDBACK_BINARY_PACKET_SIZE);
    decodePacket();
    float checksum = 0.0f;
    for (size_t i = 0; i < FEEDBACK_PACKET_SIZE - 1; i++) {
      checksum += feedback_[i];
    }
    return checksum - feedback_[FEEDBACK_PACKET_SIZE - 1] < std::numeric_limits<float>::epsilon();
  }

  return false; // Read failed or not enough bytes
}

bool SerialDriver::write()
{
  ssize_t n_bytes_written;
  return write(n_bytes_written);
}

bool SerialDriver::write(ssize_t & n_bytes_written)
{
  resetLightsIfExpired();  // clear the lights once their hold time has elapsed
  encodePacket();
  if (fd_ < 0) {throw std::runtime_error("Serial port not open");}
  n_bytes_written = ::write(fd_, commands_binary_, COMMANDS_BINARY_PACKET_SIZE);
  if (n_bytes_written == -1 && (errno == ENODEV || errno == EIO || errno == EBADF)) {
    perror("Serial port disconnected");
    close();
    return false;
  }
  if (n_bytes_written != COMMANDS_BINARY_PACKET_SIZE) {
    return false;
  }
  // resetLights()
  // resetFlagInPlace();
  // resetAngleOfRotation();
  return true;
}

bool SerialDriver::isOpen() const
{
  return fd_ >= 0;
}


bool SerialDriver::configurePort()
{
  struct termios tty;
  if (tcgetattr(fd_, &tty) != 0) {
    perror("tcgetattr");
    return false;
  }

  cfsetospeed(&tty, baudrateToTermios(baudrate_));
  cfsetispeed(&tty, baudrateToTermios(baudrate_));

  tty.c_cflag &= ~(PARENB | PARODD);        // shut off parity
  tty.c_cflag &= ~CSTOPB;                 // only one stop bit
  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;       // 8-bit chars
  tty.c_cflag &= ~CRTSCTS;          // No Hardware flow Control
  tty.c_cflag |= (CLOCAL | CREAD);  // ignore modem controls, enable reading

  tty.c_iflag &= ~IGNBRK;           // disable break processing
  tty.c_iflag &= ~(IXON | IXOFF | IXANY);   // shut off xon/xoff flow control

  // tty.c_lflag = 0;                  // no signaling chars, no echo, no canonical processing
  tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // NON Canonical mode

  tty.c_oflag = 0;                  // no remapping, no delays
  tty.c_oflag &= ~OPOST;            // raw output

  // From man termios:
  // If data is available, read(2) returns immediately, with the lesser of the number of bytes available, or the number of bytes requested.
  // If no data is available, read(2) returns 0.
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;


  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    perror("tcsetattr");
    return false;
  }
  return true;
}

speed_t SerialDriver::baudrateToTermios(int baudrate)
{
  switch (baudrate) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    default: return B9600;
  }
}
} // namespace demetra_serial_driver
