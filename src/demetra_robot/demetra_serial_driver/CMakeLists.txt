cmake_minimum_required(VERSION 3.8)
project(demetra_serial_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set(THIS_PKG_DEPS
  angles
  demetra_msgs
  geometry_msgs
  nav_msgs
  rclcpp
  rclcpp_components
  std_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

foreach(dep ${THIS_PKG_DEPS})
  find_package(${dep} REQUIRED)
endforeach()

include_directories(include)

add_library(serial_driver SHARED
  src/serial_driver.cpp
)
ament_target_dependencies(serial_driver ${THIS_PKG_DEPS})

add_library(kinematics SHARED
  src/kinematics.cpp
)
ament_target_dependencies(kinematics ${THIS_PKG_DEPS})

add_library(driver_node_component SHARED
  src/driver_node.cpp
)
ament_target_dependencies(driver_node_component ${THIS_PKG_DEPS})
target_link_libraries(driver_node_component
  serial_driver 
  kinematics
)

rclcpp_components_register_node(driver_node_component
  PLUGIN "demetra_serial_driver::DriverNodeComponent"
  EXECUTABLE driver_node
)

# Install our headers
install(
  DIRECTORY include/
  DESTINATION include
)
# Install our node and library
install(
  TARGETS driver_node driver_node_component serial_driver kinematics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PACKAGE_NAME}
)

install(
	DIRECTORY config
	DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(driver_node_component serial_driver kinematics)
ament_export_dependencies(${THIS_PKG_DEPS})

ament_package()
