cmake_minimum_required(VERSION 3.8)
project(serenade_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python packages
ament_python_install_package(serenade_chatbot)
ament_python_install_package(serenade_ros_api)
ament_python_install_package(serenade_walker)
ament_python_install_package(serenade_agent)

# Install Python executables
install(PROGRAMS
  serenade_chatbot/main.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME chatbot_node
)
install(PROGRAMS
  serenade_agent/main.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME agent_node
)
install(PROGRAMS
  serenade_walker/main.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME walker_node
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
