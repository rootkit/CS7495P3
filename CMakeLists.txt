cmake_minimum_required(VERSION 2.8.3)
project(ap_robot_utils)
set(BUILD_FLAGS "-std=c++0x -Wreturn-type")
SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE "Debug")

# Add additional CMake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
link_libraries(${OpenCV_LIBS})

find_package(CppUnit)
include_directories(${CPPUNIT_INCLUDE_DIR})
link_libraries(${CPPUNIT_LIBRARIES})

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS})
link_libraries(${catkin_LIBRARIES})

file(GLOB felz_source "src/libraries/felzenszwalb/*.h")
include_directories("src/libraries/felzenszwalb")
## Declare a cpp library
add_library(${PROJECT_NAME}
  src/libraries/image_segmentation.cpp
  ${felz_source})

#############
## Testing ##
#############

## Allow Testing
enable_testing()

# Grab all unit test files
file(GLOB include_source "include/")
list(SORT include_source)
file(GLOB unit_tests_source "test/unit/*.cpp")
list(SORT unit_tests_source)
file(GLOB system_tests_source "test/system/*.cpp")
list(SORT system_tests_source)

# Build full unit test
add_executable(TestAllUnits "test/test_units.cpp" ${unit_tests_source})
add_test(TestAllUnits ${EXECUTABLE_OUTPUT_PATH}/TestAllUnits)
target_link_libraries(TestAllUnits ${PROJECT_NAME})
set_target_properties(TestAllUnits PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})

# Build standalone unit tests
message(STATUS "\n-- UNIT TESTS: ")
foreach(utest_src_file ${unit_tests_source})
	get_filename_component(test_base ${utest_src_file} NAME_WE)
	message(STATUS "Adding test ${test_base}" )
	add_executable(${test_base} "test/test_units.cpp" ${utest_src_file})
	add_test(${test_base} ${EXECUTABLE_OUTPUT_PATH}/${test_base})
	#catkin_add_gtest(${test_base} ${EXECUTABLE_OUTPUT_PATH}/${test_base})
	target_link_libraries(${test_base} ${PROJECT_NAME})
	set_target_properties(${test_base} PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})
endforeach(utest_src_file)

# Build system tests
message(STATUS "\n-- System TESTS: ")
foreach(stest_src_file ${system_tests_source})
	get_filename_component(test_base ${stest_src_file} NAME_WE)
	message(STATUS "Adding test ${test_base}" )
	add_executable(${test_base} ${stest_src_file})
	add_test(${test_base} ${EXECUTABLE_OUTPUT_PATH}/${test_base})
	#catkin_add_gtest(${test_base} ${EXECUTABLE_OUTPUT_PATH}/${test_base})
	target_link_libraries(${test_base} ${PROJECT_NAME})
	set_target_properties(${test_base} PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS})
endforeach(stest_src_file)
