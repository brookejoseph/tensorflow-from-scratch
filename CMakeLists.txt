cmake_minimum_required(VERSION 3.12)
project(MiniTensorFlow)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Include backend headers
include_directories(${CMAKE_SOURCE_DIR}/backend)

# Backend source files
set(BACKEND_SOURCES
    backend/tensor.cc
    backend/operation.cc
)

# Binder source file
set(BINDER_SOURCES
    binders/pybinder.cc
)

# Create the Pybind11 module
pybind11_add_module(minitf ${BACKEND_SOURCES} ${BINDER_SOURCES})

# Link required libraries
target_link_libraries(minitf PRIVATE Python3::Python pybind11::module)

# Output message
message(STATUS "Project configured successfully!")
