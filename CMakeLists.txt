cmake_minimum_required(VERSION 3.18)
project(rts_smoother LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pybind11_add_module(_core src/kalman.cpp)

install(
  TARGETS _core
  LIBRARY DESTINATION rts_smoother
  RUNTIME DESTINATION rts_smoother
  ARCHIVE DESTINATION rts_smoother
)
