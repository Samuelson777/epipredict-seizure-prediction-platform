cmake_minimum_required(VERSION 3.16)
project(EpiPredictEdgeDaemon VERSION 2.5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(edge_dsp_daemon edge_dsp_daemon.cpp)

if(UNIX AND NOT APPLE)
    target_link_libraries(edge_dsp_daemon PRIVATE pthread m)
endif()