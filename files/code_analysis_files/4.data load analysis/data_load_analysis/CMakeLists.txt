cmake_minimum_required(VERSION 3.17)
project(data_load_analysis C)

set(CMAKE_C_STANDARD 99)

add_executable(data_load_analysis
        main.c
        src/utils.h
        src/utils.c)

# 在Clion中生成的CMakeList.txt, 添加pthread库.
find_package(Threads)
target_link_libraries(data_load_analysis ${CMAKE_THREAD_LIBS_INIT})