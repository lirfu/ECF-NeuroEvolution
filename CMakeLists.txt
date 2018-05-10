cmake_minimum_required(VERSION 3.8)
project(NeuroEvolution)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp
        CellDivision/ReducedCellDivisionEval.cpp CellDivision/ReducedCellDivisionEval.h
        problems/XORProblem.cpp problems/XORProblem.h problems/IProblem.h)

find_package(Boost)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

include_directories(lib/nnpp)

add_executable(NeuroEvolution ${SOURCE_FILES})

target_link_libraries(NeuroEvolution /usr/local/lib/libecf.a)
