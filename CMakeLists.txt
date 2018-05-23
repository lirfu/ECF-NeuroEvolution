cmake_minimum_required(VERSION 3.8)
project(NeuroEvolution)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp
        CellDivision/ReducedCellDivisionEval.cpp CellDivision/ReducedCellDivisionEval.h
        CellDivision/CellDivisionEval.cpp CellDivision/CellDivisionEval.h
        lib/nnpp/matrix/MatrixDimension.cpp
        NetworkCenter.cpp NetworkCenter.h
        DisplayBestOperator.cpp DisplayBestOperator.h
        PrintBestArchitectureOperator.cpp PrintBestArchitectureOperator.h
        problems/IProblem.h
        problems/XORProblem.cpp problems/XORProblem.h
        problems/RegressionProblem.cpp problems/RegressionProblem.h
        problems/WineQualityProblem.cpp problems/WineQualityProblem.h
        DirectTest.cpp DirectTest.h)

find_package(Boost)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

add_definitions(-DOUTPUT_RESULTS)

add_subdirectory(lib/nnpp)
include_directories(lib/nnpp)

add_executable(NeuroEvolution ${SOURCE_FILES})

target_link_libraries(NeuroEvolution /usr/local/lib/libecf.a)

