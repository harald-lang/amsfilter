cmake_minimum_required(VERSION 3.5)
project(amsfilter-samples)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -march=native -fext-numeric-literals -Wall -Wno-deprecated -Wno-endif-labels -Wno-unused-function -DSQLITE_ENABLE_JSON1 -DSQLITE_OMIT_LOAD_EXTENSION")

# Database Template Library (DTL)
include_directories(thirdparty/dtl/src)
include_directories(thirdparty/dtl/thirdparty)

# Include the AMS-Filter.
set(AMSFILTER_NO_MAGIC 1) # Disables MAGIC addressing.
set(AMSFILTER_PARTIAL_BUILD 0) # Compiles only a small number of filters.
add_subdirectory(thirdparty/dtl/amsfilter amsfilter)
include_directories(thirdparty/dtl/amsfilter/src)
set_target_properties(amsfilter PROPERTIES COMPILE_FLAGS "-march=native")

set(LIBS
        pthread
        numa
        )


#===------------------------------------------------------------------------===#
# CUDA
#===------------------------------------------------------------------------===#
find_package(CUDA 9)

if (CUDA_FOUND)
    message("CUDA found")

    # avoid implicitly synchronization with other streams
    set(CUDA_NVCC_FLAGS --default-stream per-thread)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --restrict --std=c++14  -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO -D_MWAITXINTRIN_H_INCLUDED)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --source-in-ptx)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xptxas -dlcm=ca)

    # enable NVCC warnings
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -Wall)

    # the target architecture(s)  - (see https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_52,code=sm_52) # e.g. GTX 970
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_60,code=sm_60) # e.g. Tesla P100
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-code arch=compute_61,code=sm_61) # e.g. GTX 1080Ti

    # don't show deprecation warnings (which are a lot with CUDA 9)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-declarations)

    set(CUDA_NVCC_FLAGS_DEBUG -g --generate-line-info --keep -DDEBUG)
    set(CUDA_NVCC_FLAGS_RELEASE -O3 --use_fast_math -DNDEBUG)


    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_SEPARABLE_COMPILATION OFF)

    include_directories(/usr/local/cuda/include)

    set(LIBS ${LIBS} cudart)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_CUDA")


else()
    message("CUDA found")
endif()


cuda_add_executable(amsfilter_lite_sample src/amsfilter_lite_sample.cpp)
set_target_properties(amsfilter_lite_sample
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(amsfilter_lite_sample PROPERTIES COMPILE_FLAGS "-march=native")
target_link_libraries(amsfilter_lite_sample ${LIBS} amsfilter)

cuda_add_executable(amsfilter_lite_benchmark_device src/benchmark_device.cpp)
set_target_properties(amsfilter_lite_benchmark_device
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(amsfilter_lite_benchmark_device PROPERTIES COMPILE_FLAGS "-march=native")
target_link_libraries(amsfilter_lite_benchmark_device ${LIBS} amsfilter)

cuda_add_executable(amsfilter_lite_benchmark_host src/benchmark_host.cpp)
set_target_properties(amsfilter_lite_benchmark_host
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(amsfilter_lite_benchmark_host PROPERTIES COMPILE_FLAGS "-march=native")
target_link_libraries(amsfilter_lite_benchmark_host ${LIBS} amsfilter)

cuda_add_executable(amsfilter_lite_benchmark_coproc src/benchmark_coproc.cpp)
set_target_properties(amsfilter_lite_benchmark_coproc
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(amsfilter_lite_benchmark_coproc PROPERTIES COMPILE_FLAGS "-march=native")
target_link_libraries(amsfilter_lite_benchmark_coproc ${LIBS} amsfilter)

