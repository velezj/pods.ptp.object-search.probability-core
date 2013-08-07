cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME probability-core)
include(cmake/pods.cmake)

# automatically build LCM types.  This also defines a number of CMake
# variables, see cmake/lcmtypes.cmake for details
include(cmake/lcmtypes.cmake)
lcmtypes_build()

# build two LCM test programs
add_executable(send_message src/send_message.c)
pods_use_pkg_config_packages(send_message lcm lcmtypes_${POD_NAME})

add_executable(listener src/listener.c)
pods_use_pkg_config_packages(listener lcm lcmtypes_${POD_NAME})

# make exeutables public
pods_install_executables(send_message listener)



#set(CMAKE_CXX_FLAGS "-std=c++0x -pedantic -Wall -O2 -g")
#set(CMAKE_CXX_FLAGS "-std=c++0x -pedantic -Wall -O0 -g3")

add_definitions( -std=c++0x -pedantic -Wall )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

# The probability-core library
add_library( probability-core SHARED
  src/core.cpp
  src/gamma.cpp
  src/beta.cpp
  src/poisson.cpp
  src/negative_binomial.cpp
  src/gaussian.cpp
  src/discrete.cpp
  src/gamma_conjugate_prior.cpp
  src/distributions.cpp
  src/distribution_utils.cpp
  src/toms462.cpp
  )
pods_install_headers( 
  src/core.hpp
  src/gamma.hpp
  src/beta.hpp
  src/poisson.hpp
  src/negative_binomial.hpp
  src/gaussian.hpp
  src/discrete.hpp
  src/gamma_conjugate_prior.hpp
  src/distributions.hpp
  src/distribution_utils.hpp
  src/rejection_sampler.hpp
  src/toms462.hpp
  DESTINATION
  probability-core )
pods_use_pkg_config_packages(probability-core gsl math-core lcmtypes_probability-core lcmtypes_math-core boost-1.51.0)
pods_install_libraries( probability-core )
pods_install_pkg_config_file(probability-core
    CFLAGS
    LIBS -lprobability-core
    REQUIRES gsl math-core boost-1.51.0
    VERSION 0.0.1)



# The probability-core-graphics library
add_library( probability-core-graphics SHARED
  src/graphics/lcmgl_distributions.cpp
  )
pods_install_headers( 
  src/graphics/lcmgl_distributions.hpp
  DESTINATION
  probability-core-graphics )
pods_use_pkg_config_packages(probability-core-graphics gsl math-core probability-core lcmtypes_probability-core lcmtypes_math-core bot2-lcmgl-client)
pods_install_libraries( probability-core-graphics )
pods_install_pkg_config_file(probability-core-graphics
    CFLAGS
    LIBS -lprobability-core-graphics
    REQUIRES gsl math-core bot2-lcmgl-client
    VERSION 0.0.1)



# add the test subdirectory
add_subdirectory( test )
