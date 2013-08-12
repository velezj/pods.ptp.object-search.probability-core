cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME p2l-probability-core)
include(cmake/pods.cmake)

# automatically build LCM types.  This also defines a number of CMake
# variables, see cmake/lcmtypes.cmake for details
include(cmake/lcmtypes.cmake)
lcmtypes_build()




#set(CMAKE_CXX_FLAGS "-std=c++0x -pedantic -Wall -O2 -g")
#set(CMAKE_CXX_FLAGS "-std=c++0x -pedantic -Wall -O0 -g3")

add_definitions( -Wall -fdiagnostics-show-option )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

option ( USE_PEDANTIC "Turn on -pendantic mode in gcc. This will spit out *lots* of warnings from lcm :-(, but hopefully none from the rest of the code" OFF)
if( USE_PEDANTIC )
    add_definitions( -pedantic )
endif (USE_PEDANTIC)


# The probability-core library
add_library( p2l-probability-core SHARED
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
pods_use_pkg_config_packages(p2l-probability-core gsl p2l-math-core lcmtypes_p2l-probability-core lcmtypes_p2l-math-core boost-1.54.0)
pods_install_libraries( p2l-probability-core )
pods_install_pkg_config_file(p2l-probability-core
    CFLAGS
    LIBS -lp2l-probability-core
    REQUIRES gsl-1.16 p2l-math-core boost-1.54.0
    VERSION 0.0.1)



# The probability-core-graphics library
add_library( p2l-probability-core-graphics SHARED
  src/graphics/lcmgl_distributions.cpp
  )
pods_install_headers( 
  src/graphics/lcmgl_distributions.hpp
  DESTINATION
  probability-core-graphics )
pods_use_pkg_config_packages(p2l-probability-core-graphics 
    gsl-1.16 
    p2l-math-core 
    p2l-probability-core 
    lcmtypes_p2l-probability-core 
    lcmtypes_p2l-math-core 
    bot2-lcmgl-client)
pods_install_libraries( p2l-probability-core-graphics )
pods_install_pkg_config_file(p2l-probability-core-graphics
    CFLAGS
    LIBS -lp2l-probability-core-graphics
    REQUIRES gsl-1.16 p2l-math-core bot2-lcmgl-client p2l-probability-core
    VERSION 0.0.1)



# add the test subdirectory
add_subdirectory( test )
