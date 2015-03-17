cmake_minimum_required(VERSION 2.6.0)

cmake_policy( SET CMP0045 OLD )
cmake_policy( SET CMP0026 OLD )


# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME object-search.probability-core)
include(cmake/pods.cmake)




#set(CMAKE_CXX_FLAGS "-std=c++0x -pedantic -Wall -O2 -g")
#set(CMAKE_CXX_FLAGS "-std=c++0x -pedantic -Wall -O0 -g3")

add_definitions( -std=c++0x -Wall -fdiagnostics-show-option -Wno-unused-local-typedefs -fPIC )
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")

option ( USE_PEDANTIC "Turn on -pendantic mode in gcc. This will spit out *lots* of warnings from lcm :-(, but hopefully none from the rest of the code" OFF)
if( USE_PEDANTIC )
    add_definitions( -pedantic )
endif (USE_PEDANTIC)


# The probability-core library
add_library( object-search.probability-core SHARED
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
  src/slice_sampler.cpp
  src/gibbs_sampler.cpp
  src/EM.cpp
  )
pods_install_headers(
  src/types.hpp
  src/core.hpp
  src/gamma.hpp
  src/beta.hpp
  src/poisson.hpp
  src/negative_binomial.hpp
  src/gaussian.hpp
  src/discrete.hpp
  src/gamma_conjugate_prior.hpp
  src/uniform.hpp
  src/distributions.hpp
  src/distribution_utils.hpp
  src/rejection_sampler.hpp
  src/toms462.hpp
  src/slice_sampler.hpp
  src/gibbs_sampler.hpp
  src/bumps.hpp
  src/autocorrelation.hpp
  src/EM.hpp
  DESTINATION
  probability-core )
pods_use_pkg_config_packages(object-search.probability-core
  gsl
  object-search.common
  object-search.math-core
  boost-1.54.0
  boost-1.54.0-system
  boost-1.54.0-thread)
pods_install_libraries( object-search.probability-core )
pods_install_pkg_config_file(object-search.probability-core
    CFLAGS
    LIBS -lobject-search.probability-core
    REQUIRES gsl-1.16 object-search.common object-search.math-core boost-1.54.0 boost-1.54.0-system boost-1.54.0-thread
    VERSION 0.0.1)



# add the test subdirectory
add_subdirectory( test )
