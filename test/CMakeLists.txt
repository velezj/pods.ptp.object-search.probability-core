
add_executable( test-gamma
  test-gamma.cpp )
pods_use_pkg_config_packages( test-gamma gsl probability-core lcmtypes_probability-core)
pods_install_executables( test-gamma )


add_executable( test-poisson
  test-poisson.cpp )
pods_use_pkg_config_packages( test-poisson gsl probability-core lcmtypes_probability-core)
pods_install_executables( test-poisson )

add_executable( test-gaussian
  test-gaussian.cpp )
pods_use_pkg_config_packages( test-gaussian gsl probability-core lcmtypes_probability-core)
pods_install_executables( test-gaussian )


add_executable( test-graphics
  test-graphics.cpp )
pods_use_pkg_config_packages( test-graphics probability-core-graphics )
pods_install_executables( test-graphics )


add_executable( test-rejection-sampler
  test-rejection-sampler.cpp )
pods_use_pkg_config_packages( test-rejection-sampler probability-core lcmtypes_probability-core math-core lcmtypes_math-core boost-1.51.0)
pods_install_executables( test-rejection-sampler )



add_executable( probability-core-test-gamma-conjugate-prior
  test-gamma-conjugate-prior.cpp )
pods_use_pkg_config_packages( probability-core-test-gamma-conjugate-prior gsl probability-core lcmtypes_probability-core)
pods_install_executables( probability-core-test-gamma-conjugate-prior )


add_executable( probability-core-test-gamma-conjugate-prior-update
  test-gamma-conjugate-prior-update.cpp )
pods_use_pkg_config_packages( probability-core-test-gamma-conjugate-prior-update gsl probability-core lcmtypes_probability-core)
pods_install_executables( probability-core-test-gamma-conjugate-prior-update )


add_executable( probability-core-test-gamma-conjugate-prior-sampler
  test-gamma-conjugate-prior-sampler.cpp )
pods_use_pkg_config_packages( probability-core-test-gamma-conjugate-prior-sampler gsl probability-core lcmtypes_probability-core)
pods_install_executables( probability-core-test-gamma-conjugate-prior-sampler )



add_executable( probability-core-test-sample-mean-gaussian-prior
  test-sample-mean-gaussian-prior.cpp )
pods_use_pkg_config_packages( probability-core-test-sample-mean-gaussian-prior gsl probability-core lcmtypes_probability-core)
pods_install_executables( probability-core-test-sample-mean-gaussian-prior )
