
add_executable( test-gamma
  test-gamma.cpp )
pods_use_pkg_config_packages( test-gamma gsl-1.16 p2l-probability-core lcmtypes_p2l-probability-core)
pods_install_executables( test-gamma )


add_executable( test-poisson
  test-poisson.cpp )
pods_use_pkg_config_packages( test-poisson gsl-1.16 p2l-probability-core  lcmtypes_p2l-probability-core )
pods_install_executables( test-poisson )

add_executable( test-gaussian
  test-gaussian.cpp )
pods_use_pkg_config_packages( test-gaussian gsl-1.16 p2l-probability-core  lcmtypes_p2l-probability-core )
pods_install_executables( test-gaussian )


add_executable( test-graphics
  test-graphics.cpp )
pods_use_pkg_config_packages( test-graphics p2l-probability-core-graphics )
pods_install_executables( test-graphics )


add_executable( test-rejection-sampler
  test-rejection-sampler.cpp )
pods_use_pkg_config_packages( test-rejection-sampler p2l-probability-core  lcmtypes_p2l-probability-core  p2l-math-core lcmtypes_p2l-math-core boost-1.54.0)
pods_install_executables( test-rejection-sampler )



add_executable( probability-core-test-gamma-conjugate-prior
  test-gamma-conjugate-prior.cpp )
pods_use_pkg_config_packages( probability-core-test-gamma-conjugate-prior gsl-1.16 p2l-probability-core  lcmtypes_p2l-probability-core )
pods_install_executables( probability-core-test-gamma-conjugate-prior )


add_executable( probability-core-test-gamma-conjugate-prior-update
  test-gamma-conjugate-prior-update.cpp )
pods_use_pkg_config_packages( probability-core-test-gamma-conjugate-prior-update gsl-1.16 p2l-probability-core  lcmtypes_p2l-probability-core )
pods_install_executables( probability-core-test-gamma-conjugate-prior-update )


add_executable( probability-core-test-gamma-conjugate-prior-sampler
  test-gamma-conjugate-prior-sampler.cpp )
pods_use_pkg_config_packages( probability-core-test-gamma-conjugate-prior-sampler gsl-1.16 p2l-probability-core  lcmtypes_p2l-probability-core )
pods_install_executables( probability-core-test-gamma-conjugate-prior-sampler )



add_executable( probability-core-test-sample-mean-gaussian-prior
  test-sample-mean-gaussian-prior.cpp )
pods_use_pkg_config_packages( probability-core-test-sample-mean-gaussian-prior gsl-1.16 p2l-probability-core  lcmtypes_p2l-probability-core )
pods_install_executables( probability-core-test-sample-mean-gaussian-prior )



add_executable( probability-core-test-slice-sampler
  test-slice-sampler.cpp )
pods_use_pkg_config_packages( probability-core-test-slice-sampler gsl-1.16 p2l-probability-core lcmtypes_p2l-probability-core p2l-math-core lcmtypes_p2l-math-core boost-1.54.0 )
pods_install_executables( probability-core-test-slice-sampler )


add_executable( probability-core-test-gibbs-sampler
  test-gibbs-sampler.cpp )
pods_use_pkg_config_packages( probability-core-test-slice-sampler gsl-1.16 p2l-probability-core lcmtypes_p2l-probability-core p2l-math-core lcmtypes_p2l-math-core boost-1.54.0 )
pods_install_executables( probability-core-test-gibbs-sampler )