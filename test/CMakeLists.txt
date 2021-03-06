
add_executable( test-gamma
  test-gamma.cpp )
pods_use_pkg_config_packages( test-gamma gsl-1.16 object-search.probability-core )
pods_install_executables( test-gamma )


add_executable( test-poisson
  test-poisson.cpp )
pods_use_pkg_config_packages( test-poisson gsl-1.16 object-search.probability-core   )
pods_install_executables( test-poisson )

add_executable( test-gaussian
  test-gaussian.cpp )
pods_use_pkg_config_packages( test-gaussian gsl-1.16 object-search.probability-core   )
pods_install_executables( test-gaussian )


add_executable( test-rejection-sampler
  test-rejection-sampler.cpp )
pods_use_pkg_config_packages( test-rejection-sampler object-search.probability-core    object-search.math-core  boost-1.54.0)
pods_install_executables( test-rejection-sampler )



add_executable( probability-core-test-gamma-conjugate-prior
  test-gamma-conjugate-prior.cpp )
pods_use_pkg_config_packages( probability-core-test-gamma-conjugate-prior gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-gamma-conjugate-prior )


add_executable( probability-core-test-gamma-conjugate-prior-update
  test-gamma-conjugate-prior-update.cpp )
pods_use_pkg_config_packages( probability-core-test-gamma-conjugate-prior-update gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-gamma-conjugate-prior-update )


add_executable( probability-core-test-gamma-conjugate-prior-sampler
  test-gamma-conjugate-prior-sampler.cpp )
pods_use_pkg_config_packages( probability-core-test-gamma-conjugate-prior-sampler gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-gamma-conjugate-prior-sampler )



add_executable( probability-core-test-sample-mean-gaussian-prior
  test-sample-mean-gaussian-prior.cpp )
pods_use_pkg_config_packages( probability-core-test-sample-mean-gaussian-prior gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-sample-mean-gaussian-prior )



add_executable( probability-core-test-slice-sampler
  test-slice-sampler.cpp )
pods_use_pkg_config_packages( probability-core-test-slice-sampler gsl-1.16 object-search.probability-core  object-search.math-core  boost-1.54.0 )
pods_install_executables( probability-core-test-slice-sampler )


add_executable( probability-core-test-gibbs-sampler
  test-gibbs-sampler.cpp )
pods_use_pkg_config_packages( probability-core-test-gibbs-sampler gsl-1.16 object-search.probability-core  object-search.math-core  boost-1.54.0 )
pods_install_executables( probability-core-test-gibbs-sampler )


add_executable( probability-core-test-gcp
  test-gcp.cpp )
pods_use_pkg_config_packages( probability-core-test-gcp 
  boost-1.54.0 gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-gcp )


add_executable( probability-core-test-autoscale-rejection-sampler
  test-autoscale-rejection-sampler.cpp )
pods_use_pkg_config_packages( probability-core-test-autoscale-rejection-sampler
  boost-1.54.0 gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-autoscale-rejection-sampler )


add_executable( probability-core-test-bumps
  test-bumps.cpp )
pods_use_pkg_config_packages( probability-core-test-bumps
  boost-1.54.0 gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-bumps )


add_executable( probability-core-test-slice-sampler-ac
  test-slice-sampler-ac.cpp )
pods_use_pkg_config_packages( probability-core-test-slice-sampler-ac
  boost-1.54.0 gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-slice-sampler-ac )


add_executable( probability-core-test-em
  test-em.cpp )
pods_use_pkg_config_packages( probability-core-test-em
  boost-1.54.0 gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-em )


add_executable( probability-core-test-negative-binomial-mle
  test-negative-binomial-mle.cpp )
pods_use_pkg_config_packages( probability-core-test-negative-binomial-mle
  boost-1.54.0 gsl-1.16 object-search.probability-core   )
pods_install_executables( probability-core-test-negative-binomial-mle )
