
#define BOOST_TEST_MODULE bumps
#include <boost/test/included/unit_test.hpp>


#include "common-likelihoods.hpp"
#include <iostream>
#include <probability-core/slice_sampler.hpp>
#include <probability-core/autocorrelation.hpp>
#include <boost/function.hpp>


using namespace probability_core;
using namespace math_core;



//========================================================================

BOOST_AUTO_TEST_SUITE( slice_sampler_suite )

//========================================================================

BOOST_AUTO_TEST_CASE( slice_sample_ac_test )
{
  size_t max_samples = 1e3;

  bumpy_likelihood_known_mode_t lik = bumpy_likelihood_known_mode_t( 1, 100.0 );
  boost::function1<double,double> lik_f = lik;
  std::pair<double,double> support_range = std::make_pair( lik.low_support, lik.high_supprt );
  slice_sampler_workplace_t<double> workplace( support_range );
  
  // get the samples from the slice sampler
  std::vector<double> samples;
  for( size_t i = 0; i < max_samples; ++i ) {
    double s = slice_sample_1d<double,double>( lik_f, workplace );
    samples.push_back( s );
  }

  // compute the autocorrelation
  std::vector<double> ac = biased_autocorrelation_sample_mean_var( samples );
  
  // make sure the autocorrelation decreases in t
  BOOST_CHECK_GT( fabs(ac[0]), fabs(ac[ac.size()-1]) );

  // std::cout << "ac = [";
  // for( auto a : ac ) {
  //   std::cout << a << ",";
  // }
  // std::cout << "];" << std::endl;
  // std::cout << std::endl << std::endl << std::endl;

  // std::cout << "samples = [";
  // for( auto a : samples ) {
  //   std::cout << a << ",";
  // }
  // std::cout << "];" << std::endl;

}

//========================================================================
//========================================================================
//========================================================================
//========================================================================
//========================================================================
//========================================================================
//========================================================================
//========================================================================
//========================================================================
//========================================================================

BOOST_AUTO_TEST_SUITE_END()
