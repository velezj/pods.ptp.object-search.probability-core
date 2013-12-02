
#define BOOST_TEST_MODULE bumps
#include <boost/test/included/unit_test.hpp>

#include "common-likelihoods.hpp"
#include <iostream>
#include <probability-core/bumps.hpp>
#include <boost/function.hpp>


using namespace probability_core;


//========================================================================

BOOST_AUTO_TEST_SUITE( find_single_bump_bumpy_lik_suite )

//========================================================================

BOOST_AUTO_TEST_CASE( find_single_bump_test_simple_1 )
{
  simple_lik lik;
  boost::function1<double,double> lik_f = lik;
  boost::function3<std::vector<double>,double,double,std::pair<double,double> > neigh_f = neighborhood::uniform_radius<double>();
  
  // since it only has a single mode, try to find the single bump
  double max_step = 1.0;
  double min_step = 1e-10;
  size_t max_iter = 1e5;
  find_bump_status_t<double,double> status;
  double bump_loc =
    find_single_bump( lik_f,
		      lik.low_support,
		      std::make_pair( lik.low_support, lik.high_supprt ),
		      max_step,
		      min_step,
		      neigh_f,
		      max_iter,
		      status );
  
  BOOST_CHECK_LE( status.iterations, max_iter );
  //BOOST_CHECK_LT( status.seconds, 10.0 );
  BOOST_CHECK_CLOSE( bump_loc, lik.mode, 1.0 );
  
}

//========================================================================

BOOST_AUTO_TEST_CASE( find_single_bump_test_1 )
{
  size_t modes = 1;
  bumpy_likelihood_known_mode_t lik( modes, 1.0 );
  boost::function1<double,double> lik_f = lik;
  boost::function3<std::vector<double>,double,double,std::pair<double,double> > neigh_f = neighborhood::uniform_radius<double>();
  
  // since it only has a single mode, try to find the single bump
  double max_step = 1.0;
  double min_step = 1e-10;
  size_t max_iter = 1e5;
  find_bump_status_t<double,double> status;
  double bump_loc =
    find_single_bump( lik_f,
		      lik.low_support,
		      std::make_pair( lik.low_support, lik.high_supprt ),
		      max_step,
		      min_step,
		      neigh_f,
		      max_iter,
		      status );
  
  BOOST_CHECK_LE( status.iterations, max_iter );
  //BOOST_CHECK_LT( status.seconds, 10.0 );
  BOOST_CHECK_CLOSE( bump_loc, lik.mode, 1.0 );
  
}

//========================================================================

BOOST_AUTO_TEST_SUITE_END()

//========================================================================

BOOST_AUTO_TEST_SUITE( find_bumps_bumpy_lik_suite )

//========================================================================

BOOST_AUTO_TEST_CASE( find_bumps_test_1 )
{
  size_t modes = 3;
  bumpy_likelihood_known_mode_t lik( modes, 1.0 );
  boost::function1<double,double> lik_f = lik;
  boost::function3<std::vector<double>,double,double,std::pair<double,double> > neigh_f = neighborhood::uniform_radius<double>();
  uniform_distribution_t<double> u = uniform_distribution( lik.low_support,
							   lik.high_supprt );
  boost::function0<double> start_loc = [u]() { return sample_from(u); };
  boost::function2<double,double,double> distance_f = [](const double&a,
							 const double&b)
    { return fabs(a-b); };

  // since it only has a single mode, try to find the single bump
  double max_step = 1.0;
  double min_step = 1e-4;
  size_t max_iter = 1e5;
  size_t num_restarts = 1e5;
  std::vector<find_bump_status_t<double,double> > status;
  std::vector<double> bumps =
    find_bumps_using_restarts
    ( lik_f,
      std::make_pair( lik.low_support, lik.high_supprt ),
      num_restarts,
      max_step,
      min_step,
      neigh_f,
      max_iter,
      start_loc,
      distance_f,
      status );
  
  BOOST_CHECK_EQUAL( bumps.size(), modes );
}

//========================================================================

BOOST_AUTO_TEST_CASE( find_bumps_test_100 )
{
  size_t modes = 7;
  bumpy_likelihood_known_mode_t lik( modes, 100.0 );
  boost::function1<double,double> lik_f = lik;
  boost::function3<std::vector<double>,double,double,std::pair<double,double> > neigh_f = neighborhood::uniform_radius<double>();
  uniform_distribution_t<double> u = uniform_distribution( lik.low_support,
							   lik.high_supprt );
  boost::function0<double> start_loc = [u]() { return sample_from(u); };
  boost::function2<double,double,double> distance_f = [](const double&a,
							 const double&b)
    { return fabs(a-b); };

  // since it only has a single mode, try to find the single bump
  double max_step = 1.0;
  double min_step = 1e-4;
  size_t max_iter = 1e5;
  size_t num_restarts = modes * 100;
  std::vector<find_bump_status_t<double,double> > status;
  std::vector<double> bumps =
    find_bumps_using_restarts
    ( lik_f,
      std::make_pair( lik.low_support, lik.high_supprt ),
      num_restarts,
      max_step,
      min_step,
      neigh_f,
      max_iter,
      start_loc,
      distance_f,
      status );
  
  BOOST_CHECK_EQUAL( bumps.size(), modes );

  // std::cout << "lik = [";
  // for( double x = lik.low_support; x < lik.high_supprt; x = x + 0.001 * (lik.high_supprt - lik.low_support ) ) {
  //   std::cout << lik(x) << ",";
  // }
  // std::cout << "];" << std::endl;
}

//========================================================================

BOOST_AUTO_TEST_SUITE_END()

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
//========================================================================
//========================================================================
//========================================================================
//========================================================================
//========================================================================
