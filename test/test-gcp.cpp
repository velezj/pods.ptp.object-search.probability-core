

#define BOOST_TEST_MODULE gamma-conjugate-prior
#include <boost/test/included/unit_test.hpp>

#include <probability-core/gamma_conjugate_prior.hpp>
#include <iostream>
#include <probability-core/uniform.hpp>
#include <probability-core/rejection_sampler.hpp>
#include <probability-core/slice_sampler.hpp>


using namespace probability_core;
using namespace math_core;
using namespace math_core::mpt;



// A fixture which creates a set of commong gamma conjugate priors
struct fixture_common_gcp
{
  fixture_common_gcp() {
    gcp_1_1_1_1.p = gcp_1_1_1_1.q = gcp_1_1_1_1.r = gcp_1_1_1_1.s = 1;
    gcp_2_1230_15_10.p = 2.0;
    gcp_2_1230_15_10.q = 1.230;
    gcp_2_1230_15_10.r = 1.5;
    gcp_2_1230_15_10.s = 1.0;
  }
  ~fixture_common_gcp() {
  }
  gamma_conjugate_prior_t gcp_1_1_1_1;
  gamma_conjugate_prior_t gcp_2_1230_15_10;
};



//=======================================================================


BOOST_AUTO_TEST_SUITE( gcp_likelihood )


//=======================================================================

BOOST_FIXTURE_TEST_CASE( gcp_1_1_1_1_likelihood_approx, fixture_common_gcp )
{
  
  // We want to test our "real" likelihood with the 
  // approximation for different gammas
  std::vector<gamma_distribution_t> gammas;
  std::vector<double> likelihoods_approx;
  std::vector<mp_float> likelihoods_true;
  for( double i = 0.0001; i <= 10; i *= 1.1 ) {
    for( double j = 0.0001; j <= 10; j *= 1.1 ) {
      
      gamma_distribution_t g;
      g.shape = i;
      g.rate = j;
      gammas.push_back( g );
      
      double approx_lik = likelihood( g, gcp_1_1_1_1 );
      mp_float true_lik = testing_true::likelihood( g, gcp_1_1_1_1 );
      likelihoods_approx.push_back( approx_lik );
      likelihoods_true.push_back( true_lik );

      BOOST_CHECK_CLOSE( approx_lik, true_lik.convert_to<double>(), 0.001 );

    }
  }

  
}



//=======================================================================


BOOST_FIXTURE_TEST_CASE( gcp_2_1230_15_10_likelihood_approx, fixture_common_gcp )
{
  
  // We want to test our "real" likelihood with the 
  // approximation for different gammas
  std::vector<gamma_distribution_t> gammas;
  std::vector<double> likelihoods_approx;
  std::vector<mp_float> likelihoods_true;
  for( double i = 0.0001; i <= 10; i *= 1.1 ) {
    for( double j = 0.0001; j <= 10; j *= 1.1 ) {
      
      gamma_distribution_t g;
      g.shape = i;
      g.rate = j;
      gammas.push_back( g );
      
      double approx_lik = likelihood( g, gcp_2_1230_15_10 );
      mp_float true_lik = testing_true::likelihood( g, gcp_2_1230_15_10 );
      likelihoods_approx.push_back( approx_lik );
      likelihoods_true.push_back( true_lik );

      BOOST_CHECK_CLOSE( approx_lik, true_lik.convert_to<double>(), 0.001 );

    }
  }
}


//=======================================================================


BOOST_FIXTURE_TEST_CASE( gcp_varied_likelihood_approx, fixture_common_gcp )
{

  uniform_distribution_t<double> gen_sample = uniform_distribution( 1.0,
								    10.0 );
  uniform_distribution_t<double> gen_num_sample = uniform_distribution( 2.0,
									100.0 );
  
  for( int n = 0; n < 10; ++n ) {
    
    // generate the number of samples
    int num_samples = static_cast<int>( sample_from( gen_num_sample ) );
    
    // generate samples
    std::vector<double> samples;
    for( int i = 0; i < num_samples; ++i ) {
      samples.push_back( sample_from( gen_sample ) );
    }

    // compute statistics for gcp prior
    double mult = 1.0;
    double sum = 0.0;
    for( double x : samples ) {
      mult *= x;
      sum += x;
    }
    
    // create a gcp from samples
    gamma_conjugate_prior_t gcp;
    gcp.p = mult;
    gcp.q = sum;
    gcp.r = gcp.s = num_samples;

    // ok, now test the likelihoods for some gammas
    for( double i = 0.0001; i <= 10; i *= 1.1 ) {
      for( double j = 0.0001; j <= 10; j *= 1.1 ) {
	
	gamma_distribution_t g;
	g.shape = i;
	g.rate = j;

	mp_float true_lik = testing_true::likelihood( g, gcp );
	double approx_lik = likelihood( g, gcp );
	
	if( approx_lik > 0.0 ) {
	  BOOST_CHECK_CLOSE( approx_lik, true_lik.convert_to<double>(), 1.0 );
	}
      }
    }
  }

}


//=======================================================================


BOOST_FIXTURE_TEST_CASE( single_g_varied_gcp_likelihood_approx, fixture_common_gcp )
{

  uniform_distribution_t<double> gen_sample = uniform_distribution( 1.0,
								    100.0 );
  uniform_distribution_t<double> gen_num_sample = uniform_distribution( 2.0,
									1000.0 );
  
  gamma_distribution_t g;
  g.shape = 1.0;
  g.rate = 1.0;
  
  for( int n = 0; n < 1000; ++n ) {
    
    // generate the number of samples
    int num_samples = static_cast<int>( sample_from( gen_num_sample ) );
    
    // generate samples
    std::vector<double> samples;
    for( int i = 0; i < num_samples; ++i ) {
      samples.push_back( sample_from( gen_sample ) );
    }

    // compute statistics for gcp prior
    double mult = 1.0;
    double sum = 0.0;
    for( double x : samples ) {
      mult *= x;
      sum += x;
    }
    
    // create a gcp from samples
    gamma_conjugate_prior_t gcp;
    gcp.p = mult;
    gcp.q = sum;
    gcp.r = gcp.s = num_samples;
  
    mp_float true_lik = testing_true::likelihood( g, gcp );
    double approx_lik = likelihood( g, gcp );

    BOOST_CHECK_GT( true_lik, mp_float(0.0) );
    if( approx_lik > 0.0 ) {
      BOOST_CHECK_CLOSE( approx_lik, true_lik.convert_to<double>(), 1.0 );
    }
  }

}



//=======================================================================

BOOST_AUTO_TEST_SUITE_END()

//=======================================================================


namespace test_functions {
  double fixed_gcp_1_1_1_1_true_likelihood( const nd_point_t& gx ) {
    gamma_distribution_t g;
    assert( !undefined( gx ) );
    assert( gx.n == 2 );
    g.shape = gx.coordinate[0];
    g.rate = gx.coordinate[1];

    gamma_conjugate_prior_t gcp;
    gcp.p = gcp.q = gcp.r = gcp.s = 1.0;
    
    return testing_true::likelihood( g, gcp ).convert_to<double>();
  }

  double fixed_gcp_1_1_1_1_approx_likelihood( const nd_point_t& gx ) {
    gamma_distribution_t g;
    assert( !undefined( gx ) );
    assert( gx.n == 2 );
    g.shape = gx.coordinate[0];
    g.rate = gx.coordinate[1];

    gamma_conjugate_prior_t gcp;
    gcp.p = gcp.q = gcp.r = gcp.s = 1.0;
    
    return likelihood( g, gcp );
  }


  struct uniform_domain_sampler
  {
    uniform_distribution_t<nd_point_t> _u;
    uniform_domain_sampler( const double& low = 1.0e-5,
			    const double& high = 1.0e5)
    {
      _u = uniform_distribution( point( low, low ),
				 point( high, high ) );
    }
    nd_point_t operator()() const {
      return sample_from( _u );
    }
  };
 
}


//=======================================================================


BOOST_AUTO_TEST_SUITE( gcp_sampling )


//=======================================================================


BOOST_FIXTURE_TEST_CASE( gcp_slice_sampler, fixture_common_gcp )
{
  
  int max_samples = 1000;
  
  std::vector<nd_point_t> slice_samples_100;
  std::vector<nd_point_t> reject_samples_100;
  std::vector<nd_point_t> slice_samples_1000;
  std::vector<nd_point_t> reject_samples_1000;
  
  for( int n = 0; n < max_samples; ++n ) {

    boost::function1<double, nd_point_t> lik_f 
      = test_functions::fixed_gcp_1_1_1_1_true_likelihood;
    boost::function0<nd_point_t> uni_f
      = test_functions::uniform_domain_sampler();
    rejection_sampler_status_t rejection_sampler_status;
    nd_point_t r_sample = 
      rejection_sample<nd_point_t>( lik_f, uni_f, rejection_sampler_status );

    BOOST_WARN_LT( rejection_sampler_status.seconds, 0.01 );
    BOOST_CHECK_LT( rejection_sampler_status.seconds, 1.0 );

    gamma_distribution_t s_g_sample =
      slice_sample_from( gcp_1_1_1_1 );
    nd_point_t s_sample = point( s_g_sample.shape,
				 s_g_sample.rate );
    
    if( n < 100 ) {
      slice_samples_100.push_back( s_sample );
      reject_samples_100.push_back( r_sample );
    }
    slice_samples_1000.push_back( s_sample );
    reject_samples_1000.push_back( r_sample );

  }


  // compute the means
  nd_point_t slice_mean_100;
  nd_point_t reject_mean_100;
  nd_point_t slice_mean_1000;
  nd_point_t reject_mean_1000;
  nd_point_t sum = point(0.0,0.0);
  for( nd_point_t x : slice_samples_100 ) {
    sum = sum + ( x - point( 0.0, 0.0 ) );
  }
  slice_mean_100 = point(0.0,0.0) + ( ( sum - point(0.0,0.0) ) * (1.0/100) );
  sum = point(0.0,0.0);
  for( nd_point_t x : reject_samples_100 ) {
    sum = sum + ( x - point( 0.0, 0.0 ) );
  }
  reject_mean_100 = point(0.0,0.0) + ( ( sum - point(0.0,0.0) ) * (1.0/100) );
  sum = point(0.0,0.0);
  for( nd_point_t x : slice_samples_1000 ) {
    sum = sum + ( x - point( 0.0, 0.0 ) );
  }
  slice_mean_1000 = point(0.0,0.0) + ( ( sum - point(0.0,0.0) ) * (1.0/1000) );
  sum = point(0.0,0.0);
  for( nd_point_t x : reject_samples_1000 ) {
    sum = sum + ( x - point( 0.0, 0.0 ) );
  }
  reject_mean_1000 = point(0.0,0.0) + ( ( sum - point(0.0,0.0) ) * (1.0/1000) );

  

  BOOST_CHECK_GT( distance( reject_mean_100, slice_mean_100 ),
		  distance( reject_mean_1000, slice_mean_1000 ) );
  BOOST_CHECK_CLOSE( reject_mean_1000.coordinate[0], slice_mean_1000.coordinate[0], 1.0 );
  BOOST_CHECK_CLOSE( reject_mean_1000.coordinate[1], slice_mean_1000.coordinate[1], 1.0 );
 
}


//=======================================================================


BOOST_AUTO_TEST_SUITE_END()

