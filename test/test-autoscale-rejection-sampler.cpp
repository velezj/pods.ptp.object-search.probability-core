
#define BOOST_TEST_MODULE gamma-conjugate-prior
#include <boost/test/included/unit_test.hpp>

#include <iostream>
#include <probability-core/rejection_sampler.hpp>
#include <probability-core/beta.hpp>
#include <probability-core/uniform.hpp>
#include <probability-core/gamma.hpp>
#include <math-core/geom.hpp>
#include <math-core/utils.hpp>
#include <boost/math/distributions/beta.hpp>
#include <limits>


using namespace probability_core;
using namespace math_core;


//============================================================================

// Description:
// A "random" multimode likelihood functor which
// has a we-defined maximum height
struct random_1d_bounded_likelihood_t
{
  size_t num_modes;
  double max_height;
  double low_support;
  double high_support;
  std::vector<beta_distribution_t> betas;
  double max_beta_sum;
  random_1d_bounded_likelihood_t( size_t modes = 100,
				  double height = 1000.0,
				  double low = 0.0,
				  double high = 1.0e5 )
    : num_modes( modes ),
      max_height( height ),
      low_support( low ),
      high_support( high )
  {
    for( size_t i = 0; i < num_modes; ++i ) {
      beta_distribution_t b;
      b.alpha = sample_from( uniform_distribution( 2.0, 10.0 ) );
      b.beta = sample_from( uniform_distribution( 2.0, 10.0 ) );
      betas.push_back( b );
      max_beta_sum += ( b.alpha - 1 ) / ( b.alpha + b.beta - 2 );
    }
  }

  double operator() ( const double& x ) {

    //std::cout << "r_1d( " << x << " ) [" << low_support << "," << high_support << "," << max_beta_sum << "," << num_modes << "] ";

    //std::cout << std::endl;
    //for( auto b : betas ) {
    //  std::cout << b.alpha << " <> " << b.beta << std::endl;
    //}
    
    // renormalize x to be from zero to one
    if( x < low_support || x > high_support ) {
      //std::cout << std::endl;
      return 0.0;
    }
    double x_norm = ( x - low_support ) / ( high_support - low_support );
    
    //std::cout << x_norm << " ";
    
    double sum = 0.0;
    for( auto b : betas ) {
      double p = pdf( x_norm, b );
      double a = gsl_ran_beta_pdf( x_norm, b.alpha, b.beta );
      sum += p;
      //std::cout << sum << "(" << p << " / " << a << ")|";
    }

    //std::cout << "{" << ( max_height / max_beta_sum ) << "}";
    
    // rescale hieght to max-height
    sum *= ( max_height / max_beta_sum );

    //std::cout << sum << std::endl;
    
    return sum;
  }
};



struct random_bounded_likelihood_t
{
  std::vector<random_1d_bounded_likelihood_t> liks;
  nd_aabox_t window;
  random_bounded_likelihood_t( const size_t modes,
			       const double& height ,
			       const nd_aabox_t& window )
    : window( window )
  {
    assert( !undefined( window ) );
    for( int i = 0; i < window.n; ++i ) {
      random_1d_bounded_likelihood_t l( modes, 
					pow( height, 1.0 / window.n ),
					window.start.coordinate[i],
					window.end.coordinate[i] );
      liks.push_back( l );
    }
  }

  double operator()( const nd_point_t& x )
  {
    double prod = 1.0;
    for( size_t i = 0; i < liks.size(); ++i ) {
      prod *= liks[i]( x.coordinate[i] );
    }
    return prod;
  }
};


//============================================================================

// Description:
// A "random" multimode likelihood functor which
// has a we-defined maximum height, using gammas
struct random_gamma_1d_bounded_likelihood_t
{
  size_t num_modes;
  double max_height;
  double low_support;
  double high_support;
  std::vector<gamma_distribution_t> gammas;
  double max_gamma_sum;
  random_gamma_1d_bounded_likelihood_t( size_t modes = 100,
					double height = 1000.0,
					double low = 0.0,
					double high = 1.0e5 )
    : num_modes( modes ),
      max_height( height ),
      low_support( low ),
      high_support( high )
  {
    max_gamma_sum = 0.0;
    for( size_t i = 0; i < num_modes; ++i ) {
      gamma_distribution_t g;
      g.shape = sample_from( uniform_distribution( 1.1, 10.0 ) );
      g.rate = sample_from( uniform_distribution( 1.1, 10.0 ) );
      gammas.push_back( g );
      max_gamma_sum += ( g.shape - 1.0 ) / g.rate;
    }
  }

  double operator() ( const double& x ) {
    
    // renormalize x to be from zero to infinity
    if( x < low_support || x > high_support ) {
      return 0.0;
    }
    double x_norm = ( x - low_support ) + std::numeric_limits<double>::epsilon();
    
    double sum = 0.0;
    for( auto g : gammas ) {
      double p = pdf( x_norm, g );
      sum += p;
    }
    
    // rescale hieght to max-height
    sum *= ( max_height / max_gamma_sum );
    
    return sum;
  }
};



struct random_gamma_bounded_likelihood_t
{
  std::vector<random_gamma_1d_bounded_likelihood_t> liks;
  nd_aabox_t window;
  random_gamma_bounded_likelihood_t( const size_t modes,
				     const double& height ,
				     const nd_aabox_t& window )
    : window( window )
  {
    assert( !undefined( window ) );
    for( int i = 0; i < window.n; ++i ) {
      random_gamma_1d_bounded_likelihood_t l( modes, 
					      pow( height, 1.0 / window.n ),
					      window.start.coordinate[i],
					      window.end.coordinate[i] );
      liks.push_back( l );
    }
  }
  
  double operator()( const nd_point_t& x )
  {
    double prod = 1.0;
    for( size_t i = 0; i < liks.size(); ++i ) {
      prod *= liks[i]( x.coordinate[i] );
    }
    return prod;
  }
};

//============================================================================

BOOST_AUTO_TEST_SUITE( gamma_suite )

//============================================================================

BOOST_AUTO_TEST_CASE( gamma_distribution )
{
  gamma_distribution_t gamma;
  gamma.shape = gamma.rate = 2.0;
  
  for( double x = 0.01; x < 10.0; x += 0.1 ) {
    BOOST_CHECK_GT( pdf( x, gamma ), 0.0 );
    BOOST_CHECK_LT( pdf( x, gamma ), 1.0 );
  }

}

//============================================================================

BOOST_AUTO_TEST_CASE( random_gamma_1d__lik )
{
  size_t modes = 10;
  double height = 100;
  double low = -10;
  double high = 10;
  random_gamma_1d_bounded_likelihood_t lik( modes, height, low, high );
  
  for( double x = low; x < high; x += 0.01*fabs(high-low) ) {
    BOOST_CHECK_GT( lik( x ), 0.0 );
    BOOST_CHECK_LT( lik( x ), height );
  }
  
}

//============================================================================


//============================================================================

BOOST_AUTO_TEST_SUITE_END()

// BOOST_AUTO_TEST_SUITE( beta )

// //============================================================================

// BOOST_AUTO_TEST_CASE( gsl_beta_test )
// {
//   for( double x = 0.0001; x < 1.0; x = x + 0.01 * x ) {
//     BOOST_CHECK_LT( gsl_ran_beta_pdf( x, 2.0, 2.0 ), 1.0 );
//   }
// }

// //============================================================================

// BOOST_AUTO_TEST_CASE( boost_beta_test )
// {
//   boost::math::beta_distribution<double> beta( 2.0, 2.0 );
//   for( double x = 0.0001; x < 1.0; x = x + 0.01 * x ) {
//     BOOST_CHECK_GT( boost::math::pdf( beta, x ), 0.0 );
//     BOOST_CHECK_LT( boost::math::pdf( beta, x ), 1.0 );
//   }
// }



// //============================================================================

// BOOST_AUTO_TEST_CASE( beta_distribution )
// {
//   beta_distribution_t beta;
//   beta.alpha = beta.beta = 2.0;
  
//   for( double x = 1e-5; x < 1.0; x += 0.001 ) {
//     BOOST_CHECK_GT( pdf( x, beta ), 0.0 );
//   }

// }


// //============================================================================

// BOOST_AUTO_TEST_SUITE_END()

// BOOST_AUTO_TEST_SUITE( random_beta_suite )

// //============================================================================

// BOOST_AUTO_TEST_CASE( random_1d_betas_lik )
// {
//   size_t modes = 10;
//   double height = 100;
//   double low = -10;
//   double high = 10;
//   random_1d_bounded_likelihood_t lik( modes, height, low, high );
  
//   for( double x = low; x < high; x += 0.01*fabs(x) ) {
//     BOOST_CHECK_GT( lik( x ), 0.0 );
//     BOOST_CHECK_LT( lik( x ), height );
//   }

// }


// //============================================================================

// BOOST_AUTO_TEST_SUITE_END()

//============================================================================

//============================================================================

//============================================================================


//============================================================================

BOOST_AUTO_TEST_SUITE( autoscale_randomlik )


//============================================================================

BOOST_AUTO_TEST_CASE( random_height_bound_gamma_1d )
{
  
  double low = -10;
  double high = 10;
  double height = 200;
  size_t modes = 100;
  
  random_gamma_1d_bounded_likelihood_t lik( modes,
					    height,
					    low, high );
  
  for( double x = low; x <= high; x = x + 0.001 * (high-low) ) {
    double lp = lik( x );
    BOOST_CHECK_GT( lp, 0.0 );
    BOOST_CHECK_LT( lp, height );
  }
}

//============================================================================

BOOST_AUTO_TEST_CASE( random_height_bound_gamma )
{
  nd_aabox_t window = aabox( point( -10.0, -10.0 ),
			     point( 10.0, 10.0 ) );
  size_t modes = 100;
  double height = 500;
  
  random_gamma_bounded_likelihood_t lik( modes,
					 height,
					 window );
  
  for( double x = window.start.coordinate[0];
       x <= window.end.coordinate[0];
       x = x + 0.1 * (window.end.coordinate[0] - window.start.coordinate[0]) ) {
    for( double y = window.start.coordinate[1];
	 y <= window.end.coordinate[1];
	 y = y + 0.1 * (window.end.coordinate[1] - window.start.coordinate[1]) ) {
      
      double lp = lik( point( x, y ) );
      BOOST_CHECK_GT( lp, 0.0 );
      BOOST_CHECK_LT( lp, height );
      
    }
  }
}

//============================================================================


BOOST_AUTO_TEST_SUITE_END()

//============================================================================
//============================================================================

BOOST_AUTO_TEST_SUITE( autoscale_rejection_sampler_suite )

//============================================================================

struct fixture_common_random_gamma_1d_bounded_likelihood
{
  random_gamma_1d_bounded_likelihood_t bounded_1d_lik_1;
  random_gamma_1d_bounded_likelihood_t bounded_1d_lik_100;
  random_gamma_1d_bounded_likelihood_t bounded_1d_lik_001;
  size_t modes;
  double low;
  double high;
  fixture_common_random_gamma_1d_bounded_likelihood() 
  {
    modes = 97;
    low = -25.3;
    high = 17.1;
    bounded_1d_lik_1 = random_gamma_1d_bounded_likelihood_t( modes,
							     1.0,
							     low,
							     high );
    bounded_1d_lik_100 = random_gamma_1d_bounded_likelihood_t( modes,
							       100.0,
							       low,
							       high );
    bounded_1d_lik_001 = random_gamma_1d_bounded_likelihood_t( modes,
							       0.001,
							       low,
							       high );
    
  }
  ~fixture_common_random_gamma_1d_bounded_likelihood()
  {
  }
};


struct uniform_domain_sampler
{
  uniform_distribution_t<double> _u;
  uniform_domain_sampler( const double& low = 1.0e-5,
			  const double& high = 1.0e5)
  {
    _u = uniform_distribution( low, high );
  }
  double operator()() const {
    return sample_from( _u );
  }
};

//============================================================================

BOOST_FIXTURE_TEST_CASE( scaled_rejection_sampler_test, 
			 fixture_common_random_gamma_1d_bounded_likelihood )
{

  boost::function1<double,double> lik = bounded_1d_lik_1;
  boost::function0<double> uni_f = uniform_domain_sampler( low, high );

  // Make sure the scaled rejection sampler works like rejection sampler
  std::vector<double> scaled_samples;
  std::vector<double> reject_samples;
  size_t num_samples = 10000;
  for( size_t n = 0; n < num_samples; ++n ) {

    rejection_sampler_status_t status, scaled_status;
    double r_sample = rejection_sample<double>( lik, uni_f, status );
    double s_sample = scaled_rejection_sample<double>( lik, 1.0, uni_f, scaled_status );

    reject_samples.push_back( r_sample );
    scaled_samples.push_back( s_sample );
  }

  // make sure means are close, and variances
  BOOST_CHECK_CLOSE( mean( reject_samples ), mean( scaled_samples ), 1.0 );
  BOOST_CHECK_CLOSE( variance( reject_samples ), variance( scaled_samples ), 1.0 );
  
}

//============================================================================

BOOST_FIXTURE_TEST_CASE( autoscaled_rejection_sampler_test_100, 
			 fixture_common_random_gamma_1d_bounded_likelihood )
{

  boost::function1<double,double> lik = bounded_1d_lik_100;
  boost::function0<double> uni_f = uniform_domain_sampler( low, high );

  // Make sure the scaled rejection sampler works like rejection sampler
  std::vector<double> scaled_samples;
  std::vector<double> autoscaled_samples;
  std::vector<double> scaled_iters;
  std::vector<double> autoscaled_iters;
  double scaled_total_seconds;
  double autoscaled_total_seconds;
  size_t num_samples = 10000;
  for( size_t n = 0; n < num_samples; ++n ) {

    rejection_sampler_status_t scaled_status;
    autoscaled_rejection_sampler_status_t autoscaled_status;
    double s_sample = scaled_rejection_sample<double>( lik, 110.0, uni_f, scaled_status );
    double as_sample = autoscale_rejection_sample<double>( lik, low, high, autoscaled_status );

    scaled_samples.push_back( s_sample );
    autoscaled_samples.push_back( as_sample );
    scaled_iters.push_back( scaled_status.iterations );
    autoscaled_iters.push_back( autoscaled_status.iterations );
    scaled_total_seconds += scaled_status.seconds;
    autoscaled_total_seconds += autoscaled_status.seconds;

    BOOST_CHECK_LE( autoscaled_status.scale, 100.0 );
    BOOST_CHECK_CLOSE( autoscaled_status.scale, 100.0, 1.0 );
  }

  // make sure means are close, and variances
  BOOST_CHECK_CLOSE( mean( scaled_samples ), mean( autoscaled_samples ), 1.0 );
  BOOST_CHECK_CLOSE( variance( scaled_samples ), variance( autoscaled_samples ), 1.0 );
  BOOST_CHECK_LT( mean( autoscaled_iters ), mean( scaled_iters ) );
  BOOST_CHECK_LT( autoscaled_total_seconds, scaled_total_seconds );
  
}

//============================================================================
//============================================================================

BOOST_AUTO_TEST_SUITE_END()

//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
//============================================================================
