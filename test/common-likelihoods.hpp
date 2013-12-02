
#if !defined( __P2L_PROBABILITY_CORE_TEST_comon_likelihoods_HPP__ )
#define __P2L_PROBABILITY_CORE_TEST_comon_likelihoods_HPP__

#include <probability-core/beta.hpp>
#include <probability-core/uniform.hpp>
#include <probability-core/gamma.hpp>
#include <math-core/geom.hpp>
#include <math-core/utils.hpp>
#include <vector>

using namespace probability_core;
using namespace math_core;

//========================================================================

struct simple_lik
{
  
  double low_support;
  double high_supprt;
  double mode;
  simple_lik()
  {
    low_support = -10;
    high_supprt = 10;
    mode = 0.0;
  }
  double operator()( const double& x ) 
  {
    return -fabs(x);
  }
};

//============================================================================

struct bumpy_likelihood_known_mode_t
{
  size_t bumps;
  double mode;
  double mode_likelihood;
  double low_support;
  double high_supprt;
  std::vector<gamma_distribution_t> gammas;
  double scaling;

  bumpy_likelihood_known_mode_t()
  {
  }

  bumpy_likelihood_known_mode_t( size_t bumps,
				 double mode_lik )
  {
    assert( bumps > 0 );
    this->bumps = bumps;
    this->mode_likelihood = mode_lik;
    low_support = 1.0;
    high_supprt = 40.0;
    double low = low_support;
    double high = high_supprt;

    // place non-mode bumps
    for( size_t i = 0; i < bumps - 1; ++i ) {
      double loc = ( ( high - low ) / 2.0 ) * ( ( (double)(2*i) ) / (2*bumps) ) + low_support;
      gamma_distribution_t g;
      g.rate = ( bumps + 2.0 );
      //g.rate = 2.0 * ( high - low ) / bumps;
      g.shape = loc * g.rate + 1;
      gammas.push_back( g );
    }
    
    // create the mode gamma
    gamma_distribution_t g;
    g.rate = (bumps + 2.0);
    g.shape = 11.0 * ( high - low ) / 17.0 * g.rate + 1.0;
    gammas.push_back( g );

    
    mode = ( gammas[0].shape - 1.0 ) / gammas[0].rate + low_support;

    // get teh scaling for hte mode
    double sum = 0;
    for( auto g : gammas ) {
      double x_norm = ( mode- low_support ) + std::numeric_limits<double>::epsilon();
      sum += pdf( x_norm, g );
    }
    scaling = mode_likelihood / sum;

    // print out the modes
    // std::cout << "bumpy_lik()" << std::endl;
    // for( auto g : gammas ) {
    //   std::cout << "  mode = " << ( g.shape - 1.0 ) / g.rate 
    // 		<< " var = " << ( g.shape ) / ( g.rate * g.rate ) 
    // 		<< std::endl;
    // }
  }
  
  double operator() ( const double& x )
  {
    double x_norm = ( x - low_support ) + std::numeric_limits<double>::epsilon();
    double sum = 0;
    for( auto g : gammas ) {
      sum += pdf( x_norm, g );
    }
    sum *= scaling;
    return sum;
  }
};


//========================================================================

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
      //double a = gsl_ran_beta_pdf( x_norm, b.alpha, b.beta );
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


//============================================================================


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

//============================================================================


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



#endif

