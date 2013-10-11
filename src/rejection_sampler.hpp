
#if !defined( __PROBABILITY_CORE_REJECTION_SAMPLER_HPP__ )
#define __PROBABILITY_CORE_REJECTION_SAMPLER_HPP__


#include <boost/function.hpp>
#include "core.hpp"
#include "distribution_utils.hpp"
#include <math-core/extrema.hpp>
#include <math-core/gsl_utils.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/optional.hpp>
#include <time.h>
#include <iostream>
#include <cmath>
#include <math-core/io.hpp>
#include <stdexcept>

namespace probability_core {


  // Description:
  // A uniform sampler for a range of double precision numbers
  class uniform_sampler_within_range
  {
  public:
    double min, max;
    
    uniform_sampler_within_range( const double& min, const double& max)
      : min(min), max(max)
    {}

    inline double operator()() const
    {
      return min + (max-min) * gsl_rng_uniform( global_rng() );
    }
  };

  
  // Description:
  // A uniform integer sampler within range
  // from [min, max) (so max is NOT included!)
  class uniform_unsigned_long_sampler_within_range
  {
  public:
    unsigned long min, max;
    
    uniform_unsigned_long_sampler_within_range( const unsigned long& min,
						const unsigned long& max )
      : min(min),
	max(max)
    {}

    inline unsigned long operator()() const
    {
      return min + gsl_rng_uniform_int( global_rng(), max-min );
    }
  };


  // Description:
  // A uniform sampler for nd_point_t structures where each coordinate
  // is bound within some window
  class uniform_point_sampler_within_window
  {
  public:
    math_core::nd_aabox_t window;
    
    uniform_point_sampler_within_window( const math_core::nd_aabox_t& window )
      : window(window)
    {
    }

    inline math_core::nd_point_t operator()() const
    {
      // generate each coordinate eseparately
      math_core::nd_point_t p = math_core::zero_point( window.start.n );
      for( std::size_t i = 0; (long)i < p.n; ++i ) {
	p.coordinate[i] = window.start.coordinate[i] + (window.end.coordinate[i] - window.start.coordinate[i]) * gsl_rng_uniform( global_rng() );
      }
      return p;
    }

  };


  // Description:
  // Status and infromation about a rejection sampling run
  struct rejection_sampler_status_t
  {
    double iterations;
    double seconds;
    
    rejection_sampler_status_t()
      : iterations(0),
	seconds(0)
    {}
  };


  // Description:
  // A rejection sampler wich just takes in a distribution
  // function (this must be a TRUE PROBABILITY DISTRIBUTION! 
  // -- hence normalized!!)
  // and a uniform sampler for the domain
  template< typename T_Domain >
  T_Domain rejection_sample( const boost::function1<double, const T_Domain&>& probability_density_function,
			     const boost::function0<T_Domain>& uniform_sampler,
			     rejection_sampler_status_t& status = rejection_sampler_status_t() )
  {

    clock_t start_clock = clock();

    bool verbose = true;

    // ignore Overflow/Underflow for this
    //gsl_error_handler_scope( gsl_ignore_representation_errors );
    gsl_error_handler_t* old_gsl_error_handler = gsl_set_error_handler_off();

    std::vector<T_Domain> sampled_x;
    std::vector<double> sampled_p;
    T_Domain max_x = T_Domain();
    double max_p = double();

    int max_samples = 100000;
    size_t max_history_size = 0;
    int num_samples_to_warn = max_samples / 10;

    // loop while we have not picked a good sample
    while( true ) {

      // uniformly sample an element of domain
      T_Domain proposed_sample = uniform_sampler();
      
      // Ok, now flip a coin based on the pdf of this element
      // and return it as a sample if the coin is heads (biased)
      double p = probability_density_function( proposed_sample );

      // treat NaN evelly!!!!!
      if( std::isnan(p) ) {
	std::cout << "NaN Found: " << proposed_sample << " " << p << std::endl;
	throw std::domain_error( "Probability cannot be NaN for rejection samplign!" );
      }

      // Ok, flip baised coin with probability p to see if we do not reject
      if( flip_coin( p ) ) {
	clock_t end_clock = clock();
	status.seconds = (double)( end_clock - start_clock ) / CLOCKS_PER_SEC;
	
	// debug
	//std::cout << "    --rej: " << status.iterations << " (" << status.seconds << ")" << std::endl;

	gsl_set_error_handler( old_gsl_error_handler );

	return proposed_sample;
      }

      // increment iteration
      status.iterations += 1.0;

      // keep max
      if( sampled_x.empty() ||
	  p > max_p ) {
	max_x = proposed_sample;
	max_p = p;
      }

      // add sampled to history
      if( sampled_x.size() < max_history_size ) {
	sampled_x.push_back( proposed_sample );
	sampled_p.push_back( p );
      }
      
      if( (int)(status.iterations) % num_samples_to_warn == 0 &&
	  status.iterations >= num_samples_to_warn &&
	  verbose ) {
	std::cout << "[" << status.iterations << "] max x: " << max_x << " p=" << max_p << std::endl;
      }

      // if too many iterations, just return max sampled
      if( verbose && status.iterations > max_samples ) {
	std::cout << "**  rejection sampler returning max sampled!!!" << std::endl;
	clock_t end_clock = clock();
	status.seconds = (double)( end_clock - start_clock ) / CLOCKS_PER_SEC;

	gsl_set_error_handler( old_gsl_error_handler );

	if( sampled_x.empty() ) {
	  return uniform_sampler();
	} else {
	  return max_x;
	}
      }
    }

    // we won't reach here since eventualyl we will sample

  }


  // Description:
  // A rejection sampler which takes ina scalign factor for the
  // given likelihood ( so it does NOT have to be normalized to
  // go from zero to 1, but the scaling must not make the function exceed 1
  template< typename T_Domain >
  T_Domain
  scaled_rejection_sample
  ( const boost::function1<double, const T_Domain&>& likelihood_function,
    const double& scale,
    const boost::function0<T_Domain>& uniform_sampler,
    rejection_sampler_status_t& status = rejection_sampler_status_t() )
  {
    return rejection_sample<T_Domain>
      ( boost::function1<double,T_Domain>
	(boost::lambda::bind( likelihood_function, 
			      boost::lambda::_1 ) / scale),
	uniform_sampler,
	status);
  }


  // Description:
  // A rejection sampler which takes in just a 1D likelihood function 
  // (so it does NOT have to be normalized) and firts finds
  // the maximum values, then scales it and uses the above rejection
  // sampler.
  // Notice also we need to explicitly give the domain of the function
  // since we can not longer only use a sampler.
  // Note: the T_Domain must be castable to a double because of the
  //       way we find hte maximum
  template< typename T_Domain >
  T_Domain 
  autoscale_rejection_sample
  ( const boost::function1<double, const T_Domain&>& likelihood_function,
    const T_Domain& low, const T_Domain& high,
    rejection_sampler_status_t& status = rejection_sampler_status_t() )
  {
    
    // Ok, first we will find the max
    double max_lik_location = math_core::find_max<double,double>( likelihood_function, low, high, low + (high - low) / 2.0 );
    double max_lik = likelihood_function( max_lik_location );

    // now we create a uniform sampler if we need to
    boost::function0<double> uniform_sampler = uniform_sampler_within_range( (double)low, (double)high );

    // rescale the lieklihood and then rejection sample
    return 
      scaled_rejection_sample<double>
      ( likelihood_function, max_lik, uniform_sampler, status );
  }

}


#endif

