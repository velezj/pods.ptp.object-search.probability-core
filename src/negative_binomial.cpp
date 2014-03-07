
#include "negative_binomial.hpp"
#include "core.hpp"
#include <math-core/roots.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <math-core/mpt.hpp>
#include <iostream>


namespace probability_core {
  
  // Description:
  // Negative Binomial Distribution
  
  double
  pdf( const unsigned int& x,
       const negative_binomial_distribution_t& nb ) 
  {
    
    return gsl_ran_negative_binomial_pdf( x, 1.0 - nb.p, nb.r );
  }

  unsigned int
  sample_from( const negative_binomial_distribution_t& nb )
  {
    return gsl_ran_negative_binomial( global_rng(), 1.0 - nb.p, nb.r );
  }

  double
  mean( const negative_binomial_distribution_t& nb )
  {
    return nb.p * nb.r / ( 1 - nb.p );
  }

  double
  variance( const negative_binomial_distribution_t& nb )
  {
    return nb.p * nb.r / ( (1- nb.p) * (1 - nb.p) );
  }


  double dlik_r( const double& r, 
		 const std::vector<size_t>& k)
  {
    size_t N = k.size();
    double sum_dg = 0;
    double avg_k = 0;
    for( size_t i = 0; i < k.size(); ++i ) {
      double kr = k[i] + r;
      sum_dg += boost::math::digamma( kr );
      avg_k += k[i];
    }
    avg_k /= N;
    double ndgr = N * boost::math::digamma( r );
    double nlnr = N * log( r / ( r + avg_k ) );
    double res = ( sum_dg + ndgr + nlnr ); 
    return res;
  }


  bool
  mle( const std::vector<size_t>& k,
       negative_binomial_distribution_t& nb)
  {
    size_t N = k.size();
    double sum_k = 0;
    for( size_t i = 0; i < k.size(); ++i ) {
      sum_k += k[i];
    }

    // Ok, numerically find the root for mle_r
    std::function<double(const double&)> f 
      = std::bind(dlik_r, std::placeholders::_1, k );
    size_t max_iters = 1000;
    double l = 1;
    double u = sum_k * N * 100;
    long mle_r;
    bool found_r = 
      math_core::find_integer_root_with_guess( f, 
					       max_iters, 
					       l, 
					       u, 
					       mle_r );

    // ok, if we could not find integer root, just find some
    // interval
    if( !found_r ) {
      bool found_interval = 
	math_core::find_root_with_guess( f,
					 10,
					 max_iters,
					 l,
					 u);
      if( !found_interval ) {
	// this is pretty bad! ... hmmm ...
	// let us throw an exception
	throw std::logic_error("could not find integer root, *or* interval, for nle estimate of negative binomial");
	return false;
      }
      
      // take the middle of the interval as the root
      mle_r = (long)( l + (u - l) / 2 );
    }

    // p is easy, once we know r :-)
    double mle_p = sum_k / ( N * mle_r + sum_k );

    // debug
    for( size_t test_r = 1; test_r < 30; ++test_r) {
      std::cout << "  dlik_r(" << test_r << ") = " << f(test_r) << std::endl;
    }

    // return a new negative_binomial_distribution_t
    nb.r = mle_r;
    nb.p = mle_p;
    return true;
  }

}

