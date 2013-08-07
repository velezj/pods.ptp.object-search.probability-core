
#include "discrete.hpp"
#include "core.hpp"
#include <gsl/gsl_statistics_double.h>


namespace probability_core {

  // Description:
  // Discrete Distribution
  
  double
  pdf( const int32_t& x,
       const discrete_distribution_t& d ) 
  {
    return d.prob[ x ];
  }

  size_t
  sample_from( const discrete_distribution_t& d )
  {
    gsl_ran_discrete_t* gd = gsl_ran_discrete_preproc( d.n, &d.prob[0] );
    double sample = gsl_ran_discrete( global_rng(), gd );
    gsl_ran_discrete_free( gd );
    return sample;
  }

  double 
  mean( const discrete_distribution_t& d )
  {
    return gsl_stats_mean( &d.prob[0], 1, d.n );
  }

  double 
  variance( const discrete_distribution_t& d )
  {
    return gsl_stats_variance( &d.prob[0], 1, d.n );
  }

}
