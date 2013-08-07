
#include "poisson.hpp"
#include "core.hpp"

namespace probability_core {

  // Description:
  // Poisson Distribution
  
  double
  pdf( const unsigned int& x,
       const poisson_distribution_t& pos ) 
  {
    return gsl_ran_poisson_pdf( x, pos.lambda );
  }

  double
  pdf( const int& x,
       const poisson_distribution_t& pos ) 
  {
    if( x < 0 )
      return 0.0;
    return gsl_ran_poisson_pdf( (unsigned int)x, pos.lambda );
  }

  
  unsigned int
  sample_from( const poisson_distribution_t& pos )
  {
    return gsl_ran_poisson( global_rng(), pos.lambda );
  }

  double
  mean( const poisson_distribution_t& pos )
  {
    return pos.lambda;
  }

  double 
  variance( const poisson_distribution_t& pos )
  {
    return pos.lambda;
  }

}
