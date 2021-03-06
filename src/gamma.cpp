
#include "gamma.hpp"
#include "core.hpp"
#include <limits>

//=========================================================================

namespace probability_core {

  // Description:
  // Gamma Distribution
  
  double 
  pdf( const double& x,
       const gamma_distribution_t& gamma ) 
  {
    double scale = 1.0 / gamma.rate;
    return gsl_ran_gamma_pdf( x, gamma.shape, scale );
  }
  
  double
  sample_from( const gamma_distribution_t& gamma )
  {
    double scale = 1.0 / gamma.rate;
    return gsl_ran_gamma( global_rng(), gamma.shape, scale );
  }

  double
  mean( const gamma_distribution_t& gamma ) 
  {
    return gamma.shape / gamma.rate;
  }

  double
  variance( const gamma_distribution_t& gamma )
  {
    return gamma.shape / (gamma.rate * gamma.rate);
  }

  double
  mode( const gamma_distribution_t& gamma )
  {
    if( gamma.shape > 1 ) {
      return ( gamma.shape - 1 ) / gamma.rate;
    } else {
      return std::numeric_limits<double>::epsilon();
    }
  }
}


