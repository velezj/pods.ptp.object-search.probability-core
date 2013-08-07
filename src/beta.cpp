
#include "beta.hpp"
#include "core.hpp"

namespace probability_core {

  // Description:
  // Beta Distribution
  
  double
  pdf( const double& x,
       const beta_distribution_t& beta ) 
  {
    
    return gsl_ran_beta_pdf( x, beta.alpha, beta.beta );
  }

  double
  sample_from( const beta_distribution_t& beta )
  {
    return gsl_ran_beta( global_rng(), beta.alpha, beta.beta );
  }

  double
  mean( const beta_distribution_t& beta )
  {
    return beta.alpha / ( beta.alpha + beta.beta );
  }

  double
  variance( const beta_distribution_t& beta )
  {
    double s = beta.alpha + beta.beta;
    return beta.alpha * beta.beta / ( s * s * (s+1) );
  }

}
