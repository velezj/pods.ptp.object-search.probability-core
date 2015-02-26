#if !defined( __PROBABILITY_CORE_BETA_HPP__ )
#define __PROBABILITY_CORE_BETA_HPP__


#include "types.hpp"


namespace probability_core {

  // Description:
  // Beta Distribution
  
  double
  pdf( const double& x,
       const beta_distribution_t& beta );

  double
  sample_from( const beta_distribution_t& beta );

  double
  mean( const beta_distribution_t& d );
  
  double 
  variance( const beta_distribution_t& d );
}


#endif
