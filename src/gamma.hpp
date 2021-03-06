#if !defined( __PROBABILITY_CORE_GAMMA_HPP__ )
#define __PROBABILITY_CORE_GAMMA_HPP__

#include "types.hpp"

namespace probability_core {

  // Description:
  // Gamma Distribution
  
  double 
  pdf( const double& x,
       const gamma_distribution_t& gamma ); 
  
  double
  sample_from( const gamma_distribution_t& gamma );
  

  double
  mean( const gamma_distribution_t& d );
  
  double 
  variance( const gamma_distribution_t& d );

  double
  mode( const gamma_distribution_t& d );

}

#endif

