#if !defined( __PROBABILITY_CORE_POISSON_HPP__ )
#define __PROBABILITY_CORE_POISSON_HPP__

#include "lcmtypes/p2l_probability_core.hpp"

namespace probability_core {

  // Description:
  // Poisson Distribution
  
  double
  pdf( const unsigned int& x,
       const poisson_distribution_t& pos );

  double
  pdf( const int& x,
       const poisson_distribution_t& pos );

  unsigned int
  sample_from( const poisson_distribution_t& pos );

  double
  mean( const poisson_distribution_t& d );
  
  double 
  variance( const poisson_distribution_t& d );

}


#endif

