#if !defined( __PROBABILITY_CORE_DISCRETE_HPP__ )
#define __PROBABILITY_CORE_DISCRETE_HPP__

#include "lcmtypes/p2l_probability_core.hpp"

namespace probability_core {

  // Description:
  // Discrete Distribution
  
  double
  pdf( const int32_t& x,
       const discrete_distribution_t& d );

  size_t
  sample_from( const discrete_distribution_t& d );


  double
  mean( const discrete_distribution_t& d );
  
  double 
  variance( const discrete_distribution_t& d );
}


#endif

