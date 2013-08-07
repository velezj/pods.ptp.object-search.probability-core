
#if !defined( __PROBABILITY_CORE_CORE_HPP__ )
#define __PROBABILITY_CORE_CORE_HPP__


#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


namespace probability_core {
  
  gsl_rng* global_rng();
    
}

#endif

