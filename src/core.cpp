
#include "core.hpp"


namespace probability_core {



  //======================================================================
  
  gsl_rng* global_rng() {
    static gsl_rng* _rng = NULL;
    static const gsl_rng_type* _rng_T = gsl_rng_default;
    if( _rng == NULL ) {
      _rng = gsl_rng_alloc( _rng_T );
    }
    return _rng;
  }
  
  //======================================================================
  
  
}
