
#if !defined( __P2L_PROBABILITY_CORE_autocorrelation_HPP__ )
#define __P2L_PROBABILITY_CORE_autocorrelation_HPP__

#include <vector>
#include <math-core/utils.hpp>

namespace probability_core {

  //=======================================================================

  // Description:
  // Returns the autocorrelation for a given vector ( a signal )
  // given the *true known* mean and variance of the signal
  template<typename T,
	   typename T_Dist = double >
  std::vector<T_Dist>
  unbiased_autocorrelation( const std::vector<T>& signal,
			    const T& true_mean,
			    const T_Dist& true_variance )
  {
    size_t n = signal.size();
    assert( n > 1 );
    std::vector<T_Dist> ac;
    for( size_t k = 0; k < n-1; ++k ) {
      double sum = 0;
      for( size_t t = 0; t < ( n - k - 1 ); ++t ) {
	sum += ( signal[t] - true_mean ) * ( signal[t+k+1] - true_mean );
      }
      double coeff = 1.0 / ( (n-k-1) * true_variance ) * sum;
      ac.push_back( coeff );
    }

    return ac;
  }

  //=======================================================================

  // Description:
  // Returns the autocorrelation for a given vector ( a signal )
  // This will be a biased estimate since we will use the sample 
  // mean and variacne as the true mean and variance
  template<typename T,
	   typename T_Dist = double >
  std::vector<T_Dist>
  biased_autocorrelation_sample_mean_var( const std::vector<T>& signal )
  {
    using namespace math_core;
    return unbiased_autocorrelation( signal, 
				     math_core::mean<T>(signal), 
				     math_core::variance<T>(signal));
  }
  

  //=======================================================================

}


#endif

