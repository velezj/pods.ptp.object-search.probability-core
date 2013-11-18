#if !defined( __PROBABILITY_CORE_GAMMA_CONJUGATE_PRIOR_HPP__ )
#define __PROBABILITY_CORE_GAMMA_CONJUGATE_PRIOR_HPP__

#include "lcmtypes/p2l_probability_core.hpp"
#include <math-core/mpt.hpp>

namespace probability_core {

  // Description:
  // Gamma Conjugate Prior Distribution
  
  //double 
  //pdf( const gamma_distribution_t& gamma,
  //     const gamma_conjugate_prior_t& gcp ); 

  double 
  likelihood( const gamma_distribution_t& gamma,
	      const gamma_conjugate_prior_t& gcp ); 
  
  
  gamma_distribution_t
  sample_from( const gamma_conjugate_prior_t& gcp );

  gamma_distribution_t
  slice_sample_from( const gamma_conjugate_prior_t& gcp );

  //double
  //mean( const gamma_conjugate_prior_t& gcp );
  
  //double 
  //variance( const gamma_conjugate_prior_t& gcp );

  // Description:
  // Estimate statistics (mean and variance) for a Gcp
  // of teh final result of the sampled gamma
  void estimate_gamma_conjugate_prior_sample_stats
  ( const gamma_conjugate_prior_t& gcp, 
    double& mean,
    double& var,
    const int num_samples = 100 );



  // Description:
  // "Fix" some numerical issues when we get things being too big.
  // This will reduce the counts (r and s) to be <= 10 
  // if they are above 20
  gamma_conjugate_prior_t
  fix_numerical_reduce_counts( const gamma_conjugate_prior_t& gcp );



  // Description:
  // Some "true" alsogirmths for testing
  namespace testing_true {

    // the likelihood
    math_core::mpt::mp_float 
    likelihood( const gamma_distribution_t& gamma, 
		const gamma_conjugate_prior_t& gcp );
    
    
  }

}


#endif

