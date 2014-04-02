
#if !defined( __P2L_PROBABILITY_CORE_posterior_HPP__ )
#define __P2L_PROBABILITY_CORE_posterior_HPP__

#include "distributions.hpp"

namespace probability_core {

  //==========================================================
  
  template<class PostT, class PriorT, class LikT, class DataT>
  PostT posterior( const DataT& x,
		   const LikT& lik, 
		   const PriorT& prior );
  
  //==========================================================

  template<>
  gaussian_distribution_t
  posterior( const std::vector<math_core::nd_point_t>& x,
	     const gaussian_distribution_t& lik,
	     const gaussian_distribution_t& prior );

  //==========================================================

  template<>
  gamma_distribution_t
  posterior( const std::vector<unsigned long>& x,
	     const poisson_distribution_t& lik,
	     const gamma_distribution_t& prior );


  //==========================================================

  template<>
  gamma_conjugate_prior_t
  posterior( const std::vector<double>& x,
	     const gamma_distribution_t& lik,
	     const gamma_conjugate_prior_t& prior );

  //==========================================================
  //==========================================================
  //==========================================================
  //==========================================================
  //==========================================================
  //==========================================================
  //==========================================================
  //==========================================================
  //==========================================================
  //==========================================================
  //==========================================================
  //==========================================================

}

#endif

