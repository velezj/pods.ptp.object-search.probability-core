
#if !defined( __P2L_PROBABILITY_CORE_EM_HPP__ )
#define __P2L_PROBABILITY_CORE_EM_HPP__

#include <functional>
#include <math-core/geom.hpp>
#include <boost/optional.hpp>

namespace probability_core {


  //====================================================================

  // Description:
  // An EM stopping criteria.
  // For now, we usually use the max_iterations.
  struct GEM_stopping_criteria_t
  {
    boost::optional<size_t> max_iterations;
  };

  //====================================================================

  // Description:
  // Performs General Expectation Maximization (EM) for 
  // maximum likelihood estimation (MLE).  
  // This is the most generic version which internally numerically maximizes
  // the Q(.) function. 
  // This is the "clustering" version or "mixture model" version, so
  // we assume that we have a hidden variable per data which denotes which
  // mixture the data comes from.
  void run_GEM_mixture_model_MLE_numerical
  ( const GEM_stopping_criteria_t& stop,
    const std::vector<math_core::nd_point_t>& data,
    const std::vector<std::vector<double> >& initial_parameters,
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>& model_likelihood,
    std::vector<std::vector<double> >& mle_estimate );

  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================



}

#endif

