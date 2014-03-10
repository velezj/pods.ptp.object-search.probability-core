
#if !defined( __P2L_PROBABILITY_CORE_EM_HPP__ )
#define __P2L_PROBABILITY_CORE_EM_HPP__

#include <functional>
#include <math-core/geom.hpp>
#include <math-core/matrix.hpp>
#include <math-core/io.hpp>
#include <boost/optional.hpp>
#include <math-core/exception.hpp>

namespace probability_core {

  using namespace math_core;

  //====================================================================

  // Description:
  // An EM stopping criteria.
  // For now, we usually use the max_iterations.
  struct GEM_stopping_criteria_t
  {
    boost::optional<size_t> max_iterations;
    boost::optional<double> relative_likelihood_tolerance;

    GEM_stopping_criteria_t() : max_iterations(100) {}
  };

  //====================================================================

  // Description:
  // Parameters for the GEM algorothm, including stopping criteria
  struct GEM_parameters_t
  {
    GEM_stopping_criteria_t stop;
    size_t max_optimize_iterations;

    GEM_parameters_t() : stop(), max_optimize_iterations(100) {}
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
  ( const GEM_parameters_t& gem_parameters,
    const std::vector<math_core::nd_point_t>& data,
    const std::vector<std::vector<double> >& initial_parameters,
    const std::vector<std::vector<double> >& param_lower_bounds,
    const std::vector<std::vector<double> >& param_upper_bounds,
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>& model_likelihood,
    std::vector<std::vector<double> >& mle_estimate,
    std::vector<double>& mle_mixture_weights );

  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================

  // Description:
  // Exceptions and error info objects for EM
  struct GEM_exception_t : virtual math_core::base_exception_t {};

  
  typedef boost::error_info<struct tag_reason,std::string> 
  errorinfo_reason;
  
  typedef boost::error_info<struct tag_last_gem_model_parameters,
			    std::vector<double> > 
  errorinfo_last_gem_model_parameters;

  typedef boost::error_info<struct tag_last_gem_mixture_weights,
			    std::vector<double> > 
  errorinfo_last_gem_mixture_weights;

  typedef boost::error_info<struct tag_last_gem_likelihood,
			    double > 
  errorinfo_last_gem_likelihood;

  typedef boost::error_info<struct tag_last_maxmix_mixture_weights,
			    std::vector<double> > 
  errorinfo_last_maxmix_mixture_weights;
  
  typedef boost::error_info<struct tag_last_maxmix_unorm_T,
			    Eigen::MatrixXd > 
  errorinfo_last_maxmix_unorm_T;

  typedef boost::error_info<struct tag_last_Q_mixture_weights,
			    std::vector<double> > 
  errorinfo_last_Q_mixture_weights;

  typedef boost::error_info<struct tag_last_Q_model_parameters,
			    std::vector<double> > 
  errorinfo_last_Q_model_parameters;

  typedef boost::error_info<struct tag_last_Q_parameters_x,
			    std::vector<double> > 
  errorinfo_last_Q_parameters_x;
  
  typedef boost::error_info<struct tag_last_Q_likelihood,
			    double > 
  errorinfo_last_Q_log_likelihood;
  
  //====================================================================



}

#endif

