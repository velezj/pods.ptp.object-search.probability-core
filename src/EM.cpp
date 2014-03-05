
#include "EM.hpp"
#include <gsl/gsl_multimin.h>
#include "uniform.hpp"
#include <math-core/gsl_utils.hpp>
#include <iostream>

namespace probability_core {

  //====================================================================

  // struct lik_data_param_pack_t
  // {
  //   std::vector<math_core::nd_point_t> data;
  //   std::function<double(const std::vector<math_core::nd_point_t>& data,
  // 			 const std::vector<double>& params)>& lik;
  //   std::vector<std::vector<double> > param_structure;
  // };

  //====================================================================

  std::vector<double> flatten( const std::vector<std::vector<double> >& p )
  {
    std::vector<double> flat;
    size_t idx = 0;
    for( size_t i = 0; i < p.size(); ++i ) {
      for( size_t j = 0; j < p[i].size(); ++j ) {
	flat.push_back( p[i][j] );
	++idx;
      }
    }
    return flat;
  }

  //====================================================================

  std::vector<std::vector<double> >
  reconstitute_parameters( const std::vector<std::vector<double > >& structure,
			   const double* values )
  {
    std::vector<std::vector<double> > params;
    size_t idx = 0;
    for( size_t i = 0; i < structure.size(); ++i ) {
      params.push_back( std::vector<double>() );
      for( size_t j = 0; j < structure[i].size(); ++j ) {
	params[i].push_back( values[idx] );
	++idx;
      }
    }
    return params;
  }

  //====================================================================

  // // the *negative* of the likelihood since we are minimizing!
  // double _run_GEM_mixture_model_MLE_numerical_likfunc
  // ( const gsl_vector* x,
  //   void* params )
  // {
  //   lik_data_param_pack_t *pack = (lik_data_param_pack_t*)params;
  //   std::vector<std::vector<double> > lik_params 
  //     = reconstitute_parameters( *pack, x );
  //   return -pack->lik( pack->data, lik_params );
  // }

  //====================================================================

  // void _maximize_mixture_model_likelihood_numerical
  // ( const GEM_stopping_criteria_t& stop,
  //   const std::vector<math_core::nd_point_t>& data,
  //   const std::vector<std::vector<double> >& initial_parameters,
  //   std::function<double(const std::vector<math_core::nd_point_t>& data,
  // 			 const std::vector<double>& params)>& model_likelihood,
  //   std::vector<std::vector<double> >& mle_estimate )
  // {

  //   const gsl_multimin_fminimizer_type *gsl_T
  //     = gsl_multimin_fminimizer_nmsimplex2;

  //   gsl_multimin_fminimizer *gsl_minimizer = NULL;
  //   gsl_multimin_function gsl_func;

  //   size_t iteration = 0;
  //   int status;
  //   double gsl_minimizer_size;

  //   // calculate the total number of parameters
  //   size_t param_flat_size = 0;
  //   for( size_t i = 0; i < initial_parameters.size(); ++i ) {
  //     param_flat_size += initial_parameters[i].size();
  //   }

  //   // cover starting parameters to a single start point
  //   gsl_vector *gsl_params = gsl_vector_alloc( param_flat_size );
  //   size_t idx = 0;
  //   for( size_t i = 0; i < initial_parameters.size(); ++i ) {
  //     for( size_t j = 0; j < initial_parameters[i].size(); ++j ) {
  // 	gsl_vector_set( gsl_params, idx, initial_parameters[i][j] );
  // 	++idx;
  //     }
  //   }

  //   // setup initial step sizes
  //   gsl_vector *step_sizes = gsl_vector_alloc( param_flat_size );
  //   gsl_vector_set_all( step_sizes, 1.0 );

  //   // setup gsl function from likelihood
  //   lik_data_param_pack_t lik_data_param_pack;
  //   lik_data_param_pack.data = data;
  //   lik_data_param_pack.lik = model_likelihood;
  //   lik_data_param_pack.param_structure = initial_parameters;
  //   gsl_func.n = param_flat_size;
  //   gsl_func.f = _run_GEM_mixture_model_MLE_numerical_likfunc;
  //   gsl_func.params = &lik_data_param_pack;

  //   // setup minimizer
  //   gsl_minimizer = gsl_multimin_fminimizer_alloc( gsl_T, param_flat_size );
  //   gsl_multimin_fminimizer_set( gsl_minimizer, &gsl_func, gsl_params, step_sizes );

  //   // iterate minimization
  //   do {

  //     iteration++;
  //     status = gsl_multimin_fminimizer_iterate( gsl_minimizer );
      
  //     if(status)
  // 	break;
      
  //   } while( status == GSL_CONTINUE && iteration < (*stop.max_iterations) );

  //   // ok, place found "min" into parameter ouput
  //   mle_estimate = reconstitute_parameters( lik_data_param_pack, gsl_params->data );

  //   // release resources
  //   gsl_vector_free(gsl_params);
  //   gsl_vector_free(step_sizes);
  //   gsl_multimin_fminimizer_free( gsl_minimizer );
  // }

  //====================================================================

  // Description:
  // Compute Q(.) for mixture models
  class _GEM_mixture_Q_t
  {
  public:

    std::vector<math_core::nd_point_t> data;
    std::vector<double> mixture_parameters;
    std::vector< std::vector<double> > model_parameters;
    std::function<double(const math_core::nd_point_t& data,
			 const std::vector<double>& params)> lik;
    
    _GEM_mixture_Q_t
    ( const std::vector<math_core::nd_point_t>& data,
      const std::vector<double>& mixture_parameters,
      const std::vector<std::vector< double > >& model_parameters,
      std::function<double(const math_core::nd_point_t& single_data,
			   const std::vector<double>& params)>& lik )
      : data(data),
        mixture_parameters(mixture_parameters),
	model_parameters(model_parameters),
	lik(lik)
    {}

    double operator() (const std::vector<double>& flat_params ) const
    {
      // reconstruct the parameter structure
      std::vector<std::vector<double > > params =
	reconstitute_parameters( model_parameters, flat_params.data() );

      // ok, now lets compute the expectation of the data
      // given the model parameter inputs as well as the
      // mixture weights used to construct this Q(.)
      double log_lik = 0;
      for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	  double w = 0;
	  w = mixture_parameters[mix_i] * lik( data[data_i],
					       params[mix_i] );
	  double norm = 0;
	  for( size_t j = 0; j < mixture_parameters.size(); ++j ) {
	    norm += mixture_parameters[j] * lik( data[data_i],
						 params[j] );
	  }
	  w = w / norm;

	  double single_lik =
	    w * lik( data[data_i],
		     params[mix_i] );
	  log_lik += single_lik;
	}
      }

      // std::cout << "  Q-GEM(";
      // for( size_t i = 0; i < flat_params.size(); ++i ) {
      // 	std::cout << flat_params[i] << ",";
      // }
      // std::cout << ") ll=" << log_lik << std::endl;

      // retunr log lik
      return log_lik;
    }

  protected:

  };

  //====================================================================

  double _x_negative_GEM_mixture_Q_t
  ( const gsl_vector* flat_params,
    void* qptr )
  {
    _GEM_mixture_Q_t* q = static_cast<_GEM_mixture_Q_t*>(qptr);
    return -q->operator()( math_core::to_vector(flat_params) );
  }

  //====================================================================

  std::vector<double>
  calculate_initial_mixture_weights
  ( const std::vector<math_core::nd_point_t>& data,
    const std::vector<std::vector<double> >& initial_parameters,
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>& model_likelihood)
  {
    // just randomly choose them for now
    std::vector<double> mix;
    double sum = 0;
    for( size_t i = 0; i < initial_parameters.size(); ++i ) {
      mix.push_back( sample_from( uniform_distribution(0.0, 1.0)));
      sum += mix[ mix.size() - 1 ];
    }
    for( size_t i = 0; i < mix.size(); ++i ) {
      mix[i] /= sum;
    }
    return mix;
  }

  //====================================================================

  std::vector<double>
  find_max_Q_numeric( const _GEM_mixture_Q_t& q,
		      const std::vector<double> flat_params,
		      const size_t max_iterations )
  {
    const gsl_multimin_fminimizer_type *gsl_T
      = gsl_multimin_fminimizer_nmsimplex2;

    gsl_multimin_fminimizer *gsl_minimizer = NULL;
    gsl_multimin_function gsl_func;

    size_t iteration = 0;
    int status;
    double gsl_minimizer_size;

    // calculate the total number of parameters
    size_t param_flat_size = flat_params.size();

    // cover starting parameters to a single start point
    gsl_vector *gsl_params = math_core::new_gsl_vector( flat_params );

    // setup initial step sizes
    gsl_vector *step_sizes = gsl_vector_alloc( param_flat_size );
    gsl_vector_set_all( step_sizes, 1.0 );

    // setup function to minimize
    gsl_func.n = param_flat_size;
    gsl_func.f = _x_negative_GEM_mixture_Q_t;
    gsl_func.params = const_cast<void*>(static_cast<const void*>(&q));

    // setup minimizer
    gsl_minimizer = gsl_multimin_fminimizer_alloc( gsl_T, param_flat_size );
    gsl_multimin_fminimizer_set( gsl_minimizer, 
				 &gsl_func, 
				 gsl_params, 
				 step_sizes );

    // iterate minimization
    do {

      iteration++;
      status = gsl_multimin_fminimizer_iterate( gsl_minimizer );
      
      if(status)
	break;

      // std::cout << "    .. status: " << status << std::endl;
      double size = gsl_multimin_fminimizer_size (gsl_minimizer);
      status = gsl_multimin_test_size (size, 1e-2);
      // std::cout << "    .. size: " << size 
      // 		<< ", f(): " << gsl_minimizer->fval 
      // 		<< ", stop status: " << status << std::endl;
      
    } while( status == GSL_CONTINUE && iteration < max_iterations );

    // std::cout << "  Q-max: statux=" << status << std::endl;

    // ok, place found "min" into parameter ouput
    std::vector<double> res
      = math_core::to_vector( gsl_minimizer->x );

    // std::cout << "  Q-max: ";
    // for( size_t i = 0; i < res.size(); ++i ) {
    //   std::cout << res[i] << " ";
    // }
    // std::cout << std::endl;

    // release resources
    gsl_vector_free(gsl_params);
    gsl_vector_free(step_sizes);
    gsl_multimin_fminimizer_free( gsl_minimizer );    

    return res;
  }

  //====================================================================

  void run_GEM_mixture_model_MLE_numerical
  ( const GEM_stopping_criteria_t& stop,
    const std::vector<math_core::nd_point_t>& data,
    const std::vector<std::vector<double> >& initial_parameters,
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>& model_likelihood,
    std::vector<std::vector<double> >& mle_estimate )
  {

    // ok, we need to store an keep up the hidden set of 
    // "weights" for each mixture
    std::vector<double> mixture_weights 
      = calculate_initial_mixture_weights( data,
					   initial_parameters,
					   model_likelihood );

    // keep track of current and old parameters
    std::vector<std::vector<double> > current_mixture_parameters 
      = initial_parameters;
    
    // loop doing E then M steps
    size_t iteration = 0;
    while( iteration < *stop.max_iterations ) {
    
      // Expectation step (E-step) which really is the conditional
      // expectation given the prameters for the mixture weights (hidden)
      // Here , we jsut create an instance of our Q(.) function
      _GEM_mixture_Q_t q( data,
			  mixture_weights,
			  current_mixture_parameters,
			  model_likelihood );
      
      // Ok, now we want to maximize the Q.
      // Since we are a GEM, we need just find a "better" parameter set,
      // so we numerically optimize Q
      size_t max_optimize_iterations = 100;
      std::vector<double> next_parameters
	= find_max_Q_numeric( q, 
			      flatten( current_mixture_parameters ), 
			      max_optimize_iterations );
      current_mixture_parameters = 
	reconstitute_parameters( current_mixture_parameters,
				 next_parameters.data() );

      // std::cout << "  M-step: p: ";
      // for( size_t i = 0; i < next_parameters.size(); ++i ) {
      // 	std::cout << next_parameters[i] << ",";
      // }
      // std::cout << std::endl;
      // std::cout << "  M-step: ";
      // for( size_t i = 0; i < current_mixture_parameters.size(); ++i ) {
      // 	for( size_t j = 0; j < current_mixture_parameters[i].size(); ++j ) {
      // 	  std::cout << current_mixture_parameters[i][j] << ",";
      // 	}
      // 	std::cout << " || ";
      // }
      // std::cout << std::endl;
	
      // increas iteration
      ++iteration;
      
    }

    // ok, set the current parameters to the output ones
    mle_estimate = current_mixture_parameters;
  }


  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================


}
