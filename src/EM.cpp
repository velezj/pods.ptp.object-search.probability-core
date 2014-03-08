
#include "EM.hpp"
#include <gsl/gsl_multimin.h>
#include "uniform.hpp"
#include <math-core/gsl_utils.hpp>
#include <math-core/matrix.hpp>
#include <math-core/extrema.hpp>
#include <math-core/mpt.hpp>
#include <iostream>
#include <limits>

using namespace math_core::mpt;

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


    // Descripiton:
    // This computes the Q( b | b^t ), the expectation of the
    // log function conditioned on the previous parameters b^t
    // as well as the hidden variables z (the mixture weights)
    // Note: we take in only the non-hidden parameters as input because
    //       the mixture weights are given to this upon creation
    //       *and* when we maximize this Q we can maximize the 
    //       hidden poarts *separately (and in lcosed form!) from the
    //       arbritary given parameters~!
    double operator() 
    (const std::vector<double>& flat_params ) const
    {
      // reconstruct the parameter structure
      std::vector<std::vector<double > > params =
	reconstitute_parameters( model_parameters, 
				 flat_params.data() );

      // ok, now lets compute the expectation of the data
      // given the model parameter inputs as well as the
      // mixture weights used to construct this Q(.)
      mp_float log_lik = 0;
      for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	  mp_float w = 0;
	  w = mixture_parameters[mix_i] * lik( data[data_i],
					       params[mix_i] );
	  mp_float norm = 0;
	  for( size_t j = 0; j < mixture_parameters.size(); ++j ) {
	    norm += mixture_parameters[j] * lik( data[data_i],
						 params[j] );
	  }
	  w = w / norm;

	  mp_float single_lik =
	    log(w) * log( lik( data[data_i],
			       params[mix_i] ));
	  log_lik += single_lik;
	}
      }

      // std::cout << "  Q-GEM(";
      // for( size_t i = 0; i < flat_params.size(); ++i ) {
      // 	std::cout << flat_params[i] << ",";
      // }
      // std::cout << ") ll=" << log_lik << std::endl;

      // retunr log lik
      if( exp(log_lik) < 1e-10 ) {
	log_lik = log( 1e-10 );
      }
      return log_lik.convert_to<double>();
    }

  protected:

  };

  //====================================================================

  // Description:
  // Computes the lileihood for a parituclar mixture weights+models
  // for the data
  double _likelihood
  (const std::vector<math_core::nd_point_t>& data,
   const std::vector<double>& mixture_parameters,
   const std::vector<std::vector< double > >& model_parameters,
   std::function<double(const math_core::nd_point_t& single_data,
			const std::vector<double>& params)>& lik)
  {
    mp_float p = 0;
    for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
      math_core::nd_point_t x = data[data_i];
      for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	double w = mixture_parameters[mix_i];
	std::vector<double> model = model_parameters[mix_i];
	p += w * lik(x,model);
      }
    }
    return p.convert_to<double>();
  }
   
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

  // Descripiton:
  // Finds the maximum parameter setting of all *non-hidden* variables!
  // This means we don NOT maximize over mixture weights,
  // that should be done (and can be done!) seperately from the 
  // given parameters.
  std::vector<double>
  find_max_Q_numeric( const _GEM_mixture_Q_t& q,
		      const std::vector<double> flat_params,
		      const std::vector<double>& lb,
		      const std::vector<double>& ub,
		      const size_t max_iterations )
  {
    double val;
    std::function<double(const std::vector<double>&)> f = q;
    std::vector<double> max_params
      = math_core::find_global_extrema
      ( math_core::stop.max_evaluations( max_iterations ),
	f,
	flat_params,
	lb,
	ub,
	math_core::extrema_maximize,
	val );
    return max_params;
  }

  //====================================================================

  std::vector<double>
  maximize_mixture_weights
  ( const std::vector<math_core::nd_point_t>& data,
    const std::vector<double>& mixture_parameters,
    const std::vector<std::vector< double > >& model_parameters,
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>& lik )
  {

    // closed form solution see wikipedia, gaussian mixture example
    // http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm


    Eigen::MatrixXd T( mixture_parameters.size(),
		       data.size() );
    for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
      for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	math_core::nd_point_t x = data[ data_i ];
	double t = mixture_parameters[ mix_i ];
	std::vector<double> params = model_parameters[ mix_i ];
	mp_float w = t * lik( x, params );
	mp_float norm_w = 0;
	for( size_t mix_j = 0; mix_j < mixture_parameters.size(); ++mix_j ) {
	  norm_w += ( mixture_parameters[mix_j] 
		      * lik( x, model_parameters[mix_j]) );
	}
	w /= norm_w;
	if( w < 1e-11 ) {
	  w = 0;
	}
	T( mix_i, data_i ) = w.convert_to<double>();
      }
    }

    // Ok, compute maximum new mixture weights given T
    std::vector<double> new_mixture_weights( mixture_parameters.size(), 0.0 );
    for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
      double w = 0;
      for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	w += T( mix_i, data_i );
      }
      new_mixture_weights[ mix_i ] = w / data.size();
    }
    
    return new_mixture_weights;
  }

  //====================================================================

  void run_GEM_mixture_model_MLE_numerical
  ( const GEM_parameters_t& gem_parameters,
    const std::vector<math_core::nd_point_t>& data,
    const std::vector<std::vector<double> >& initial_parameters,
    const std::vector<std::vector<double> >& param_lower_bounds,
    const std::vector<std::vector<double> >& param_upper_bounds,
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>& model_likelihood,
    std::vector<std::vector<double> >& mle_estimate,
    std::vector<double>& mle_mixture_weights )
  {

    // we want to ignore most "math" errros from gsl here,
    // and just try to keep on iterating!
    math_core::gsl_scoped_error_function_t
      local_gsl_error_handler( math_core::gsl_ignore_math_caveat_errors );

    // ok, we need to store an keep up the hidden set of 
    // "weights" for each mixture
    std::vector<double> mixture_weights 
      = calculate_initial_mixture_weights( data,
					   initial_parameters,
					   model_likelihood );

    // keep track of current and old parameters
    std::vector<std::vector<double> > current_mixture_parameters 
      = initial_parameters;

    // print out the initial likelihood
    std::cout << "GEM: initial lik= "
	      << _likelihood( data,
			      mixture_weights,
			      current_mixture_parameters,
			      model_likelihood )
	      << std::endl;
						      
    
    // loop doing E then M steps
    size_t iteration = 0;
    while( iteration < *gem_parameters.stop.max_iterations ) {

      // are we potentially an output loop iteration
      bool output_iter = false;
      if( *gem_parameters.stop.max_iterations >= 100 ) {
	if( (iteration) % ( (*gem_parameters.stop.max_iterations) / 100 ) == 0 )
	  output_iter = true;
      }
    
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
      std::vector<double> next_parameters
	= find_max_Q_numeric( q, 
			      flatten( current_mixture_parameters ),
			      flatten( param_lower_bounds ),
			      flatten( param_upper_bounds ),
			      gem_parameters.max_optimize_iterations );

      // Ok, now maximize the mixture weights.
      // We know a closed for solution for this so we maximize independently
      // from the above maximization
      std::vector<double> next_mixture_weights =
	maximize_mixture_weights( data,
				  mixture_weights,
				  current_mixture_parameters,
				  model_likelihood );

      // set current params to maximized set
      current_mixture_parameters = 
	reconstitute_parameters( current_mixture_parameters,
				 next_parameters.data() );
      mixture_weights = next_mixture_weights;


      // std::cout << "  M-step: p: ";
      // for( size_t i = 0; i < next_parameters.size(); ++i ) {
      // 	std::cout << next_parameters[i] << ",";
      // }
      // std::cout << std::endl;
      if( output_iter ) {
	std::cout << "  M-step: ";
	for( size_t i = 0; i < current_mixture_parameters.size(); ++i ) {
	  for( size_t j = 0; j < current_mixture_parameters[i].size(); ++j ) {
	    std::cout << current_mixture_parameters[i][j] << ",";
	  }
	  std::cout << " || ";
	}
	std::cout << std::endl;
      }

      // print out the current mxing weights
      if( output_iter ) {
	std::cout << "  M-step: mixes= ";
	for( size_t i = 0; i < mixture_weights.size(); ++i ) {
	  std::cout << mixture_weights[i] << " , ";
	}
	std::cout << std::endl;
      }
      
      // print out hte current likelihood
      if( output_iter ) {
	std::cout << "  M-step: lik= " 
		  << _likelihood( data,
				  mixture_weights,
				  current_mixture_parameters,
				  model_likelihood )
		  << std::endl;
      }
	
      // increas iteration
      ++iteration;

      if( output_iter ) {
	std::cout << "GEM[" << iteration << " " << ( (double)iteration / (*gem_parameters.stop.max_iterations) ) * 100 << "%]" << std::endl;
      }
    }

    // ok, set the current parameters to the output ones
    mle_estimate = current_mixture_parameters;
    mle_mixture_weights = mixture_weights;
  }


  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================


}
