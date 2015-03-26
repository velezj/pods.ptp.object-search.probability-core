
#include "EM.hpp"
#include <gsl/gsl_multimin.h>
#include "uniform.hpp"
#include <math-core/gsl_utils.hpp>
#include <math-core/matrix.hpp>
#include <math-core/extrema.hpp>
#include <math-core/mpt.hpp>
#include <math-core/policy_number.hpp>
#include <math-core/io.hpp>
#include <boost/thread/tss.hpp>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>

using namespace math_core::mpt;


namespace probability_core {


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

  boost::thread_specific_ptr<std::vector<double> > _Q_last_mixtures;
  boost::thread_specific_ptr<std::vector<double> > _Q_last_param_model;
  boost::thread_specific_ptr<std::vector<double> > _Q_last_param_x;
  boost::thread_specific_ptr< double > _Q_last_likelihood;

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

      // catch any exceptions and rethrow with more 
      // information about what the last things to
      // happen here were
      try {
	
	// ok, now lets compute the expectation of the data
	// given the model parameter inputs as well as the
	// mixture weights used to construct this Q(.)
	double log_lik = 0;
	for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	  for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	    double w = 0.0;
	    w = (mixture_parameters[mix_i] * lik( data[data_i],
						  params[mix_i] ));
	    double norm = 0.0;
	    for( size_t j = 0; j < mixture_parameters.size(); ++j ) {
	      norm += mixture_parameters[j] * lik( data[data_i],
						   params[j] );
	    }
	    w = w / norm;
	    
	    double the_lik = lik( data[data_i],
				  params[mix_i] );
	    double single_lik = 0.0;
	    if( w < exp(-20) ||
		the_lik < exp(-20) ||
		std::isnan(w) ||
		std::isinf(w) ||
		std::isnan( the_lik ) ||
		std::isinf( the_lik ) ) {
	      single_lik = -40;
	    } else {
	      single_lik =
		(log(w) * log( the_lik ));
	    }
	    log_lik += single_lik;
	  }
	}
	
	// std::cout << "  Q-GEM(";
	// for( size_t i = 0; i < flat_params.size(); ++i ) {
	// 	std::cout << flat_params[i] << ",";
	// }
	// std::cout << ") ll=" << log_lik << std::endl;
	
	// set last calles
	_Q_last_mixtures.reset( new std::vector<double>(mixture_parameters) );
	_Q_last_param_model.reset( new std::vector<double>(flatten(model_parameters)) );
	_Q_last_param_x.reset( new std::vector<double>(flat_params) );
	_Q_last_likelihood.reset( new double(log_lik) );
	
	return log_lik;
      }
      catch( boost::exception& e ) {
	if( _Q_last_mixtures.get() ) {
	  e << errorinfo_last_Q_mixture_weights( *_Q_last_mixtures );
	}
	if( _Q_last_param_model.get() ) {
	  e << errorinfo_last_Q_model_parameters( *_Q_last_param_model );
	}
	if( _Q_last_param_x.get() ) {
	  e << errorinfo_last_Q_parameters_x( *_Q_last_param_x );
	}
	if( _Q_last_likelihood.get() ) {
	  e << errorinfo_last_Q_log_likelihood( *_Q_last_likelihood );
	}
	throw;
      }

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
    double p;
    for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
      math_core::nd_point_t x = data[data_i];
      for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	double w = mixture_parameters[mix_i];
	std::vector<double> model = model_parameters[mix_i];
	p += w * lik(x,model);
      }
    }
    return p;
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
      mix.push_back( sample_from( uniform_distribution(1.0, 10.0)));
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

  boost::thread_specific_ptr<std::vector<double> > _last_maxmix_mixture_weights;
  boost::thread_specific_ptr<Eigen::MatrixXd> _last_maxmix_unorm_T;

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

    // catch any exception ad add information about what happened last here
    try {

      Eigen::MatrixXd T( mixture_parameters.size(),
			 data.size() );
      Eigen::MatrixXd UnormT( mixture_parameters.size(),
			      data.size() );
      for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	  math_core::nd_point_t x = data[ data_i ];
	  double t = mixture_parameters[ mix_i ];
	  std::vector<double> params = model_parameters[ mix_i ];
	  double w ( t * lik( x, params ) );
	  //double w = ( lik( x, params ) );
	  double norm_w = 0.0;
	  for( size_t mix_j = 0; mix_j < mixture_parameters.size(); ++mix_j ) {
	    norm_w += ( mixture_parameters[mix_j] 
			* lik( x, model_parameters[mix_j]) );
	    //norm_w += ( lik( x, model_parameters[mix_j]) );
	  }
	  UnormT( mix_i, data_i ) = w;
	  w /= norm_w;
	  T( mix_i, data_i ) = w;
	}
      }
      
      // Ok, compute maximum new mixture weights given T
      std::vector<double> new_mixture_weights( mixture_parameters.size(), 0.0 );
      double sum_w = 0.0;
      for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	double w = 0;
	for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	  w += T( mix_i, data_i );
	}
	new_mixture_weights[ mix_i ] = w / data.size();
	
	// cap down weights to 0 if too small!
	if( new_mixture_weights[ mix_i ] < 1.0e-20 ||
	    std::isnan( new_mixture_weights[mix_i] ) ||
	    std::isinf( new_mixture_weights[mix_i] ) ) {
	  new_mixture_weights[ mix_i ] = 0.0;
	}
	sum_w += new_mixture_weights[ mix_i ];
      }
      
      // renormalize the mixture weights in case clamping down changed
      // something in the weights
      if( sum_w > 0 ) {
	for( size_t i = 0; i < new_mixture_weights.size(); ++i ) {
	  new_mixture_weights[ i ] /= sum_w;
	}
      } else {
	//really! let's jsut uniform it then!
	// std::cout << "MIXTURES-SUM=0!!: " <<std::endl;
	// std::cout << "  Normed T:" << std::endl;
	// std::cout << T << std::endl;
	// std::cout << "  Un-Normed T:" << std::endl;
	// std::cout << UnormT << std::endl;
	for( size_t i = 0; i < new_mixture_weights.size(); ++i ) {
	  new_mixture_weights[i] = 1.0 / new_mixture_weights.size();
	}
      }
      
      Eigen::MatrixXd *storedUT = new Eigen::MatrixXd();
      *storedUT = UnormT;
      _last_maxmix_mixture_weights.reset( new std::vector<double>( new_mixture_weights ) );
      _last_maxmix_unorm_T.reset( storedUT );

      // debug
      // print out the entire process
      if( false ) {
	std::cout << "max mixture weights:" << std::endl;
	for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	  std::vector<double> params = model_parameters[mix_i];
	  std::cout << "    model[" << mix_i << "]: ";
	  for( size_t i = 0; i < params.size(); ++i ) {
	    std::cout << params[i] << " , ";
	  }
	  std::cout << std::endl;
	}
	for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	  std::cout << "    data[" << data_i << "]: "
		    << data[data_i]
		    << std::endl;
	}
	std::cout << "    initial mixture: ";
	for( double k : mixture_parameters ) {
	  std::cout << k << " , ";
	}
	std::cout << std::endl;
	std::cout << "    initial lik: "
		  << _likelihood( data, mixture_parameters, model_parameters, lik)
		  << std::endl;
	for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	  for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	    std::cout << "    T( mix " << mix_i << " , " << data_i << " ) = "
		      << T( mix_i, data_i )
		      << " [w=" << mixture_parameters[mix_i]
		      << " , lik=" << lik( data[data_i], model_parameters[mix_i])
		      << "]" << std::endl;
	  }
	}
	std::cout << "    new mixture: ";
	for( double k : new_mixture_weights ) {
	  std::cout << k << " , ";
	}
	std::cout << std::endl;
	std::cout << "    new lik: "
		  << _likelihood( data, new_mixture_weights, model_parameters, lik)
		  << std::endl;
	std::vector<double> rev_mixture_weights = new_mixture_weights;
	std::reverse( rev_mixture_weights.begin(),
		      rev_mixture_weights.end() );
	std::cout << "    reverse lik: "
		  << _likelihood( data, rev_mixture_weights, model_parameters, lik)
		  << std::endl;
      }
      
      
      return new_mixture_weights;

    } 
    catch( boost::exception& e ) {
      if( _last_maxmix_mixture_weights.get() ) {
	e << errorinfo_last_maxmix_mixture_weights( *_last_maxmix_mixture_weights );
      }
      if( _last_maxmix_unorm_T.get() ) {
	e << errorinfo_last_maxmix_unorm_T( *_last_maxmix_unorm_T );
      }
      throw;
    }
  }

  //====================================================================
  
  std::vector<double>
  maximize_mixture_weights_OLD_BROKEN
  ( const std::vector<math_core::nd_point_t>& data,
    const std::vector<double>& mixture_parameters,
    const std::vector<std::vector< double > >& model_parameters,
    std::function<double(const math_core::nd_point_t& single_data,
			 const std::vector<double>& params)>& lik )
  {

    // closed form solution see wikipedia, gaussian mixture example
    // http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm

    // catch any exception ad add information about what happened last here
    try {

      Eigen::MatrixXd T( mixture_parameters.size(),
			 data.size() );
      Eigen::MatrixXd UnormT( mixture_parameters.size(),
			      data.size() );
      for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	  math_core::nd_point_t x = data[ data_i ];
	  double t = mixture_parameters[ mix_i ];
	  std::vector<double> params = model_parameters[ mix_i ];
	  double w ( t * lik( x, params ) );
	  double norm_w;
	  for( size_t mix_j = 0; mix_j < mixture_parameters.size(); ++mix_j ) {
	    norm_w += ( mixture_parameters[mix_j] 
	    		* lik( x, model_parameters[mix_j]) );
	    
	  }
	  UnormT( mix_i, data_i ) = w;
	  w /= norm_w;
	  T( mix_i, data_i ) = w;
	}
      }
      
      // Ok, compute maximum new mixture weights given T
      std::vector<double> new_mixture_weights( mixture_parameters.size(), 0.0 );
      double sum_w = 0.0;
      for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	double w = 0;
	for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	  w += T( mix_i, data_i );
	}
	new_mixture_weights[ mix_i ] = w / data.size();
	
	// cap down weights to 0 if too small!
	if( new_mixture_weights[ mix_i ] < 1.0e-20 ||
	    std::isnan( new_mixture_weights[mix_i] ) ||
	    std::isinf( new_mixture_weights[mix_i] ) ) {
	  new_mixture_weights[ mix_i ] = 0.0;
	}
	sum_w += new_mixture_weights[ mix_i ];
      }
      
      // renormalize the mixture weights in case clamping down changed
      // something in the weights
      if( sum_w > 0 ) {
	for( size_t i = 0; i < new_mixture_weights.size(); ++i ) {
	  new_mixture_weights[ i ] /= sum_w;
	}
      } else {
	//really! let's jsut uniform it then!
	// std::cout << "MIXTURES-SUM=0!!: " <<std::endl;
	// std::cout << "  Normed T:" << std::endl;
	// std::cout << T << std::endl;
	// std::cout << "  Un-Normed T:" << std::endl;
	// std::cout << UnormT << std::endl;
	for( size_t i = 0; i < new_mixture_weights.size(); ++i ) {
	  new_mixture_weights[i] = 1.0 / new_mixture_weights.size();
	}
      }
      
      Eigen::MatrixXd *storedUT = new Eigen::MatrixXd();
      *storedUT = UnormT;
      _last_maxmix_mixture_weights.reset( new std::vector<double>( new_mixture_weights ) );
      _last_maxmix_unorm_T.reset( storedUT );

      // debug
      // print out the entire process
      std::cout << "max mixture weights:" << std::endl;
      for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	std::vector<double> params = model_parameters[mix_i];
	std::cout << "    model[" << mix_i << "]: ";
	for( size_t i = 0; i < params.size(); ++i ) {
	  std::cout << params[i] << " , ";
	}
	std::cout << std::endl;
      }
      for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	std::cout << "    data[" << data_i << "]: "
		  << data[data_i]
		  << std::endl;
      }
      std::cout << "    initial mixture: ";
      for( double k : mixture_parameters ) {
	std::cout << k << " , ";
      }
      std::cout << std::endl;
      std::cout << "    initial lik: "
		<< _likelihood( data, mixture_parameters, model_parameters, lik)
		<< std::endl;
      for( size_t mix_i = 0; mix_i < mixture_parameters.size(); ++mix_i ) {
	for( size_t data_i = 0; data_i < data.size(); ++data_i ) {
	  std::cout << "    T( mix " << mix_i << " , " << data_i << " ) = "
		    << T( mix_i, data_i )
		    << " [w=" << mixture_parameters[mix_i]
		    << " , lik=" << lik( data[data_i], model_parameters[mix_i])
		    << "]" << std::endl;
	}
      }
      std::cout << "    new mixture: ";
      for( double k : new_mixture_weights ) {
	std::cout << k << " , ";
      }
      std::cout << std::endl;
      std::cout << "    new lik: "
		<< _likelihood( data, new_mixture_weights, model_parameters, lik)
		<< std::endl;
      std::vector<double> rev_mixture_weights = new_mixture_weights;
      std::reverse( rev_mixture_weights.begin(),
		    rev_mixture_weights.end() );
      std::cout << "    reverse lik: "
		<< _likelihood( data, rev_mixture_weights, model_parameters, lik)
		<< std::endl;
      
      
      return new_mixture_weights;

    } 
    catch( boost::exception& e ) {
      if( _last_maxmix_mixture_weights.get() ) {
	e << errorinfo_last_maxmix_mixture_weights( *_last_maxmix_mixture_weights );
      }
      if( _last_maxmix_unorm_T.get() ) {
	e << errorinfo_last_maxmix_unorm_T( *_last_maxmix_unorm_T );
      }
      throw;
    }
  }


  //====================================================================

  boost::thread_specific_ptr<std::vector<double> > _last_GEM_mixtures_weights;
  boost::thread_specific_ptr<std::vector<double> > _last_GEM_model_parameters;
  boost::thread_specific_ptr<double > _last_GEM_likelihood;
 
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

    //  catch any exceptions and add information about what last
    // happened here the nrethrow
    try {

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


      // store last params
      _last_GEM_mixtures_weights.reset( new std::vector<double>( mixture_weights ) );    
      _last_GEM_model_parameters.reset( new std::vector<double>( flatten(current_mixture_parameters) ) );

						      
    
      // loop doing E then M steps
      double previous_likelihood = 0.0;
      size_t iteration = 0;
      while( iteration < *gem_parameters.stop.max_iterations ) {

	// are we potentially an output loop iteration
	bool output_iter = false;
	if( *gem_parameters.stop.max_iterations >= 100 &&
	    ( gem_parameters.stop.relative_likelihood_tolerance == false || 
	      iteration > 0.25 * ( *gem_parameters.stop.max_iterations ) ) ) {
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


	// calculate the new likleihood
	double new_likelihood =
	  _likelihood( data,
		       mixture_weights,
		       current_mixture_parameters,
		       model_likelihood );


	// store last parameters
	_last_GEM_mixtures_weights.reset( new std::vector<double>( mixture_weights ) );
	_last_GEM_model_parameters.reset( new std::vector<double>( flatten(current_mixture_parameters) ) );
	_last_GEM_likelihood.reset( new double( new_likelihood ) );
	
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
		    << new_likelihood
		    << std::endl;
	}
	
	// increas iteration
	++iteration;

	if( output_iter ) {
	  std::cout << "GEM[" << iteration << " " << ( (double)iteration / (*gem_parameters.stop.max_iterations) ) * 100 << "%]" << std::endl;
	}

	// check if we need to stop because of relative tolerance 
	// on the likelihood
	if( gem_parameters.stop.relative_likelihood_tolerance ) {
	  double tol = 
	    (*gem_parameters.stop.relative_likelihood_tolerance) 
	    * previous_likelihood;
	  if( fabs( new_likelihood - previous_likelihood ) < tol ) {
	    // we are done! tolerance reached
	    break;
	  }
	}
      
	previous_likelihood = new_likelihood;
      }

      // ok, set the current parameters to the output ones
      mle_estimate = current_mixture_parameters;
      mle_mixture_weights = mixture_weights;

    } 
    catch( boost::exception& e ) {
      if( _last_GEM_mixtures_weights.get() ) {
	e << errorinfo_last_gem_mixture_weights( *_last_GEM_mixtures_weights );
	std::cout << "  last GEM-mix: " << *_last_GEM_mixtures_weights << std::endl;
      }
      if( _last_GEM_model_parameters.get() ) {
	e << errorinfo_last_gem_model_parameters( *_last_GEM_model_parameters );
	std::cout << "  last GEM-mod: " << *_last_GEM_model_parameters << std::endl;
      }
      if( _last_GEM_likelihood.get() ) {
	e << errorinfo_last_gem_likelihood( *_last_GEM_likelihood );
	std::cout << "  last GEM-lik: " << *_last_GEM_likelihood << std::endl;
      }
      if( _last_maxmix_mixture_weights.get() ) {
	e << errorinfo_last_maxmix_mixture_weights( *_last_maxmix_mixture_weights );
	std::cout << "  last MM-mix: " << *_last_maxmix_mixture_weights << std::endl;
      }
      if( _last_maxmix_unorm_T.get() ) {
	e << errorinfo_last_maxmix_unorm_T( *_last_maxmix_unorm_T );
	std::cout << "  last MM-UT: " << *_last_maxmix_unorm_T << std::endl;
      }
      if( _Q_last_mixtures.get() ) {
	e << errorinfo_last_Q_mixture_weights( *_Q_last_mixtures );
	std::cout << "  last Q-mix: " << *_Q_last_mixtures << std::endl;
      }
      if( _Q_last_param_model.get() ) {
	e << errorinfo_last_Q_model_parameters( *_Q_last_param_model );
	std::cout << "  last Q-mod: " << *_Q_last_param_model << std::endl;
      }
      if( _Q_last_param_x.get() ) {
	e << errorinfo_last_Q_parameters_x( *_Q_last_param_x );
	std::cout << "  last Q-x: " << *_Q_last_param_x << std::endl;
      }
      if( _Q_last_likelihood.get() ) {
	e << errorinfo_last_Q_log_likelihood( *_Q_last_likelihood );
	std::cout << "  last Q-log-lik: " << *_Q_last_likelihood << std::endl;
      }
      throw;
    }
  
  }


  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================


}
