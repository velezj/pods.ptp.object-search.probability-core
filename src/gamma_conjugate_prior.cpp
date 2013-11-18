
#include "gamma_conjugate_prior.hpp"
#include "core.hpp"
#include "rejection_sampler.hpp"
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_monte.h>
//#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>
#include "slice_sampler.hpp"
#include <boost/bind.hpp>
#include <limits>
#include <p2l-common/stat_counter.hpp>
#include <p2l-common/context.hpp>


using namespace math_core;

namespace probability_core {
  
  //====================================================================

  double 
  likelihood( const gamma_distribution_t& gamma,
	      const gamma_conjugate_prior_t& gcp )
  {
    P2L_COMMON_push_function_context();

    // Ok, if the rate or shape of hte gamma aar too large,
    // fake them out using the same ratio but smaller
    if( gamma.shape > 20 &&
	gamma.rate > 20 )  {
      gamma_distribution_t new_gamma;
      new_gamma.rate = 1;
      new_gamma.shape = gamma.shape / gamma.rate;
      return likelihood( new_gamma, gcp );
    }
    
    // if only one of rate or chape is large, we just return a small value
    if( gamma.shape > 20 ||
	gamma.rate > 20 ) {
      return 1e-30;
    }

    // compute the likelihood
    double b = gamma.rate;
    double p_power = pow( gcp.p, gamma.shape - 1.0 );
    double q_exp = exp( - (b) * gcp.q );
    double shape_gamma = gsl_sf_gamma( gamma.shape );
    double r_power = pow( shape_gamma, gcp.r );
    double s_power = pow( (b), - gamma.shape * gcp.s );
    double result = (p_power * q_exp) / ( r_power * s_power );

    // if the result is NaN, return zero
    if( std::isnan( result ) )
      return 0;
    if( std::isinf( result ) )
      return 0;

    return result;
  } 
  
  
  //====================================================================

  // Description:
  // A uniform sampler of gammas by sampling their parameters
  // uniformely and independently.
  class uniform_gamma_sampler
  {
  public:
    boost::function0<double> shape_sampler;
    boost::function0<double> rate_sampler;
    uniform_gamma_sampler( const boost::function0<double>& shape,
			   const boost::function0<double>& rate )
      : shape_sampler( shape ),
	rate_sampler( rate )
    {}

    gamma_distribution_t operator() () const
    {
      gamma_distribution_t gamma;
      gamma.shape = shape_sampler();
      gamma.rate = rate_sampler();
      return gamma;
    }
  };


  // Description:
  // The gamma conjugate prior likelihood for monte carlo integration
  double
  gamma_conjugate_prior_likelihood_mc
  ( double* x,
    size_t dim,
    void* params )
  {
    gamma_conjugate_prior_t* gcp = (gamma_conjugate_prior_t*)params;
    gamma_distribution_t gamma;
    gamma.shape = x[0];
    gamma.rate = x[1];
    
    return likelihood( gamma, *gcp );
  }

  gamma_distribution_t
  sample_from( const gamma_conjugate_prior_t& gcp )
  {
    P2L_COMMON_push_function_context();

    // we are going to use rejection sampling here
    // but first we need to estimate a scaling for the sampling.

    // compute the range to integrate over
    double low[2] = { 0.1, 0.01 };
    double high[2] = { 20, 20 };
    
    // use monte carlo integration to get a sense of the normalizing
    // factor for the gamma prior
    gsl_monte_function F = { &gamma_conjugate_prior_likelihood_mc,
			     2,
			     const_cast<void*>((const void*)&gcp) };

    // the number of calls to the monte calro samples
    size_t num_samples = 500;
    size_t warmup_samples = 500;
    size_t num_tries = 0;
    size_t max_tries = 3;

    double norm, norm_err;
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc( 2 );
    gsl_monte_vegas_integrate( &F, low, high, 2,
			       warmup_samples, global_rng(), s,
			       &norm, &norm_err );
    do {

      gsl_monte_vegas_integrate( &F, low, high, 2,
				 num_samples, global_rng(), s,
				 &norm, &norm_err );
      ++num_tries;

    } while( fabs( gsl_monte_vegas_chisq(s) - 1.0 ) > 0.5 &&
	     num_tries < max_tries);

    STAT_LVL( trace, "mc.vegas.tries", (double)num_tries );
    STAT_LVL( debug, "mc.vegas.chisq", gsl_monte_vegas_chisq(s) );

    // free resources
    gsl_monte_vegas_free( s );

    // handle NaN by assuming just a really small value
    if( std::isnan( norm ) ) {
      norm = 1e-5;
      std::cout << "hadled NaN in gamma conjugate prior norm " << gcp << std::endl;
    }

    // fix zero estaimtes
    if( norm == 0 ) {
      norm = 1e-5;
      std::cout << "   gcp sampling: vegas mc got 0 norm, setting to " << norm << std::endl;
    }


    // debug
    //std::cout << "Gamma Conjugate Prior: " << gcp << " norm: " << norm << " (err:" << norm_err << ") " << std::endl;

    // ok, now scale rejection sample usign the compute norm
    rejection_sampler_status_t status;
    uniform_sampler_within_range shape_sampler(low[0],high[0]);
    uniform_sampler_within_range rate_sampler(low[1],high[1]);
    uniform_gamma_sampler uniform_sampler( shape_sampler, rate_sampler );

    gamma_distribution_t sample = 
      scaled_rejection_sample<gamma_distribution_t>
      ( boost::lambda::bind( likelihood, boost::lambda::_1, gcp ),
	norm,
	uniform_sampler,
	status );

    // debug
    //std::cout << "sampled gamma: " << status.iterations << " (" << status.seconds << ")" << std::endl;

    return sample;
  }
  

  //====================================================================

  void estimate_gamma_conjugate_prior_sample_stats
  ( const gamma_conjugate_prior_t& gcp, 
    double& mean_result,
    double& var_result,
    const int num_samples )
  {
    P2L_COMMON_push_function_context();

    mean_result = 0;
    var_result = 0;
    for( int i = 0; i < num_samples; ++i ) {
      gamma_distribution_t g = sample_from( gcp );
      double m = mean( g );
      double v = variance( g );
      mean_result += m;
      var_result += v;
    }
    mean_result /= num_samples;
    var_result /= num_samples;
  }

  //====================================================================

  gamma_conjugate_prior_t
  fix_numerical_reduce_counts( const gamma_conjugate_prior_t& gcp )
  {
    double max_count = 80;
    double norm_count = 40;

    // nothing to fix if less than max count
    if( std::max( gcp.r, gcp.s ) < max_count ) {
      return gcp;
    }

    // ok, take away "average" samples until we get norm_count
    double diff_counts = std::max( gcp.r, gcp.s ) - norm_count;
    double avg_elt_q = gcp.q / std::max( gcp.r, gcp.s );
    double avg_elt_p = std::pow( gcp.p, 1.0 / std::max( gcp.r, gcp.s ) );
    
    gamma_conjugate_prior_t fixed = gcp;
    fixed.r -= diff_counts;
    fixed.s -= diff_counts;
    fixed.q -= ( diff_counts * avg_elt_q );
    fixed.p = std::pow( avg_elt_p, norm_count );
    
    // debug
    // std::cout << "  ** fixing Gcp: " << gcp << std::endl;
    // std::cout << "  **   diff: " << diff_counts << ", avg q: " << avg_elt_q << " p: " << avg_elt_p << std::endl;
    // std::cout << "  **   fixed: " << fixed << std::endl;

    return fixed;
  }

  //====================================================================
  //====================================================================

  double temp_lik( const nd_point_t& args,
		   const gamma_conjugate_prior_t& gcp )
  {
    assert( args.n == 2 );
    assert( !undefined( args ) );
    if( undefined( args ) )
      return 0.0;
    if( args.n != 2 )
      return 0.0;
    gamma_distribution_t g;
    g.shape = args.coordinate[0];
    g.rate = args.coordinate[1];
    return likelihood( g, gcp );
  }

  gamma_distribution_t
  slice_sample_from( const gamma_conjugate_prior_t& gcp ) 
  {
    P2L_COMMON_push_function_context();

    static std::pair<nd_point_t,nd_point_t> support 
      = std::make_pair( point( 1.0e-4, 1.0e-4 ),
			point( 1000.0, 1000.0 ) );
    static slice_sampler_workplace_t<nd_point_t> workspace(support);
    boost::function<double(const nd_point_t&)> lik = 
      boost::bind( temp_lik, _1, gcp );
    nd_point_t params = slice_sample( lik,
				      workspace,
				      0.001 );
    gamma_distribution_t gamma;
    gamma.shape = params.coordinate[0];
    gamma.rate = params.coordinate[1];
    return gamma;
  }

  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================

}
