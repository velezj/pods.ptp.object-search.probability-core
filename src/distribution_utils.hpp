
#if !defined( __DISTRIBUTION_UTILS_HPP__ )
#define __DISTRIBUTION_UTILS_HPP__

#include "distributions.hpp"
#include <iosfwd>
#include <gsl/gsl_sf_gamma.h>
#include <cmath>
#include <string>

namespace probability_core {


  // Description:
  // Create a discrete distribution from weights
  discrete_distribution_t discrete_distribution( const std::vector<double>& w );
  discrete_distribution_t discrete_distribution( size_t n,
						 const double* w );


  // Description:
  // Flip a coin and return true iff heads, otehrwise tails
  bool flip_coin( const double& p );



  // Description:
  // Sample a guassian distribution from a
  // Gaussian mean and Gamma precision prior
  gaussian_distribution_t 
  sample_gaussian_from( const gaussian_distribution_t& mean_distribution,
			const gamma_distribution_t& precision_distribution );

  
  // Descripiton:
  // Samples a poisson distribution from a gamma prior
  poisson_distribution_t 
  sample_poisson_from( const gamma_distribution_t& lambda_distribution );


  // Description:
  // Samples a gamma distribution from a conjugate prior over
  // gammas
  gamma_distribution_t
  sample_gamma_from( const gamma_conjugate_prior_t& gamma_distribution );


  // Description:
  // Samples a new gaussian prior for a gaussian mean distribution.
  // This is used to update hyperparameters and priors.
  // For example, if you have a mean that has a guassian prior,
  // then this will sample a new prior given some observed means.
  //
  // This is only for 1D gaussians!
  gaussian_distribution_t
  sample_mean_gaussian_prior
  ( const std::vector< double > observed_means,
    const double& current_variance,
    const gamma_distribution_t& precision_distribution,
    const double& prior_mean,
    const double& prior_variance );



  // Description:
  // Samples a new gaussian prior for a gaussian mean distribution.
  // This is used to update hyperparameters and priors.
  // For example, if you have a mean that has a guassian prior,
  // then this will sample a new prior given some observed means.
  //
  // This assumes a SINGLE covariance value, so COV = sigma * I;
  gaussian_distribution_t
  sample_mean_gaussian_prior
  ( const std::vector< math_core::nd_point_t > observed_means,
    const double& current_variance,
    const gamma_distribution_t& precision_distribution,
    const math_core::nd_point_t& prior_mean, 
    const double& prior_variance );
    


  // Description:
  // The explicit posterior form for the shape of a gamma
  // given some precisions as data.
  // This is used for sampling a posterior gamma prior
  // for a precision variable ( so for example, if you
  // have a distribution of precision as a gamma, then 
  // this will givne you the posterior of the shape paramter 
  // of this gamma given some observed precicions )
  //
  class precision_shape_posterior_t
  {
  public:

    // Description:
    // The set of observed precision with which to compute likleihoods and get
    // a posterior over the shape
    std::vector<double> observed_precisions;

    // Descripion:
    // The rate of the gamma we are resampling,
    // or just the rate of the gamma we are getting hte posterior
    // shape parameter over
    double rate;

    // Description:
    // The precision factor,
    // precomputed, this shoulkd NOT work but we are trying it
    // since it seems to give better answers
    double factor;
    
    precision_shape_posterior_t( const std::vector<double>& observed_precisions,
				 double rate,
				 double factor )
      : observed_precisions(observed_precisions),
  	rate(rate),
	factor( factor )
    {}
    
    double operator() (double b) const
    {
      double k = observed_precisions.size();
      double h = gsl_sf_gamma( b/2.0 );
      if( h > 1000 )
  	return 0.0;
      if( h < 0.00001 )
  	return 0.0;
      double r = 1.0 / std::pow( h, k );
      r *= std::pow( b * rate / 2.0, (k * b - 3.0) / 2.0 );
      r *= std::exp( - 1.0 / ( 2.0 * b ) );
      // double factor =1 ;
      // for( std::size_t i = 0; i < observed_precisions.size(); ++i ) {
      // 	double prec = observed_precisions[i];
      // 	factor *= pow( prec, b/2.0) * exp( - b * rate * prec / 2.0 );
      // }
      r *= factor;
      if( r > 10000 )
  	return 0;
      return r;
    }
  };

  
  // Description:
  // Sample the prior hyperparameters for a precision distribution
  // given a vector of precision (as data) a precision distribution (likelihood)
  // and a current gamma prior, returns a sampled gamma prior
  gamma_distribution_t
  sample_precision_gamma_prior
  ( const std::vector<double> precisions,
    const gamma_distribution_t current_prior,
    const double prior_variance );

  
  std::ostream& operator<< (std::ostream& os,
			    const gaussian_distribution_t& gaussian );
  std::ostream& operator<< (std::ostream& os,
			    const poisson_distribution_t& pos );
  std::ostream& operator<< (std::ostream& os,
			    const gamma_distribution_t& gamma );
  std::ostream& operator<< (std::ostream& os,
			    const discrete_distribution_t& dist );
  std::ostream& operator<< (std::ostream& os,
			    const beta_distribution_t& beta );
  std::ostream& operator<< (std::ostream& os,
			    const negative_binomial_distribution_t& ng );
  std::ostream& operator<< (std::ostream& os,
			    const gamma_conjugate_prior_t& gcp );


  std::string to_json( const gaussian_distribution_t& gauss );
  std::string to_json( const gamma_distribution_t& gamma );
  std::string to_json( const gamma_conjugate_prior_t& gcp );
  std::string to_json( const poisson_distribution_t& pos );
  std::string to_json( const negative_binomial_distribution_t& nb );
  std::string to_json( const beta_distribution_t& beta );
  std::string to_json( const discrete_distribution_t& dist );

}

#endif

