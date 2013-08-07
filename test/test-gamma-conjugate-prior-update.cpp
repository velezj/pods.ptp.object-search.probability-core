#include <probability-core/distributions.hpp>
#include <probability-core/distribution_utils.hpp>
#include <iostream>
#include <stdexcept>

using namespace probability_core;


// Description:
// The posterior for a gamma conjugate prior likelihood
// and a poisson prior for a single parameter for the gamma
// conjugate prior (this is for hyperparameter resampling!)
class single_parameter_gcp_likelihood_gamma_prior_posterior_t
{
public:
  std::vector<gamma_distribution_t> gamma_data_points;
  gamma_conjugate_prior_t base_likelihood_gcp;
  gamma_distribution_t prior;
  char param;
  
  single_parameter_gcp_likelihood_gamma_prior_posterior_t
  ( const std::vector<gamma_distribution_t>& points,
    const gamma_conjugate_prior_t& base_lik,
    const gamma_distribution_t& prior,
    const char param)
    : gamma_data_points(points),
      base_likelihood_gcp( base_lik ),
      prior( prior ),
      param(param)
  {
  }      
  
  double operator() ( const double& x ) const 
  {
    double mult = 1;
    gamma_conjugate_prior_t likelihood_gcp = base_likelihood_gcp;
    switch( param ) {
    case 'p':
      likelihood_gcp.p = x;
      break;
    case 'q':
      likelihood_gcp.q = x;
      break;
    case 'r':
      likelihood_gcp.r = x;
      break;
    case 's':
      likelihood_gcp.s = x;
      break;
    default:
      throw std::domain_error( "Unknown gamma conjugate prior parameter!" );
    }
    
    for( size_t i = 0; i < gamma_data_points.size(); ++i ) {
      mult *= likelihood( gamma_data_points[i], likelihood_gcp );
    }	
    mult *= pdf( x, prior );
    return mult;
  }
};


int main()
{

  // create prior
  double p = 1;
  double q = 2;
  double r = 2;
  double s = 2;
  gamma_conjugate_prior_t gcp = { p, q, r, s };
 
  
  // ok, create gamma priors for the parameters
  gamma_distribution_t p_prior;
  p_prior.shape = 1;
  p_prior.rate = 0.5;
  gamma_distribution_t q_prior;
  q_prior.shape = 1;
  q_prior.rate = 0.5;
  gamma_distribution_t r_prior;
  r_prior.shape = 1;
  r_prior.rate = 0.5;
  gamma_distribution_t s_prior;
  s_prior.shape = 1;
  s_prior.rate = 0.5;

  // now, get some "observed" gammas
  std::vector<gamma_distribution_t> observed_gammas;
  gamma_distribution_t g0;
  g0.shape = 3;
  g0.rate = 1;
  observed_gammas.push_back( g0 );
  
  // compute statistics about the "p" posterior
  single_parameter_gcp_likelihood_gamma_prior_posterior_t
    p_posterior( observed_gammas,
		 gcp,
		 p_prior,
		 'p' );
  double p_min = 0.001;
  double p_max = 1000;
  double p_step = 0.1;
  // for( double p = p_min; p < p_max; p += p_step ) {
  //   double p_lik = p_posterior( p );
  //   std::cout << p << " " << p_lik << std::endl;
  // }

  single_parameter_gcp_likelihood_gamma_prior_posterior_t
    q_posterior( observed_gammas,
		 gcp,
		 q_prior,
		 'q' );
  double q_min = 0.00001;
  double q_max = 10;
  double q_step = 0.01;
  for( double q = q_min; q < q_max; q += q_step ) {
    double q_lik = q_posterior( q );
    std::cout << q << " " << q_lik << std::endl;
  }


  return 0;
}
