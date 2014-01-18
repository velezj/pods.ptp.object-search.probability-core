
#include "distribution_utils.hpp"
#include <math-core/geom.hpp>
#include <math-core/io.hpp>
#include <math-core/matrix.hpp>
#include <iostream>
#include <sstream>
#include <gsl/gsl_randist.h>
#include "core.hpp"
#include "rejection_sampler.hpp"

#define DEBUG_VERBOSE false
#define PRINT_MEAN_VAR false

using namespace math_core;


namespace probability_core {


  //=======================================================================

  discrete_distribution_t discrete_distribution( const std::vector<double>& w )
  {
    discrete_distribution_t d;
    d.n = w.size();
    d.prob = w;
    double sum = 0;
    for( size_t i = 0; i < w.size(); ++i ) {
      sum += w[i];
    }
    if( sum != 1.0 ) {
      for( size_t i = 0; i < w.size(); ++i ) {
	d.prob[i] /= sum;
      }
    }
    return d;
  }

  //=======================================================================
  
  discrete_distribution_t discrete_distribution( size_t n,
						 const double* w )
  {
    discrete_distribution_t d;
    d.n = n;
    d.prob = std::vector<double>();
    double sum = 0;
    for( size_t i = 0; i < n; ++i ) {
      sum += w[i];
    }
    for( size_t i = 0; i < n; ++i ) {
      d.prob.push_back( w[i] / sum );
    }
    return d;
  }


  //=======================================================================

  bool flip_coin( const double& p )
  {
    return gsl_ran_bernoulli( global_rng(), p ) == 1;
  }

  //========================================================================

  // Description:
  // Given a mean and precision distribution (gaussian and gamma)
  // samples a new gausiant form it (1D gaussian)
  gaussian_distribution_t sample_gaussian_from( const gaussian_distribution_t& mean_distribution,
						const gamma_distribution_t& precision_distribution ) 
  {
    nd_point_t mean = sample_from( mean_distribution );
    double prec = sample_from( precision_distribution );
    gaussian_distribution_t gauss;
    gauss.dimension = mean.n;
    gauss.means = mean.coordinate;
    double cov = 1.0 / prec;
    if( cov <= 1e-10 ) {
      cov = 1e-10;
    }
    if( isnan(cov) ) {
      // assume becasue of divide by 0
      cov = 1.0 / 1e-10; 
    }
    gauss.covariance = to_dense_mat( Eigen::MatrixXd::Identity( mean.n, mean.n ) * cov );
    
    for( size_t i = 0; i < gauss.dimension; ++i ) {
      if( std::isnan( gauss.means[i] ) ) {
	std::cout << "!NaN in sampled gaiussian mean!" << std::endl;
      }
    }
    for( size_t i = 0; i < gauss.covariance.num_elements; ++i ) {
      if( std::isnan( gauss.covariance.data[i] ) ) {
	std::cout << "!NaN in sampled gaiussian covariance!" << std::endl;
      }
    }

    return gauss;
  }

  //========================================================================

  poisson_distribution_t sample_poisson_from( const gamma_distribution_t& lambda_distribution ) 
  {
    double lam = sample_from( lambda_distribution );
    poisson_distribution_t pos;
    pos.lambda = lam;
    return pos;
  }

  //=======================================================================

  gamma_distribution_t sample_gamma_from( const gamma_conjugate_prior_t& gamma_distribution )
  {
    return slice_sample_from( gamma_distribution );
  }

  //=======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const gamma_distribution_t& gamma ) 
  {
    os << "G(" << gamma.shape << "," << gamma.rate << ") [mu:" << mean(gamma) << ", var:" << variance(gamma) << "]";
    return os;
  }

  //=======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const gaussian_distribution_t& gauss )
  {
    os << "N( " << point(gauss.means) << " [";
    for( size_t i = 0; (long)i < gauss.covariance.num_elements; ++i ) {
      os << gauss.covariance.data[i];
      if( (long)(i + 1) < gauss.covariance.num_elements ) 
	os << ", ";
    }
    os << "] )";
    return os;
  }

  //=======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const poisson_distribution_t& pos )
  {
    os << "Pos(" << pos.lambda << ")";
    return os;
  }

  //=======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const discrete_distribution_t& dist )
  {
    os << "Dis(";
    for( size_t i = 0; (long)i < dist.n; ++i ) {
      os << dist.prob[i];
      if( (long)(i + 1) < dist.n ) {
	os << ",";
      }
    }
    os << ")";
    return os;
  }

  //=======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const beta_distribution_t& beta )
  {
    os << "B(" << beta.alpha << "," << beta.beta << ")";
    return os;
  }

  //=======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const negative_binomial_distribution_t& ng )
  {
    os << "NG(" << ng.r << "," << ng.p << ")";
    return os;
  }

  //=======================================================================

  std::ostream& operator<< (std::ostream& os,
			    const gamma_conjugate_prior_t& gcp  )
  {
    double mean = -1;
    double var = -1;
    if( PRINT_MEAN_VAR ) {
      estimate_gamma_conjugate_prior_sample_stats( gcp, mean, var );
    }
    os << "Gcp(" << gcp.p << "," << gcp.q << "," << gcp.r << "," << gcp.s << ") [mu: " << mean << " var:" << var << "]";
    return os;
  }


  //=======================================================================
  //=======================================================================
  //=======================================================================
  //=======================================================================
  
  gaussian_distribution_t
  sample_mean_gaussian_prior
  ( const std::vector< double > observed_means,
    const double& current_variance,
    const gamma_distribution_t& precision_distribution,
    const double& prior_mean, 
    const double& prior_variance )
  {
    
    double previous_precision = 1.0 / current_variance;
    double hyperprior_precision = 1.0 / prior_variance;
    
    // ok, sum the means of the current mixtures
    double mean_sum = 0;
    for( std::size_t i = 0; i < observed_means.size(); ++i ) {
      mean_sum += observed_means[i];
    }

    // compute the new variance of the distribution over means
    double new_variance = 
      1.0 / ( previous_precision * observed_means.size() + hyperprior_precision );
    
    // create the new gaussian for hte mean
    gaussian_distribution_t new_mean_dist;
    new_mean_dist.dimension = 1;
    new_mean_dist.means.push_back( ( previous_precision * mean_sum + prior_mean * hyperprior_precision ) * new_variance );
    new_mean_dist.covariance = to_dense_mat( Eigen::MatrixXd::Identity(1,1) * new_variance );

    // sample a new mean
    nd_point_t new_mean = sample_from( new_mean_dist );
    
    // sum the sqaured ereror to this new mean
    // to compute distribution of new precision
    double sum_diff = 0;
    for( std::size_t i = 0; i < observed_means.size(); ++i ) {
      sum_diff += distance_sq( point(observed_means[i]),
  			       new_mean );
    }
    
    // create the precision distribution
    gamma_distribution_t new_precision_dist;
    new_precision_dist.shape = ( observed_means.size() / 2.0 + precision_distribution.shape );
    new_precision_dist.rate = sum_diff / 2.0 + precision_distribution.rate;
    
    // sample a new precision
    double new_precision = sample_from( new_precision_dist );    
    
    // return a new gaussian
    gaussian_distribution_t sampled_mean_dist;
    sampled_mean_dist.dimension = new_mean_dist.dimension;
    sampled_mean_dist.means = new_mean.coordinate;
    sampled_mean_dist.covariance = to_dense_mat( Eigen::MatrixXd::Identity(1,1) * 1.0 / new_precision );


    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "     sample mean gauss prior:" << std::endl;
      std::cout << "      obs: ";
      for( size_t i = 0; i < observed_means.size(); ++i ) {
	std::cout << observed_means[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "      Precision prev: " << previous_precision << ", hyper: " << hyperprior_precision << std::endl;
      std::cout << "      New Mean dist: " << new_mean_dist << " new mean: " << new_mean << std::endl;
      std::cout << "      New Prec dist: " << new_precision_dist << " [mean: " << mean(new_precision_dist) << ", var: " << variance( new_precision_dist ) << "]  new prec: " << new_precision << std::endl;
      std::cout << "      Sample: " << sampled_mean_dist << std::endl;
    }
      
    return sampled_mean_dist;

  }

  //=======================================================================

  gaussian_distribution_t
  sample_mean_gaussian_prior
  ( const std::vector< math_core::nd_point_t > observed_means,
    const double& current_variance,
    const gamma_distribution_t& precision_distribution,
    const math_core::nd_point_t& prior_mean,
    const double& prior_variance )

  {
    
    int dim = prior_mean.n;
    double previous_precision = 1.0 / current_variance;
    double hyperprior_precision = 1.0 / prior_variance;
    
    // ok, sum the means of the current mixtures
    nd_vector_t mean_sum = zero_vector(dim);
    for( std::size_t i = 0; i < observed_means.size(); ++i ) {
      mean_sum = mean_sum + (observed_means[i] - zero_point(dim));
    }

    // compute the new variance of the distribution over means
    double new_variance = 
      1.0 / ( previous_precision * observed_means.size() + hyperprior_precision );
    
    // create the new gaussian for hte mean
    gaussian_distribution_t new_mean_dist;
    new_mean_dist.dimension = dim;
    new_mean_dist.means = ( ( mean_sum * previous_precision + (prior_mean - zero_point(dim)) * hyperprior_precision ) * new_variance ).component;
    new_mean_dist.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * new_variance );

    // sample a new mean
    nd_point_t new_mean = sample_from( new_mean_dist );
    
    // sum the sqaured ereror to this new mean
    // to compute distribution of new precision
    double sum_diff = 0;
    for( std::size_t i = 0; i < observed_means.size(); ++i ) {
      sum_diff += distance_sq( observed_means[i],
  			       new_mean );
    }
    
    // create the precision distribution
    gamma_distribution_t new_precision_dist;
    new_precision_dist.shape = ( observed_means.size() / 2.0 + precision_distribution.shape );
    new_precision_dist.rate = sum_diff / 2.0 + precision_distribution.rate;
    
    // sample a new precision
    double new_precision = sample_from( new_precision_dist );
    
    // return a new gaussian
    gaussian_distribution_t sampled_mean_dist;
    sampled_mean_dist.dimension = dim;
    sampled_mean_dist.means = new_mean.coordinate;
    sampled_mean_dist.covariance = to_dense_mat( Eigen::MatrixXd::Identity(dim,dim) * 1.0 / new_precision );

    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "     sample mean gauss prior:" << std::endl;
      std::cout << "      obs: ";
      for( size_t i = 0; i < observed_means.size(); ++i ) {
	std::cout << observed_means[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "      Precision prev: " << previous_precision << ", hyper: " << hyperprior_precision << std::endl;
      std::cout << "      New Mean dist: " << new_mean_dist << " new mean: " << new_mean << std::endl;
      std::cout << "      New Prec dist: " << new_precision_dist << " [mean: " << mean(new_precision_dist) << ", var: " << variance( new_precision_dist ) << "]  new prec: " << new_precision << std::endl;
      std::cout << "      Sample: " << sampled_mean_dist << std::endl;
    }


    return sampled_mean_dist;

  }
  
  //=======================================================================

  gamma_distribution_t
  sample_precision_gamma_prior
  ( const std::vector<double> precisions,
    const gamma_distribution_t current_prior,
    const double prior_variance )
  {
    
    double b = current_prior.shape;
    double w = current_prior.rate;
    
    // sum the precisions of each data point
    double prec_sum = 0;
    double prec_factor = 1;
    for( std::size_t i = 0; i < precisions.size(); ++i ) {
      double prec = precisions[i]; 
      prec_sum += prec;
      prec_factor *= pow( prec, b/2.0) * exp( - b * w * prec / 2.0 );
    }
    
    precision_shape_posterior_t lik(precisions, prec_factor, w );
    
    // Hack, just sample some values and sample from those
    uniform_sampler_within_range uniform( 0.00001, 100 );
    std::vector<double> sample_precs;
    std::vector<double> sample_vals;
    for( std::size_t i = 0; i < 100; ++i ) {
      double x = uniform();
      double l = lik(x);
      sample_vals.push_back( x );
      sample_precs.push_back( l );
    }
    discrete_distribution_t dist;
    dist.n = sample_precs.size();
    dist.prob = sample_precs;
    int idx = sample_from(dist);
    double new_shape = sample_vals[idx];
    
    
    // now build up the distribution for the rate of the precision
    gamma_distribution_t new_rate_dist;
    new_rate_dist.shape = ( precisions.size() * new_shape + 1 ) / 2.0;
    new_rate_dist.rate = 2 * 1.0 / ( new_shape * prec_sum + 1.0 / prior_variance );
    
    
    // sample a new precision rate
    double new_rate = sample_from(new_rate_dist);
    
    // return hte new distribution
    gamma_distribution_t new_dist;
    new_dist.shape = new_shape;
    new_dist.rate = new_rate;

    // debug
    if( DEBUG_VERBOSE ) {
      std::cout << "     Sample Precision Prior: " << std::endl;
      std::cout << "      prec obs: ";
      for( size_t i = 0; i < precisions.size(); ++i ) {
	std::cout << precisions[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "      Prior: " << current_prior << ", prior var: " << prior_variance << std::endl;
      std::cout << "      new shape: " << new_shape << " p(..): " << sample_precs[idx] << std::endl;
      std::cout << "      new rate dist: " << new_rate_dist << ", sampled: " << new_rate << std::endl;
      std::cout << "      Sample: " << new_dist << std::endl;
    }
    
    return new_dist;

  }
  
  //=======================================================================
  //=======================================================================

  std::string to_json( const gaussian_distribution_t& gauss ) {
    std::ostringstream oss;
    oss << "{ \"object_class\" : \"gaussian_distribution_t\" , ";
    oss << "  \"mean\" : [ ";
    for( int i = 0; i < gauss.dimension; ++i ) {
      oss << gauss.means[0];
      if( i < gauss.dimension - 1 ) {
	oss << ",";
      }
    }
    oss << "], ";
    oss << "  \"covariance\" : " << to_json( gauss.covariance );
    oss << " }";
    return oss.str();
  }

  //=======================================================================

  std::string to_json( const gamma_distribution_t& gamma ) {
    std::ostringstream oss;
    oss << "{ \"object_class\" : \"gamma_distribution_t\" , ";
    oss << "  \"shape\" : " << gamma.shape << " , ";
    oss << "  \"rate\" : " << gamma.rate;
    oss << " }";
    return oss.str();
  }

  //=======================================================================

  std::string to_json( const gamma_conjugate_prior_t& gcp ) {
    std::ostringstream oss;
    oss << "{ \"object_class\" : \"gamma_conjugate_prior_t\" , ";
    oss << "  \"p\" : " << gcp.p << " , ";
    oss << "  \"q\" : " << gcp.q << " , ";
    oss << "  \"r\" : " << gcp.r << " , ";
    oss << "  \"s\" : " << gcp.s;
    oss << " }";
    return oss.str();
  }

  //=======================================================================

  std::string to_json( const beta_distribution_t& beta ) {
    std::ostringstream oss;
    oss << "{ \"object_class\" : \"beta_distribution_t\" , ";
    oss << "  \"alpha\" : " << beta.alpha << " , ";
    oss << "  \"beta\" : " << beta.beta;
    oss << " }";
    return oss.str();
  }

  //=======================================================================

  std::string to_json( const poisson_distribution_t& pos ) {
    std::ostringstream oss;
    oss << "{ \"object_class\" : \"poisson_distribution_t\" , ";
    oss << "  \"lambda\" : " << pos.lambda;
    oss << " }";
    return oss.str();
  }

  //=======================================================================

  std::string to_json( const negative_binomial_distribution_t& nb ) {
    std::ostringstream oss;
    oss << "{ \"object_class\" : \"negative_binomial_distribution_t\" , ";
    oss << "  \"r\" : " << nb.r << " , ";
    oss << "  \"p\" : " << nb.p;
    oss << " }";
    return oss.str();
  }

  //=======================================================================
  
  std::string to_json( const discrete_distribution_t& dist )
  {
    std::ostringstream oss;
    oss << "{ \"object_class\" : \"discrete_distribution_t\" , ";
    oss << "  \"prob\" : [ ";
    for( size_t i = 0; i < dist.prob.size(); ++i ) {
      oss << dist.prob[i];
      if( i < dist.prob.size() - 1 ) {
	oss << ",";
      }
    }
    oss << "] }";
    return oss.str();
  }
  
  //=======================================================================
  //=======================================================================
  //=======================================================================



}
