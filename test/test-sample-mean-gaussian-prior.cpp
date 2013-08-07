
#include <probability-core/distribution_utils.hpp>
#include <math-core/utils.hpp>
#include <iostream>


using namespace probability_core;
using namespace math_core;

int main()
{

  // priors
  nd_point_t prior_mean = point( 10.0 );
  double prior_variance = 25.0;
  
  // the current variance
  double current_variance = 16.0;
  
  // The precision distribution
  gamma_distribution_t precision_distribution;
  precision_distribution.shape = 1.0/64.0;
  precision_distribution.rate = 1.0/4.0;

  // Some observations of the mean
  std::vector<nd_point_t> obs;
  obs.push_back( point( 2 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1.5 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1 ) );
  obs.push_back( point( 1 ) );
  
  // Ok, now sample some gaussians
  // and keep track of their mean mean and mean variance
  double mean_mean = 0;
  double mean_var = 0;
  std::vector<double> means;
  std::vector<double> vars;
  int num_samples = 1000;
  for( int i = 0; i < num_samples; ++i ) {
    gaussian_distribution_t g =
      sample_mean_gaussian_prior( obs,
				  current_variance,
				  precision_distribution,
				  prior_mean,
				  prior_variance );
    mean_mean += g.means[0];
    mean_var += g.covariance.data[0];
    means.push_back( g.means[0] );
    vars.push_back( g.covariance.data[0] );
    
    if( num_samples < 100 ) {
      std::cout << "  sample: " << g << std::endl;
    }
  }
  mean_mean /= num_samples;
  mean_var /= num_samples;
  
  std::cout << "Mean {mean: " << mean_mean << ", var: " << mean_var << "}" << std::endl;
  std::cout << "Var {mean: " << variance( means ) << ", var: " << variance( vars ) << "} " << std::endl;

  return 0;
}
