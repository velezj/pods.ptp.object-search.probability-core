
#include <probability-core/distributions.hpp>
#include <probability-core/distribution_utils.hpp>
#include <iostream>

using namespace probability_core;

int main()
{

  // create prior
  double p = 1;
  double q = 2;
  double r = 2;
  double s = 2;
  gamma_conjugate_prior_t gcp = { p, q, r, s };
  

  for( size_t i = 0; i < 10; ++i ) {
    gamma_distribution_t gamma = sample_from( gcp );
    std::cout << "sampled: " << gamma << "  lik(.)=" << likelihood( gamma, gcp ) << std::endl;
  }


  // Ok, now we will test the posterior update
  double fixed_data = 1;
  int num_observations = 20;
  int num_samples = 10;
  for( int i = 0; i < num_observations; ++i ) {
    double x = fixed_data;
    std::cout << "Prior: " << gcp << "  obs: " << x << std::endl;
    gcp.p *= x;
    gcp.q += x;
    gcp.r += 1;
    gcp.s += 1;
    std::cout << "Posterior:  " << gcp << std::endl;
    
    std::vector<double> sample_means;
    std::vector<double> sample_vars;
    std::cout << "   samples: ";
    for( int k = 0; k < num_samples; ++k ) {
      gamma_distribution_t g = sample_from( gcp );
      std::cout << g << "  ";
      sample_means.push_back( mean(g) );
      sample_vars.push_back( variance( g ) );
    }
    std::cout << std::endl;
    std::cout << "   means: ";
    for( int k = 0; k < num_samples; ++k ) {
      std::cout << sample_means[k] << "  ";
    }
    std::cout << std::endl;
    std::cout << "   vars : ";
    for( int k = 0; k < num_samples; ++k ) {
      std::cout << sample_vars[k] << "  ";
    }
    std::cout << std::endl;
    
    double mean = 0, mean_var = 0;
    for( int k = 0; k < num_samples; ++k ) {
      mean += sample_means[k];
      mean_var += sample_vars[k];
    }
    mean /= num_samples;
    mean_var /= num_samples;
    std::cout << " + MEAN= " << mean << "  Mean VAR= " << mean_var << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
