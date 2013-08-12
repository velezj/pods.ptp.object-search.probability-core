

#include <probability-core/distributions.hpp>
#include <probability-core/distribution_utils.hpp>
#include <iostream>
#include <sstream>

using namespace probability_core;

int main( int argn, char** argv )
{

  // read in arguments for prior
  double p = -1;
  double q = -1;
  double r = -1;
  double s = -1;
  std::istringstream iss1( argv[1] );
  iss1 >> p;
  std::istringstream iss2( argv[2] );
  iss2 >> q;
  std::istringstream iss3( argv[3] );
  iss3 >> r;
  std::istringstream iss4( argv[4] );
  iss4 >> s;
  std::cout << "raw args: ";
  for( int i = 0 ; i < argn; ++i ) {
    std::cout << argv[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "args: " << p << " " << q << " " << r << " " << s << std::endl;

  // creat the prior
  gamma_conjugate_prior_t gcp = { p, q, r, s };
  

  // sample gamma from the prior
  int num_gamma_samples = 10;
  for( size_t i = 0; (long)i < num_gamma_samples; ++i ) {
    gamma_distribution_t gamma = sample_from( gcp );
    std::cout << "sampled: " << gamma << "  lik(.)=" << likelihood( gamma, gcp ) << std::endl;
  }


  // Ok, now calculate the resulting mean/variance
  double m,v;
  int num_samples = 10000;
  estimate_gamma_conjugate_prior_sample_stats( gcp, m, v, num_samples );
  std::cout << "Rollout Estimate Mean: " << m << " , Var: " << v << std::endl;

  return 0;
}
