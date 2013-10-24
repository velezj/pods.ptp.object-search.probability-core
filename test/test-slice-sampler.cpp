
#include <probability-core/slice_sampler.hpp>
#include <probability-core/distributions.hpp>
#include <math-core/utils.hpp>
#include <iostream>
#include <vector>


using namespace probability_core;
using namespace math_core;


double func_foo( const double& x )
{
  beta_distribution_t true_beta;
  true_beta.alpha = 2.0;
  true_beta.beta = 5.0;
  return pdf( x, true_beta );
}

int main( int argc, char** argv )
{

  std::pair<double,double> support( 0, 1.0);
  slice_sampler_workplace_t<double> workplace( support );

  beta_distribution_t true_beta;
  true_beta.alpha = 2.0;
  true_beta.beta = 5.0;

  std::vector<double> slice_samples;
  std::vector<double> true_samples;

  boost::function<double (const double&)> func_foo_f = &func_foo;

  std::cout << "First sample: " << slice_sample_1d( func_foo_f, workplace ) << std::endl;
  
  int num_samples = 10000;
  for( int i = 0; i < num_samples; ++i ) {
    slice_samples.push_back( slice_sample_1d( func_foo_f, workplace ) );
    true_samples.push_back( sample_from( true_beta ) );
  }

  std::cout << "Slice Stats: mean=" << mean( slice_samples ) << " var=" << variance( slice_samples ) << std::endl;
  std::cout << "GSL Stats: mean=" << mean( true_samples ) << " var=" << variance( true_samples ) << std::endl;
  

  return 0;
}
