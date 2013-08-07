
#include <probability-core/rejection_sampler.hpp>
#include <math-core/utils.hpp>
#include <iostream>


using namespace probability_core;


class gamma_pdf
{
public:
  gamma_distribution_t gamma;
  
  gamma_pdf( const gamma_distribution_t& g )
    : gamma(g)
  {}

  inline double operator()( const double& x )
  {
    return pdf( x, gamma );
  }
};

int main()
{
  std::vector<double> samples, rejection_samples, as_samples;
  
  gamma_distribution_t gamma;
  gamma.shape = 2;
  gamma.rate = 0.5;
  gamma_pdf pdf( gamma );
  uniform_sampler_within_range uniform( 0, 1000 );
  
  rejection_sampler_status_t rstatus;
  for( std::size_t i = 0; i < 1000000; ++i ) {
    samples.push_back( sample_from( gamma ) );
    rejection_samples.push_back( rejection_sample<double>( pdf, uniform, rstatus ) );
    as_samples.push_back( autoscale_rejection_sample<double>( boost::function1<double,double>(pdf), 0.0, 1000.0, rstatus ) );
    if( i % 10000 == 0 ) {
      std::cout << "sampled: " << samples[ samples.size()-1 ] << "  " << rejection_samples[ rejection_samples.size() - 1 ] << "  " << as_samples[ as_samples.size() -1 ]<< std::endl;
    }
  }

  // ok, compute the mean and variance and see if they are close
  double samples_mean = mean( samples );
  double rejection_mean = mean( rejection_samples );
  double as_mean = mean( as_samples );
  double samples_variace = variance( samples );
  double rejection_variance = variance( rejection_samples );
  double as_variance = variance( as_samples );
  std::cout << "True: " << mean( gamma ) << "  " << variance( gamma ) << std::endl;
  std::cout << "Samples: " << samples_mean << "  " << samples_variace << std::endl;
  std::cout << "Rejection: " << rejection_mean << "  " << rejection_variance << std::endl;
  std::cout << "Autoscale: " << as_mean << "  " << as_variance << std::endl;
}
