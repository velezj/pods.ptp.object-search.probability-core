
#include <probability-core/slice_sampler.hpp>
#include <probability-core/distributions.hpp>
#include <math-core/utils.hpp>
#include <math-core/io.hpp>
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

double func_nd( const nd_point_t& x )
{
  beta_distribution_t true_beta;
  true_beta.alpha = 2.0;
  true_beta.beta = 5.0;
  beta_distribution_t true_beta_b;
  true_beta_b.alpha = 0.5;
  true_beta_b.beta = 1.0;
  return pdf( x.coordinate[0], true_beta ) * pdf( x.coordinate[1], true_beta_b );
}

int main( int argc, char** argv )
{

  std::pair<double,double> support( 0, 1.0);
  slice_sampler_workplace_t<double> workplace( support );

  beta_distribution_t true_beta;
  true_beta.alpha = 2.0;
  true_beta.beta = 5.0;
  beta_distribution_t true_beta_b;
  true_beta_b.alpha = 0.5;
  true_beta_b.beta = 1.0;


  std::vector<double> slice_samples;
  std::vector<double> true_samples;

  boost::function<double (const double&)> func_foo_f = &func_foo;



  std::cout << "First sample: " << slice_sample_1d( func_foo_f, workplace ) << std::endl;
  
  int num_samples = 10;
  for( int i = 0; i < num_samples; ++i ) {
    slice_samples.push_back( slice_sample_1d( func_foo_f, workplace ) );
    true_samples.push_back( sample_from( true_beta ) );
  }

  std::cout << "Slice Stats: mean=" << mean( slice_samples ) << " var=" << variance( slice_samples ) << std::endl;
  std::cout << "GSL Stats: mean=" << mean( true_samples ) << " var=" << variance( true_samples ) << std::endl;
  


  // test the mutli-dim slice sampler
  std::pair<nd_point_t,nd_point_t> nd_support( point(0.0,0.0),
					       point(1.0,1.0) );
  slice_sampler_workplace_t<nd_point_t> nd_workplace( nd_support );
  boost::function<double (const nd_point_t&)> func_nd_f = &func_nd;
  
  std::vector<nd_point_t> slice_points;
  std::vector<nd_point_t> true_points;
  num_samples = 1000000;
  for( int i = 0; i < num_samples; ++i ) {
    slice_points.push_back( slice_sample( func_nd_f, nd_workplace, 0.001 ) );
    true_points.push_back( point( sample_from( true_beta ),
				  sample_from( true_beta_b ) ) );
    //std::cout << "Slice Sample: " << slice_points[ i ] << std::endl;
    //std::cout << "True Sampler: " << true_points[ i ] << std::endl;
  }
  nd_point_t mean_slices = slice_points[0];
  nd_point_t mean_true = true_points[0];
  for( int i = 0; i < slice_points.size(); ++i ) {
    for( int j = 0; j < mean_slices.n; ++j ) {
      mean_slices.coordinate[j] += slice_points[i].coordinate[j];
      mean_true.coordinate[j] += true_points[i].coordinate[j];
    }
  }
  for( int j = 0; j < mean_slices.n; ++j ) {
    mean_slices.coordinate[j] /= num_samples;
    mean_true.coordinate[j] /= num_samples;
  }
  std::cout << "Slice ND mean= " << mean_slices << std::endl;
  std::cout << "GSL mean= " << mean_true << std::endl;
  
  
  return 0;
}
