
#include <probability-core/EM.hpp>
#include <probability-core/distribution_utils.hpp>
#include <gsl/gsl_randist.h>
#include <iostream>


using namespace probability_core;
using namespace math_core;

double test_lik( const nd_point_t& x,
		 const std::vector<double>& params )
{
  beta_distribution_t beta;
  beta.alpha = params[0];
  beta.beta = params[1];
  if( beta.alpha <= 0 ) {
    beta.alpha = 1.0e-10;
  }
  if( beta.beta <= 0 ) {
    beta.beta = 1.0e-10;
  }
  return pdf( x.coordinate[0], beta );
}

double test_lik_gaus( const nd_point_t& x,
		      const std::vector<double>& params )
{
  double sigma = params[1];
  if( sigma < 0.2 )
    return 1.0e-10;
  return gsl_ran_gaussian_pdf( x.coordinate[0] - params[0],
			       sigma );
}

int main( int argc, char** argv )
{
  std::vector< nd_point_t > data = 
    { point(0.05), point(0.2), point(0.1), 
      point(0.9), point(0.8), point(0.85), point(0.95) };

  // std::vector< std::vector<double> > init_params =
  //   { { 1, 1 }, 
  //     { 3, 6 } };
  std::vector< std::vector<double> > init_params =
    { { 1.0, 0.1 }, 
      { 4.0, 0.1 },
      { 2.0, 0.1 } };
  std::vector< std::vector<double> > lb =
    { { -10.0, 0 },
      { -10.0, 0 },
      { -10.0, 0 } };
  std::vector< std::vector<double> > ub =
    { { 10.0, 20 },
      { 10.0, 20 },
      { 10.0, 20 } };
  
  
  std::vector<std::vector<double> > mle_params;
  std::vector<double> mle_mixtures;

  GEM_parameters_t gem_params;
  gem_params.max_optimize_iterations = 100;
  gem_params.stop.max_iterations = 100;
  
  std::function< double(const math_core::nd_point_t& single_data,
			const std::vector<double>& params) > 
    test_lik_f = &test_lik_gaus;
  run_GEM_mixture_model_MLE_numerical( gem_params,
				       data,
				       init_params,
				       lb,
				       ub,
				       test_lik_f,
				       mle_params,
				       mle_mixtures);
 
  // beta_distribution_t beta0, beta1;
  // beta0.alpha = mle_params[0][0];
  // beta0.beta = mle_params[0][1];
  // beta1.alpha = mle_params[1][0];
  // beta1.beta = mle_params[1][1];

  // std::cout << "EM MLE: " << std::endl;
  // std::cout << "    " << beta0 
  // 	    << " mean:" << mean(beta0) 
  // 	    << " var:" << variance(beta0) << std::endl;
  // std::cout << "    " << beta1 
  // 	    << " mean:" << mean(beta1) 
  // 	    << " var:" << variance(beta1) << std::endl;  

  std::cout << "EM MLE: ";
  for( size_t i = 0; i < mle_params.size(); ++i ) {
    for( size_t j = 0; j < mle_params[i].size(); ++j ) {
      std::cout << mle_params[i][j] << " , ";
    }
    std::cout << std::endl;
    std::cout << "        ";
  }
  std::cout << std::endl;
  std::cout << "   Mix: ";
  for( size_t i = 0; i < mle_mixtures.size(); ++i ) {
    std::cout << mle_mixtures[i] << " , ";
  }
  std::cout << std::endl;
}
