
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
			       sigma ) 
    * gsl_ran_gaussian_pdf( params[0], 1.0 );
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
    { { 0.3, 0.1 }, 
      { 0.7, 0.1 },
      { 0.5, 0.1 } };

  std::vector<std::vector<double> > mle_params;
  
  GEM_stopping_criteria_t stop;
  stop.max_iterations = 10000;
  
  std::function< double(const math_core::nd_point_t& single_data,
			const std::vector<double>& params) > 
    test_lik_f = &test_lik_gaus;
  run_GEM_mixture_model_MLE_numerical( stop,
				       data,
				       init_params,
				       test_lik_f,
				       mle_params);
 
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
  }
  std::cout << std::endl;
}
