
#include <probability-core/EM.hpp>
#include <probability-core/distribution_utils.hpp>
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

int main( int argc, char** argv )
{
  std::vector< nd_point_t > data = 
    { point(0.05), point(0.2), point(0.1), 
      point(0.9), point(0.8), point(0.85), point(0.95) };

  std::vector< std::vector<double> > init_params =
    { { 1, 1 }, 
      { 3, 6 } };

  std::vector<std::vector<double> > mle_params;
  
  GEM_stopping_criteria_t stop;
  stop.max_iterations = 10;
  
  std::function< double(const math_core::nd_point_t& single_data,
			const std::vector<double>& params) > 
    test_lik_f = &test_lik;
  run_GEM_mixture_model_MLE_numerical( stop,
				       data,
				       init_params,
				       test_lik_f,
				       mle_params);
 
  beta_distribution_t beta0, beta1;
  beta0.alpha = mle_params[0][0];
  beta0.beta = mle_params[0][1];
  beta1.alpha = mle_params[1][0];
  beta1.beta = mle_params[1][1];

  std::cout << "EM MLE: " << std::endl;
  std::cout << "    " << beta0 
	    << " mean:" << mean(beta0) 
	    << " var:" << variance(beta0) << std::endl;
  std::cout << "    " << beta1 
	    << " mean:" << mean(beta1) 
	    << " var:" << variance(beta1) << std::endl;  
}
