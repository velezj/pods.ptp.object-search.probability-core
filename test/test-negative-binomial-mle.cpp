
#include <probability-core/negative_binomial.hpp>
#include <probability-core/distribution_utils.hpp>
#include <math-core/utils.hpp>
#include <iostream>

using namespace probability_core;

int main( int argc, char** argv )
{
  // std::vector<size_t> k =
  //   { 1, 3, 4, 3, 2, 5, 6, 3, 4, 5, 1, 1, 2 };
  std::vector<size_t> k;
  negative_binomial_distribution_t nb0;
  nb0.r = 10;
  nb0.p = 0.1;
  for( size_t i = 0; i < 1e3; ++i ) {
    k.push_back( sample_from( nb0 ) );
  }

  
  negative_binomial_distribution_t nb;
  bool res  = mle( k, nb );
  std::cout << "NB MLE: " << nb << std::endl;
  std::cout << "NB GT : " << nb0 << std::endl;
  std::cout << "NB MLE mean/var: " << mean(nb) << " " << variance(nb) << std::endl;
  std::cout << "K      mean/var: " << math_core::mean(k) << " " 
	    << math_core::variance(k) << std::endl;
  std::cout << "NB GT  mean/var: " << mean(nb0) << " " << variance(nb0) << std::endl;

  // compute lileihoods of data
  double lik_nb0 = 0;
  double lik_nb = 0;
  for( size_t i = 0; i < k.size(); ++i ) {
    size_t k_i = k[i];
    lik_nb0 += pdf( k_i, nb0 );
    lik_nb += pdf( k_i, nb );
  }
  std::cout << "Lik(k | MLE)= " << lik_nb << std::endl;
  std::cout << "Lik(k | GT )= " << lik_nb0 << std::endl;

  // sample som data from each
  std::cout << "MLE ~: ";
  for( size_t i = 0; i < 20; ++i ) {
    std::cout << sample_from( nb ) << ",";
  } 
  std::cout << std::endl;
  std::cout << "GT  ~: ";
  for( size_t i = 0; i < 20; ++i ) {
    std::cout << sample_from( nb0 ) << ",";
  } 
  std::cout << std::endl;


  return 0;
}
