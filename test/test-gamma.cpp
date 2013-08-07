
#include <probability-core/distributions.hpp>
#include <iostream>


using namespace probability_core;
using namespace std;


int main( int argc, char** argv )
{

  // craete a new gamma distribution
  gamma_distribution_t gamma0;
  gamma0.shape = 2.0;
  gamma0.rate = 1.0/2.0;
  
  // ask for the pdf for x = 1.0
  double px = pdf( 1.0, gamma0 );
  cout << "P(1.0) = " << px << endl;
  
  // sample some points
  for( int i = 0; i < 10; ++i ) {
    double sx = sample_from( gamma0 );
    cout << "~gamma0: " << sx << endl;
  }

  return 0;
}
