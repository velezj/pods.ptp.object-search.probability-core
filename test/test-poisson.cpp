
#include <probability-core/distributions.hpp>
#include <iostream>


using namespace probability_core;
using namespace std;


int main( int argc, char** argv )
{

  // craete a new poisson distribution
  poisson_distribution_t pos0;
  pos0.lambda = 2.0;
  
  // ask for the pdf for x = 1.0
  double px = pdf( 1, pos0 );
  cout << "P(1) = " << px << endl;
  
  // sample some points
  for( int i = 0; i < 10; ++i ) {
    unsigned int sx = sample_from( pos0 );
    cout << "~pos0: " << sx << endl;
  }

  return 0;
}
