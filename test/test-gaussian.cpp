
#include <probability-core/distributions.hpp>
#include <math-core/io.hpp>
#include <iostream>


using namespace probability_core;
using namespace math_core;
using namespace std;


int main( int argc, char** argv )
{

  // craete a new 2D gaussian distribution
  gaussian_distribution_t gaussian0;
  gaussian0.dimension = 2;
  gaussian0.means = std::vector<double>();
  gaussian0.means.push_back( 10 );
  gaussian0.means.push_back( 20 );
  gaussian0.covariance.rows = 2;
  gaussian0.covariance.cols = 2;
  gaussian0.covariance.num_elements = 4;
  gaussian0.covariance.data = std::vector<double>();
  gaussian0.covariance.data.push_back( 1 );
  gaussian0.covariance.data.push_back( 0.1 );
  gaussian0.covariance.data.push_back( 0.1 );
  gaussian0.covariance.data.push_back( 5 );
  
  // ask for the pdf for x = ( 10, 20 )
  double px = pdf( point( 10, 20 ), gaussian0 );
  cout << "P(" << point( 10,20 ) << ") = " << px << endl;
  
  // sample some points
  for( int i = 0; i < 10; ++i ) {
    nd_point_t sx = sample_from( gaussian0 );
    cout << "~gaussian0: " << sx << endl;
  }

  return 0;
}
