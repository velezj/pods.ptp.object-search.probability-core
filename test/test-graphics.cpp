
#include <probability-core-graphics/lcmgl_distributions.hpp>


using namespace probability_core;
using namespace probability_core::graphics;


int main( int argc, char** argv )
{

  // create lcm and bot_lcmgl
  lcm_t* lcm = lcm_create(NULL);
  bot_lcmgl_t* lcmgl = bot_lcmgl_init( lcm, "TEST-GRAPHICS" );
  
  
  // draw a gamma and a poisson
  gamma_distribution_t gamma = { 6, 2 };
  poisson_distribution_t pos = { 3.0 };
  beta_distribution_t beta = { 0.5, 0.5 };
  gaussian_distribution_t gaussian;
  gaussian.dimension = 1;
  gaussian.means.push_back( 3 );
  gaussian.covariance.rows = 1;
  gaussian.covariance.cols = 1;
  gaussian.covariance.num_elements = 1;
  gaussian.covariance.data.push_back( 1 );

  lcmglColor3f( 0, 0, 1 );
  draw_distribution( lcmgl, beta );
  
  lcmglPushMatrix();
  lcmglScalef( 1.0, 10, 1.0 );
  lcmglColor3f( 1, 0, 0 );  
  draw_distribution( lcmgl, gamma );
  lcmglColor3f( 0, 1, 0 );
  draw_distribution( lcmgl, pos );
  lcmglColor3f( 1, 0, 1 );
  draw_distribution( lcmgl, gaussian );
  lcmglPopMatrix();
  
  bot_lcmgl_switch_buffer( lcmgl );
  
  while( lcm_handle(lcm) == 0 ) {
  }

  return 0;
}
