
#include "lcmgl_distributions.hpp"
#include <cmath>


namespace probability_core {
  namespace graphics {


    //=======================================================================

    // Description:
    // Given a variance and mean,
    // and a set of points to sample in each sigma level (0,1,2,3..)
    // returns points evenly within each sigma level 
    // Does not return poitns outside of given min,max
    std::vector<double> variance_spread_points
    (const std::vector<unsigned long> num_per_level,
     const double& variance,
     const double& mean,
     const double& min, const double& max)
    {
      std::vector<double> points;
      double sigma = std::sqrt(variance);

      // negative sigma
      for( long i = num_per_level.size() - 1; i >= 0; --i ) {
	double step = sigma / num_per_level[i];
	for( long k = num_per_level[i] - 1; k >= 0; --k ) {
	  double p = mean - ( i * sigma + k * step );
	  if( p >= min && p <= max ) {
	    //std::cout << " varpoint- p: " << p << " i=" << i << " sigma=" << sigma << " k=" << k << " mean=" << mean << " step=" << step << std::endl;
	    points.push_back( p );
	  }
	}
      }

      // positive sigma
      for( long i = 0; i < num_per_level.size(); ++i ) {
	double step = sigma / num_per_level[i];
	for( long k = 0; k < num_per_level[i]; ++k ) {
	  double p = mean + ( i * sigma + k * step );
	  if( p >= min && p <= max ) {
	    //std::cout << " varpoint+ p: " << p << " i=" << i << " sigma=" << sigma << " k=" << k << " mean=" << mean << " step=" << step << std::endl;
	    points.push_back( p );
	  }
	}
      }

      return points;
    }


    //=======================================================================

    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const gamma_distribution_t& gamma )
    {

      double m = mean(gamma);
      double var = variance(gamma);
      std::vector<unsigned long> var_samples;
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      var_samples.push_back( 5 );
      var_samples.push_back( 5 );
      std::vector<double> points = variance_spread_points( var_samples, var, m, 0, 100000 );
      draw_1d_piecewise( lcmgl, gamma, points );
    }
    

    //=======================================================================

    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const beta_distribution_t& beta )
    {

      double m = mean(beta);
      double var = variance(beta);
      std::vector<unsigned long> var_samples;
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      var_samples.push_back( 5 );
      var_samples.push_back( 5 );
      std::vector<double> points = variance_spread_points( var_samples, var, m, 0, 1 );
      draw_1d_piecewise( lcmgl, beta, points );
    }

    //=======================================================================

    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const poisson_distribution_t& pos )
    {
      double m = mean(pos);
      double var = variance(pos);
      std::vector<unsigned long> var_samples;
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      std::vector<double> points_d = variance_spread_points( var_samples, var, m, 0, 100000 );
      std::vector<unsigned int> points( points_d.begin(),
					points_d.end() );
      draw_1d_step( lcmgl, pos, points );
    }


    //=======================================================================

    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const negative_binomial_distribution_t& nb )
    {
      double m = mean(nb);
      double var = variance(nb);
      std::vector<unsigned long> var_samples;
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      var_samples.push_back( 10 );
      std::vector<double> points_d = variance_spread_points( var_samples, var, m, 0, 100000 );
      std::vector<unsigned int> points( points_d.begin(),
					points_d.end() );
      draw_1d_step( lcmgl, nb, points );
    }


    //=======================================================================

    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const gaussian_distribution_t& gaussian )
    {
      // we onl can draw 1D gaussians. Any higher dimension is not going
      // to be drawn
      if( gaussian.dimension > 1 ) {
	return;
      }

      double m = mean(gaussian).coordinate[0];
      double var = variance(gaussian);
      std::vector<unsigned long> var_samples;
      var_samples.push_back( 28 );
      var_samples.push_back( 18 );
      var_samples.push_back( 3 );
      var_samples.push_back( 1 );
      std::vector<double> x = variance_spread_points( var_samples, var, m, -10000, 100000 );
      
      // draw the line loop
      lcmglBegin( 0x3 );
      for( std::size_t i = 0; i < x.size(); ++i ) {
	math_core::nd_point_t p = math_core::point( x[i] );
	lcmglVertex2d( x[i], pdf( p, gaussian ) );
      }
      lcmglEnd();
    }
    

    //=======================================================================

    void
    draw_distribution( bot_lcmgl_t* lcmgl,
		       const discrete_distribution_t& dist )
    {
      std::vector<unsigned int> points;
      for( std::size_t i = 0; i < dist.n; ++i ) {
	points.push_back(i);
      }
      draw_1d_step( lcmgl, dist, points );
    }

    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    //=======================================================================
    


  }
}
