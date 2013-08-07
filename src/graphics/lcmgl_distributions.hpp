
#if !defined( __PROBABILITY_CORE_GRAPHICS_LCMGL_DISTRIBUTIONS_HPP__ )
#define __PROBABILITY_CORE_GRAPHICS_LCMGL_DISTRIBUTIONS_HPP__


#include <probability-core/distribution_utils.hpp>
#include <bot_lcmgl_client/lcmgl.h>
#include <iostream>

namespace probability_core {
  namespace graphics {


    // Description:
    // Draw a distribution to hte lcmgl channel
    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const gaussian_distribution_t& gaussian );
    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const gamma_distribution_t& gamma );
    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const poisson_distribution_t& pos );
    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const beta_distribution_t& beta );
    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const negative_binomial_distribution_t& nb );
    void draw_distribution( bot_lcmgl_t* lcmgl,
			    const discrete_distribution_t& dist );
    
    
    // Description:
    // Draws a distribution by sampling given points
    // and drawing piecewise lines in between
    template< typename T_Distribution >
    void draw_1d_piecewise( bot_lcmgl_t* lcmgl,
			    const T_Distribution& dist,
			    const std::vector<double>& points )
    {
      //std::cout << "1d piecewise " << dist << std::endl;
      lcmglBegin( 0x3 );
      for( std::size_t i = 0; i < points.size(); ++i ) {
	lcmglVertex2d( points[i], pdf(points[i],dist) );
	//std::cout << "    [" << i << "]: pdf(" << points[i] << ") = " << pdf(points[i],dist) << std::endl;
      }
      lcmglEnd();
    }

    // Description:
    // Draws a distribution by sampling given points
    // and drawing steps in between
    template< typename T_Distribution >
    void draw_1d_step( bot_lcmgl_t* lcmgl,
		       const T_Distribution& dist,
		       const std::vector<unsigned int>& points )
    {
      if( points.empty() )
	return;
      
      //std::cout << "1d step " << dist << std::endl;
      lcmglBegin( 0x3 );
      for( std::size_t i = 0; i < points.size() - 1; ++i ) {
	lcmglVertex2d( points[i], pdf(points[i],dist) );
	lcmglVertex2d( points[i+1], pdf(points[i],dist) );
	//std::cout << "   [" << i << "]  (" << points[i] << "," << points[i+1] << "), pdf = " << pdf( points[i], dist ) << std::endl;
      }
      // last point is special
      std::size_t i = points.size() - 1;
      if( points.size() > 1 ) {
	lcmglVertex2d( points[i], pdf(points[i],dist) );
	lcmglVertex2d( points[i] + abs(points[i] - points[i-1]), 
		       pdf(points[i],dist) );
	//std::cout << "   [" << i << "]  (" << points[i] << "," << points[i] + abs(points[i] - points[i-1]) << "), pdf = " << pdf( points[i], dist ) << std::endl;
      } else {
	lcmglVertex2d( points[i], pdf(points[i],dist) );
	lcmglVertex2d( points[i] + 1, pdf(points[i],dist) );
	//std::cout << "   [" << i << "]  (" << points[i] << "," << points[i] + 1 << "), pdf = " << pdf( points[i], dist ) << std::endl;
      }
      lcmglEnd();
    }


  }
}

#endif

