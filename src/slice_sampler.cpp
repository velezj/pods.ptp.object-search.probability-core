
#include "slice_sampler.hpp"
#include <math-core/geom.hpp>
#include <math-core/io.hpp>
#include <iostream>

using namespace math_core;

namespace probability_core {

  //========================================================================

  struct multi_slice_t {
    nd_point_t l,u;
    multi_slice_t( const nd_point_t& a, const nd_point_t& b )
      : l(a), u(b)
    {}
    nd_point_t lower() const { return l; }
    nd_point_t upper() const { return u; }
    std::pair<nd_point_t,nd_point_t> pair() const {
      return std::make_pair( l, u );
    }
  };
  
  nd_point_t
  slice_sample( const boost::function<double (const nd_point_t&)>& f,
		slice_sampler_workplace_t<nd_point_t>& workplace,
		double initial_slice_fraction )
  {
    P2L_COMMON_push_function_context();
    typedef multi_slice_t slice_t;
    
    // get the height of the previous x, then sample uniformly for a level
    double max_y = f( workplace.previous_x );

    int count_level_finds = 0;
    int max_level_finds = 10000;
    while( max_y == 0 && count_level_finds < max_level_finds ) {
      std::cout << "slice_sample | bad max_y " << max_y << " at " << workplace.previous_x << " [" << count_level_finds << "]" << std::endl;
      workplace.reset();
      max_y = f( workplace.previous_x );
      ++count_level_finds;
    }
    double level_y = sample_from( uniform_distribution<double>( 0.0, max_y ) ); 

    //std::cout << "slice_sample | prev_x: " << workplace.previous_x << " max_y: " << max_y << " level: " << level_y << std::endl;
    STAT( "max_y", max_y );
    STAT( "level_y", level_y );
    STAT( "num-level-finds", (double)count_level_finds );
    
    
    // Find the window (the slice)
    // We will initialize it to a small section around the 
    // x, and then we will double it in a random direction until
    // the ends are at points below level_y
    nd_vector_t w = initial_slice_fraction * ( workplace.support.second - workplace.support.first );
    slice_t slice = slice_t(workplace.previous_x + (-1.0 * w), workplace.previous_x + w);
    slice_t max_slice = slice_t( workplace.support.first, workplace.support.second );

    //STAT( "w", w );
    //STAT( "initial-slice.low", slice.lower() );
    //STAT( "initial-slice.high", slice.upper() );
    //STAT( "initial-slice.span", ( slice.upper() - slice.lower() ) );
    //STAT( "max-slice.low", max_slice.lower() );
    //STAT( "max-slice.high", max_slice.upper() );
    //STAT( "max-slice.span", ( max_slice.upper() - max_slice.lower() ) );

    double p_level;
    size_t count_window_doubles = 0;
    size_t max_doubling = 100;
    do {
      
      // double the slice in a random direction
      nd_vector_t len = slice.upper() - slice.lower();
      if( flip_coin() ) {
	slice = slice_t( slice.lower() + (-1.0 * len), slice.upper() );
      } else {
	slice = slice_t( slice.lower(), slice.upper() + len );
      }

      // make sure we are still isndei support
      nd_point_t p = slice.lower();
      for( int64_t i = 0; i < p.n; ++i ) {
	if( p.coordinate[i] < max_slice.lower().coordinate[i] ) {
	  p.coordinate[i] = max_slice.lower().coordinate[i];
	}
      }
      slice = slice_t( p, slice.upper() );
      p = slice.upper();
      for( int64_t i = 0; i < p.n; ++i ) {
	if( p.coordinate[i] > max_slice.upper().coordinate[i] ) {
	  p.coordinate[i] = max_slice.upper().coordinate[i];
	}
      }
      slice = slice_t( slice.lower(), p );
      

      // compute a random point in hte slice and see if it
      // is inside level set
      p = sample_from( uniform_distribution( slice.pair() ) );
      p_level = f( p );

      //std::cout << "  slice: [" << slice.lower() << " , " << slice.upper() << "]" << " " << p << " = " << p_level  << std::endl;

      ++count_window_doubles;

    } while( p_level >= level_y && count_window_doubles < max_doubling );
    
    // Ok, we've found the slice, now uniformly sample within it
    nd_point_t sampled_x = sample_from( uniform_distribution( slice.pair() ) );
    double sampled_y = f( sampled_x );
    size_t count_shrinks = 0;
    size_t max_shrinks = 100;
    while( sampled_y < level_y && count_shrinks < max_shrinks ) {
      
      // we need to shrink our slice since we sampled a bad
      // location (not on the level_y )
      nd_point_t low = slice.lower();
      nd_point_t up = slice.upper();
      for( int64_t i = 0; i < sampled_x.n; ++i ) {
	if( sampled_x.coordinate[i] < workplace.previous_x.coordinate[i] ) {
	  low.coordinate[i] = sampled_x.coordinate[i];
	} else {
	  up.coordinate[i] = sampled_x.coordinate[i];
	}
      }
      slice = slice_t( low, up );
      
      sampled_x = sample_from( uniform_distribution( slice.pair() ) );
      sampled_y = f( sampled_x );
      
      //std::cout << "  shrink slice: [" << slice.lower() << " , " << slice.upper() << "]  sampled: " << sampled_x << " (level: " << sampled_y << ")" << std::endl;

      ++count_shrinks;
    }

    // print bad things here
    if( count_shrinks >= max_shrinks ) {
      std::cout << "slice_sampler: max shrinks, doubling: " << count_window_doubles << ", shrinks: " << count_shrinks << std::endl;
    }
    
    // store this as the previous x in hte workplace
    workplace.previous_x = sampled_x;

    //std::cout << "Doubling: " << count_window_doubles << "\t Shrinks: " << count_shrinks << std::endl;
    STAT( "num-doubling", (double)count_window_doubles );
    STAT( "num-shrinks", (double)count_shrinks );
    
    // return the sample
    return sampled_x;
  }


  //========================================================================

}
