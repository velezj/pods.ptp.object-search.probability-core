
#if !defined( __P2L_PROBABILITY_CORE_slice_sampler_HPP__ )
#define __P2L_PROBABILITY_CORE_slice_sampler_HPP__

#include "uniform.hpp"
#include "distribution_utils.hpp"
#include <boost/numeric/interval.hpp>
#include <boost/function.hpp>
#include <utility>
#include <iostream>
#include <p2l-common/stat_counter.hpp>
#include <p2l-common/context.hpp>


namespace probability_core {


  //=======================================================================

  // Description:
  // The workspace for a slice sampler.
  // This simply stores the last sampled x location so that
  // we get a true markov chain.
  template<class Support_Type>
  class slice_sampler_workplace_t
  {
  public:

    // Description:
    // Create a new slice sampler workspace that works within
    // the giuven support (domain) as [low,high]
    slice_sampler_workplace_t( const std::pair<Support_Type,Support_Type>& support_range )
      : previous_x(),
	support(support_range)
    {
      reset();
    }

    // Description:
    // Reset the workspace
    void reset()
    { 
      previous_x = sample_from( uniform_distribution( support ) );
    }


    // Description:
    // The previusly selected domain X.
    Support_Type previous_x;
    std::pair<Support_Type,Support_Type> support;
  };
  
  //=======================================================================
  
  // Description:
  // Sample from a 1D function using the Slice Sampling method
  template<class Support_Type, class Range_Type = double>
  Support_Type
  slice_sample_1d( const boost::function<Range_Type (const Support_Type&)>& f,
		   slice_sampler_workplace_t<Support_Type>& workplace,
		   double initial_slice_fraction = 0.00001 )
  {
    P2L_COMMON_push_function_context();
    
    using namespace boost::numeric;
    using namespace interval_lib;
    typedef interval<Support_Type> slice_t;
    
    // get the height of the previous x, then sample uniformly for a level
    Range_Type max_y = f( workplace.previous_x );
    int count_level_finds = 0;
    while( max_y == 0 ) {
      if( (count_level_finds + 1) % 1000 == 0 ) {
	std::cout << "slice_sample_1d | bad max_y " << max_y << " at " << workplace.previous_x << " [" << count_level_finds << "]" << std::endl;
      }
      workplace.reset();
      max_y = f( workplace.previous_x );
      ++count_level_finds;
    }
    if( count_level_finds > 0 ) {
      //std::cout << "final slice_sample_1d | bad max_y " << max_y << " at " << workplace.previous_x << " [" << count_level_finds << "]" << std::endl;
    }
    Range_Type level_y = sample_from( uniform_distribution<Range_Type>( 0.0, max_y ) ); 

    //std::cout << "slice_sample_1d | prev_x: " << workplace.previous_x << " max_y: " << max_y << " level: " << level_y << std::endl;
    
    // Find the window (the slice)
    // We will initialize it to a small section around the 
    // x, and then we will double it in a random direction until
    // the ends are at points below level_y
    Support_Type w = initial_slice_fraction * ( workplace.support.second - workplace.support.first );
    slice_t slice = slice_t(workplace.previous_x - w, workplace.previous_x + w);
    slice_t max_slice = slice_t( workplace.support.first, workplace.support.second );

    STAT_LVL( trace, "max_y", max_y );
    STAT_LVL( debug, "level_y", level_y );
    STAT_LVL( trace, "previous_x", workplace.previous_x );
    STAT_LVL( trace, "w", w );
    STAT_LVL( trace, "max_slice.low", workplace.support.first );
    STAT_LVL( trace, "max_slice.high", workplace.support.second );
    STAT_LVL( trace, "max_slice.span", (workplace.support.second - workplace.support.first ) );

    Range_Type low_level, high_level;
    size_t count_window_doubles = 0;
    size_t max_doubling = 100;
    do {
      
      // double the slice in a random direction
      Range_Type len = slice.upper() - slice.lower();
      if( flip_coin() ) {
	slice = slice_t( slice.lower() - len, slice.upper() );
      } else {
	slice = slice_t( slice.lower(), slice.upper() + len );
      }
      if( slice.lower() < max_slice.lower() ) {
	slice = slice_t( max_slice.lower(), slice.upper() );
      }
      if( slice.upper() > max_slice.upper() ) {
	slice = slice_t( slice.lower(), max_slice.upper() );
      }
      

      // compute hte levels at the end
      low_level = f( slice.lower() );
      high_level = f( slice.upper() );

      //std::cout << "  slice: [" << slice.lower() << " , " << slice.upper() << "]" << "  (" << low_level << "," << high_level << ") " << std::endl;

      ++count_window_doubles;

    } while( low_level >= level_y 
	     && high_level >= level_y 
	     &&
	     ( slice.upper() < max_slice.upper() 
	       || slice.lower() > max_slice.lower() ) 
	     && count_window_doubles < max_doubling );
    
    // Ok, we've found the slice, now uniformly sample within it
    std::pair<Support_Type,Support_Type> slice_range( slice.lower(),
						      slice.upper() );
    Support_Type sampled_x = sample_from( uniform_distribution( slice_range ) );
    Range_Type sampled_y = f( sampled_x );

    STAT_LVL( trace, "slice_range.low", slice.lower() );
    STAT_LVL( trace, "slice_range.high", slice.upper() );
    STAT_LVL( trace, "slice_range.span", (slice.upper() - slice.lower()) );

    size_t count_shrinks = 0;
    size_t max_shrinks = 100;
    while( sampled_y < level_y && count_shrinks < max_shrinks) {
      
      // we need to shrink our slice since we sampled a bad
      // location (not on the level_y )
      if( sampled_x < workplace.previous_x ) {
	slice = slice_t( sampled_x, slice.upper() );
      } else {
	slice = slice_t( slice.lower(), sampled_x );
      }
      
      // ok, resample an x from the newly shrunk slice
      slice_range = std::make_pair( slice.lower(),
				    slice.upper() );
      sampled_x = sample_from( uniform_distribution( slice_range ) );
      sampled_y = f( sampled_x );
      
      //std::cout << "  shrink slice: [" << slice.lower() << " , " << slice.upper() << "]  sampled: " << sampled_x << " (level: " << sampled_y << ")" << std::endl;

      ++count_shrinks;
    }
    
    // store this as the previous x in hte workplace
    workplace.previous_x = sampled_x;

    //std::cout << "Doubling: " << count_window_doubles << "\t Shrinks: " << count_shrinks << std::endl;
    
    STAT_LVL( debug, "num-doubling", (double)count_window_doubles );
    STAT_LVL( debug, "num-shrinks", (double)count_shrinks );
    
    // return the sample
    return sampled_x;
  }

  //=======================================================================

  // Description:
  // Sample from a joint using a joint slice sampling method.
  math_core::nd_point_t
  slice_sample( const boost::function<double (const math_core::nd_point_t&)>& f,
		slice_sampler_workplace_t<math_core::nd_point_t>& workplace,
		double initial_slice_fraction = 0.00001 );  

  //=======================================================================
  

}

#endif
