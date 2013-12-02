
#if !defined( __P2L_PROBABILITY_CORE_bumps_HPP__ )
#define __P2L_PROBABILITY_CORE_bumps_HPP__

#include <boost/function.hpp>
#include <limits>
#include <utility>
#include <random>
#include "uniform.hpp"
#include "distribution_utils.hpp"
#include <iostream>


namespace probability_core {

  //====================================================================

  // Description:
  // Implementation of is_inside for a pair range
  template<typename T>
  bool is_inside( const T& x,
		  const std::pair<T,T>& range )
  {
    return ( x >= range.first &&
	     x <= range.second );
  }

  //====================================================================

  template<typename T_Domain, typename T>
  struct find_bump_status_t
  {
    size_t iterations;
    T bump_height;
    T_Domain bump_location;
    double seconds;
    std::vector<T_Domain> trace;
    find_bump_status_t()
      : iterations(0),
	bump_height(),
	bump_location(),
	seconds( std::numeric_limits<double>::quiet_NaN() ),
	trace()
    {
    }
  };

  //====================================================================

  // Description:
  // Find a particular "bump" in a function when starting at the
  // given location.
  template< typename T_Domain, 
	    typename T = double, 
	    typename T_Dist = double, 
	    typename T_Support = std::pair<T_Domain,T_Domain> >
  T_Domain
  find_single_bump
  ( const boost::function1<T,T_Domain>& f,
    const T_Domain& start_x,
    const T_Support& support,
    const T_Dist& max_step_size,
    const T_Dist& min_step_size,
    const boost::function3<std::vector<T_Domain>,T_Domain,T_Dist,T_Support>& neighborhood_f,
    const size_t& max_iterations,
    find_bump_status_t<T_Domain,T>& status )
  {
    status.trace.push_back( start_x );
    T_Domain last_x = start_x;
    T last_x_value = f( start_x );
    //std::cout << "find_single_bump(): " << last_x << " = " << last_x_value << std::endl;
    for( ; status.iterations < max_iterations;
	 ++status.iterations ) {

      // get the neighborhood
      std::vector<T_Domain> neighbors 
	= neighborhood_f( last_x, max_step_size, support );
      assert( neighbors.size() > 0 );

      // filter out neighbors outside of the support
      auto outside_support = [support] ( const T_Domain& a ) {
	return !is_inside( a, support );
      };
      neighbors.erase( std::remove_if( neighbors.begin(),
				       neighbors.end(),
				       outside_support ),
		       neighbors.end() );
      
      // calculate the neighbors values
      std::vector<T> values;
      for( auto x : neighbors ) {
	values.push_back( f(x) );
      }
      assert( neighbors.size() == values.size() );

      //std::cout << "  neigh: ";
      //for( size_t i = 0; i < neighbors.size(); ++i ) {
      //  std::cout << neighbors[i] << "=" << values[i] << "   ";
      //}
      //std::cout << std::endl;

      // find the best one
      boost::optional<T_Domain> max_loc;
      boost::optional<T> max_val;
      if( neighbors.empty() == false ) {
	max_loc.reset( neighbors[0] );
	max_val.reset( values[0] );
	for( size_t i = 1; i < values.size(); ++i ) {
	  if( values[i] > max_val ) {
	    max_loc.reset( neighbors[i] );
	    max_val.reset( values[i] );
	  }
	}
      }

      // how does this step compre to our previous one?
      if( max_val && max_loc && (*max_val) > last_x_value ) {

	//std::cout << "  max neigh: " << *max_loc << "=" << *max_val << std::endl;
	
	// this was a good step, yay!
	last_x = *max_loc;
	last_x_value = *max_val;
	status.trace.push_back( last_x );

      } else {

	// hmm, not a good step, let's see if we can 
	// shrink our step size
	if( max_step_size > 2.0 * min_step_size ) {

	  // yay, recurse with a smaller step size
	  return find_single_bump
	    ( f, 
	      last_x,
	      support,
	      max_step_size / 2.0,
	      min_step_size,
	      neighborhood_f,
	      max_iterations,
	      status );

	} else {

	  //std::cout << "  found: " << last_x << "=" << last_x_value << std::endl;
	  // ok, we cannot shrink, so we actually are at the bump
	  return last_x;

	}

      }

    }

    //std::cout << "  giving up (iters): " << last_x << "=" << last_x_value << std::endl;

    // getting here means we used up all our iterations
    return last_x;

  }

  //====================================================================

  // Description:
  // Tries to find a set of bumps of a function by calling find_single_bump
  // from random start points in the support space.
  template< typename T_Domain, 
	    typename T = double, 
	    typename T_Dist = double, 
	    typename T_Support = std::pair<T_Domain,T_Domain> >
  std::vector< T_Domain >
  find_bumps_using_restarts
  ( const boost::function1<T,T_Domain>& f,
    const T_Support& support,
    const size_t& num_restarts,
    const T_Dist& max_step_size,
    const T_Dist& min_step_size,
    const boost::function3<std::vector<T_Domain>,T_Domain,T_Dist,T_Support>& neighborhood_f,
    const size_t& max_single_restart_iterations,
    const boost::function0<T_Domain>& start_location_sampler,
    const boost::function2<T_Dist,T_Domain,T_Domain>& domain_distance_f,
    std::vector<find_bump_status_t<T_Domain,T> >& status )
  {
    std::vector<T_Domain> bumps;

    // std::streamsize org_prec = std::cout.precision();
    // std::cout.precision( 20 );
    // std::cout << "find_bumps_using_restarts(): min_step_size=" << min_step_size << std::endl;
    
    // This function is pretty simple, just
    // sample a new start location and call find_single_bump.
    for( size_t i = 0; i < num_restarts; ++i ) {
      
      find_bump_status_t<T_Domain,T> single_status;
      T_Domain start = start_location_sampler();
      T_Domain bump 
	= find_single_bump<T_Domain, T, T_Dist, T_Support >
	( f,
	  start,
	  support,
	  max_step_size,
	  min_step_size,
	  neighborhood_f,
	  max_single_restart_iterations,
	  single_status );

      // std::cout << "  found bump: " << bump << " ";


      // we need to chekc if the found bump is within 2*min_step_size
      // of any other previsouly found bump, in which case we decide it
      // is *the same* bump and do not store a second copy in the resulting
      // vector of bumps
      auto same_bump_predicate =
	[bump,domain_distance_f,min_step_size](const T_Domain& x ) 
	{
	  return domain_distance_f( x, bump ) <= 2.0 * min_step_size;
	};
      if( std::none_of( bumps.begin(), bumps.end(), same_bump_predicate ) ) {

	// a new bump was found, store it and it's trace
	status.push_back( single_status );
	bumps.push_back( bump );

	// std::cout << " [stored]!";
      }

      // std::cout << std::endl;
    }
    
    // std::cout.precision( org_prec );

    return bumps;
  }

  //====================================================================

  // Description:
  // Neghborhood sampling functions for find_*_bump
  namespace neighborhood {

    //====================================================================

    // Description:
    // The simplest, traditional neighborhood function
    // which samples aroudn a radius a given number of points
    // at the given distance
    template<typename T_Domain, 
	     typename T_Dist = double,
	     typename T_Support = std::pair<T_Domain,T_Domain> >
    struct uniform_radius
    {
      size_t num_samples;
      uniform_radius( const size_t& num_samples = 0 ) 
	: num_samples( num_samples )
      {
	if( this->num_samples < 1 ) {
	  this->num_samples = 2;
	}
      }
      
      std::vector<T_Domain>
      operator() ( const T_Domain& x,
		   const T_Dist& r,
		   const T_Support& /* support */ )
      {
	std::vector<T_Domain> neigh;
	if( num_samples == 1 ) {
	  if( flip_coin() ) {
	    neigh.push_back( x + r );
	  } else {
	    neigh.push_back( x - r );
	  }
	} else {
	  neigh.push_back( x + r );
	  neigh.push_back( x - r );
	}
	return neigh;
      }
    };

    //====================================================================
    

  }

  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  //====================================================================
  
  

}

#endif

