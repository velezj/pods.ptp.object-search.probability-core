#if !defined( __PROBABILITY_CORE_uniform_HPP__ )
#define __PROBABILITY_CORE_uniform_HPP__

#include <utility>
#include <boost/random.hpp>
#include <boost/numeric/interval.hpp>
#include <math-core/geom.hpp>
//#include <boost/static_assert.hpp>

namespace probability_core {

  // Description:
  // Uniform Distribution

  template<class Support_Type>
  class uniform_distribution_t
  {
  public:
    std::pair<Support_Type,Support_Type> support;
  };

  template<class Support_Type>
  uniform_distribution_t<Support_Type> 
  uniform_distribution( const std::pair<Support_Type,Support_Type>& support )
  {
    uniform_distribution_t<Support_Type> dist;
    dist.support = support;
    return dist;
  }

  template<class Support_Type>
  uniform_distribution_t<Support_Type> 
  uniform_distribution( const Support_Type& low,
			const Support_Type& high )
  {
    uniform_distribution_t<Support_Type> dist;
    dist.support = std::make_pair( low, high );
    return dist;
  }

  template<class Support_Type>
  uniform_distribution_t<Support_Type> 
  uniform_distribution( const boost::numeric::interval<Support_Type>& support )
  {
    uniform_distribution_t<Support_Type> dist;
    dist.support = std::make_pair( support.lower(),
				   support.upper() );
    return dist;
  }



  
  template<class Support_Type>
  double 
  pdf( const Support_Type& x,
       const  uniform_distribution_t<Support_Type>& u )
  {
    return (1.0) / ( u.support.second - u.support.first );
  } 
  
  template<class Support_Type>
  Support_Type
  sample_from( const uniform_distribution_t<Support_Type>& u )
  {
    //BOOST_STATIC_ASSERT( false, "uniform distribution support type not implemented" );
  }
  template<>
  double
  sample_from( const uniform_distribution_t<double>& u )
  {
    static boost::random::mt19937 engine;
    boost::random::uniform_real_distribution<double> real( u.support.first, u.support.second );
    return real(engine);
  }
  template<>
  math_core::nd_point_t
  sample_from( const uniform_distribution_t<math_core::nd_point_t>& u )
  {
    std::vector<double> c;
    for( int64_t i = 0; i < u.support.first.n; ++i ) {
      c.push_back( sample_from( uniform_distribution( u.support.first.coordinate[i],
						      u.support.second.coordinate[i]) ) );
    }
    return math_core::point( c );
  }
  

  template<class Support_Type>
  Support_Type
  mean( const uniform_distribution_t<Support_Type>& u )
  {
    //BOOST_STATIC_ASSERT( false, "uniform distribution support type not implemented" );
  }
  
  template<class Support_Type>
  double 
  variance( const uniform_distribution_t<Support_Type>& u )
  {
    //BOOST_STATIC_ASSERT( false, "uniform distribution support type not implemented" );
  }


}

#endif

