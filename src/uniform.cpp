
#include "uniform.hpp"

namespace probability_core {


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
  

}
