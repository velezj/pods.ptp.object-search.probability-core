
#if !defined( __PROBABILITY_CORE_GAUSSIAN_HPP__ )
#define __PROBABILITY_CORE_GAUSSIAN_HPP__

#include "types.hpp"
#include <math-core/types.hpp>
#include <math-core/geom.hpp>
#include <math-core/math_function.hpp>
#include <math-core/matrix.hpp>
#include <boost/optional.hpp>

namespace probability_core {


  class gaussian_distribution_t
  {
  public:
    size_t                dimension;
    std::vector< double > means;
    math_core::dense_matrix_t covariance;
    Eigen::MatrixXd inverse_covariance() const {
      if( !_inverse_covariance ) {
	const_cast<gaussian_distribution_t*>(this)->_inverse_covariance 
	  = to_eigen_mat( covariance ).inverse();
      }
      return *_inverse_covariance;
    }
  protected:
     boost::optional<Eigen::MatrixXd> _inverse_covariance;
  };
  
  // Description:
  // Gaussian Distribution

  using namespace math_core;
  
  double
  pdf( const nd_point_t& x,
       const gaussian_distribution_t& gaussian );

  nd_point_t
  sample_from( const gaussian_distribution_t& gaussian );

  nd_point_t
  mean( const gaussian_distribution_t& d );

  nd_point_t
  mode( const gaussian_distribution_t& d );
  
  dense_matrix_t 
  covariance( const gaussian_distribution_t& d );

  double
  variance( const gaussian_distribution_t& d ); // ONLY fo 1D Gaussians!!!  

  double
  cdf( const nd_point_t& x, 
       const gaussian_distribution_t& d );
}


// Description:
// math_function_t interface for PDFs
namespace functions {


  // Description:
  // Internal implementation functor for the gaussian pdf
  namespace impl {
    class gaussian_pdf_t
      : public math_core::math_function_t<math_core::nd_point_t,
					  double>
    {
    public:
      probability_core::gaussian_distribution_t g;
      gaussian_pdf_t( const probability_core::gaussian_distribution_t& g )
	: g(g)
      {}
      virtual ~gaussian_pdf_t() {};
      double operator() ( const math_core::nd_point_t& x ) const
      {
	return probability_core::pdf( x, g );
      }
    };
  }

   
  // Description:
  // API function to get a guassian pdf as a math_function_t
  boost::shared_ptr<math_core::math_function_t<math_core::nd_point_t,
					       double> >
  gaussian_pdf( const probability_core::gaussian_distribution_t& g );
}

#endif

