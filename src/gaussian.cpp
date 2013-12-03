
#include "gaussian.hpp"
#include "core.hpp"
#include <math-core/matrix.hpp>
#include <math-core/geom.hpp>
#include <stdexcept>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include "toms462.hpp"



namespace probability_core {

  // Description:
  // Gaussian Distribution

  using namespace math_core;
  
  double
  pdf( const nd_point_t& x,
       const gaussian_distribution_t& gaussian ) 
  {
    assert( x.n == gaussian.dimension );
    unsigned long n = gaussian.dimension;
    Eigen::MatrixXd cov = to_eigen_mat( gaussian.covariance );
    nd_point_t mean = point( n, gaussian.means );
    Eigen::VectorXd mean_diff = to_eigen_mat( x - mean );
    Eigen::MatrixXd temp = ( mean_diff.adjoint() * cov.inverse() * mean_diff ); 
    double exponent = - 0.5 * temp(0);
    double norm_t0 = pow( 2.0 * M_PI, - (double)n / 2.0 );
    double norm_t1 = ( 1.0 / sqrt( cov.determinant() ) );
    double norm = norm_t0 * norm_t1;
    return norm * exp( exponent );
  }

  double
  cdf( const nd_point_t& x,
       const gaussian_distribution_t& d )
  {
    assert(x.n <= 2);
    assert( x.n == d.dimension );
    if( x.n > 2 || x.n < 1 )
      throw std::domain_error( "Can only take CDF of 1D or 2D gaussian!");
    if( x.n != d.dimension )
      throw std::domain_error( "Dimensions of point and gaussian do not mathc!" );

    // simple case, 1D gaussian CDF use GSL
    if( x.n == 1 )
      return gsl_cdf_gaussian_P( (x - point(d.means)).component[0],
				 sqrt(d.covariance.data[0]) );

    // Ok, for 2D CDF, we need to standardize the input
    // ( zero mean centered, divide by standard div )
    // THIS is ONLY for DIAGONAL CoVARIANCES

    // SLOW because of FREE? (operatore delete() )
    // assert( is_diagonal( d.covariance ) );
    // if( !is_diagonal( d.covariance ) ) {
    //   throw std::domain_error( "Can only take CDF of 2D gaussian with DIAGONAL convariance!" );
    // }
    // int dim = x.n;
    // Eigen::MatrixXd cov = to_eigen_mat( d.covariance );
    // nd_point_t xo = x + (-1.0 * (point( d.means ) - zero_point(dim)));
    // for( int i = 0; i < dim; ++i ) {
    //   xo.coordinate[i] /= sqrt( cov( i,i ) );
    // }
    // double correlation_p = 0;
    // return bivnor( xo.coordinate[0], xo.coordinate[1], correlation_p );

    double cov0, cov1;
    cov0 = d.covariance.data[0];
    cov1 = d.covariance.data[3];
    double xo0 = x.coordinate[0] - d.means[0];
    double xo1 = x.coordinate[1] - d.means[1];
    xo0 /= sqrt(cov0);
    xo1 /= sqrt(cov1);
    double correlation_p = 0;
    return bivnor( xo0, xo1, correlation_p );
  }

  nd_point_t
  sample_from( const gaussian_distribution_t& gaussian )
  {
    Eigen::MatrixXd cov = to_eigen_mat( gaussian.covariance );
    Eigen::VectorXd mean = to_eigen_mat( point( gaussian.dimension, gaussian.means ) - zero_point( gaussian.dimension ) );

    // decompose covariance into AA' = Cov form
    Eigen::LLT< Eigen::MatrixXd > lltOfCov( cov );
    Eigen::MatrixXd A = lltOfCov.matrixL();
    
    // draw standard normals
    Eigen::VectorXd draws( gaussian.dimension );
    for( int64_t i = 0; i < gaussian.dimension; ++i ) {
      draws( i ) = gsl_ran_gaussian( global_rng(), 1.0 );
    }

    // now add the mean and the decomposed covariance
    Eigen::VectorXd sample = mean + A * draws;
    
    // return the sample as a point
    nd_point_t p;
    p.n = gaussian.dimension;
    p.coordinate = std::vector<double>();
    for( int64_t i = 0; i < gaussian.dimension; ++i ) {
      p.coordinate.push_back( sample(i) );
    }
    return p;
  }

  nd_point_t
  mean( const gaussian_distribution_t& gaussian )
  {
    return point( gaussian.means );
  }

  nd_point_t
  mode( const gaussian_distribution_t& gaussian )
  {
    return mean( gaussian );
  }

  dense_matrix_t
  covariance( const gaussian_distribution_t& gaussian )
  {
    return gaussian.covariance;
  }

  double
  variance( const gaussian_distribution_t& gaussian )
  {
    if( gaussian.dimension == 1 )
      return gaussian.covariance.data[0];
    
    // error!
    throw std::domain_error("Cannot take variance of a N-Dimension Gaussian!!");
  }
}
  

namespace functions {

  boost::shared_ptr<math_core::math_function_t<math_core::nd_point_t,
					       double> >
  gaussian_pdf( const probability_core::gaussian_distribution_t& g ) {
    return 
      boost::shared_ptr<math_core::math_function_t<math_core::nd_point_t,
						   double> >
      ( new impl::gaussian_pdf_t( g ) );
  }
  
}
