
#if !defined( __DISTRIBUTIONS_HPP__ )
#define __DISTRIBUTIONS_HPP__

#include "lcmtypes/probability_core.hpp"
#include <lcmtypes/math_core.hpp>
#include "core.hpp"
#include <math-core/geom.hpp>
#include <math-core/matrix.hpp>
#include <iostream>


//========================================================================

// Description:
// The basic API for distributions.
// There are two basic functions:
//   pdf( x, distribution )
//   sample_from( distribution )
//
// There is also a global random number generator
//   global_rng()

namespace probability_core {


  // Description:
  // The traits for the pdf function
  template< typename T_Input, typename T_Distribution >
  struct pdf_traits {
    typedef double result_type;
  };

  // Description:
  // Returns hte pdf for a particular x and sitribution
  template<typename T_Input, typename T_Distribution>
  typename pdf_traits<T_Input,T_Distribution>::result_type
  pdf( const T_Input& x,
       const T_Distribution& distribution );


  // Description:
  // The traits for the sample_from function
  template< typename T_Distribution >
  struct sample_from_traits {
    // typedef FOO result_type
  };

  // Description:
  // Samples from a given distribution
  template< typename T_Distribution >
  typename sample_from_traits< T_Distribution >::result_type
  sample_from( const T_Distribution& distribution );


  
}




//=========================================================================

namespace probability_core {

  // Description:
  // Gamma Distribution
  
  template<>
  typename pdf_traits<double,gamma_distribution_t>::result_type
  pdf( const double& x,
       const gamma_distribution_t& gamma ) 
  {
    double scale = 1.0 / gamma.rate;
    return gsl_ran_gamma_pdf( x, gamma.shape, scale );
  }

  template<>
  struct sample_from_traits<gamma_distribution_t> {
    typedef double result_type;
  };
  
  template<>
  typename sample_from_traits< gamma_distribution_t >::result_type
  sample_from( const gamma_distribution_t& gamma )
  {
    double scale = 1.0 / gamma.rate;
    return gsl_ran_gamma( global_rng(), gamma.shape, scale );
  }

}


//=========================================================================

namespace probability_core {

  // Description:
  // Beta Distribution
  
  template<>
  typename pdf_traits<double,beta_distribution_t>::result_type
  pdf( const double& x,
       const beta_distribution_t& beta ) 
  {
    
    return gsl_ran_beta_pdf( x, beta.alpha, beta.beta );
  }

  template<>
  struct sample_from_traits<beta_distribution_t> {
    typedef double result_type;
  };
  
  template<>
  typename sample_from_traits< beta_distribution_t >::result_type
  sample_from( const beta_distribution_t& beta )
  {
    return gsl_ran_beta( global_rng(), beta.alpha, beta.beta );
  }

}
  
//=========================================================================

namespace probability_core {

  // Description:
  // Poisson Distribution
  
  template<>
  typename pdf_traits<unsigned int,poisson_distribution_t>::result_type
  pdf( const unsigned int& x,
       const poisson_distribution_t& pos ) 
  {
    return gsl_ran_poisson_pdf( x, pos.lambda );
  }

  template<>
  typename pdf_traits<int,poisson_distribution_t>::result_type
  pdf( const int& x,
       const poisson_distribution_t& pos ) 
  {
    if( x < 0 )
      return 0.0;
    return gsl_ran_poisson_pdf( (unsigned int)x, pos.lambda );
  }

  template<>
  struct sample_from_traits<poisson_distribution_t> {
    typedef unsigned int result_type;
  };
  
  template<>
  typename sample_from_traits< poisson_distribution_t >::result_type
  sample_from( const poisson_distribution_t& pos )
  {
    return gsl_ran_poisson( global_rng(), pos.lambda );
  }

}


//=========================================================================

namespace probability_core {

  // Description:
  // Gaussian Distribution

  using namespace math_core;
  
  template<>
  typename pdf_traits<nd_point_t,gaussian_distribution_t>::result_type
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

  template<>
  struct sample_from_traits<gaussian_distribution_t> {
    typedef nd_point_t result_type;
  };
  
  template<>
  typename sample_from_traits< gaussian_distribution_t >::result_type
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

}
  

//=========================================================================

namespace probability_core {

  // Description:
  // Discrete Distribution
  
  template<>
  typename pdf_traits<int32_t,discrete_distribution_t>::result_type
  pdf( const int32_t& x,
       const discrete_distribution_t& d ) 
  {
    return d.prob[ x ];
  }

  template<>
  struct sample_from_traits<discrete_distribution_t> {
    typedef size_t result_type;
  };
  
  template<>
  typename sample_from_traits< discrete_distribution_t >::result_type
  sample_from( const discrete_distribution_t& d )
  {
    gsl_ran_discrete_t* gd = gsl_ran_discrete_preproc( d.n, &d.prob[0] );
    double sample = gsl_ran_discrete( global_rng(), gd );
    gsl_ran_discrete_free( gd );
    return sample;
  }

}

//=========================================================================

namespace probability_core {
  
  // Description:
  // Negative Binomial Distribution
  
  template<>
  typename pdf_traits<unsigned int,negative_binomial_distribution_t>::result_type
  pdf( const unsigned int& x,
       const negative_binomial_distribution_t& nb ) 
  {
    
    return gsl_ran_negative_binomial_pdf( x, nb.p, nb.r );
  }

  template<>
  struct sample_from_traits<negative_binomial_distribution_t> {
    typedef unsigned int result_type;
  };
  
  template<>
  typename sample_from_traits< negative_binomial_distribution_t >::result_type
  sample_from( const negative_binomial_distribution_t& nb )
  {
    return gsl_ran_negative_binomial( global_rng(), nb.p, nb.r );
  }

}



#endif

