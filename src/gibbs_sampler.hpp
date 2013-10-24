
#if !defined( __P2L_PROBABILITY_CORE_gibbs_sampler_HPP__ )
#define __P2L_PROBABILITY_CORE_gibbs_sampler_HPP__


#include <vector>
#include <boost/function.hpp>

namespace probability_core {


  //========================================================================

  class gibbs_sampler_workspace_t
  {
  public:

    // Description:
    // Creates a new gibbs sampler workspace with
    // the given parameters set
    gibbs_sampler_workspace_t
    ( const std::vector<double>& params,
      const std::vector<boost::function<double (const boost::function<double (double)>&)> >& samplers )
      : parameters( params ),
	samplers( samplers)
    {}

    // Description:
    // Reset the parameters to given
    void reset( const std::vector<double>& params )
    {
      parameters = params;
    }

    // Description:
    // The current set of parameter values
    std::vector<double> parameters;

    // Description:
    // The samplers to use to sample from conditional distributions.
    // A sampler is jsut a function whcih returns a sample given a 
    // distribution function (or conditional)
    std::vector<boost::function<double (const boost::function<double (double)>&)> > samplers;
  };


  //========================================================================

  class gibbs_sampler_joint_t
  {
  public:

    // Description:
    // Creates a new gibbs sampler joint from a complete set of
    // conditional functions of the parameters.
    gibbs_sampler_joint_t
    ( const size_t& num_params, 
      const boost::function< boost::function<double (const double&) > ( const size_t&, const std::vector<double>& ) >& conditionals )
      : _num_free_parameters( num_params ), 
	_conditional_func( conditionals )
    {}

    // Description:
    // The number of free parameters for this joint function.
    size_t num_free_parameters() const
    {
      return _num_free_parameters;
    }
    
    
    // Description:
    // The conditional density for all but hte given parameter index being
    // fixed at the given values
    boost::function<double (const double&)>
    conditional( const size_t& index,
		 const std::vector<double>& parameters ) const
    {
      return _conditional_func( index, parameters );
    }

  protected:

    size_t _num_free_parameters;

    // Description:
    // The complete set of conditionals used for gibbs sampling
    // This is a function which returns the conditional distribution
    // given tall the parameters and the index of the conditional to
    // calculate (the index of the only free parameter left)
    boost::function< boost::function<double (const double&) > ( const size_t&, const std::vector<double>& ) > _conditional_func;
			 
  };

  //========================================================================


  // Description:
  // Sample the given joint distribution using gibbs sampling.
  std::vector<double>
  gibbs_sample( const gibbs_sampler_joint_t& joint,
		gibbs_sampler_workspace_t& workspace );
  
  //========================================================================


}

#endif

