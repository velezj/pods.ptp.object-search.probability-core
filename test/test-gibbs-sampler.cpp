
#include <probability-core/gibbs_sampler.hpp>
#include <probability-core/slice_sampler.hpp>
#include <iostream>
#include <boost/bind.hpp>



using namespace probability_core;


double func_x( const double& x, const double& y )
{
  return 1.0;
}

double func_y( const double& y, const double& x )
{
  return 1.0 / ( abs( x - y ) + 1.0 );
}

boost::function<double (const double&)>
conditionals( const size_t index,
	      const std::vector<double>& parameters )
{
  if( index == 0 ) {
    return boost::bind( func_x, _1, parameters[1] );
  } else {
    return boost::bind( func_y, _1, parameters[0] );
  }
}


int main( int argc, char** argv )
{

  std::vector<double> params;
  params.push_back( 0.0 );
  params.push_back( 5.0 );
  std::pair<double,double> support = std::make_pair( -10.0, 10.0 );
  std::vector<slice_sampler_workplace_t<double> > slice_workspace;
  slice_workspace.push_back( slice_sampler_workplace_t<double>( support ) );
  slice_workspace.push_back( slice_sampler_workplace_t<double>( support ) );
  std::vector< boost::function< double (const boost::function<double (double)>&)> > samplers;
  samplers.push_back( boost::bind( slice_sample_1d<double,double>, _1, slice_workspace[0], 0.001 ) );
  samplers.push_back( boost::bind( slice_sample_1d<double,double>, _1, slice_workspace[1], 0.001 ) );
  gibbs_sampler_workspace_t workspace( params, samplers );
  gibbs_sampler_joint_t joint( 2, &conditionals );
  
  // ok, now sample using gibbs
  size_t num_samples = 10;
  std::vector< std::vector<double> > samples;
  for( size_t i = 0; i < num_samples; ++i ) {
    std::vector<double> sample = gibbs_sample( joint, workspace );
    samples.push_back( sample );
    std::cout << "( " << sample[0] << " , " << sample[1] << " )" << std::endl;
  }
  
  return 0;
}
