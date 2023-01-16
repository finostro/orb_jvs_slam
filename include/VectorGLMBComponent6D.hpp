
/*
 * Software License Agreement (New BSD License)
 *
 * Copyright (c) 2022, Felipe Inostroza
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Advanced Mining Technology Center (AMTC), the
 *       Universidad de Chile, nor the names of its contributors may be
 *       used to endorse or promote products derived from this software without
 *       specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE AMTC, UNIVERSIDAD DE CHILE, OR THE COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 * THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#pragma once

#include <vector>
#include <boost/bimap.hpp>
#include <boost/container/allocator.hpp>


#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/BetweenFactor.h>


#include "OrbslamMapPoint.hpp"
#include "OrbslamPose.hpp"


namespace rfs
{


/**
 *  The weight and index of a landmarks from which a measurement can come from
 */
struct AssociationProbability {
	int i; /**< index of a landmark */
	double l; /**< log probability of association*/
};
struct AssociationProbabilities {
	std::vector<int> i; /**< index of a landmark , index of measurement in reverse form*/
	std::vector<double> l; /**< log probability of association*/
};



/**
 * Struct to store a single component of a VGLMB , with its own g2o optimizer
 */
struct VectorGLMBComponent6D {
	typedef gtsam::Point3 PointType;
	typedef gtsam::Pose3 PoseType;
	typedef gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3> StereoMeasurementEdge;
	typedef gtsam::BetweenFactor<gtsam::Pose3> OdometryEdge;



 	gtsam::NonlinearFactorGraph graph;
	gtsam::LevenbergMarquardtOptimizer optimizer;





	std::vector<boost::bimap<int, int, boost::container::allocator<int>>>
			DA_bimap_, prevDA_bimap_; /**< Bimap containing data association hypothesis at time k  */

	std::vector<std::vector<AssociationProbabilities> > DAProbs_; /**< DAProbs_ [k][nz] are is the association probabilities of measurement
	 nz at time k, used for switching using gibbs sampling*/
	std::vector<std::vector<AssociationProbabilities> > reverseDAProbs_; /**< same as before but from landmarks to measurements*/ 

	std::vector<OrbslamPose> poses_;
	std::vector<OrbslamPose*> keyposes_;
	std::vector<OdometryEdge *> odometries_;
	
	std::vector<OrbslamMapPoint> landmarks_;
	std::vector<double> landmarksResetProb_, landmarksInitProb_;

	double logweight_ = -std::numeric_limits<double>::infinity(),
			prevLogWeight_ = -std::numeric_limits<double>::infinity();
	int numPoses_, numPoints_;
	bool reverted_ = false;
	std::vector<std::pair<int, int>> tomerge_; /**< proportional to probability of merge heuristic*/

    int maxpose_; /**< maximum pose to optimize to */




    void saveAsTUM(std::string filename, PoseType base_link_to_cam0_se3){
		std::ofstream file;
		file.open(filename);
		file << std::setprecision(20) ;

		for (int k =0; k<maxpose_; k++){
			auto &pose =poses_[k];
			PoseType p = pose.pose*base_link_to_cam0_se3.inverse();
			auto q = p.rotation().toQuaternion();
			
			file 	<< pose.stamp << " "
					<< p.translation().x() << " "
					<< p.translation().y() << " "
					<< p.translation().z() << " "
					<< q.x() << " "
					<< q.y() << " "
					<< q.z() << " "
					<< q.w() << "\n";
		}

	}





};

}
