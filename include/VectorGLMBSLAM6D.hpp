/*
 * Software License Agreement (New BSD License)
 *
 * Copyright (c) 2014, Keith Leung, Felipe Inostroza
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
#ifndef VECTORGLMBSLAM6D_HPP
#define VECTORGLMBSLAM6D_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Core>
#include <math.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include "ceres/ceres.h"
#include <unordered_map>
#include <math.h>
#include "GaussianGenerators.hpp"
#include "AssociationSampler.hpp"
#include "OrbslamMapPoint.hpp"
#include "OrbslamPose.hpp"
#include "VectorGLMBComponent6D.hpp"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/core/robust_kernel_impl.h"

#include "g2o/types/sba/types_six_dof_expmap.h" // se3 poses

#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>

#include <boost/bimap.hpp>
#include <boost/container/allocator.hpp>
#include <yaml-cpp/yaml.h>

#include "misc/EigenYamlSerialization.hpp"
#include <misc/termcolor.hpp>

#include <external/ORBextractor.h>
#include <external/Converter.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "sophus/se3.hpp"

#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>

#include <visualization_msgs/Marker.h>

#ifdef _PERFTOOLS_CPU
#include <gperftools/profiler.h>
#endif
#ifdef _PERFTOOLS_HEAP
#include <gperftools/heap-profiler.h>
#endif

namespace rfs
{

	static const int orb_th_low = 50;
	static const int orb_th_high = 100;
	static constexpr int thOrbDist = (orb_th_high + orb_th_low) / 2;

	struct bimap_less
	{
		bool operator()(
			const boost::bimap<int, int, boost::container::allocator<int>> x,
			const boost::bimap<int, int, boost::container::allocator<int>> y) const
		{

			return x.left < y.left;
		}
	};
	struct StampedPose{
		double stamp;
		g2o::SE3Quat pose;
	};

	// for profiler
int opt1(g2o::SparseOptimizer *optimizer, int ni){
	return optimizer->optimize(ni);
}
int opt2(g2o::SparseOptimizer *optimizer, int ni){
	return optimizer->optimize(ni);
}

int opt3(g2o::SparseOptimizer *optimizer, int ni){
	return optimizer->optimize(ni);
}


	struct TrajectoryWeight{
		double weight;
		std::vector<StampedPose> trajectory;

    void loadTUM(std::string filename, g2o::SE3Quat base_link_to_cam0_se3, double initstamp){
		std::ifstream file;
		file.open(filename);
		std::string line;
		double t;
		StampedPose stampedPose;
		StampedPose initPose;
		initPose.stamp = -std::numeric_limits<double>::infinity();
		while (file.good() && !file.eof()){
			//std::cout << "read line\n";
			file >> stampedPose.stamp;
			Eigen::Vector3d t;
			file >> t[0];
			file >> t[1];
			file >> t[2];
			Eigen::Quaterniond q;
			file >> q.x();
			file >> q.y();
			file >> q.z();
			file >> q.w();
			stampedPose.pose.setTranslation(t);
			stampedPose.pose.setRotation(q);
			stampedPose.pose = stampedPose.pose*base_link_to_cam0_se3;
			trajectory.push_back(stampedPose);


			if (abs(stampedPose.stamp-initstamp) < abs(initPose.stamp-initstamp) ){
				initPose = stampedPose;
			}
		}
		//std::cout << termcolor::magenta << "mindiff: " <<abs(initPose.stamp-initstamp)  << "\n" << termcolor::reset;
		//initPose = trajectory[0];
		for(auto &p:trajectory){
			p.pose = initPose.pose.inverse() * p.pose ;
		}

	}
	};

	template< class MapType >
void print_map(const MapType & m)
{
    typedef typename MapType::const_iterator const_iterator;
    for( const_iterator iter = m.begin(), iend = m.end(); iter != iend; ++iter )
    {
        std::cout << iter->first << "-->" << iter->second << std::endl;
    }
}



	// Computes the Hamming distance between two ORB descriptors
	static int descriptorDistance(const cv::Mat &a, const cv::Mat &b)
	{
		const int *pa = a.ptr<int32_t>();
		const int *pb = b.ptr<int32_t>();

		int dist = 0;

		for (int i = 0; i < 8; i++, pa++, pb++)
		{
			unsigned int v = *pa ^ *pb;
			v = v - ((v >> 1) & 0x55555555);
			v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
			dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
		}

		return dist;
	}
	inline double dist2loglikelihood(int d){
		double dd=d/255.0;
		static  double a = std::log(1.0/8.0);
		static  double b = std::log(7.0/8.0);
		static  double c = std::log(1.0/2.0);
		double ret =14.0+20.0*(dd*a+(1-dd)*b+c);
		return  ret;
	}

	/**
	 *  \class VectorGLMBSLAM6D
	 *  \brief Random Finite Set  optimization using ceres solver for  feature based SLAM
	 *
	 *
	 *  \author  Felipe Inostroza
	 */
	class VectorGLMBSLAM6D
	{
	public:
		typedef g2o::VertexSBAPointXYZ PointType;
		typedef g2o::VertexSE3Expmap PoseType;
		typedef g2o::EdgeProjectXYZ2UV MonocularMeasurementEdge;
		typedef g2o::EdgeStereoSE3ProjectXYZ StereoMeasurementEdge;
		typedef g2o::EdgeSE3Expmap OdometryEdge;

		typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>> SlamBlockSolver;
		typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
		/**
		 * \brief Configurations for this  optimizer
		 */
		struct Config
		{

			/** The threshold used to determine if a possible meaurement-landmark
			 *  pairing is significant to worth considering
			 */
			double MeasurementLikelihoodThreshold_;

			double logKappa_; /**< intensity of false alarm poisson model*/

			//double PE_; /**<  landmark existence probability*/
			double logExistenceOdds; /**<  landmark existence probability*/

			double PD_; /**<  landmark detection probability*/

			double maxRange_; /**< maximum sensor range */

			int numComponents_;

			int birthDeathNumIter_; /**< apply birth and death every n iterations */

			std::vector<double> xlim_, ylim_;

			int numLandmarks_; /**< number of landmarks per dimension total landmarks will be numlandmarks^2 */

			int numGibbs_;				 /**< number of gibbs samples of the data association */
			int numLevenbergIterations_; /**< number of gibbs samples of the data association */
			int crossoverNumIter_;
			int numPosesToOptimize_; /**< number of poses to optimize data associations */

			int lmExistenceProb_;
			int numIterations_; /**< number of iterations of main algorithm */
			double initTemp_;
			double tempFactor_;
			Eigen::Matrix3d anchorInfo_;		   /** information for anchor edges, should be low*/
			Eigen::Matrix3d stereoInfo_;		   /** information for stereo uvu edges */
			Eigen::Matrix<double, 6, 6> odomInfo_; /** information for odometry edges */

			std::string finalStateFile_;

			g2o::CameraParameters *g2o_cam_params;
			double viewingCosLimit_;

			std::string eurocFolder_, eurocTimestampsFilename_;

			bool use_gui_;


			Eigen::MatrixXd base_link_to_cam0;
			g2o::SE3Quat base_link_to_cam0_se3;
			// camera distortion params

			struct CameraParams
			{
				double fx;
				double fy;
				double cx;
				double cy;
				double k1;
				double k2;
				double p1;
				double p2;

				cv::Size originalImSize, newImSize;
				cv::Mat opencv_distort_coeffs, opencv_calibration;
				cv::Mat M1, M2;

				cv::Mat cv_c0_to_camera;

				Eigen::MatrixXd cv_c0_to_camera_eigen;
			};
			std::vector<CameraParams> camera_parameters_;

			double stereo_baseline;
			double stereo_baseline_f;
			double stereo_init_max_depth;

			Sophus::SE3<float> pose_left_to_right, pose_right_to_left;
			Eigen::Matrix<float, 3, 3> rotation_matrix_left_to_right;
			Eigen::Vector3f mtlr;
			/*
			 * Rectification stuff
			 */
			cv::Mat M1l_, M2l_;
			cv::Mat M1r_, M2r_;

			// add some noise to traj before association
			double perturbTrans ;
			double perturbRot ;


			// ORB params
			struct ORBExtractor
			{
				int nFeatures;
				double scaleFactor;
				int nLevels;
				int iniThFAST;
				int minThFAST;

			} orb_extractor;

			//frames to load
			int minframe;
			int maxframe;
			int staticframes;
			double maxWeightDifference;
		} config;

		/**
		 * Constructor
		 */
		VectorGLMBSLAM6D();

		/** Destructor */
		~VectorGLMBSLAM6D();

		/**
		 *  Load a yaml style config file
		 * @param filename filename of the yaml config file
		 */
		void
		loadConfig(std::string filename);

		/**
		 *  Load imaged from an euroc dataset , folder set in config file
		 *
		 */
		void
		loadEuroc();

		/**
		 * initialize the components , set the initial data associations to all false alarms
		 */
		void initComponents();

		/**
		 * delete  the components at end , to check we are not leaking memory
		 */
		void deleteComponents();

		void deleteLandmarks(VectorGLMBComponent6D &c);
		void deleteMeasurements(VectorGLMBComponent6D &c);

		/**
		 * run the optimization over the possible data associations.
		 * @param numsteps number of iterations in algorithm.
		 */
		void run(int numsteps);
		/**
		 * Do n iterations
		 * @param ni number of iterations of the optimizer
		 */

		void optimize(int ni);

		/**
		 * Select nearest neighbor at the last included time step (maxpose_)
		 * @param c the GLMB component
		 */
		void selectNN(VectorGLMBComponent6D &c);
		/**
		 * Sample n data associations from the already visited group, in order to perform gibbs sampler on each.
		 * @param ni number of iterations of the optimizer
		 */
		void sampleComponents();

		/**
		 *
		 * Update the ROS style poses, run every time after running g2o optimizer
		 * @param c the GLMB component
		 */
		void updatePoses(VectorGLMBComponent6D &c);

		void updateDescriptors(VectorGLMBComponent6D &c);
		void perturbTraj(VectorGLMBComponent6D &c);
		/**
		 * Computes the stereo matches in the orb measurements stored in c
		 * @param pose an orbslam pose with keypoints and descriptors
		 */
		void computeStereoMatches(OrbslamPose &pose);

		/**
		 * Use the data association stored in DA_ to create the graph.
		 * @param c the GLMB component
		 */
		void constructGraph(VectorGLMBComponent6D &c);
		/**
		 * @brief update all the undetected landmark poses
		 * 
		 * @param c the GLMB component
		 */
		void moveBirth(VectorGLMBComponent6D &c);
		/**
		 *
		 * Initialize a VGLMB component , setting the data association to all false alarms
		 * @param c the GLMB component
		 */
		void init(VectorGLMBComponent6D &c);

		/**
		 * Calculate the probability of each measurement being associated with a specific landmark
		 * @param c the GLMB component
		 */
		void updateDAProbs(VectorGLMBComponent6D &c, int minpose, int maxpose);

		/**
		 * Calculate the FoV at each time
		 * @param c the GLMB component
		 */
		void updateFoV(VectorGLMBComponent6D &c);

		/**
		 * update descriptor and viewing angle according to associations
		 * @param c the GLMB component
		 */
		void updateMetaStates(VectorGLMBComponent6D &c);

		/**
		 * Use the probabilities calculated using updateDAProbs to sample a new data association through gibbs sampling
		 * @param c the GLMB component
		 */
		double sampleDA(VectorGLMBComponent6D &c);	
		
		/**
		 * Use the probabilities calculated using updateDAProbs to sample a new data association through gibbs sampling
		 * sample for each existing landmark instead of measurement
		 * @param c the GLMB component
		 */
		double reverseSampleDA(VectorGLMBComponent6D &c);

		/**
		 * Merge two data associations into a third one, by selecting a random merge time.
		 */
		std::vector<boost::bimap<int, int, boost::container::allocator<int>>> sexyTime(
			VectorGLMBComponent6D &c1, VectorGLMBComponent6D &c2);

		/**
		 * Use the probabilities calculated in sampleDA to reset all the detections of a single landmark to all false alarms.
		 * @param c the GLMB component
		 */
		double sampleLMDeath(VectorGLMBComponent6D &c);

		/**
		 * Randomly merge landmarks in order to improve the sampling algorithm
		 * @param c the GLMB component
		 */
		double mergeLM(VectorGLMBComponent6D &c);

		/**
		 * Use the probabilities calculated in sampleDA to initialize landmarks from  false alarms.
		 * @param c the GLMB component
		 */
		double sampleLMBirth(VectorGLMBComponent6D &c);

		/**
		 * Change the data association in component c to the one stored on da.
		 */
		void changeDA(VectorGLMBComponent6D &c,
					  const std::vector<
						  boost::bimap<int, int, boost::container::allocator<int>>> &da);

		/**
		 * print the data association in component c
		 * @param c the GLMB component
		 */
		void printDA(VectorGLMBComponent6D &c, std::ostream &s = std::cout);
		/**
		 * Check data association in graph and bimap match
		 * @param c the GLMB component
		 */
		void checkDA(VectorGLMBComponent6D &c, std::ostream &s = std::cout);
		/**
		 * Check g2o graph is consistent
		 * @param c the GLMB component
		 */
		void checkGraph(VectorGLMBComponent6D &c, std::ostream &s = std::cout);
		/**
		 * Check g2o graph is consistent
		 * @param c the GLMB component
		 */
		void checkNumDet(VectorGLMBComponent6D &c, std::ostream &s = std::cout);
		/**
		 * print the data association in component c
		 * @param c the GLMB component
		 */
		void printDAProbs(VectorGLMBComponent6D &c);
		/**
		 * print the data association in component c
		 * @param c the GLMB component
		 */
		void printFoV(VectorGLMBComponent6D &c);
		/**
		 * Use the data association hipothesis and the optimized state to calculate the component weight.
		 * @param c the GLMB component
		 */
		void calculateWeight(VectorGLMBComponent6D &c);
		/**
		 * Use the new sampled data association to update the g2o graph
		 * @param c the GLMB component
		 */
		void updateGraph(VectorGLMBComponent6D &c);

		void publishMarkers(VectorGLMBComponent6D &c);

		/**
		 * @brief waits for the gui to exit
		 *
		 */
		void waitForGuiClose();

		/**
		 * Calculate the range between a pose and a landmark, to calculate the probability of detection.
		 * @param pose A 6D pose
		 * @param lm A 3D landmark
		 * @return the distance between pose and landmark
		 */
		static double distance(PoseType *pose, PointType *lm);

		/**
		 *  convert stereo measurement into xyz
		 * @param measurement
		 * @param trans_xyz
		 */
		bool cam_unproject(const StereoMeasurementEdge &measurement, Eigen::Vector3d &trans_xyz);


		void initStereoEdge(OrbslamPose &pose, int numMatch);

		/**
		 * initialize a map point using a pose and stereo match
		 * @param pose
		 * @param numMatch
		 * @param lm
		 * @param newId
		 * @return
		 */
		bool initMapPoint(OrbslamPose &pose, int numMatch, OrbslamMapPoint &lm, int newId);

		int nThreads_; /**< Number of threads  */

		VectorGLMBComponent6D initial_component_;

		std::vector<VectorGLMBComponent6D> components_; /**< VGLMB components */
		double bestWeight_ = -std::numeric_limits<double>::infinity();
		std::vector<boost::bimap<int, int, boost::container::allocator<int>>> best_DA_;
		int best_DA_max_detection_time_ = 0; /**< last association time */

		std::map<
			std::vector<boost::bimap<int, int, boost::container::allocator<int>>>,
			TrajectoryWeight>
			visited_;
		
		TrajectoryWeight gt_traj;
		double startingStamp;
		double temp_;
		int minpose_ = 0; /**< sample data association from this pose  onwards*/
		int maxpose_ = 2; /**< optimize only up to this pose */
		int maxpose_prev_ = 2;
		int iteration_ = 0;
		int iterationBest_ = 0;
		double insertionP_ = 0.5;

		// euroc dataset

		// filenames
		std::vector<std::string> vstrImageLeft;
		std::vector<std::string> vstrImageRight;
		// timestamps

		std::vector<double> vTimestampsCam;
		int nImages;

		ORB_SLAM3::ORBextractor *mpORBextractorLeft;
		ORB_SLAM3::ORBextractor *mpORBextractorRight;
		std::vector<float> orbExtractorInvScaleFactors;
		std::vector<float> orbExtractorScaleFactors;

		ros::NodeHandle n;
		ros::Publisher measurements_pub, trajectory_pub, gt_trajectory_pub, map_pub, association_pub;
	};

	//////////////////////////////// Implementation ////////////////////////

	VectorGLMBSLAM6D::VectorGLMBSLAM6D() : n("~")
	{
		nThreads_ = 1;

#ifdef _OPENMP
		nThreads_ = omp_get_max_threads();
#endif
		measurements_pub = n.advertise<sensor_msgs::PointCloud2>("measurements", 1);
		trajectory_pub = n.advertise<nav_msgs::Path>("est_traj", 1);
		gt_trajectory_pub = n.advertise<nav_msgs::Path>("gt_traj", 1);
		association_pub = n.advertise<visualization_msgs::Marker>("association", 1);
		map_pub = n.advertise<sensor_msgs::PointCloud2>("map", 1);
	}

	VectorGLMBSLAM6D::~VectorGLMBSLAM6D()
	{
	}

	void VectorGLMBSLAM6D::publishMarkers(VectorGLMBComponent6D &c)
	{

		//print_map(c.DA_bimap_[0].left);
		// latest pose tf
		ros::spinOnce();
		static tf2_ros::StaticTransformBroadcaster br;
		ros::Time now = ros::Time::now();
		geometry_msgs::TransformStamped transformStamped;

		transformStamped.header.stamp = now;
		transformStamped.header.frame_id = "map";
		transformStamped.child_frame_id = "camera";

		transformStamped.transform.translation.x = c.poses_[c.maxpose_-1].pPose->estimate().inverse().translation()[0];
		transformStamped.transform.translation.y = c.poses_[c.maxpose_-1].pPose->estimate().inverse().translation()[1];
		transformStamped.transform.translation.z = c.poses_[c.maxpose_-1].pPose->estimate().inverse().translation()[2];

		transformStamped.transform.rotation.x = c.poses_[c.maxpose_-1].pPose->estimate().inverse().rotation().x();
		transformStamped.transform.rotation.y = c.poses_[c.maxpose_-1].pPose->estimate().inverse().rotation().y();
		transformStamped.transform.rotation.z = c.poses_[c.maxpose_-1].pPose->estimate().inverse().rotation().z();
		transformStamped.transform.rotation.w = c.poses_[c.maxpose_-1].pPose->estimate().inverse().rotation().w();

		br.sendTransform(transformStamped);

		// measurements

		// static pcl::PointCloud<pcl::PointXYZI>::Ptr z_cloud(new pcl::PointCloud<pcl::PointXYZI>);
		// z_cloud->height = 1;
		// z_cloud->width = c.poses_[c.maxpose_-1].point_camera_frame.size();
		// z_cloud->is_dense = false;
		// //z_cloud->resize(c.poses_[c.maxpose_-1].point_camera_frame.size());
		// z_cloud->clear();
		// for(int k = 0 ; k <c.maxpose_ ; k++){

		// 	z_cloud->reserve(z_cloud->size()+c.poses_[k].point_camera_frame.size());
		// 	for (int i = 0; i < c.poses_[k].point_camera_frame.size(); i++)
		// 	{
		// 		auto point_world_frame = c.poses_[k].pPose->estimate().inverse().map(c.poses_[k].point_camera_frame[i]);
				
		// 		pcl::PointXYZI point ;
		// 		point.x = point_world_frame[0];
		// 		point.y = point_world_frame[1];
		// 		point.z = point_world_frame[2];
		// 		point.intensity = k;
		// 		z_cloud->push_back(point);
		// 		//std::cout << "z: " << z_cloud->at(i).x  << " , "<< z_cloud->at(i).y  << " , "<< z_cloud->at(i).z  << " \n";
		// 	}
		// }
		// std::cout << "z size " << z_cloud->size() << "\n";
		pcl::PCLPointCloud2 pcl_debug;
		// sensor_msgs::PointCloud2 ros_debug;
		// pcl::toPCLPointCloud2(*z_cloud, pcl_debug);
		// pcl_conversions::fromPCL(pcl_debug, ros_debug);

		// ros_debug.header.stamp = now;
		// ros_debug.header.frame_id = "map";

		// measurements_pub.publish(ros_debug);

		// map

		static pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZI>);
		map_cloud->height = 1;
		map_cloud->is_dense = false;
		int numlm = 0;
		for (int i = 0; i < c.landmarks_.size(); i++)
		{
			if (c.landmarks_[i].numDetections_ > 0)
			{
				numlm++;
			}
		}
		map_cloud->width = numlm;
		map_cloud->resize(numlm);
		std::cout << termcolor::green <<" = nlandmarks: " <<numlm << "\n" <<termcolor::reset;
		//numlm = 0;
		int nmap=0;
		for (int i = 0; i < c.landmarks_.size(); i++)
		{
			auto &lm=c.landmarks_[i];
			if (c.landmarks_[i].numDetections_ > 0)
			{

				map_cloud->at(nmap).x = c.landmarks_[i].pPoint->estimate()[0];
				map_cloud->at(nmap).y = c.landmarks_[i].pPoint->estimate()[1];
				map_cloud->at(nmap).z = c.landmarks_[i].pPoint->estimate()[2];
				
				map_cloud->at(nmap).intensity = c.landmarks_[i].numDetections_ / (float)c.landmarks_[i].numFoV_;

				// map_cloud->at(nmap).intensity = std::binary_search(lm.is_in_fov_.begin(), lm.is_in_fov_.end(), c.maxpose_-1);

				nmap++;
			}

		}

		sensor_msgs::PointCloud2 ros_map_cloud;
		pcl::toPCLPointCloud2(*map_cloud, pcl_debug);
		pcl_conversions::fromPCL(pcl_debug, ros_map_cloud);

		ros_map_cloud.header.stamp = now;
		ros_map_cloud.header.frame_id = "map";

		map_pub.publish(ros_map_cloud);

		// trajectory
		nav_msgs::Path path;
		path.header.stamp = now;
		path.header.frame_id = "map";

		path.poses.resize(c.poses_.size());
		for (int i = 0; i < c.poses_.size(); i++)
		{
			path.poses[i].header.stamp = now;
			path.poses[i].header.frame_id = "map";
			path.poses[i].pose.position.x = c.poses_[i].pPose->estimate().inverse().translation()[0];
			path.poses[i].pose.position.y = c.poses_[i].pPose->estimate().inverse().translation()[1];
			path.poses[i].pose.position.z = c.poses_[i].pPose->estimate().inverse().translation()[2];
			path.poses[i].pose.orientation.x = c.poses_[i].pPose->estimate().inverse().rotation().x();
			path.poses[i].pose.orientation.y = c.poses_[i].pPose->estimate().inverse().rotation().y();
			path.poses[i].pose.orientation.z = c.poses_[i].pPose->estimate().inverse().rotation().z();
			path.poses[i].pose.orientation.w = c.poses_[i].pPose->estimate().inverse().rotation().w();
		}
		trajectory_pub.publish(path);

	// gt trajectory
		nav_msgs::Path gtpath;
		gtpath.header.stamp = now;
		gtpath.header.frame_id = "map";

		gtpath.poses.resize(gt_traj.trajectory.size());
		for (int i = 0; i < gt_traj.trajectory.size(); i++)
		{
			gtpath.poses[i].header.stamp = now;
			gtpath.poses[i].header.frame_id = "map";
			gtpath.poses[i].pose.position.x = gt_traj.trajectory[i].pose.translation()[0];
			gtpath.poses[i].pose.position.y = gt_traj.trajectory[i].pose.translation()[1];
			gtpath.poses[i].pose.position.z = gt_traj.trajectory[i].pose.translation()[2];
			gtpath.poses[i].pose.orientation.x = gt_traj.trajectory[i].pose.rotation().x();
			gtpath.poses[i].pose.orientation.y = gt_traj.trajectory[i].pose.rotation().y();
			gtpath.poses[i].pose.orientation.z = gt_traj.trajectory[i].pose.rotation().z();
			gtpath.poses[i].pose.orientation.w = gt_traj.trajectory[i].pose.rotation().w();
		}
		gt_trajectory_pub.publish(gtpath);


		// Associations
		visualization_msgs::Marker as_marker;
		as_marker.header.frame_id = "map";
		as_marker.header.stamp = ros::Time();
		as_marker.ns = "my_namespace";
		as_marker.id = 0;
		as_marker.type = visualization_msgs::Marker::LINE_LIST;
		as_marker.action = visualization_msgs::Marker::ADD;
		as_marker.pose.position.x = 0;
		as_marker.pose.position.y = 0;
		as_marker.pose.position.z = 0;
		as_marker.pose.orientation.x = 0.0;
		as_marker.pose.orientation.y = 0.0;
		as_marker.pose.orientation.z = 0.0;
		as_marker.pose.orientation.w = 1.0;
		as_marker.scale.x = 0.02;
		as_marker.scale.y = 0.02;
		as_marker.scale.z = 0.02;
		as_marker.color.a = 1.0; // Don't forget to set the alpha!
		as_marker.color.r = 0.0;
		as_marker.color.g = 1.0;
		as_marker.color.b = 0.0;
		int a=0;

		for(int k=0; k<c.maxpose_; k++){
		as_marker.points.resize(as_marker.points.size()+2*c.DA_bimap_[k].size());

		for (auto iter = c.DA_bimap_[k].left.begin(), iend = c.DA_bimap_[k].left.end(); iter != iend;
			 ++iter)
		{
		

			as_marker.points[a].x = c.landmarks_[iter->second - c.landmarks_[0].id].pPoint->estimate()[0];
			as_marker.points[a].y = c.landmarks_[iter->second - c.landmarks_[0].id].pPoint->estimate()[1];
			as_marker.points[a].z = c.landmarks_[iter->second - c.landmarks_[0].id].pPoint->estimate()[2];

			auto point_world_frame = c.poses_[k].pPose->estimate().inverse().map(c.poses_[k].point_camera_frame[iter->first]);

			as_marker.points[a+1].x = point_world_frame[0];
			as_marker.points[a+1].y = point_world_frame[1];
			as_marker.points[a+1].z = point_world_frame[2];

			a+=2;
		}

		}
		std::cout << "n ass: " << c.DA_bimap_[c.maxpose_-1].size()  << "  nz: "  << c.poses_[c.maxpose_-1].Z_.size() << "\n";

		association_pub.publish(as_marker);
		ros::spinOnce();
	}

	void VectorGLMBSLAM6D::changeDA(VectorGLMBComponent6D &c,
									const std::vector<
										boost::bimap<int, int, boost::container::allocator<int>>> &da)
	{
		// update bimaps!!!
		c.DA_bimap_ = da;

		// update the numdetections
		for (auto &mp : c.landmarks_)
		{
			mp.numDetections_ = 0;
		}
		for (auto &bimap : da)
		{
			for (auto it = bimap.begin(), it_end = bimap.end(); it != it_end;
				 it++)
			{
				c.landmarks_[it->right - c.landmarks_[0].pPoint->id()].numDetections_++;
			}
		}
		updateGraph(c);
		for(int i=0; i< c.maxpose_ && i < 1 ; i++){
			c.poses_[i].pPose->setFixed(true);
		}
		c.optimizer_->initializeOptimization();
		//c.optimizer_->computeInitialGuess();
		c.optimizer_->setVerbose(false);
		if (!c.optimizer_->verifyInformationMatrices(true)){
			std::cerr << "info is bad\n";
		}
		checkDA(c);
		//int g2o_result = c.optimizer_->optimize(config.numLevenbergIterations_);
		//assert(g2o_result > 0);
		//checkDA(c);
	}
	void VectorGLMBSLAM6D::sampleComponents()
	{

		// do logsumexp on the components to calculate probabilities

		std::vector<double> probs(visited_.size(), 0);
		double maxw = -std::numeric_limits<double>::infinity();
		for (auto it = visited_.begin(), it_end = visited_.end(); it != it_end;
			 it++)
		{
			if (it->second.weight > maxw)
				maxw = it->second.weight;
		}
		int i = 1;
		probs[0] = std::exp((visited_.begin()->second.weight - maxw) / temp_);
		for (auto it = std::next(visited_.begin()), it_end = visited_.end();
			 it != it_end; i++, it++)
		{
			if (it->second.weight > maxw)
				maxw = it->second.weight;
			probs[i] = probs[i - 1] + std::exp((it->second.weight - maxw) / temp_);
		}
		/*
		 std:: cout << "maxw " << maxw <<  "  temp " << temp_ << "\n";
		 std::cout << "probs " ;
		 for (auto p:probs){std::cout <<"  " << p;}
		 std::cout <<"\n";*/

		boost::uniform_real<> dist(0.0,
								   probs[probs.size() - 1] / components_.size());

		double r = dist(randomGenerators_[0]);
		int j = 0;
		auto it = visited_.begin();
		for (int i = 0; i < components_.size(); i++)
		{
			while (probs[j] < r)
			{
				j++;
				it++;
			}

			// add data association j to component i
			changeDA(components_[i], it->first);
			for(int numpose = 0 ; numpose < components_[i].poses_.size() ; numpose++){
				components_[i].poses_[numpose].pPose->setEstimate(it->second.trajectory[numpose].pose);	
				if (numpose>0){
					double dist = (it->second.trajectory[numpose].pose.translation()-it->second.trajectory[numpose-1].pose.translation()).norm();
					//assert (dist<0.1);
				}
			}
			components_[i].logweight_ = it->second.weight;

			// std::cout  << "sample w: " << it->second << " j " << j  << " r " << r <<" prob "  << probs[j]<< "\n";

			r += probs[probs.size() - 1] / components_.size();
		}
	}

	void VectorGLMBSLAM6D::loadEuroc()
	{
		std::string pathCam0 = config.eurocFolder_ + "/mav0/cam0/data";
		std::string pathCam1 = config.eurocFolder_ + "/mav0/cam1/data";

		// Loading image filenames and timestamps
		std::ifstream fTimes;
		fTimes.open(config.eurocTimestampsFilename_.c_str());
		vTimestampsCam.reserve(5000);
		vstrImageLeft.reserve(5000);
		vstrImageRight.reserve(5000);
		int numFrame=0;
		while (!fTimes.eof())
		{

			std::string s;
			std::getline(fTimes, s);
			if (!s.empty())
			{
				if (numFrame > config.minframe && numFrame%1==0){
				std::stringstream ss;
				ss << s;
				vstrImageLeft.push_back(pathCam0 + "/" + ss.str() + ".png");
				vstrImageRight.push_back(pathCam1 + "/" + ss.str() + ".png");
				double t;
				ss >> t;
					vTimestampsCam.push_back(t / 1e9);
				}
				numFrame++;
				if (numFrame > config.maxframe )
					break;
			}
		}
		startingStamp = vTimestampsCam[0];
		nImages = vstrImageLeft.size();

		cv::Mat imLeft, imRight;
		cv::Mat imLeft_rect, imRight_rect;

		// Seq loop
		double t_resize = 0;
		double t_rect = 0;
		double t_track = 0;
		int num_rect = 0;

		initial_component_.poses_.resize(nImages);
		initial_component_.numPoses_ = nImages;
		initial_component_.numPoints_ = 0;

		for (int ni = 0; ni < nImages; ni++)
		{
			// Read left and right images from file
			imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED);	//,cv::IMREAD_UNCHANGED);
			imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

			if (imLeft.empty())
			{
				std::cerr << std::endl
						  << "Failed to load image at: "
						  << std::string(vstrImageLeft[ni]) << std::endl;
				exit(1);
			}

			if (imRight.empty())
			{
				std::cerr << std::endl
						  << "Failed to load image at: "
						  << std::string(vstrImageRight[ni]) << std::endl;
				exit(1);
			}

			double tframe = vTimestampsCam[ni];

			cv::remap(imLeft, imLeft_rect, config.camera_parameters_[0].M1,
					  config.camera_parameters_[0].M2, cv::INTER_LINEAR);
			cv::remap(imRight, imRight_rect, config.camera_parameters_[1].M1,
					  config.camera_parameters_[1].M2, cv::INTER_LINEAR);

			initial_component_.poses_[ni];
			initial_component_.poses_[ni].stamp = tframe;
			std::cout << std::setprecision(20) ;
			std::cout << "time " << initial_component_.poses_[ni].stamp << "\n";
			std::vector<int> vLapping_left = {0, 0};
			std::vector<int> vLapping_right = {0, 0};
			cv::Mat mask_left, mask_right;

			//		std::thread threadLeft(&ORB_SLAM3::ORBextractor::extract,
			//				mpORBextractorLeft, &imLeft_rect, &mask_left,
			//				&initial_component_.poses_[ni].keypoints_left,
			//				&initial_component_.poses_[ni].descriptors_left,
			//				&vLapping_left);
			//		std::thread threadRight(&ORB_SLAM3::ORBextractor::extract,
			//				mpORBextractorRight, &imRight_rect, &mask_right,
			//				&initial_component_.poses_[ni].keypoints_right,
			//				&initial_component_.poses_[ni].descriptors_right,
			//				&vLapping_right);
			//		threadLeft.join();
			//		threadRight.join();
			cv::Mat desc_left , desc_right;
			ORB_SLAM3::ORBextractor::extract(mpORBextractorLeft,
											 &imLeft_rect, &mask_left,
											 &initial_component_.poses_[ni].keypoints_left,
											 &desc_left,
											 &vLapping_left);
			ORB_SLAM3::ORBextractor::extract(mpORBextractorRight,
											 &imRight_rect, &mask_right,
											 &initial_component_.poses_[ni].keypoints_right,
											 &desc_right,
											 &vLapping_right);
			
			
			initial_component_.poses_[ni].descriptors_left.resize(initial_component_.poses_[ni].keypoints_left.size());
			for (int nk=0; nk<initial_component_.poses_[ni].keypoints_left.size() ;nk++){
				initial_component_.poses_[ni].descriptors_left[nk].from_mat(desc_left.row(nk));
			}
			initial_component_.poses_[ni].descriptors_right.resize(initial_component_.poses_[ni].keypoints_right.size());
			for (int nk=0; nk<initial_component_.poses_[ni].keypoints_right.size() ;nk++){
				initial_component_.poses_[ni].descriptors_right[nk].from_mat(desc_right.row(nk));
			}
			computeStereoMatches(initial_component_.poses_[ni]);

			initial_component_.numPoints_ += initial_component_.poses_[ni].matches_left_to_right.size();

			initial_component_.poses_[ni].mnMinX = 0.0f;
			initial_component_.poses_[ni].mnMaxX = imLeft.cols;
			initial_component_.poses_[ni].mnMinY = 0.0f;
			initial_component_.poses_[ni].mnMaxY = imLeft.rows;

			initial_component_.poses_[ni].mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
			// std::cout << "invsig:\n";
			// for( auto sig: initial_component_.poses_[ni].mvInvLevelSigma2 ) {
			// std::cout << sig << " ";
			// }


			// plot stereo matches
			cv::Mat imLeftKeys, imRightKeys, imMatches;
			cv::Scalar kpColor = cv::Scalar(255, 0, 0);

			// cv::drawMatches(imLeft_rect,
			// 				initial_component_.poses_[ni].keypoints_left, imRight_rect,
			// 				initial_component_.poses_[ni].keypoints_right,
			// 				initial_component_.poses_[ni].matches_left_to_right, imMatches);

			// cv::drawKeypoints(imRight_rect,
			// 				  initial_component_.poses_[ni].keypoints_right, imRightKeys,
			// 				  kpColor);
			// cv::drawKeypoints(imLeft_rect,
			// 				  initial_component_.poses_[ni].keypoints_left, imLeftKeys,
			// 				  kpColor);
			//  cv::imshow("matches", imMatches);
			 //cv::imshow("imLeft", imLeft);
			 //cv::imshow("imLeft_rect", imLeft_rect);

			 cv::waitKey(1); // Wait for a keystroke in the window
			std::cout << ni + 1 << "/" << nImages << "                                   \r";
		}
		std::cout << "\n";
	}

	void VectorGLMBSLAM6D::computeStereoMatches(OrbslamPose &pose)
	{

		pose.uRight = std::vector<float>(pose.keypoints_left.size(), -1.0f);
		pose.depth = std::vector<float>(pose.keypoints_left.size(), -1.0f);

		pose.mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
		pose.mnScaleLevels = mpORBextractorLeft->GetLevels();

		static constexpr int thOrbDist = (orb_th_high + orb_th_low) / 2;

		const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

		std::vector<std::vector<std::size_t>> vRowIndices(nRows,
														  std::vector<std::size_t>());

		for (int i = 0; i < nRows; i++)
			vRowIndices[i].reserve(200);

		const int Nl = pose.keypoints_left.size();
		const int Nr = pose.keypoints_right.size();

		for (int iR = 0; iR < Nr; iR++)
		{
			const cv::KeyPoint &kp = pose.keypoints_right[iR];
			const float &kpY = kp.pt.y;
			const float r =
				2.0f * mpORBextractorLeft->GetScaleFactors()[pose.keypoints_right[iR].octave];
			const int maxr = ceil(kpY + r);
			const int minr = floor(kpY - r);

			for (int yi = minr; yi <= maxr; yi++)
				vRowIndices[yi].push_back(iR);
		}

		// Set limits for search
		const float minZ = config.stereo_baseline;
		const float minD = 0;
		const float maxD = config.stereo_baseline_f / minZ;

		// For each left keypoint search a match in the right image
		std::vector<std::pair<int, int>> vDistIdx;
		vDistIdx.reserve(Nl);

		for (int iL = 0; iL < Nl; iL++)
		{
			const cv::KeyPoint &kpL = pose.keypoints_left[iL];
			const int &levelL = kpL.octave;
			const float &vL = kpL.pt.y;
			const float &uL = kpL.pt.x;

			const std::vector<std::size_t> &vCandidates = vRowIndices[vL];

			if (vCandidates.empty())
				continue;

			const float minU = uL - maxD;
			const float maxU = uL - minD;

			if (maxU < 0)
				continue;

			int bestDist = orb_th_high;
			size_t bestIdxR = 0;

			const auto &dL = pose.descriptors_left[iL];

			// Compare descriptor to right keypoints
			for (size_t iC = 0; iC < vCandidates.size(); iC++)
			{
				const size_t iR = vCandidates[iC];
				const cv::KeyPoint &kpR = pose.keypoints_right[iR];

				if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
					continue;

				const float &uR = kpR.pt.x;

				if (uR >= minU && uR <= maxU)
				{
					const auto &dR = pose.descriptors_right[iR];
					const int dist = ORBDescriptor::distance(dL,dR);

					if (dist < bestDist)
					{
						bestDist = dist;
						bestIdxR = iR;
					}
				}
			}

			// Subpixel match by correlation
			if (bestDist < thOrbDist)
			{
				// coordinates in image pyramid at keypoint scale
				const float uR0 = pose.keypoints_right[bestIdxR].pt.x;
				const float scaleFactor = orbExtractorInvScaleFactors[kpL.octave];
				const float scaleduL = round(kpL.pt.x * scaleFactor);
				const float scaledvL = round(kpL.pt.y * scaleFactor);
				const float scaleduR0 = round(uR0 * scaleFactor);

				// sliding window search
				const int w = 5;
				cv::Mat IL =
					mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(
																	  scaledvL - w, scaledvL + w + 1)
						.colRange(
							scaleduL - w, scaleduL + w + 1);

				int bestDist = INT_MAX;
				int bestincR = 0;
				const int L = 5;
				std::vector<float> vDists;
				vDists.resize(2 * L + 1);

				const float iniu = scaleduR0 + L - w;
				const float endu = scaleduR0 + L + w + 1;
				if (iniu < 0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
					continue;

				for (int incR = -L; incR <= +L; incR++)
				{
					cv::Mat IR =
						mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(
																		   scaledvL - w, scaledvL + w + 1)
							.colRange(
								scaleduR0 + incR - w, scaleduR0 + incR + w + 1);

					float dist = cv::norm(IL, IR, cv::NORM_L1);
					if (dist < bestDist)
					{
						bestDist = dist;
						bestincR = incR;
					}

					vDists[L + incR] = dist;
				}

				if (bestincR == -L || bestincR == L)
					continue;

				// Sub-pixel match (Parabola fitting)
				const float dist1 = vDists[L + bestincR - 1];
				const float dist2 = vDists[L + bestincR];
				const float dist3 = vDists[L + bestincR + 1];

				const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

				if (deltaR < -1 || deltaR > 1)
					continue;

				// Re-scaled coordinate
				float bestuR = orbExtractorScaleFactors[kpL.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);

				float disparity = (uL - bestuR);

				if (disparity >= minD && disparity < maxD)
				{
					if (disparity <= 0)
					{
						disparity = 0.01;
						bestuR = uL - 0.01;
					}
					pose.depth[iL] = config.stereo_baseline_f / disparity;
					pose.uRight[iL] = bestuR;
					if (pose.depth[iL]<60.0){
					pose.matches_left_to_right.push_back(
						cv::DMatch(iL, bestIdxR, bestDist));

					vDistIdx.push_back(std::pair<int, int>(bestDist, iL));
					}
				}
			}
		}
		sort(pose.matches_left_to_right.begin() , pose.matches_left_to_right.end() );


		sort(vDistIdx.begin(), vDistIdx.end());

		const float median = vDistIdx[vDistIdx.size() / 2].first;
		const float thDist = 1.5f * 1.4f * median;

		cv::DMatch thres(0, 0, thDist);
		auto it = std::lower_bound(pose.matches_left_to_right.begin() , pose.matches_left_to_right.end() , thres );
		pose.matches_left_to_right.resize((it-pose.matches_left_to_right.begin()));
	
		for (int i = vDistIdx.size() - 1; i >= 0; i--)
		{
			if (vDistIdx[i].first < thDist)
				break;
			else
			{
				pose.uRight[vDistIdx[i].second] = -1;
				pose.depth[vDistIdx[i].second] = -1;
			}
		}
	}

	void VectorGLMBSLAM6D::loadConfig(std::string filename)
	{

		YAML::Node node = YAML::LoadFile(filename);

		config.MeasurementLikelihoodThreshold_ =
			node["MeasurementLikelihoodThreshold"].as<double>();
		config.lmExistenceProb_ = node["lmExistenceProb"].as<double>();
		config.logKappa_ = node["logKappa"].as<double>();
		config.logExistenceOdds = node["logExistenceOdds"].as<double>();
		config.PD_ = node["PD"].as<double>();
		config.maxRange_ = node["maxRange"].as<double>();
		config.numComponents_ = node["numComponents"].as<int>();
		config.birthDeathNumIter_ = node["birthDeathNumIter"].as<int>();
		config.numLandmarks_ = node["numLandmarks"].as<int>();
		config.numGibbs_ = node["numGibbs"].as<int>();
		config.numIterations_ = node["numIterations"].as<int>();
		config.numLevenbergIterations_ = node["numLevenbergIterations"].as<int>();
		config.xlim_.push_back(node["xlim"][0].as<double>());
		config.xlim_.push_back(node["xlim"][1].as<double>());
		config.ylim_.push_back(node["ylim"][0].as<double>());
		config.ylim_.push_back(node["ylim"][1].as<double>());
		config.initTemp_ = node["initTemp"].as<double>();
		config.tempFactor_ = node["tempFactor"].as<double>();
		config.minframe = node["minframe"].as<int>();
		config.maxframe = node["maxframe"].as<int>();
		config.staticframes = node["staticframes"].as<int>();

		config.maxWeightDifference = node["maxWeightDifference"].as<double>();

		config.crossoverNumIter_ = node["crossoverNumIter"].as<int>();
		config.numPosesToOptimize_ = node["numPosesToOptimize"].as<int>();
		config.finalStateFile_ = node["finalStateFile"].as<std::string>();

		if (!YAML::convert<Eigen::Matrix3d>::decode(node["anchorInfo"],
													config.anchorInfo_))
		{
			std::cerr << "could not load anchor info matrix \n";
			exit(1);
		}
		if (!YAML::convert<Eigen::Matrix<double, 6, 6>>::decode(node["odomInfo"],
																config.odomInfo_))
		{
			std::cerr << "could not load odom info matrix \n";
			exit(1);
		}

		if (!YAML::convert<Eigen::Matrix3d>::decode(node["stereoInfo"],
													config.stereoInfo_))
		{
			std::cerr << "could not load stereo info matrix \n";
			exit(1);
		}

		double focal_length = node["camera.focal_length"].as<double>();

		config.viewingCosLimit_ = node["viewingCosLimit"].as<double>();

		config.perturbTrans = node["perturbTrans"].as<double>();
		config.perturbRot = node["perturbRot"].as<double>();


		config.eurocFolder_ = node["eurocFolder"].as<std::string>();
		config.use_gui_ = node["use_gui"].as<bool>();

		config.eurocTimestampsFilename_ = node["eurocTimestampsFilename"].as<std::string>();

		if (!YAML::convert<Eigen::MatrixXd>::decode(node["base_link_to_cam0"],
														config.base_link_to_cam0))
		{
			std::cerr << "could not load base_link_to_cam0 \n";
			exit(1);
		}
		config.base_link_to_cam0_se3.setTranslation( config.base_link_to_cam0.col(3).head(3));
		Eigen::Matrix3d rotMat= config.base_link_to_cam0.block(0,0,3,3);
		Eigen::Quaterniond q(rotMat)  ;
		config.base_link_to_cam0_se3.setRotation( q );


		for (auto camera : node["camera_params"])
		{
			Config::CameraParams params;
			params.fx = camera["fx"].as<double>();
			params.fy = camera["fy"].as<double>();
			params.cx = camera["cx"].as<double>();
			params.cy = camera["cy"].as<double>();
			params.k1 = camera["k1"].as<double>();
			params.k2 = camera["k2"].as<double>();
			params.p1 = camera["p1"].as<double>();
			params.p2 = camera["p2"].as<double>();
			params.originalImSize.width = camera["width"].as<int>();
			params.originalImSize.height = camera["height"].as<int>();

			params.newImSize = params.originalImSize;

			if (!YAML::convert<Eigen::MatrixXd>::decode(camera["cv_c0_to_camera"],
														params.cv_c0_to_camera_eigen))
			{
				std::cerr << "could not load cv_c0_to_camera \n";
				exit(1);
			}
			cv::eigen2cv(params.cv_c0_to_camera_eigen, params.cv_c0_to_camera);

			cv::Mat dist_coeffs(4, 1, CV_64F);
			dist_coeffs.at<float>(0, 0) = params.k1;
			dist_coeffs.at<float>(1, 0) = params.k2;
			dist_coeffs.at<float>(2, 0) = params.p1;
			dist_coeffs.at<float>(3, 0) = params.p2;
			params.opencv_distort_coeffs =
				(cv::Mat_<double>(4, 1) << params.k1, params.k2, params.p1, params.p2);

			params.opencv_calibration =
				(cv::Mat_<double>(3, 3) << (float)params.fx, 0.f, (float)params.cx, 0.f, (float)params.fy, (float)params.cy, 0.f, 0.f, 1.f);

			config.camera_parameters_.push_back(params);
		}

		Sophus::SE3d Tlr(config.camera_parameters_[1].cv_c0_to_camera_eigen);

        std::cout << "Tlr0: "
				  << Tlr.matrix3x4() << "\n";
		cv::Mat cvTlr ;

        cv::eigen2cv(Tlr.inverse().matrix3x4(),cvTlr);
        std::cout << "cvTlr1: "
				  << cvTlr << "\n";
		cv::Mat R12 = cvTlr.rowRange(0, 3).colRange(0, 3);
		R12.convertTo(R12, CV_64F);
		cv::Mat t12 = cvTlr.rowRange(0, 3).col(3);
		t12.convertTo(t12, CV_64F);

		config.stereo_baseline = Tlr.translation().norm();
		config.stereo_baseline_f = config.stereo_baseline * config.camera_parameters_[0].fx;

		Eigen::Vector2d principal_point = {config.camera_parameters_[0].cx,
										   config.camera_parameters_[0].cy};

		config.g2o_cam_params = new g2o::CameraParameters(
			config.camera_parameters_[0].fx, principal_point,
			config.stereo_baseline);
		config.g2o_cam_params->setId(0);

		cv::Mat R_r1_u1, R_r2_u2;
		cv::Mat P1, P2, Q;

		cv::stereoRectify(config.camera_parameters_[0].opencv_calibration,
						  config.camera_parameters_[0].opencv_distort_coeffs,
						  config.camera_parameters_[1].opencv_calibration,
						  config.camera_parameters_[1].opencv_distort_coeffs,
						  config.camera_parameters_[0].newImSize, R12, t12, R_r1_u1, R_r2_u2,
						  P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1,
						  config.camera_parameters_[0].newImSize);

		cv::initUndistortRectifyMap(config.camera_parameters_[0].opencv_calibration,
									config.camera_parameters_[0].opencv_distort_coeffs, R_r1_u1,
									P1.rowRange(0, 3).colRange(0, 3),
									config.camera_parameters_[0].newImSize, CV_32F,
									config.camera_parameters_[0].M1, config.camera_parameters_[0].M2);
		cv::initUndistortRectifyMap(config.camera_parameters_[1].opencv_calibration,
									config.camera_parameters_[1].opencv_distort_coeffs, R_r2_u2,
									P2.rowRange(0, 3).colRange(0, 3),
									config.camera_parameters_[1].newImSize, CV_32F,
									config.camera_parameters_[1].M1, config.camera_parameters_[1].M2);

        
        std::cout << "config.camera_parameters_[1].cv_c0_to_camera: "
				  << config.camera_parameters_[1].cv_c0_to_camera << "\n";    

        std::cout << "cvTlr: "
				  << cvTlr << "\n";
        std::cout << "Tlr: "
				  << Tlr.matrix3x4() << "\n";

        std::cout << "R12: "
				  << R12 << "\n";
        std::cout << "t12: "
				  << t12 << "\n";

		std::cout << "cam0 opencv_calibration: "
				  << config.camera_parameters_[0].opencv_calibration << "\n";
		std::cout << "cam0 opencv_distort_coeffs: "
				  << config.camera_parameters_[0].opencv_distort_coeffs << "\n";
		std::cout << "cam0 R_r1_u1: "
				  << R_r1_u1 << "\n";

		std::cout << "cam1 opencv_calibration: "
				  << config.camera_parameters_[1].opencv_calibration << "\n";
		std::cout << "cam1 opencv_distort_coeffs: "
				  << config.camera_parameters_[1].opencv_distort_coeffs << "\n";
		std::cout << "cam0 R_r2_u2: "
				  << R_r2_u2 << "\n";

		//    std::cout << "cam0 M1: " << config.camera_parameters_[0].M1 << "\n";
		//    std::cout << "cam0 M2: " << config.camera_parameters_[0].M2 << "\n";
		//    std::cout << "cam1 M1: " << config.camera_parameters_[1].M1 << "\n";
		//    std::cout << "cam1 M2: " << config.camera_parameters_[1].M2 << "\n";

		config.orb_extractor.nFeatures = node["ORBextractor.nFeatures"].as<int>();
		config.orb_extractor.scaleFactor = node["ORBextractor.scaleFactor"].as<double>();
		config.orb_extractor.nLevels = node["ORBextractor.nLevels"].as<int>();
		config.orb_extractor.iniThFAST = node["ORBextractor.iniThFAST"].as<int>();
		config.orb_extractor.minThFAST = node["ORBextractor.minThFAST"].as<int>();
		config.stereo_init_max_depth = node["stereo_init_max_depth"].as<double>();

		mpORBextractorLeft = new ORB_SLAM3::ORBextractor(
			config.orb_extractor.nFeatures, config.orb_extractor.scaleFactor,
			config.orb_extractor.nLevels, config.orb_extractor.iniThFAST,
			config.orb_extractor.minThFAST);
		mpORBextractorRight = new ORB_SLAM3::ORBextractor(
			config.orb_extractor.nFeatures, config.orb_extractor.scaleFactor,
			config.orb_extractor.nLevels, config.orb_extractor.iniThFAST,
			config.orb_extractor.minThFAST);

		orbExtractorInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
		orbExtractorScaleFactors = mpORBextractorLeft->GetScaleFactors();
	}

	double VectorGLMBSLAM6D::distance(PoseType *pose, PointType *lm)
	{

		Eigen::Vector3d point_in_camera_frame = pose->estimate().map(
			lm->estimate());
		return point_in_camera_frame.norm();
	}

	inline void VectorGLMBSLAM6D::deleteComponents(){
	for (auto &c : components_)
		{
			deleteLandmarks(c);
			deleteMeasurements(c);
			delete c.solverLevenberg_;
			delete c.optimizer_;
		}
		components_.clear();
	}
	inline void VectorGLMBSLAM6D::deleteLandmarks(VectorGLMBComponent6D &c){
		for (auto &lm: c.landmarks_ ){
			delete lm.pPoint;
		}
		c.landmarks_.clear();

	}
	inline void VectorGLMBSLAM6D::deleteMeasurements(VectorGLMBComponent6D &c){
		for(int k =0; k<c.poses_.size(); k++){
			for(auto &z: c.poses_[k].Z_){
				delete z;
			}
			delete c.poses_[k].pPose;
			
		}
		for(auto odo:c.odometries_){
			delete odo;
		}
		c.odometries_.clear();
		c.poses_.clear();
	}


	inline void VectorGLMBSLAM6D::initComponents()
	{
		components_.resize(config.numComponents_);
		int i = 0;
		for (auto &c : components_)
		{
			init(c);
			constructGraph(c);

			// optimize once at the start to calculate the hessian.
			c.poses_[0].pPose->setFixed(true);
			for(int k = 0; k< config.staticframes-config.minframe ; k++){
				c.poses_[k].pPose->setFixed(true);
			}

			c.optimizer_->initializeOptimization();
			//c.optimizer_->computeInitialGuess();
			c.optimizer_->setVerbose(false);
			std::string filename = std::string("init") + std::to_string(i++) + ".g2o";
			c.optimizer_->save(filename.c_str(), 0);
		if (!c.optimizer_->verifyInformationMatrices(true)){
			std::cerr << "info is bad\n";
		}
			// int niterations = c.optimizer_->optimize(1);
			// std::cout << "niterations  " << niterations << "\n";
			// assert(niterations > 0);
		}

		// Visualization
		if (config.use_gui_)
		{
		}
	}
	inline void VectorGLMBSLAM6D::run(int numSteps)
	{

		for (int i = 0; i < numSteps && ros::ok(); i++)
		{
			maxpose_prev_ = maxpose_;
			maxpose_ = 2 + components_[0].poses_.size() * i / (numSteps * 0.95);

			if (best_DA_max_detection_time_ + config.numPosesToOptimize_/2 < maxpose_)
			{
				maxpose_ = best_DA_max_detection_time_ +10;// config.numPosesToOptimize_/2;
			}
			if (maxpose_ > components_[0].poses_.size())
				maxpose_ = components_[0].poses_.size();
			if (best_DA_max_detection_time_ == components_[0].poses_.size() - 1)
			{
				minpose_ = 0;
			}
			else
			{
				minpose_ = std::max(0,
									std::min(maxpose_ - 2 * config.numPosesToOptimize_,
											 best_DA_max_detection_time_ - config.numPosesToOptimize_));
			}
			for (auto &c : components_)
			{
				c.maxpose_ = maxpose_;
			}

			// minpose_ = 0;
			std::cout << "maxpose: " << maxpose_ << " max det:  "
					  << best_DA_max_detection_time_ << "  " << minpose_ << "\n";
			std::cout << "iteration: " << iteration_ << " / " << numSteps << "\n";
			optimize(config.numLevenbergIterations_);
		}
	}

	void VectorGLMBSLAM6D::updatePoses(VectorGLMBComponent6D &c){
		for(int k=0; k<c.poses_.size() ;k++){
			c.poses_[k].invPose = c.poses_[k].pPose->estimate().inverse();
		}

	}

	void VectorGLMBSLAM6D::updateDescriptors(VectorGLMBComponent6D &c){
		
	}
	void VectorGLMBSLAM6D::perturbTraj(VectorGLMBComponent6D &c){
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif

		boost::uniform_real<> uni_dist(0, 1);
		int startk = std::max(std::max(minpose_ , config.staticframes-config.minframe ) ,1);
		

		for(int k = startk ; k < c.poses_.size() ; k++){
			//g2o::Vector6 p_v = (c.poses_[k-1].pPose->estimate().toMinimalVector()+c.poses_[k].pPose->estimate().toMinimalVector()+c.poses_[k+1].pPose->estimate().toMinimalVector())*(1.0/3.0);
			double d = uni_dist(randomGenerators_[threadnum]);
			g2o::Vector6 p_v;

			p_v = c.poses_[k].pPose->estimate().inverse().toMinimalVector();
	

			p_v[0] += gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbTrans;
			p_v[1] += gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbTrans;
			p_v[2] += gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbTrans;
			p_v[3] += gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbRot;
			p_v[4] += gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbRot;
			p_v[5] += gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbRot;
			g2o::SE3Quat pos(p_v);
			c.poses_[k].pPose->setEstimate(pos.inverse());
		}

		// auto displacement = c.poses_[maxpose_].pPose->estimate()*c.poses_[maxpose_-1].pPose->estimate().inverse();
		// auto current_pose = c.poses_[maxpose_].pPose->estimate();

		// for(int k = maxpose_ ; k < c.poses_.size() ; k++){
		// 	//g2o::Vector6 p_v = (c.poses_[k-1].pPose->estimate().toMinimalVector()+c.poses_[k].pPose->estimate().toMinimalVector()+c.poses_[k+1].pPose->estimate().toMinimalVector())*(1.0/3.0);
		// 	current_pose = current_pose*displacement;


		// 	c.poses_[k].pPose->setEstimate(current_pose);
		// }

	}

	void VectorGLMBSLAM6D::selectNN(VectorGLMBComponent6D &c)
	{
		std::vector<boost::bimap<int, int, boost::container::allocator<int>>> out;
		out.resize(c.DA_bimap_.size());
		for (int i = 0; i < maxpose_; i++)
		{
			out[i] = c.DA_bimap_[i];
		}
		int max_detection_time = maxpose_ - 1;
		while (max_detection_time > 0 && c.DA_bimap_[max_detection_time].size() == 0)
		{
			max_detection_time--;
		}

		for (int k = max_detection_time + 1; k < maxpose_; k++)
		{

			updateGraph(c);
			for(int k = 0; k< config.staticframes-config.minframe ; k++){
				c.poses_[k].pPose->setFixed(true);
			}
			c.optimizer_->initializeOptimization();
			//c.optimizer_->computeInitialGuess();
			c.optimizer_->setVerbose(false);
			if (!c.optimizer_->verifyInformationMatrices(true)){
				std::cerr << "info is bad\n";
			}
			int niterations = c.optimizer_->optimize(30);
			assert(niterations > 0);
			// calculateWeight(c);
			moveBirth(c);
			updateDAProbs(c, k, k + 1);

			AssociationProbabilities probs;
			for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
			{
				// std::cout << "selectNN\n";
				probs.i.clear();
				probs.l.clear();
				double maxprob = -std::numeric_limits<double>::infinity();
				int maxprobi = 0;
				auto it = c.DA_bimap_[k].left.find(nz);
				double selectedProb;
				int selectedDA = -5;
				if (it != c.DA_bimap_[k].left.end())
				{
					selectedDA = it->second;
				}
				double maxlikelihood = -std::numeric_limits<double>::infinity();
				int maxlikelihoodi = 0;
				for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++)
				{
					double likelihood = c.DAProbs_[k][nz].l[a];

					if (maxlikelihood < likelihood)
					{
						maxlikelihood = likelihood;
						maxlikelihoodi = a;
					}
					if (c.DAProbs_[k][nz].i[a] == -5)
					{
						probs.i.push_back(c.DAProbs_[k][nz].i[a]);
						probs.l.push_back(c.DAProbs_[k][nz].l[a]);
						if (c.DAProbs_[k][nz].l[a] > maxprob)
						{
							maxprob = c.DAProbs_[k][nz].l[a];
							maxprobi = a;
						}
					}
					else if (c.DAProbs_[k][nz].i[a] == selectedDA)
					{
						probs.i.push_back(c.DAProbs_[k][nz].i[a]);
						if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].numDetections_ == 1)
						{
							likelihood += config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].numFoV_) * std::log(1 - config.PD_);
							// std::cout <<" single detection: increase:  " <<logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].pPoint->id()].numFoV_)*std::log(1-config.PD_) <<"\n";
							probs.l.push_back(likelihood);
						}
						else
						{
							probs.l.push_back(c.DAProbs_[k][nz].l[a]);
						}
						if (likelihood > maxprob)
						{
							maxprob = likelihood;
							maxprobi = a;
						}
					}
					else
					{
						if (c.DA_bimap_[k].right.count(c.DAProbs_[k][nz].i[a]) == 0)
						{ // landmark is not already associated to another measurement
							probs.i.push_back(c.DAProbs_[k][nz].i[a]);
							if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].numDetections_ == 0)
							{
								likelihood +=
									config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].numFoV_) * std::log(1 - config.PD_);
								// std::cout <<" 0 detection: increase:  " << logExistenceOdds+ (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].pPoint->id()].numFoV_)*std::log(1-config.PD_)<<"\n";
								probs.l.push_back(likelihood);
							}
							else
							{
								probs.l.push_back(c.DAProbs_[k][nz].l[a]);
							}
							if (likelihood > maxprob)
							{
								maxprob = likelihood;
								maxprobi = a;
							}
						}
					}
				}
				int newdai = c.DAProbs_[k][nz].i[maxprobi];

				if (newdai == -5)
				{
					// std::cout << "false alarm\n";
				}
				// std::cout << "ass\n";
				if (newdai != selectedDA)
				{ // if selected association, change bimap

					if (newdai >= 0)
					{
						c.landmarks_[newdai - c.landmarks_[0].pPoint->id()].numDetections_++;
						if (selectedDA < 0)
						{
							c.DA_bimap_[k].insert({nz, newdai});
						}
						else
						{

							c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_--;
							assert(c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_ >= 0);
							c.DA_bimap_[k].left.replace_data(it, newdai);
						}
					}
					else
					{ // if a change has to be made and new DA is false alarm, we need to remove the association
						c.DA_bimap_[k].left.erase(it);
					if(selectedDA>=0){
						c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_--;
						assert(c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_ >= 0);
					}
					}
				}
			}
		}
	}

	std::vector<boost::bimap<int, int, boost::container::allocator<int>>> VectorGLMBSLAM6D::sexyTime(
		VectorGLMBComponent6D &c1, VectorGLMBComponent6D &c2)
	{

		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif
		if (maxpose_ == 0)
		{
			std::vector<boost::bimap<int, int, boost::container::allocator<int>>> out(
				c1.DA_bimap_);
			return out;
		}
		boost::uniform_int<> random_merge_point(-maxpose_, maxpose_);
		std::vector<boost::bimap<int, int, boost::container::allocator<int>>> out;
		out.resize(c1.DA_bimap_.size());
		int merge_point = random_merge_point(rfs::randomGenerators_[threadnum]);

		if (merge_point >= 0)
		{
			// first half from da1 second from da2
			for (int i = merge_point; i < c2.DA_bimap_.size(); i++)
			{
				out[i] = c2.DA_bimap_[i];
			}
			for (int i = 0; i < merge_point; i++)
			{
				out[i] = c1.DA_bimap_[i];
			}
		}
		else
		{
			// first half from da2 second from da1
			for (int i = 0; i < -merge_point; i++)
			{
				out[i] = c2.DA_bimap_[i];
			}
			for (int i = -merge_point; i < c2.DA_bimap_.size(); i++)
			{
				out[i] = c1.DA_bimap_[i];
			}
		}
		return out;
	}
	inline void VectorGLMBSLAM6D::optimize(int ni)
	{

		std::cout << "visited  " << visited_.size() << "\n";
		if (visited_.size() > 0)
		{
			sampleComponents();
			//visited_.clear();
			std::cout << "sampling compos \n";
		}

// 		if (iteration_ % config.crossoverNumIter_ == 0 and false)
// 		{

// 			std::cout << termcolor::magenta
// 					  << " =========== SexyTime!============\n"
// 					  << termcolor::reset;
// 			for (int i = 0; i < components_.size(); i++)
// 			{
// 				auto &c = components_[i];
// 				c.maxpose_ = maxpose_;

// 				boost::uniform_int<> random_component(0, components_.size() - 1);
// 				int secondComp;
// 				do
// 				{
// 					secondComp = random_component(rfs::randomGenerators_[0]);
// 				} while (secondComp == i);
// 				auto da = sexyTime(c, components_[secondComp]);
// 				changeDA(c, da);
// 			}
// #pragma omp parallel for
// 			for (int i = 0; i < components_.size(); i++)
// 			{

// 				if (maxpose_prev_ != maxpose_)
// 				{
// 					auto &c = components_[i];
// 					selectNN( c);
// 					updateGraph(c);
// 					for(int k=0; k< c.maxpose_ && k < 1 ; k++){
// 						c.poses_[k].pPose->setFixed(true);
// 					}
// 					c.optimizer_->initializeOptimization();
// 					//c.optimizer_->computeInitialGuess();
// 					c.optimizer_->setVerbose(false);
// 					if (!c.optimizer_->verifyInformationMatrices(true)){
// 						std::cerr << "info is bad\n";
// 					}
// 					int niterations = c.optimizer_->optimize(ni);
// 					assert(niterations > 0);

// 					updateFoV(c);
// 					// std::cout << "fov update: \n";

// 					moveBirth(c);
// 					updateDAProbs(c, minpose_, maxpose_);
// 					// std::cout << "da update: \n";
// 					c.prevDA_bimap_ = c.DA_bimap_;
// 					double expectedChange = 0;
// 					bool inserted;
// 					selectNN(c);
// 					//	std::cout << "nn update: \n";
// 					//updateGraph(c);
// 					for(int k=0; k< c.maxpose_ && k < 1 ; k++){
// 						c.poses_[k].pPose->setFixed(true);
// 					}
// 					c.optimizer_->initializeOptimization();
// 					//c.optimizer_->computeInitialGuess();
// 					c.optimizer_->setVerbose(false);
// 					if (!c.optimizer_->verifyInformationMatrices(true)){
// 						std::cerr << "info is bad\n";
// 					}
// 					niterations = c.optimizer_->optimize(ni);
// 					assert(niterations > 0);
// 					calculateWeight(c);

// 					std::map<
// 					std::vector<boost::bimap<int, int, boost::container::allocator<int>>>,
// 					TrajectoryWeight>::iterator it;
// 					#pragma omp critical(bestweight)
// 					{
// 						if (c.logweight_ > bestWeight_)
// 						{
// 							bestWeight_ = c.logweight_;
// 							best_DA_ = c.DA_bimap_;
// 							best_DA_max_detection_time_ = maxpose_ - 1;
// 							while (best_DA_max_detection_time_ > 0 && best_DA_[best_DA_max_detection_time_].size() == 0)
// 							{
// 								best_DA_max_detection_time_--;
// 							}
// 							std::stringstream filename;

// 							filename << "video/beststate_" << std::setfill('0')
// 									 << std::setw(5) << iterationBest_++ << ".g2o";
// 							c.optimizer_->save(filename.str().c_str(), 0);
// 							std::cout << termcolor::yellow << "========== newbest:"
// 									  << bestWeight_ << " ============\n"
// 									  << termcolor::reset;
// 							//printDA(c);
// 							if (config.use_gui_)
// 							{
// 								std::cout << termcolor::yellow << "========== piblishingmarkers:"
// 										  << bestWeight_ << " ============\n"
// 										  << termcolor::reset;
// 								publishMarkers(c);
// 							}
// 						}
// 					}
// 					TrajectoryWeight to_insert;
// 					to_insert.weight = c.logweight_;
// 					to_insert.trajectory.resize(c.poses_.size() );
// 					for(int numpose =0; numpose< maxpose_; numpose++){
// 						to_insert.trajectory[numpose]=c.poses_[numpose].pPose->estimate();
// 					}
// 					for(int numpose = maxpose_; numpose< c.poses_.size(); numpose++){
// 						to_insert.trajectory[numpose]=c.poses_[maxpose_-1].pPose->estimate();
// 					}

// 					auto pair = std::make_pair(c.DA_bimap_, to_insert);
// 					#pragma omp critical(insert)
// 					{
// 						std::tie(it, inserted) = visited_.insert(pair);
// 						insertionP_ = insertionP_ * 0.9;
// 						if (inserted)
// 							insertionP_ += 0.1;
// 					}
// 					//it->second = c.logweight_;
// 				}
// 			}
// 		}
		#pragma omp parallel for
		for (int i = 0; i < components_.size(); i++)
		{


			
			auto &c = components_[i];
			// for (int k=0; k< minpose_;k++){
			// 	if(k%5!=0){
			// 		 for(int nz=0; nz < c.poses_[k].Z_.size();nz++){
			// 		 	c.poses_[k].Z_[nz]->setLevel(2);
			// 			// c.optimizer_->removeEdge(c.poses_[k].Z_[nz]);
			// 		 }
			// 		//c.optimizer_->removeVertex(c.poses_[k].pPose);
			// 		//c.poses_[k].pPose->setMarginalized(true);
			// 	}
			// }
			for(int k = 1; k< maxpose_ ; k++){
				c.odometries_[k-1]->setLevel(0);
			}
			for(int k = maxpose_; k< c.poses_.size() ; k++){
				//c.poses_[k].pPose->setFixed(true);
				c.odometries_[k-1]->setLevel(2);
			}
			int threadnum = 0;
			#ifdef _OPENMP
			threadnum = omp_get_thread_num();
			#endif
			checkGraph(c);
			updateGraph(c);
			moveBirth(c);
			checkGraph(c);
			for(int k = 0; k< config.staticframes-config.minframe ; k++){
				c.poses_[k].pPose->setFixed(true);
			}
			// for(int k = std::max(config.staticframes-config.minframe,0); k< minpose_ ; k++){
			// 	c.poses_[k].pPose->setFixed(true);
			// }
			c.poses_[0].pPose->setFixed(true);
			c.optimizer_->initializeOptimization();
			//c.optimizer_->computeInitialGuess();
			c.optimizer_->setVerbose(false);
			//std::cout  << "optimizing \n";
			//c.optimizer_->save("initial.g2o");

			//int niterations = opt1(c.optimizer_ , 1);
			//assert(niterations > 0);
			perturbTraj(c);
			//std::cout  << "optimized \n";
			//c.optimizer_->save("01.g2o");
			updateMetaStates(c);

			moveBirth(c);
			checkGraph(c);
			//checkNumDet(c);
			updateFoV(c);
			checkNumDet(c);
			checkGraph(c);
			//std::cout  << "optimizing \n";
			// if (!c.optimizer_->verifyInformationMatrices(true)){
			// 	std::cerr << "info is bad\n";
			// }
			// niterations = c.optimizer_->optimize(ni);
			// assert(niterations > 0);
			//std::cout  << "optimized \n";
			//c.optimizer_->save("02.g2o");
			if (!c.reverted_)
				updateDAProbs(c, minpose_, maxpose_);
			for (int p = 0; p < maxpose_; p++)
			{
				c.prevDA_bimap_[p] = c.DA_bimap_[p];
			}
			double expectedChange = 0;
			bool inserted;
			std::map<
			std::vector<boost::bimap<int, int, boost::container::allocator<int>>>,
			TrajectoryWeight>::iterator it;

			{
				// do{
				// checkGraph(c);
				 //selectNN(c);
				// checkGraph(c);
				expectedChange += sampleDA(c);
				/*
				 std::cout << termcolor::magenta <<" =========== SexyTime!============\n" << termcolor::reset;
				 boost::uniform_int<> random_component(0, components_.size()-1);
				 int secondComp;
				 do{
				 secondComp=random_component(rfs::randomGenerators_[threadnum]);
				 }while(secondComp==i);
				 sexyTime(c,components_[secondComp]);
				 */

				if (iteration_ % config.birthDeathNumIter_ == 0)
				{
					switch ((iteration_ / config.birthDeathNumIter_) % 3)
					{
					case 0:
						 expectedChange += sampleLMDeath(c);
						// expectedChange += mergeLM(c);
						break;
					case 1:
						 expectedChange += sampleLMBirth(c);
						// expectedChange += mergeLM(c);
						break;

					case 2:
						 expectedChange += mergeLM(c);
						break;
					}


					checkGraph(c);
					updateGraph(c);
					moveBirth(c);
					checkGraph(c);
					// for(int k = 0; k< config.staticframes-config.minframe ; k++){
					// 	c.poses_[k].pPose->setFixed(true);
					// }
					// c.poses_[0].pPose->setFixed(true);
					c.optimizer_->initializeOptimization();
					int niterations = opt2(c.optimizer_ , 1);
					updatePoses(c);
					updateMetaStates(c);

					moveBirth(c);
					checkGraph(c);
					updateFoV(c);
					checkGraph(c);	
					updateDAProbs(c, minpose_, maxpose_);

				}

				for (int ng = 1; ng < config.numGibbs_; ng++)
				{
					if(ng%2==0){
						expectedChange += sampleDA(c);
						//reverseSampleDA(c);
					}else{
						reverseSampleDA(c);
						//expectedChange += sampleDA(c);
					}


					// if (i%7==0){
					// 	checkGraph(c);
					// 	updateGraph(c);
					// 	checkGraph(c);
					// 	for(int k = 0; k< minpose_ ; k++){
					// 		c.poses_[k].pPose->setFixed(true);
					// 	}
					// 	c.poses_[0].pPose->setFixed(true);
					// 	c.optimizer_->initializeOptimization();
					// 	int niterations = c.optimizer_->optimize(1);
					// 	// updateMetaStates(c);
					// 	// moveBirth(c);
					// 	// checkGraph(c);
					// 	// updateFoV(c);
					// 	// checkGraph(c);	
					// 	updateDAProbs(c, minpose_, maxpose_);
					// }
				}
				// for(int k = config.staticframes - config.minframe; k< minpose_ ; k++){
				// 			c.poses_[k].pPose->setFixed(false);
							
				// 		}
				
				// if (iteration_ % config.birthDeathNumIter_ == 0)
				// {
				// 	switch ((iteration_ / config.birthDeathNumIter_) % 3)
				// 	{
				// 	case 0:
				// 		 expectedChange += sampleLMBirth(c);
				// 		 expectedChange += mergeLM(c);
				// 		break;
				// 	case 1:
				// 		 expectedChange += sampleLMDeath(c);
				// 		 expectedChange += mergeLM(c);
				// 		break;

				// 	case 2:
				// 		 expectedChange += mergeLM(c);
				// 		break;
				// 	}
				// }
			}
			// expectedChange += sampleLMDeath(c);
			// expectedChange += sampleLMBirth(c);
			// expectedChange += sampleLMDeath(c);
			


			/*
			 if (!inserted) {
			 std::cout << "data association already inserted\n";
			 }
			 */
			//}while(!inserted);
			// printFoV(c);
			/*  print data association
			 if(i==0){
			 std::ofstream dafile;
			 std::stringstream filename;
			 filename << "DA__" << iteration_ << ".txt";
			 dafile.open(filename.str());
			 std::cout<<" iteraton " <<iteration_++  << " :::: \n";
			 printDA(c,dafile);
			 }
			 */
			// printDAProbs(c);
			checkGraph(c);
			updateGraph(c);
			moveBirth(c);
			checkGraph(c);
			for(int k = 0; k< config.staticframes-config.minframe ; k++){
				c.poses_[k].pPose->setFixed(true);
			}
			// for(int k = std::max(config.staticframes-config.minframe,0); k< minpose_ ; k++){
			// 	c.poses_[k].pPose->setFixed(true);
			// }
			c.poses_[0].pPose->setFixed(true);
			c.optimizer_->initializeOptimization();
			//c.optimizer_->computeInitialGuess();
			c.optimizer_->setVerbose(false);
			if (!c.optimizer_->verifyInformationMatrices(true)){
				std::cerr << "info is bad\n";
			}
			int niterations = opt3(c.optimizer_ , ni);
			updatePoses(c);
			updateMetaStates(c);
			updateFoV(c);
			//assert(niterations > 0);
			// std::cout <<"niterations  " <<c.optimizer_->optimize(ni) << "\n";
			calculateWeight(c);

			TrajectoryWeight to_insert;
			to_insert.weight = c.logweight_;
			to_insert.trajectory.resize(c.poses_.size() );
			for(int numpose =0; numpose< maxpose_; numpose++){
				to_insert.trajectory[numpose].pose=c.poses_[numpose].pPose->estimate();
			}
			for(int numpose = maxpose_; numpose< c.poses_.size(); numpose++){
				to_insert.trajectory[numpose].pose=c.poses_[maxpose_-1].pPose->estimate();
				c.poses_[numpose].pPose->setEstimate(c.poses_[maxpose_-1].pPose->estimate());
			}
			auto pair = std::make_pair(c.DA_bimap_, to_insert);

			for ( int numpose =1 ; numpose < maxpose_; numpose++){
				double dist = (c.poses_[numpose].pPose->estimate().inverse().translation()-c.poses_[numpose-1].pPose->estimate().inverse().translation()).norm();
				if (dist>0.1){
					std::cout << termcolor::red << "dist to high setting w to -inf"  << "\n";
					std::cout << "chi2 " << c.odometries_[numpose-1]->chi2() << "\n";
					std::cout << "globalchi2 " << c.optimizer_->activeChi2() << "\n";
					std::cout << "dist " << dist << "\n" << termcolor::reset ;
				
					c.logweight_ = -std::numeric_limits<double>::infinity();
				}
				

				//assert (dist<0.1);
			}
			
			std::cout << termcolor::blue << "========== current:"
							  <<  c.logweight_ << " ============\n"
							  << termcolor::reset;
			
			checkGraph(c);
			#pragma omp critical(insert)
			{
			}

			#pragma omp critical(bestweight)
			{
				if (c.logweight_ > bestWeight_-config.maxWeightDifference){
					std::tie(it, inserted) = visited_.insert(pair);
					insertionP_ = insertionP_ * 0.99;
					if (inserted){
						insertionP_ += 0.01;
						it->second.weight = c.logweight_;
					}
				}
				if (c.logweight_ > bestWeight_)
				{
					bestWeight_ = c.logweight_;
					best_DA_ = c.DA_bimap_;
					std::stringstream filename;

					best_DA_max_detection_time_ = std::max(maxpose_ - 1, 0);
					while (best_DA_max_detection_time_ > 0 && best_DA_[best_DA_max_detection_time_].size() == 0)
					{
						best_DA_max_detection_time_--;
					}
					filename << "video/beststate_" << std::setfill('0')
							 << std::setw(5) << iterationBest_++ << ".g2o";
					c.optimizer_->save(filename.str().c_str(), 0);
					std::cout << termcolor::yellow << "========== newbest:"
							  << bestWeight_ << " ============\n"
							  << termcolor::reset;
					std::cout << "globalchi2 " << c.optimizer_->activeChi2() << "\n";
					std::cout <<"  determinant: " << c.linearSolver_->_determinant<< "\n";
					// for ( int numpose =1 ; numpose < maxpose_; numpose++){
					// 	double dist = (c.poses_[numpose].pPose->estimate().translation()-c.poses_[numpose-1].pPose->estimate().translation()).norm();
					// 	std::cout << "chi2 " << c.odometries_[numpose-1]->chi2() << "\n";
					// 	std::cout << "globalchi2 " << c.optimizer_->activeChi2() << "\n";
					// 	std::cout << "dist " << dist << "\n";

					// 	assert (dist<0.1);
					// }
					//printDA(c);
					std::stringstream name;
					static int numbest=0;
					name <<  "best__" << numbest++ << ".tum"; 
					//c.optimizer_->save(name.str().c_str());
					c.saveAsTUM(name.str(),config.base_link_to_cam0_se3);

					if (config.use_gui_)
					{
						std::cout << termcolor::yellow << "========== piblishingmarkers:"
								  << bestWeight_ << " ============\n"
								  << termcolor::reset;
						//perturbTraj(c);
						publishMarkers(c);
					}

					for (auto i = visited_.begin(), last = visited_.end(); i != last; ) {
					if (i->second.weight < bestWeight_-config.maxWeightDifference ) {
						i = visited_.erase(i);
					} else {
						++i;
					}
					}

				}
			}
			// double accept = std::min(1.0 ,  std::exp(c.logweight_-c.prevLogWeight_ - std::min(expectedChange, 0.0) ));

			/*
			 boost::uniform_real<> uni_dist(0, 1);

			 std::cout << "accept: " << accept << "thred " << threadnum<<"\n";
			 std::cout << "weight: " << c.logweight_ << " prevWeight: " << c.prevLogWeight_ << " expectedChange " << expectedChange << "   chi2:  " <<c.optimizer_->activeChi2() << "  determinant: " << c.linearSolver_->_determinant<< "\n";


			 if(uni_dist(rfs::randomGenerators_[threadnum]) > accept){
			 std::cout << "===================================================REVERT========================================================\n";
			 revertDA(c);
			 }else{
			 c.reverted_= false;
			 } */
		}

		iteration_++;

		// temp_ *= config.tempFactor_;

		std::cout << "insertionp: " << insertionP_ << " temp: " << temp_ << "\n";

		// if (insertionP_ > 0.95 && temp_ > 5e-1)
		// {
		// 	temp_ *= 0.5;
		// 	std::cout << "reducing: \n";
		// }
		// if (insertionP_ < 0.05 && temp_ < 1e5)
		// {
		// 	temp_ *= 2;
		// 	std::cout << "augmenting: ";
		// }

		// if (config.use_gui_)
		// {
		// 	std::cout << termcolor::yellow << "piblishingmarkers\n"
		// 			  << termcolor::reset;
		// 	publishMarkers(components_[0]);
		// }
	}
	inline void VectorGLMBSLAM6D::calculateWeight(VectorGLMBComponent6D &c)
	{
		double logw = 0;
		double pd_logw =0;
		double kappa_logw =0;
		double info_logw =0;
		double descriptor_logw=0;
		double scale_logw=0;
		for (int k = 0; k < c.poses_.size(); k++)
		{
			for (int nz = 0; nz < c.poses_[k].Z_.size(); nz++)
			{
				auto it = c.DA_bimap_[k].left.find(nz);
				int selectedDA = -5;
				if (it != c.DA_bimap_[k].left.end())
				{
					selectedDA = it->second;
				}
				if (selectedDA < 0)
				{
					kappa_logw += config.logKappa_;
				}
				else
				{
					info_logw +=
						-0.5 * (c.poses_[k].Z_[nz]->dimension() * std::log(2 * M_PI) - std::log(
																						   c.poses_[k].Z_[nz]->information().determinant()));


										StereoMeasurementEdge  &z = *c.poses_[k].Z_[nz];

						auto &lm = c.landmarks_[selectedDA - c.landmarks_[0].id];
						int lmFovIdx =-1;
						for (int n=0;n< c.poses_[k].fov_.size();n++){
							if(c.poses_[k].fov_[n]==lm.id){
								lmFovIdx=n;
								break;
							}
						}
						assert(lmFovIdx>=0);
						assert(c.poses_[k].predicted_scales.size() == c.poses_[k].fov_.size() );
						if (lmFovIdx<0){
							// std::cout << "ERROR lmid: " << lm.id << "\n";
							// std::cout << "k: " << k << "\n";
							// std::cout << "fov: ";
							// for (int kk:c.poses_[k].fov_) {
							// 	std::cout << kk << "  ";
							// 	}
							// std::cout << "\n";
							scale_logw += -std::numeric_limits<double>::infinity();
						}else{
							int predictedScale = c.poses_[k].predicted_scales[lmFovIdx];
							int scalediff = abs(predictedScale-c.poses_[k].keypoints_left[c.poses_[k].matches_left_to_right[nz].queryIdx].octave );

							scale_logw += 20-20.0*scalediff;
						}
						//int dist = ORBDescriptor::distance(lm.descriptor, c.poses_[k].descriptors_left[c.poses_[k].matches_left_to_right[nz].queryIdx]);
						double desclikelihood = lm.descriptor.likelihood(c.poses_[k].descriptors_left[c.poses_[k].matches_left_to_right[nz].queryIdx] , c.poses_[k].descriptors_right[c.poses_[k].matches_left_to_right[nz].trainIdx]);

						descriptor_logw += desclikelihood;
						if (!std::isfinite(descriptor_logw)){
							for (int kk:lm.is_in_fov_){
								auto it = c.DA_bimap_[kk].right.find(lm.id);
								if (it != c.DA_bimap_[kk].right.end()){
									std::cout << "k " << kk << " nz  " <<it->second << "\n";
								}
							}
							assert(0);
						}


				}
			}

			for (int lm = 0; lm < c.poses_[k].fov_.size(); lm++)
			{

				if (c.DA_bimap_[k].right.count(c.poses_[k].fov_[lm]) > 0)
				{
					pd_logw += std::log(config.PD_);
				}
				else
				{
					bool exists = c.landmarks_[c.poses_[k].fov_[lm] - c.landmarks_[0].pPoint->id()].numDetections_ > 0;
					if (exists)
					{
						pd_logw += std::log(1 - config.PD_);
					}
				}
			}
		}
		logw += pd_logw;
		logw += kappa_logw;
		logw += info_logw;
		logw += descriptor_logw;
		logw += scale_logw;
		double lm_exist_w=0;
		for (int lm = 0; lm < c.landmarks_.size(); lm++)
		{
			bool exists = c.landmarks_[lm].numDetections_ > 0;
			if (exists)
			{
				lm_exist_w += config.logExistenceOdds;
			}
		}
		logw += lm_exist_w;
		double chi_logw=-0.5 * (c.optimizer_->activeChi2());
		double det_logw = -0.5* c.linearSolver_->_determinant;

		logw += chi_logw+det_logw;
		 std::cout << termcolor::blue << "weight: " <<logw << " det_logw: " <<     det_logw  
		 	<< " pd_logw: " <<     pd_logw  << " kappa_logw: " <<     kappa_logw  
			<< " info_logw: " <<     info_logw 
			<< " lm_exist_w: " <<     lm_exist_w 
			<< " descriptor_logw: " <<     descriptor_logw 
			<< " scale_logw: " <<     scale_logw << "\n"      <<termcolor::reset <<"\n";
		assert(!isnan(logw));
		c.prevLogWeight_ = c.logweight_;
		c.logweight_ = logw;
	}

	inline void VectorGLMBSLAM6D::updateGraph(VectorGLMBComponent6D &c)
	{
		checkGraph(c);
		
		for (int k = 0; k < maxpose_; k++)
		{
			if (c.optimizer_->vertex(c.poses_[k].pPose->id())==NULL){
				c.optimizer_->addVertex(c.poses_[k].pPose);

			}
			if(k>0){
				if (c.optimizer_->edges().find(c.odometries_[k-1]) == c.optimizer_->edges().end())
					c.optimizer_->addEdge(c.odometries_[k-1]);
			}
		}
		
		for (int k = 0; k < maxpose_; k++)
		{
			int prevedges =  c.poses_[k].pPose->edges().size();
			int numselections=0;
			int addededges = 0;
			for (int nz = 0; nz < c.poses_[k].Z_.size(); nz++)
			{
				int selectedDA = -5;
				auto it = c.DA_bimap_[k].left.find(nz);

				if (it != c.DA_bimap_[k].left.end())
				{
					selectedDA = it->second;
				}
				if (selectedDA>=0){
					numselections++;
				}
				int previd =
					c.poses_[k].Z_[nz]->vertex(0) ? c.poses_[k].Z_[nz]->vertex(0)->id() : -5; /**< previous data association */
				if (previd >= 0)
				{
					assert(c.optimizer_->edges().find(c.poses_[k].Z_[nz]) != c.optimizer_->edges().end() );
				}else{
					assert(c.optimizer_->edges().find(c.poses_[k].Z_[nz]) == c.optimizer_->edges().end() );
				}
				if (previd == selectedDA)
				{
					continue;
				}
				if (selectedDA >= 0)
				{
					// if vertex is not in graph add it
					auto vertex_it = c.optimizer_->vertices().find( selectedDA);
					if ( vertex_it == c.optimizer_->vertices().end() ){
						c.optimizer_->addVertex(c.landmarks_[selectedDA-c.landmarks_[0].id].pPoint);
					}
					auto vertex_pt = dynamic_cast<g2o::OptimizableGraph::Vertex *>(c.optimizer_->vertices().find(selectedDA)->second);
					assert( vertex_pt == c.landmarks_[selectedDA-c.landmarks_[0].id].pPoint );
					// if edge was already in graph, modify it
					if (previd >= 0)
					{
						assert(c.optimizer_->edges().find(c.poses_[k].Z_[nz]) != c.optimizer_->edges().end() );
						c.optimizer_->setEdgeVertex(c.poses_[k].Z_[nz], 0,
													vertex_pt); // this removes the edge from the list in both vertices
						if (c.landmarks_[previd-c.landmarks_[0].id].pPoint->edges().size() == 0 ){
							c.optimizer_->removeVertex(c.landmarks_[previd-c.landmarks_[0].id].pPoint);
						}
					}
					else
					{
						addededges++;
						c.poses_[k].Z_[nz]->setVertex(0,  vertex_pt);
						
						c.optimizer_->addEdge(c.poses_[k].Z_[nz]);
						assert(c.optimizer_->edges().find(c.poses_[k].Z_[nz]) != c.optimizer_->edges().end() );
																							
					}
					assert(c.optimizer_->edges().find(c.poses_[k].Z_[nz]) != c.optimizer_->edges().end() );
				}
				else
				{
					c.optimizer_->removeEdge(c.poses_[k].Z_[nz]);
					c.poses_[k].Z_[nz]->setVertex(0, NULL);
					if (c.landmarks_[previd-c.landmarks_[0].id].pPoint->edges().size() == 0 ){
						c.optimizer_->removeVertex(c.landmarks_[previd-c.landmarks_[0].id].pPoint);
					}
					
				}
				
			}
			int expected_num_edges = c.DA_bimap_[k].size()+1;
			if (k>0 && k < maxpose_-1){
				expected_num_edges++;
			}
			if(c.poses_[k].pPose->edges().size() !=  expected_num_edges ){
				checkDA(c);
				assert(0);
			}
			
		}
		for (auto &lm:c.landmarks_){
			if (lm.numDetections_ != lm.pPoint->edges().size() ){
								//printDA(c);
								assert(0);
							}

		}

	}
	template <class MapType>
	void print_map(const MapType &m, std::ostream &s = std::cout)
	{
		typedef typename MapType::const_iterator const_iterator;
		for (const_iterator iter = m.begin(), iend = m.end(); iter != iend;
			 ++iter)
		{
			s << iter->first << "-->" << iter->second << std::endl;
		}
	}
	inline void VectorGLMBSLAM6D::printFoV(VectorGLMBComponent6D &c)
	{
		std::cout << "FoV:\n";
		for (int k = 0; k < c.poses_.size(); k++)
		{
			std::cout << k << "  FoV at:   ";
			for (int lmid : c.poses_[k].fov_)
			{
				std::cout << "  ,  " << lmid;
			}
			std::cout << "\n";
		}
	}
	inline void VectorGLMBSLAM6D::printDAProbs(VectorGLMBComponent6D &c)
	{
		for (int k = 0; k < c.DAProbs_.size(); k++)
		{
			if (k == 2)
				break;
			std::cout << k << "da probs:\n";
			for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
			{
				std::cout << "z =  " << nz << "  ;";
				for (double l : c.DAProbs_[k][nz].l)
				{
					std::cout << std::max(l, -100.0) << " , ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
	}

	inline void VectorGLMBSLAM6D::printDA(VectorGLMBComponent6D &c,
										  std::ostream &s)
	{
		s << "bimap: \n";
		for (int k = 0; k < c.DA_bimap_.size(); k++)
		{
			s << k << ":\n";
			print_map(c.DA_bimap_[k].left, s);
		}

		// s << "optimizer: \n";

		// for (int k = 0; k < c.poses_.size(); k++)
		// {
		// 	s << k << ":\n";
		// 	for(int nz =0; nz < c.poses_[k].Z_.size(); nz++){
		// 		s << nz << " : " <<  (c.poses_[k].Z_[nz]->vertex(0) ? c.poses_[k].Z_[nz]->vertex(0)->id():-10) << "\n";
		// 	}
		// }
	}

	inline void VectorGLMBSLAM6D::checkDA(VectorGLMBComponent6D &c,
										  std::ostream &s)
	{
		//s << "bimap: \n";
		for (int k = 0; k < c.DA_bimap_.size(); k++)
		{
			//s << k << ":\n";
			//print_map(c.DA_bimap_[k].left, s);
		}

		//s << "optimizer: \n";

		for (int k = 0; k < c.poses_.size(); k++)
		{
			//s << k << ":\n";
			for(int nz =0; nz < c.poses_[k].Z_.size(); nz++){
				int graph_da = (c.poses_[k].Z_[nz]->vertex(0) ? c.poses_[k].Z_[nz]->vertex(0)->id():-5);
				
				auto it = c.DA_bimap_[k].left.find(nz);
				int bimap_da = -5;
				if (it != c.DA_bimap_[k].left.end())
				{
					bimap_da = it->second;
				}
				assert(graph_da == bimap_da);
				if (graph_da>=0){
					
					assert (c.optimizer_->vertices().find(c.poses_[k].Z_[nz]->vertex(0)->id() ) != c.optimizer_->vertices().end() );
					assert (c.optimizer_->vertices().find(c.poses_[k].Z_[nz]->vertex(0)->id()) != c.optimizer_->vertices().end() );
					assert (c.optimizer_->edges().find(c.poses_[k].Z_[nz]) != c.optimizer_->edges().end() );
				}

			}
		}
	}
	inline void VectorGLMBSLAM6D::checkNumDet(VectorGLMBComponent6D &c,
										  std::ostream &s)
	{
		for (auto &lm:c.landmarks_){
			int numdet=0;
			for(int k=0;k< maxpose_;k++){
				auto it = c.DA_bimap_[k].right.find(lm.id);
				if (it != c.DA_bimap_[k].right.end()){
					numdet++;

				}

			}
			assert(numdet==lm.numDetections_);
		}
	}	
	
	inline void VectorGLMBSLAM6D::checkGraph(VectorGLMBComponent6D &c,
										  std::ostream &s)
	{

		for (int k = 0; k < c.poses_.size(); k++)
		{
			
			for(int nz =0; nz < c.poses_[k].Z_.size(); nz++){
				int graph_da = (c.poses_[k].Z_[nz]->vertex(0) ? c.poses_[k].Z_[nz]->vertex(0)->id():-5);
				


				if (graph_da>=0){
					
					assert (c.optimizer_->vertices().find(c.poses_[k].Z_[nz]->vertex(0)->id() ) != c.optimizer_->vertices().end() );
					assert (c.optimizer_->vertices().find(c.poses_[k].Z_[nz]->vertex(1)->id()) != c.optimizer_->vertices().end() );
					assert (c.optimizer_->edges().find(c.poses_[k].Z_[nz]) != c.optimizer_->edges().end() );
				}else{

					assert (c.optimizer_->edges().find(c.poses_[k].Z_[nz]) == c.optimizer_->edges().end() );
				}

			}
		}
	}

	inline double VectorGLMBSLAM6D::sampleLMBirth(VectorGLMBComponent6D &c)
	{
		double expectedWeightChange = 0;
		boost::uniform_real<> uni_dist(0, 1);
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif

		for (int i = 0; i < c.landmarks_.size(); i++)
		{
			auto &lm=c.landmarks_[i];
			if (lm.numDetections_ > 0 || lm.numFoV_ == 0 || lm.birthTime_>c.poses_[maxpose_-1].stamp || lm.birthTime_<c.poses_[minpose_].stamp)
			{
				continue;
			}
			//
			//c.landmarksInitProb_[i] = c.landmarksInitProb_[i] / (c.landmarks_[i].numFoV_ * config.PD_);
			double numdet = c.landmarksInitProb_[i];
			//c.landmarksInitProb_[i] =-(numdet)*config.logKappa_ +c.landmarksInitProb_[i]+ config.logExistenceOdds;
			
			double aux= exp(c.landmarksInitProb_[i] );
			double p = aux/(1+aux);
			if (p > uni_dist(randomGenerators_[threadnum]))
			{
				// reset all associations to false alarms
				expectedWeightChange += (config.logKappa_ + (1 - config.PD_)) * c.landmarks_[i].numFoV_;
				//int numdet = 0;
				for (int k = minpose_; k < maxpose_; k++)
				{
					double maxl = -std::numeric_limits<double>::infinity();
					int max_nz = -5;
					for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
					{
						// if measurement is associated, continue
						auto it = c.DA_bimap_[k].left.find(nz);
						if (it != c.DA_bimap_[k].left.end())
						{
							continue;
						}
						for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++)
						{
							if (c.DAProbs_[k][nz].i[a] == c.landmarks_[i].pPoint->id())
							{
								if (c.DAProbs_[k][nz].l[a] > c.DAProbs_[k][nz].l[0] && c.DAProbs_[k][nz].l[a] > maxl)
								{
									maxl = c.DAProbs_[k][nz].l[a];
									max_nz = nz;
								}
							}
						}
					}
					if (max_nz>=0){
						c.DA_bimap_[k].insert( {max_nz, c.landmarks_[i].pPoint->id()});
						expectedWeightChange += maxl - config.logKappa_;
						c.landmarks_[i].numDetections_++;
					}

				}
				expectedWeightChange += config.logExistenceOdds;
				
				 std::cout << termcolor::green << "LANDMARK BORN "
				 << termcolor::reset << " initprob: "
				 << c.landmarksInitProb_[i] << " numDet "
				 << c.landmarks_[i].numDetections_ << " nd: "
				 << numdet << " numfov: "
				 << c.landmarks_[i].numFoV_ << "  expectedChange "
				 << expectedWeightChange << "\n";
				 

				//c.landmarks_[i].numDetections_ = 0;
			}
		}

		// std::cout << "Death Change  " <<expectedWeightChange <<"\n";
		checkNumDet(c);
		return expectedWeightChange;
	}

	inline double VectorGLMBSLAM6D::mergeLM(VectorGLMBComponent6D &c)
	{
		double expectedWeightChange = 0;
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif

		
		if (c.tomerge_.size() == 0)
		{
			std::cout << termcolor::blue << "no jumps so no merge \n"
					  << termcolor::reset;
			return 0;
		}
		int nummerge=1;
		
		//std::cout  << termcolor::red << "nummerge: " << nummerge << "\n" << termcolor::reset;
		for(int i=0; i< nummerge ; i++){

		if (c.tomerge_.size() == 0)
			{
				return 0;
			}

		boost::uniform_int<> random_pair(0, c.tomerge_.size() - 1);

		int rp = random_pair(rfs::randomGenerators_[threadnum]);
		int todelete = c.tomerge_[rp].first;

		int toAddMeasurements = c.tomerge_[rp].second;

		

		if (c.landmarks_[toAddMeasurements - c.landmarks_[0].pPoint->id()].numDetections_ <  c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_ ){
			todelete = c.tomerge_[rp].second;
			toAddMeasurements = c.tomerge_[rp].first;
		}
		 auto &del_lm = c.landmarks_[todelete - c.landmarks_[0].pPoint->id()];
		 auto &add_lm = c.landmarks_[toAddMeasurements - c.landmarks_[0].pPoint->id()];
		 if (del_lm.numDetections_ ==0 ){
			continue;
		 }

		for (int k = 0; k < maxpose_; k++)
		{
			auto it = c.DA_bimap_[k].right.find(todelete);
			auto itadd = c.DA_bimap_[k].right.find(toAddMeasurements);

			if (it != c.DA_bimap_[k].right.end())
			{
				if (itadd != c.DA_bimap_[k].right.end()){
					c.DA_bimap_[k].right.erase(it);
					c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_--;
					assert(c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_ >= 0);
					continue;
				}

				// for (int l = 0; l < c.DAProbs_[k][it->second].i.size(); l++)
				// {
				// 	if (c.DAProbs_[k][it->second].i[l] == it->first)
				// 	{

				// 		expectedWeightChange -= c.DAProbs_[k][it->second].l[l];
				// 		break;
				// 	}
				// }
				c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_--;
				assert(c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_ >= 0);
				c.landmarks_[toAddMeasurements - c.landmarks_[0].pPoint->id()].numDetections_++;

				bool result = c.DA_bimap_[k].right.replace_key(it, toAddMeasurements);
				if (!result){

					std::cerr << termcolor::red  << "key not replaced\n"<< termcolor::reset;
				}
			}
		}
		if (c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_ != 0)
		{
			std::cerr << "landmark  numDetections_ not zero"
					  << c.landmarks_[todelete - c.landmarks_[0].pPoint->id()].numDetections_
					  << "\n";
		}

		for (int n=0; n <c.tomerge_.size(); n++){
			if (c.tomerge_[n].first == todelete || c.tomerge_[n].second == todelete || c.tomerge_[n].first == toAddMeasurements || c.tomerge_[n].second == toAddMeasurements){
				c.tomerge_[n] = c.tomerge_[c.tomerge_.size()-1];
				c.tomerge_.pop_back();
			}
		}
		}
		c.tomerge_.clear();
		checkNumDet(c);
		return expectedWeightChange;
	}

	inline double VectorGLMBSLAM6D::sampleLMDeath(VectorGLMBComponent6D &c)
	{
		double expectedWeightChange = 0;
		boost::uniform_real<> uni_dist(0, 1);
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif

		for (int i = 0; i < c.landmarks_.size(); i++)
		{
			if (c.landmarks_[i].numDetections_ < 1)
			{
				continue;
			}
			if (c.landmarks_[i].birthTime_< c.poses_[minpose_].stamp){
				continue;
			}

			c.landmarksResetProb_[i] =(c.landmarks_[i].numDetections_)*config.logKappa_ -(c.landmarks_[i].numDetections_) * std::log(config.PD_) - (c.landmarks_[i].numFoV_ - c.landmarks_[i].numDetections_) * std::log(1 - config.PD_) - config.logExistenceOdds;
			//c.landmarksResetProb_[i] =( (double)( c.landmarks_[i].numDetections_))/c.landmarks_[i].numFoV_;
			double det_r =( (double)( c.landmarks_[i].numDetections_))/c.landmarks_[i].numFoV_;
			// double p;
			// if (det_r > config.PD_){
			// 	p = 0.01;
			// }else{
			// 	p = 0.01+(config.PD_-det_r)/10.0;
			// }
			double aux= exp(c.landmarksResetProb_[i] );
			double p = aux/(1+aux);
			if (uni_dist(randomGenerators_[threadnum]) < p )
			{
				/*
				 c.landmarksResetProb_[i] = (1-((double)c.landmarks_[i]numDetections_)/c.landmarks_[i].numFoV_)*(config.PD_);
				 if(uni_dist(randomGenerators_[threadnum]) < c.landmarksResetProb_[i]){*/
				// reset all associations to false alarms
				expectedWeightChange += config.logKappa_ * c.landmarks_[i].numDetections_;
				int numdet = 0;
				for (int k = 0; k < maxpose_; k++)
				{
					auto it = c.DA_bimap_[k].right.find(
						c.landmarks_[i].pPoint->id());
					if (it != c.DA_bimap_[k].right.end())
					{
						// for (int l = 0; l < c.DAProbs_[k][it->second].i.size();
						// 	 l++)
						// {
						// 	if (c.DAProbs_[k][it->second].i[l] == it->first)
						// 	{
						// 		numdet++;
						// 		expectedWeightChange -=
						// 			c.DAProbs_[k][it->second].l[l];
						// 		break;
						// 	}
						// }
						c.DA_bimap_[k].right.erase(it);
					}
				}
				expectedWeightChange +=-config.logExistenceOdds;
				expectedWeightChange += -std::log(1 - config.PD_) * c.landmarks_[i].numFoV_;
				
				 std::cout << termcolor::red << "KILL LANDMARK\n" << termcolor::reset
				 << c.landmarksResetProb_[i] << " n "
				 << c.landmarks_[i].numDetections_ << " nfov:"
				 << c.landmarks_[i].numFoV_ << "  expectedChange "
				 << expectedWeightChange << "\n";
				 

				c.landmarks_[i].numDetections_ = 0;
			}
		}

		checkNumDet(c);
		// std::cout << "Death Change  " <<expectedWeightChange <<"\n";
		return expectedWeightChange;
	}
	inline double VectorGLMBSLAM6D::reverseSampleDA(VectorGLMBComponent6D &c){
		//checkNumDet(c);
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif

		//c.tomerge_.clear();
		//AssociationProbabilities probs;
		double expectedWeightChange = 0;
		for (unsigned int k = minpose_; k < maxpose_; k++){
			break;
			std::cout  << "testk " << k  << " maxpose " << maxpose_ << "\n";
			if(k>=maxpose_){
				break;
			}
			std::cout  << "testk " << k << "\n";
			for(int lmFovIdx =0; lmFovIdx <c.poses_.at(k).fov_.size();lmFovIdx++){
				int lmidx = c.poses_.at(k).fov_.at(lmFovIdx);

				auto &lm = c.landmarks_[lmidx-c.landmarks_[0].id];



				
				
					std::vector<double> P;
					AssociationProbabilities probs;




					probs.i.push_back(-5);
					probs.l.push_back(config.logKappa_);


					if (lm.numDetections_==0 ){
						if (lm.birthTime_ < c.poses_[k].stamp && lm.birthTime_ > c.poses_[k].stamp-1.0){
							probs.l[0] += -config.logExistenceOdds - (lm.numFoV_-1) * std::log(1 - config.PD_) ;
						}else{
							continue;
						}
					}



					int prevDA = -5;
					auto it = c.DA_bimap_[k].right.find(lm.id);
					if (it != c.DA_bimap_[k].right.end())
					{
						prevDA = it->second;
					}

					if(lm.numDetections_>0){
						if(&c.poses_[k] == lm.birthPose){
							assert(prevDA == lm.birthMatch);
							continue;
						}
					}


					if (lm.numDetections_==1 && prevDA >=0){
						probs.l[0] += -config.logExistenceOdds - (lm.numFoV_-1) * std::log(1 - config.PD_);
					}



					double maxprob = probs.l[0];
					int maxprobi = 0;

					for (int a = 0; a < c.reverseDAProbs_[k][lmFovIdx].i.size(); a++)
					{
						int nz= c.reverseDAProbs_[k][lmFovIdx].i[a];
						double likelihood = c.reverseDAProbs_[k][lmFovIdx].l[a];

						if (nz == prevDA){
							
							probs.i.push_back(c.reverseDAProbs_[k][lmFovIdx].i[a]);
							probs.l.push_back(likelihood);
							if (likelihood > maxprob)
							{
								maxprob = likelihood;
								maxprobi = a;
							}

							
						}else{
							if (c.DA_bimap_[k].left.count(nz) == 0)
							{ // measurement is not already associated to another landmark
								
								probs.i.push_back(c.reverseDAProbs_[k][lmFovIdx].i[a]);
								probs.l.push_back(likelihood);
								if (likelihood > maxprob)
								{
									maxprob = likelihood;
									maxprobi = a;
								}
								
							}
						}

					}

				//probs calculated now sampling

				P.resize(probs.l.size());
				double alternativeprob = 0;
				for (int i = 0; i < P.size(); i++)
				{

					P[i] = exp((probs.l[i] - maxprob)); // /temp_);

					// std::cout << p << "   ";
					alternativeprob += P[i];
				}

				size_t sample = GibbsSampler::sample(randomGenerators_[threadnum],
													 P);
				

				if (probs.i[sample] != prevDA)
				{ // if selected association, change bimap

					if (probs.i[sample] >= 0)
					{
						if (prevDA<0){
							lm.numDetections_++;
							auto result = c.DA_bimap_[k].insert({ probs.i[sample] ,  lmidx});

							assert(result.second);
						}
						else
						{
							bool result = c.DA_bimap_[k].right.replace_data(it, probs.i[sample]);
							assert(result);

						}
						if (lm.numDetections_>0 ){
							int kbirth = lm.birthPose->pPose->id();
							
							auto birth_it = c.DA_bimap_[kbirth].left.find(lm.birthMatch);
							if (birth_it == c.DA_bimap_[kbirth].left.end()){
								auto birth_it2 = c.DA_bimap_[kbirth].right.find(lmidx); 
								if (birth_it2 == c.DA_bimap_[kbirth].right.end()){
									if(lm.numDetections_== lm.numFoV_){
										std::cout << termcolor::red << "adding imposssible det_:\n" << termcolor::reset;
										printDA(c);
										assert(0);
									}
									auto result = c.DA_bimap_[kbirth].insert({lm.birthMatch, lmidx });
									assert(result.second);
									lm.numDetections_++;
								}
							}
						}
					}
					else
					{ // if a change has to be made and new DA is false alarm, we need to remove the association
						c.DA_bimap_[k].right.erase(it);
						lm.numDetections_--;
						assert(lm.numDetections_ >= 0);
					}
				}


			}
		}


		checkNumDet(c);
		return 0.0;

	}

	inline double VectorGLMBSLAM6D::sampleDA(VectorGLMBComponent6D &c)
	{

		checkNumDet(c);
		std::vector<double> P;
		boost::uniform_real<> uni_dist(0, 1);
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif

		c.tomerge_.clear();
		AssociationProbabilities probs;
		double expectedWeightChange = 0;

		std::fill(c.landmarksResetProb_.begin(), c.landmarksResetProb_.end(),
				  -config.logExistenceOdds);
		std::fill(c.landmarksInitProb_.begin(), c.landmarksInitProb_.end(), config.logExistenceOdds);
		for (int k = minpose_; k < maxpose_; k++)
		{

			for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
			{
				probs.i.clear();
				probs.l.clear();
				double maxprob = -std::numeric_limits<double>::infinity();
				int maxprobi = 0;
				auto it = c.DA_bimap_[k].left.find(nz);
				double selectedProb;
				int selectedDA = -5;
				if (it != c.DA_bimap_[k].left.end())
				{
					selectedDA = it->second;
					auto &selectedLM = c.landmarks_[selectedDA-c.landmarks_[0].id];
					if (selectedLM.numDetections_>1 && nz == selectedLM.birthMatch && selectedLM.birthPose == &(c.poses_[k]) ){
						// selected this measurement spawns landmark , cannot sample
						continue;

					}
				}
				double maxlikelihood = -std::numeric_limits<double>::infinity();
				int maxlikelihoodi = 0;
				for (int a = 0; a < c.DAProbs_[k][nz].i.size(); a++)
				{
					double likelihood = c.DAProbs_[k][nz].l[a];

					if (maxlikelihood < likelihood)
					{
						maxlikelihood = likelihood;
						maxlikelihoodi = a;
					}
					if (c.DAProbs_[k][nz].i[a] == -5)
					{
						probs.i.push_back(c.DAProbs_[k][nz].i[a]);
						probs.l.push_back(c.DAProbs_[k][nz].l[a]);
						if (c.DAProbs_[k][nz].l[a] > maxprob)
						{
							maxprob = c.DAProbs_[k][nz].l[a];
							maxprobi = a;
						}
					}
					else if (c.DAProbs_[k][nz].i[a] == selectedDA)
					{
						probs.i.push_back(c.DAProbs_[k][nz].i[a]);
						if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].numDetections_ == 1)
						{
							likelihood += config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].numFoV_) * std::log(1 - config.PD_);
							// std::cout <<" single detection: increase:  " << config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].pPoint->id()].numFoV_)*std::log(1-config.PD_) <<"\n";
							probs.l.push_back(likelihood);
						}
						else
						{
							probs.l.push_back(c.DAProbs_[k][nz].l[a]);
						}
						if (likelihood > maxprob)
						{
							maxprob = likelihood;
							maxprobi = a;
						}
					}
					else
					{
						if (c.DA_bimap_[k].right.count(c.DAProbs_[k][nz].i[a]) == 0)
						{ // landmark is not already associated to another measurement
							
							if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].numDetections_ == 0)
							{
								
								if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].birthTime_ < c.poses_[k].stamp
								&& c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].birthTime_ > c.poses_[k].stamp-1.0){
									likelihood +=
										config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].pPoint->id()].numFoV_) * std::log(1 - config.PD_);
									// std::cout <<" 0 detection: increase:  " << config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].pPoint->id()].numFoV_)*std::log(1-config.PD_)<<"\n";
									probs.i.push_back(c.DAProbs_[k][nz].i[a]);
									probs.l.push_back(likelihood);
									if (likelihood > maxprob)
									{
										maxprob = likelihood;
										maxprobi = a;
									}
								}
							}
							else
							{
								probs.i.push_back(c.DAProbs_[k][nz].i[a]);
								probs.l.push_back(c.DAProbs_[k][nz].l[a]);
								if (likelihood > maxprob)
								{
									maxprob = likelihood;
									maxprobi = a;
								}
							}
						}
					}
					if (c.DAProbs_[k][nz].i[a] == selectedDA)
					{
						expectedWeightChange -= probs.l[probs.l.size() - 1];
					}
				}

				P.resize(probs.l.size());
				double alternativeprob = 0;
				for (int i = 0; i < P.size(); i++)
				{

					P[i] = exp((probs.l[i] - maxprob)); // /temp_);

					// std::cout << p << "   ";
					alternativeprob += P[i];
				}

				size_t sample = GibbsSampler::sample(randomGenerators_[threadnum],
													 P);

				// alternativeprob=(alternativeprob -P[sample])/alternativeprob;
				if (alternativeprob < 1)
				{
					std::cout << P[maxprobi]
							  << " panicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanicpanic  \n";
				}

				expectedWeightChange += probs.l[sample];
				if (probs.i[sample] >= 0)
				{
					// c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0].pPoint->id()] *= (P[ P.size()-1] )/P[sample];
					/*
					 if(probs.i[sample] != c.DAProbs_[k][nz].i[maxprobi]){
					 c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0].pPoint->id()] +=  c.DAProbs_[k][nz].l[maxprobi] - probs.l[sample]; //(1 )/alternativeprob;

					 }else{
					 c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0].pPoint->id()] += probs.l[probs.l.size()-1] - probs.l[sample] ;
					 }*/

					c.landmarksResetProb_[probs.i[sample] - c.landmarks_[0].pPoint->id()] += std::log(
						P[sample] / alternativeprob);
				}
				else
				{

					if (c.DAProbs_[k][nz].i[maxlikelihoodi] >= 0)
					{
						// std::cout << "increasing init prob of lm " <<c.DAProbs_[k][nz].i[maxlikelihoodi] << "  by " <<maxlikelihood  << "- " << probs.l[sample]<< "\n";
						c.landmarksInitProb_[c.DAProbs_[k][nz].i[maxlikelihoodi] - c.landmarks_[0].pPoint->id()] += maxlikelihood-config.logKappa_;
					}
				}

				if (probs.i[sample] != selectedDA)
				{ // if selected association, change bimap

					if (probs.i[sample] >= 0)
					{
						c.landmarks_[probs.i[sample] - c.landmarks_[0].pPoint->id()].numDetections_++;
						if (selectedDA < 0)
						{
							auto result = c.DA_bimap_[k].insert({nz, probs.i[sample]});
							assert(result.second);
						}
						else
						{

							c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_--;
							assert(c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_ >= 0);
							auto result = c.DA_bimap_[k].left.replace_data(it, probs.i[sample]);
							assert(result);

							// add an log for possible landmark merge
							c.tomerge_.push_back(
								std::make_pair(probs.i[sample], selectedDA));
						}
						if (c.landmarks_[probs.i[sample] - c.landmarks_[0].pPoint->id()].numDetections_>0 ){
							int kbirth = c.landmarks_[probs.i[sample] - c.landmarks_[0].pPoint->id()].birthPose->pPose->id();
							auto &lm = c.landmarks_[probs.i[sample] - c.landmarks_[0].pPoint->id()];
							
							auto birth_it = c.DA_bimap_[kbirth].left.find(c.landmarks_[probs.i[sample] - c.landmarks_[0].pPoint->id()].birthMatch);
							if (birth_it == c.DA_bimap_[kbirth].left.end()){
								auto birth_it2 = c.DA_bimap_[kbirth].right.find(probs.i[sample]); 
								if (birth_it2 == c.DA_bimap_[kbirth].right.end()){
									if(lm.numDetections_== lm.numFoV_){
										std::cout << termcolor::red << "adding imposssible det_:\n" << termcolor::reset;
										printDA(c);
										assert(0);
									}
									auto result = c.DA_bimap_[kbirth].insert({c.landmarks_[probs.i[sample] - c.landmarks_[0].pPoint->id()].birthMatch, probs.i[sample] });
									assert(result.second);
									c.landmarks_[probs.i[sample] - c.landmarks_[0].pPoint->id()].numDetections_++;
								}
							}
						}
					}
					else
					{ // if a change has to be made and new DA is false alarm, we need to remove the association
						c.DA_bimap_[k].left.erase(it);
						c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_--;
							assert(c.landmarks_[selectedDA - c.landmarks_[0].pPoint->id()].numDetections_ >= 0);
					}
				}

			}
		}
		//printDA(c);

		checkNumDet(c);
		return expectedWeightChange;
	}
	inline void VectorGLMBSLAM6D::updateMetaStates(VectorGLMBComponent6D &c)
	{

		for (auto &map_point : c.landmarks_)
			{

				if (map_point.numDetections_ >0){
					
					std::vector<int> avg_desc(256,0);
					map_point.normalVector.setZero();
					double avg_depth=0;
					double max_depth=0;
					double min_depth= std::numeric_limits<double>::infinity();

					// std::vector<ORBDescriptor*> descriptors;
					// std::vector<int> distances_bef, distances_after;


					for (auto &edge: map_point.pPoint->edges()){

						int posenum = edge->vertex(1)->id()-c.poses_[0].pPose->id();
						
						auto it = c.DA_bimap_[posenum].right.find(map_point.id);
						assert ( it != c.DA_bimap_[posenum].right.end() );
						int nz = it->second;
						assert(edge == c.poses_[posenum].Z_[nz]);
						int nl = c.poses_[posenum].matches_left_to_right[nz].queryIdx;
						int nr = c.poses_[posenum].matches_left_to_right[nz].trainIdx;
						auto &desc_left = c.poses_[posenum].descriptors_left[nl];
						auto &desc_right = c.poses_[posenum].descriptors_right[nr];
						// distances_bef.push_back(ORBDescriptor::distance(map_point.descriptor , desc_left));
						// distances_bef.push_back(ORBDescriptor::distance(map_point.descriptor , desc_right));
						// descriptors.push_back(&c.poses_[posenum].descriptors_left[nl]);
						// descriptors.push_back(&c.poses_[posenum].descriptors_right[nr]);
						for(int i=0;i<256;i++){
							if (desc_left.desc[i])
								avg_desc[i]++;
							if (desc_right.desc[i])
								avg_desc[i]++;
						}
						Eigen::Vector3d lmpos = map_point.pPoint->estimate();
						Eigen::Vector3d poset = c.poses_[posenum].invPose.translation();
						int level = c.poses_[posenum].keypoints_left[nl].octave;
						double depth = (lmpos-poset).norm();
						if (max_depth<depth){
							max_depth = depth;
						}
						if (min_depth > depth){
							min_depth = depth;
						}
						avg_depth +=depth*c.poses_[posenum].mvScaleFactors[level];
						map_point.normalVector += (lmpos-poset).normalized();

					}
					assert(map_point.numDetections_ == map_point.pPoint->edges().size());
					avg_depth = avg_depth/map_point.numDetections_;
					map_point.mfMaxDistance = std::max( 1.2*avg_depth , 1.1*max_depth);
					//assert(map_point.mfMaxDistance<10.0);
					map_point.mfMinDistance = std::min(map_point.mfMaxDistance / c.poses_[0].mvScaleFactors[ c.poses_[0].mnScaleLevels - 1] * 0.8 , 0.9*min_depth);

					assert(max_depth < map_point.mfMaxDistance);
					assert(min_depth > map_point.mfMinDistance);

					for(int i=0;i<256;i++){
						map_point.descriptor.desc[i] = avg_desc[i]>map_point.numDetections_;
					}
					// int ii=0;
					// for (auto desc:descriptors){
					// 	distances_after.push_back(ORBDescriptor::distance(map_point.descriptor , *desc));
					// 	ii++;
					// 	if(ii%2 ==0 ){
					// 		assert(distances_after[ii-1]+distances_after[ii-2] <=200);
					// 	}
						
					// }
					map_point.normalVector.normalize();

				}else{
					map_point.normalVector = map_point.birthPose->point_camera_frame[map_point.birthMatch].normalized();
					map_point.descriptor = map_point.birthPose->descriptors_left[map_point.birthPose->matches_left_to_right[map_point.birthMatch].queryIdx];

				}

			}

		return;
	}

	inline void VectorGLMBSLAM6D::updateFoV(VectorGLMBComponent6D &c)
	{
		for (auto &map_point : c.landmarks_)
		{
			map_point.numFoV_ = 0;
			map_point.numFoV_ = 0;
			map_point.is_in_fov_.resize(0);
			map_point.predicted_scales.resize(0);
		}

		for (int k = 0; k < maxpose_; k++)
		{

			c.poses_[k].fov_.clear();
			c.poses_[k].predicted_scales.clear();

			if (c.poses_[k].Z_.size() > 0)
			{ // if no measurements we set FoV to empty ,

				for (int lm = 0; lm < c.landmarks_.size(); lm++)
				{
					auto it = c.DA_bimap_[k].right.find(c.landmarks_[lm].id);
					bool associated = it != c.DA_bimap_[k].right.end();
					bool isInFov = false;
					if(c.landmarks_[lm].birthTime_ > c.poses_[maxpose_-1].stamp){
						break;
					}

					if (c.landmarks_[lm].numDetections_ > 0 || (c.landmarks_[lm].birthTime_ <= c.poses_[maxpose_-1].stamp && c.landmarks_[lm].birthTime_ >= c.poses_[minpose_].stamp))
					//if ( (c.landmarks_[lm].birthTime_ <= c.poses_[maxpose_-1].stamp))
					{
						double predScale=0;
						if (associated || c.poses_[k].isInFrustum(&c.landmarks_[lm],
													config.viewingCosLimit_, config.g2o_cam_params, &predScale))
						{

							c.poses_[k].fov_.push_back(c.landmarks_[lm].pPoint->id());
							c.poses_[k].predicted_scales.push_back(predScale);
							c.landmarks_[lm].is_in_fov_.push_back(k);
							c.landmarks_[lm].predicted_scales.push_back(predScale);
							c.landmarks_[lm].numFoV_++;
							isInFov = true;
						}
					}
					// if (associated){
					// 	assert(isInFov);
					// }
				}
			}
		}
	}
	

	inline void VectorGLMBSLAM6D::updateDAProbs(VectorGLMBComponent6D &c,
												int minpose, int maxpose)
	{

		g2o::JacobianWorkspace jac_ws;
		StereoMeasurementEdge z;
		jac_ws.updateSize(2, 3 * 6);
		jac_ws.allocate();
		for (int k =0; k< minpose;k++){
			c.reverseDAProbs_[k].clear();
			c.DAProbs_[k].clear();
		}

		for (int k = minpose; k < maxpose; k++)
		{

			std::fill(c.reverseDAProbs_[k].begin(),c.reverseDAProbs_[k].end(), AssociationProbabilities() );
			c.reverseDAProbs_[k].resize(c.poses_[k].fov_.size());
			c.DAProbs_[k].resize(c.poses_[k].Z_.size());

			for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
			{
				c.DAProbs_[k][nz].i.clear();
				c.DAProbs_[k][nz].l.clear();
				c.DAProbs_[k][nz].i.reserve(c.poses_[k].fov_.size()+1);
				c.DAProbs_[k][nz].i.push_back(-5);
				c.DAProbs_[k][nz].l.push_back(config.logKappa_);
			}
			// double posHLogDet;
			// if (!c.poses_[k].pPose->fixed())
			// {
			// 	posHLogDet = std::log(c.poses_[k].pPose->hessianDeterminant());
			// }
			// PoseType::HessianBlockType poseHessian(
			// 	c.poses_[k].pPose->hessianData());


				// setting the topology of DAProbs to include all measurements in current FoV
				
				//c.DAProbs_[k][nz].i = c.poses_[k].fov_;
				//prechecking daprobs with descriptor distance

				// for(auto lmidx:c.poses_[k].fov_){
				// 	auto &lm = c.landmarks_[lmidx - c.landmarks_[0].id];
				// 	// int dist = descriptorDistance(c.poses_[k].descriptors_left.row(c.poses_[k].matches_left_to_right[nz].queryIdx),
				// 	// lm.descriptor);
				// 	int dist = ORBDescriptor::distance(lm.descriptor , c.poses_[k].descriptors_left[c.poses_[k].matches_left_to_right[nz].queryIdx]);

				// 	if (dist < 80){
				// 		c.DAProbs_[k][nz].i.push_back(lmidx); 
				// 	}
					
				// }
				// c.DAProbs_[k][nz].i.push_back(-5); // add posibility of false alarm
				// c.DAProbs_[k][nz].l.resize(c.DAProbs_[k][nz].i.size());

				
				// Eigen::Matrix<double, PoseType::HessianBlockType::RowsAtCompileTime,
				// 			  PoseType::HessianBlockType::ColsAtCompileTime>
				// 	poseHessianCopy;

				// if (!c.poses_[k].pPose->fixed())
				// {
				// 	poseHessianCopy = poseHessian;
				// }
				// if (selectedDA >= 0)
				// {
				// 	c.poses_[k].Z_[nz]->g2o::BaseBinaryEdge<3, g2o::Vector3, PointType, PoseType>::linearizeOplus(jac_ws);
				// 	StereoMeasurementEdge::JacobianXjOplusType Jpose =
				// 		c.poses_[k].Z_[nz]->jacobianOplusXj();
				// 	poseHessianCopy -= Jpose.transpose() * c.poses_[k].Z_[nz]->information() * Jpose;
				// }

				// double avg_desc_likelihood = 0;

				for(int lmFovIdx =0; lmFovIdx <c.poses_[k].fov_.size();lmFovIdx++){
					int lmidx = c.poses_[k].fov_[lmFovIdx];
					auto &lm = c.landmarks_[lmidx - c.landmarks_[0].id];

					Eigen::Vector3d predictedZ = c.poses_[0].Z_[0]->cam_project(c.poses_[k].pPose->estimate().map(lm.pPoint->estimate()), c.poses_[0].Z_[0]->bf);;
					

					for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
					{
						StereoMeasurementEdge  &z = *c.poses_[k].Z_[nz];
						double p=0.0;

						// auto it = c.DA_bimap_[k].left.find(nz);
						// int selectedDA = -5;
						// if (it != c.DA_bimap_[k].left.end())
						// {
						// 	selectedDA = it->second;
						// }
						// if (selectedDA < 0){
						// 	assert(c.poses_[k].Z_[nz]->vertex(0) == NULL);
						// 	assert(c.optimizer_->edges().find(c.poses_[k].Z_[nz]) == c.optimizer_->edges().end() );
						// }else{
						// 	assert(c.poses_[k].Z_[nz]->vertex(0)->id() == selectedDA);
						// 	assert(c.optimizer_->edges().find(c.poses_[k].Z_[nz]) != c.optimizer_->edges().end() );
						// }



						int predictedScale = c.poses_[k].predicted_scales[lmFovIdx];
						int scalediff = abs(predictedScale-c.poses_[k].keypoints_left[c.poses_[k].matches_left_to_right[nz].queryIdx].octave );
						if(scalediff >2){
							continue;
						}
						p += 20-20.0*scalediff;

						double error_scalar = (z.measurement()-predictedZ).norm();

						if (error_scalar > 50){
							continue;
						}

						//int dist = ORBDescriptor::distance(lm.descriptor, c.poses_[k].descriptors_left[c.poses_[k].matches_left_to_right[nz].queryIdx]);
						double desclikelihood = lm.descriptor.likelihood(c.poses_[k].descriptors_left[c.poses_[k].matches_left_to_right[nz].queryIdx] , c.poses_[k].descriptors_right[c.poses_[k].matches_left_to_right[nz].trainIdx]);
						if (!std::isfinite(desclikelihood) ){
							continue;
						}
						p += desclikelihood;

						p += std::log(config.PD_) - std::log(1 - config.PD_);
						// c.poses_[k].Z_[nz]->setVertex(0, lm.pPoint);

						//c.poses_[k].Z_[nz]->g2o::BaseBinaryEdge<3, g2o::Vector3, PointType, PoseType>::linearizeOplus(jac_ws);
						//c.poses_[k].Z_[nz]->computeError();
						


						// if pose is not fixed, calc updated pose and lm
						// if (false && !c.poses_[k].pPose->fixed())
						// {
						// 	PointType::HessianBlockType::PlainMatrix pointHessian;
						// 	//PointType::HessianBlockType pointHessian(h.data());

						// 	if (c.landmarks_[lmidx - c.landmarks_[0].pPoint->id()].numDetections_ > 0)
						// 	{
						// 		// assert(c.landmarks_[lmidx - c.landmarks_[0].pPoint->id()].pPoint->hessianData()!=NULL);
						// 		// new (&pointHessian) PointType::HessianBlockType(
						// 		// 	c.landmarks_[lmidx - c.landmarks_[0].pPoint->id()].pPoint->hessianData());
						// 		for (int i= 0; i < pointHessian.rows(); i++){
						// 			for (int j; j< pointHessian.cols() ;j++){
						// 				pointHessian(i,j) = lm.pPoint->hessian(i,j);
						// 			}
						// 		}
						// 		//h = lm.pPoint->hessian();

						// 		// std::cout << "g2o pointH: " << pointHessian << "\n\n\n";
						// 	}
						// 	else
						// 	{
						// 		pointHessian = config.anchorInfo_;
						// 		// std::cout << "calc pointH: " << pointHessian << "\n\n\n";
						// 	}
						// 	// std::cout << "numdetections:  " << c.landmarks_[lmidx - c.landmarks_[0].pPoint->id()].numDetections_  << "\n";

						// 	StereoMeasurementEdge::JacobianXjOplusType Jpose =
						// 		c.poses_[k].Z_[nz]->jacobianOplusXj();
						// 	StereoMeasurementEdge::JacobianXiOplusType Jpoint =
						// 		c.poses_[k].Z_[nz]->jacobianOplusXi();

						// 	Eigen::Matrix<double,
						// 				  PoseType::Dimension + PointType::Dimension,
						// 				  PoseType::Dimension + PointType::Dimension>
						// 		H;
						// 	Eigen::Matrix<double,
						// 				  PoseType::Dimension + PointType::Dimension, 1>
						// 		b, sol;
						// 	H.setZero();

						// 	H.block(0, 0, PoseType::Dimension, PoseType::Dimension) =
						// 		poseHessianCopy;
						// 	H.block(PoseType::Dimension, PoseType::Dimension,
						// 			PointType::Dimension, PointType::Dimension) =
						// 		pointHessian;

						// 	H.block(0, 0, PoseType::Dimension, PoseType::Dimension) +=
						// 		Jpose.transpose() * c.poses_[k].Z_[nz]->information() * Jpose;
						// 	H.block(PoseType::Dimension, PoseType::Dimension,
						// 			PointType::Dimension, PointType::Dimension) +=
						// 		Jpoint.transpose() * c.poses_[k].Z_[nz]->information() * Jpoint;

						// 	H.block(PoseType::Dimension, 0, PointType::Dimension,
						// 			PoseType::Dimension) = Jpoint.transpose() * c.poses_[k].Z_[nz]->information() * Jpose;
						// 	H.block(0, PoseType::Dimension, PoseType::Dimension,
						// 			PointType::Dimension) = H.block(PoseType::Dimension, 0, PointType::Dimension,
						// 											PoseType::Dimension)
						// 										.transpose();
						// 	b.block(0, 0, PoseType::Dimension, 1) =
						// 		Jpose.transpose() * omega_r;
						// 	b.block(PoseType::Dimension, 0, PointType::Dimension, 1) =
						// 		Jpoint.transpose() * omega_r;

						// 	Eigen::LLT<
						// 		Eigen::Matrix<double,
						// 					  PoseType::Dimension + PointType::Dimension,
						// 					  PoseType::Dimension + PointType::Dimension>>
						// 		lltofH(
						// 			H);
						// 	sol = lltofH.solve(b);
						// 	double poseh_det = poseHessianCopy.determinant();
						// 	double pointh_det = pointHessian.determinant();
						// 	double updated_det = lltofH.matrixL().determinant();
						// 	assert(poseh_det>0);
						// 	assert(pointh_det>0);
						// 	assert(updated_det>0);
						// 	// double increase = std::log(updated_det) - (std::log(poseh_det)+std::log(pointh_det));

						// 	// p += increase;
						// 	// if (increase > 0){
						// 	// 	assert(!isnan(p));
						// 	// }
						// 	assert(!isnan(p));
						// 	double increase = std::log( c.poses_[k].Z_[nz]->information().determinant()) ;
						// 	p += increase;
						// 	if (increase > 0){
						// 		assert(!isnan(p));
						// 	}
						// 	//increase = posHLogDet;

						// 	// if (increase > 0){
						// 	// 	assert(!isnan(p));
						// 	// }

						// 	// p += increase;
						// 	assert(!isnan(p));

						// 	p += -0.5 * (c.poses_[k].Z_[nz]->chi2() - sol.dot(b));
						// 	assert(!isnan(p));
						// 	p += -0.5 * c.poses_[k].Z_[nz]->dimension() * std::log(2 * M_PI);
						// 	assert(!isnan(p));
						// }
						// else
						{ // if pose is fixed only calculate updated landmark
							//c.landmarks_[lmidx - c.landmarks_[0].pPoint->id()].pPoint->solveDirect();
							//auto &lm = c.landmarks_[lmidx - c.landmarks_[0].id];
							
							// PointType::HessianBlockType::PlainMatrix h;
							// PointType::HessianBlockType pointHessian(h.data());

							// StereoMeasurementEdge::JacobianXiOplusType Jpoint =
							// 	c.poses_[k].Z_[nz]->jacobianOplusXi();

							// if (lm.numDetections_ != lm.pPoint->edges().size() ){
							// 	checkDA(c);
							// 	assert(0);
							// }

							// if (lm.numDetections_ > 0)
							// {
							// 	new (&pointHessian) PointType::HessianBlockType(
							// 		c.landmarks_[lmidx - c.landmarks_[0].pPoint->id()].pPoint->hessianData());

							// 	// std::cout << "g2o pointH: " << pointHessian << "\n\n\n";
							// }
							// else
							// {
							// 	h = config.anchorInfo_;
							// 	// std::cout << "calc pointH: " << pointHessian << "\n\n\n";
							// }

							// Eigen::Matrix<double, PointType::Dimension,
							// 			  PointType::Dimension>
							// 	H, ph ;
							// 	ph = pointHessian;
							// H.setZero();
							// if (pointHessian.allFinite()){
							// 	H = pointHessian + Jpoint.transpose() * c.poses_[k].Z_[nz]->information() * Jpoint;
							// }else{
							// 	H =  Jpoint.transpose() * c.poses_[k].Z_[nz]->information() * Jpoint;

							// }
							// Eigen::Matrix<double, PointType::Dimension, 1> b, sol;
							// b = Jpoint.transpose() * omega_r;

							// Eigen::LLT<
							// 	Eigen::Matrix<double, PointType::Dimension,
							// 				  PointType::Dimension>>
							// 	lltofH(H);
							// sol = lltofH.solve(b);

							// p += -std::log(
							// 	lltofH.matrixL().determinant());
							// assert(!isnan(p));
							// p +=
							// 	std::log(
							// 		c.poses_[k].Z_[nz]->information().determinant());
							// assert(!isnan(p));

							// p += -0.5 * (c.poses_[k].Z_[nz]->chi2() - sol.dot(b));
							
							 p += -0.5 * (error_scalar*error_scalar*z.information()(0,0)); //chi2, this assumes information is a*Identity(3,3)
							assert(!isnan(p));
							p += -0.5 * c.poses_[k].Z_[nz]->dimension() * std::log(2 * M_PI);
							assert(!isnan(p));


							// 	std::cerr << "jp: " << Jpoint << "\n";
							// 	std::cerr << "H: " << pointHessian << "\n";
							// if (p != p){
							// 	std::cerr << "nanerror\n";
							// 	std::cerr << "nop\n";
							// }



						}
					
					assert(!isnan(p));
					if (p  > config.logKappa_ - 1000){
						c.DAProbs_[k][nz].l.push_back(p);
						c.DAProbs_[k][nz].i.push_back(lmidx);
						c.reverseDAProbs_[k][lmFovIdx].i.push_back(nz);
						c.reverseDAProbs_[k][lmFovIdx].l.push_back(p);

						// avg_desc_likelihood += desclikelihood;
					}
				}
				// avg_desc_likelihood /= c.DAProbs_[k][nz].l.size()-1;
				// for (int nl =1; nl < c.DAProbs_[k][nz].l.size(); nl++){
				// 	c.DAProbs_[k][nz].l[nl] -= avg_desc_likelihood;
				// }





				// if (selectedDA >= 0)
				// {
				// 	c.poses_[k].Z_[nz]->setVertex(0,
				// 								  dynamic_cast<g2o::OptimizableGraph::Vertex *>(c.optimizer_->vertices().find(
				// 																											selectedDA)
				// 																					->second));
				// 	//c.poses_[k].Z_[nz]->linearizeOplus();
				// 	c.poses_[k].Z_[nz]->computeError();
				// 	assert(c.optimizer_->edges().find(c.poses_[k].Z_[nz]) != c.optimizer_->edges().end() );
				// }
				// else
				// {

				// 	c.poses_[k].Z_[nz]->setVertex(0, NULL);
				// 	assert(c.optimizer_->edges().find(c.poses_[k].Z_[nz]) == c.optimizer_->edges().end() );
					
				// }
			}
		}
	}

	void VectorGLMBSLAM6D::initStereoEdge(OrbslamPose &pose, int numMatch){

		Eigen::Vector3d uvu;
		int nl = pose.matches_left_to_right[numMatch].queryIdx;

		uvu[0] = pose.keypoints_left[nl].pt.x;
		uvu[1] = pose.keypoints_left[nl].pt.y;
		uvu[2] = pose.uRight[nl];

		StereoMeasurementEdge *stereo_edge = new StereoMeasurementEdge();
		stereo_edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(pose.pPose));
		stereo_edge->fx = config.camera_parameters_[0].fx;
		stereo_edge->fy = config.camera_parameters_[0].fy;
		stereo_edge->cx = config.camera_parameters_[0].cx;
		stereo_edge->cy = config.camera_parameters_[0].cy;
		stereo_edge->bf = config.stereo_baseline_f;
		stereo_edge->setInformation(config.stereoInfo_ * pose.mvInvLevelSigma2[pose.keypoints_left[nl].octave]);
		stereo_edge->setMeasurement(uvu);


		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		stereo_edge->setRobustKernel(rk);
		rk->setDelta(sqrt(7.8));

		pose.Z_[numMatch] = stereo_edge;

		pose.initial_lm_id[numMatch] = -1;
	}

	bool VectorGLMBSLAM6D::initMapPoint(OrbslamPose &pose, int numMatch, OrbslamMapPoint &lm, int newId)
	{

		Eigen::Vector3d uvu;
		int nl = pose.matches_left_to_right[numMatch].queryIdx;
		uvu[0] = pose.keypoints_left[nl].pt.x;
		uvu[1] = pose.keypoints_left[nl].pt.y;
		uvu[2] = pose.uRight[nl];

		lm.pPoint = new PointType();
		lm.pPoint->setId(newId);
		lm.id = newId;

		lm.birthMatch = numMatch;
		lm.birthPose = &pose;

		Eigen::Vector3d point_world_frame;

		lm.numDetections_ = 0;
		lm.numFoV_ = 0;
		int level = pose.keypoints_left[nl].octave;
		const int nLevels = pose.mnScaleLevels;

		if (!cam_unproject(*pose.Z_[numMatch], pose.point_camera_frame[numMatch]))
		{
			// std::cerr << "stereofail\n";

			return false;
		}

		lm.mfMaxDistance = pose.point_camera_frame[numMatch].norm() * pose.mvScaleFactors[level] * 1.2;
		lm.mfMinDistance = lm.mfMaxDistance / pose.mvScaleFactors[nLevels - 1] * 0.8;
		pose.initial_lm_id[numMatch] = newId;
		//pose.Z_[numMatch]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(lm.pPoint));

		// with only 2 descriptors (left,right) we can pick either one to represent the point
		// allocating new
		lm.descriptor = pose.descriptors_left[nl];
		//pose.descriptors_left.row(nl).copyTo(lm.descriptor);

		point_world_frame = pose.pPose->estimate().inverse().map(pose.point_camera_frame[numMatch]);
		lm.pPoint->setEstimate(point_world_frame);
		lm.pPoint->setMarginalized(true);
		lm.birthTime_ = pose.stamp;


		lm.normalVector = (point_world_frame - pose.pPose->estimate().inverse().translation()).normalized();
		//pose.Z_[numMatch]->computeError();

		return true;
	}


	void VectorGLMBSLAM6D::moveBirth(VectorGLMBComponent6D &c)
	{

		for( OrbslamMapPoint &lm:c.landmarks_){
			if ( lm.numDetections_ == 0){
				

				Eigen::Vector3d point_world_frame;
				point_world_frame = lm.birthPose->pPose->estimate().inverse().map(lm.birthPose->point_camera_frame[lm.birthMatch]);
				lm.pPoint->setEstimate(point_world_frame);

			}
		}
		for (int k = 0; k < c.poses_.size(); k++)
		{
			
			
			for (auto iter = c.DA_bimap_[k].left.begin(), iend = c.DA_bimap_[k].left.end(); iter != iend;
				++iter)
			{
				Eigen::Vector3d point_world_frame;
				point_world_frame = c.poses_[k].pPose->estimate().inverse().map(c.poses_[k].point_camera_frame[iter->first]);
				
				c.landmarks_[iter->second-c.landmarks_[0].id].pPoint->setEstimate(point_world_frame);
				
			}
		}


	}

	inline void VectorGLMBSLAM6D::constructGraph(VectorGLMBComponent6D &c)
	{
		c.numPoses_ = initial_component_.numPoses_;

		c.poses_ = initial_component_.poses_;
		

		int edgeid = c.numPoses_;

		for (int k = 0; k < c.poses_.size(); k++)
		{
			c.poses_[k].stamp = initial_component_.poses_[k].stamp;
			c.poses_[k].mvInvLevelSigma2 = initial_component_.poses_[k].mvInvLevelSigma2;
			// create graph pose
			c.poses_[k].pPose = new PoseType();
			g2o::SE3Quat pose_estimate;
			c.poses_[k].pPose->setEstimate(pose_estimate);

			c.poses_[k].pPose->setId(k);
			if (k < maxpose_){
				c.optimizer_->addVertex(c.poses_[k].pPose);
			}
			//
			if (k > 0)
			{
				OdometryEdge *odo = new OdometryEdge;
				odo->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(c.poses_[k - 1].pPose));
				odo->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(c.poses_[k].pPose));
				g2o::SE3Quat q;
				odo->setMeasurement(q);
				odo->setInformation(config.odomInfo_);
				c.odometries_.push_back(odo);
				if(k < maxpose_){
					if (!c.optimizer_->addEdge(odo))
					{
						std::cerr << "odo edge insert fail \n";
					}
				}

			}

			c.poses_[k].Z_.resize(c.poses_[k].matches_left_to_right.size());
			c.poses_[k].initial_lm_id.resize(c.poses_[k].matches_left_to_right.size());
			c.poses_[k].point_camera_frame.resize(c.poses_[k].matches_left_to_right.size());
			
			for (int nz = 0; nz < c.poses_[k].matches_left_to_right.size(); nz++)
			{
				OrbslamMapPoint lm;
				initStereoEdge(c.poses_[k], nz);
				cam_unproject(*c.poses_[k].Z_[nz], c.poses_[k].point_camera_frame[nz]);
		
				if (k%15==0){
					if (initMapPoint(c.poses_[k], nz, lm, edgeid))
					{
						edgeid++;
						//c.optimizer_->addVertex(lm.pPoint);
						c.landmarks_.push_back(lm);

						// if (!c.optimizer_->addEdge(c.poses_[k].Z_[nz]))
						// {
						// 	std::cerr << "initial measurement edge insert fail \n";
						// }
					}
				}
			}
		}

		c.landmarksResetProb_.resize(c.landmarks_.size(), 0.0);
		c.landmarksInitProb_.resize(c.landmarks_.size(), config.logExistenceOdds);

		for (int i = 0; i < c.landmarks_.size(); i++)
		{
			c.landmarks_[i].numDetections_ = 0;
		}

		c.DA_bimap_.resize(c.numPoses_);
		boost::bimap<int, int, boost::container::allocator<int>> empty_bimap;
		for (int p = 0; p < c.numPoses_; p++)
		{
			c.DA_bimap_[p] = empty_bimap;
		}
		c.prevDA_bimap_ = c.DA_bimap_;

		c.DAProbs_.resize(c.numPoses_);
		c.reverseDAProbs_.resize(c.numPoses_);
	}

	bool VectorGLMBSLAM6D::cam_unproject(const StereoMeasurementEdge &measurement,
										 Eigen::Vector3d &trans_xyz)
	{

		Eigen::Vector3d meas = measurement.measurement();
		trans_xyz[2] = config.stereo_baseline_f / (meas[0] - meas[2]);
		trans_xyz[0] = (meas[0] - config.camera_parameters_[0].cx) * trans_xyz[2] / config.camera_parameters_[0].fx;
		trans_xyz[1] = (meas[1] - config.camera_parameters_[0].cy) * trans_xyz[2] / config.camera_parameters_[0].fy;


		if (trans_xyz[2] > config.stereo_init_max_depth || trans_xyz[2] < 0 || trans_xyz[2] != trans_xyz[2])
		{
			return false;
		}

		return true;
	}

	void VectorGLMBSLAM6D::waitForGuiClose()
	{
		if (config.use_gui_)
		{
		}
	}

	inline void VectorGLMBSLAM6D::init(VectorGLMBComponent6D &c)
	{
		temp_ = config.initTemp_;
		auto linearSolver = g2o::make_unique<SlamLinearSolver>();
		linearSolver->setBlockOrdering(false);
		c.linearSolver_ = linearSolver.get();
		auto blockSolver = g2o::make_unique<SlamBlockSolver>(
			std::move(linearSolver));
		c.blockSolver_ = blockSolver.get();
		c.solverLevenberg_ = new g2o::OptimizationAlgorithmLevenberg(
			std::move(blockSolver));

		c.optimizer_ = new g2o::SparseOptimizer();
		c.optimizer_->setAlgorithm(c.solverLevenberg_);
	}

}
#endif
