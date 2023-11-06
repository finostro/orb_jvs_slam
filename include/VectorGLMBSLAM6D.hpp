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
#include <filesystem>
#include "ceres/ceres.h"
#include <unordered_map>
#include <math.h>
#include "GaussianGenerators.hpp"
#include "AssociationSampler.hpp"
#include "OrbslamMapPoint.hpp"
#include "OrbslamPose.hpp"
#include "VectorGLMBComponent6D.hpp"



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


#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/StereoFactor.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/dataset.h>


#ifdef _PERFTOOLS_CPU
#include <gperftools/profiler.h>
#endif
#ifdef _PERFTOOLS_HEAP
#include <gperftools/heap-profiler.h>
#endif

namespace rfs
{


	typedef gtsam::Point3 PointType;
	typedef gtsam::Pose3 PoseType;
	typedef gtsam::GenericStereoFactor<gtsam::Pose3,gtsam::Point3> StereoMeasurementEdge;
	typedef gtsam::BetweenFactor<gtsam::Pose3> OdometryEdge;

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
		gtsam::Pose3 pose;
	};



	struct EstimateWeight{
		double weight;
		std::vector<StampedPose> trajectory;
		gtsam::Values estimate;

    void loadTUM(std::string filename, gtsam::Pose3 base_link_to_cam0_se3, double initstamp){
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
			gtsam::Rot3 rot(q);
			stampedPose.pose = PoseType(rot,t);

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
			bool doCrossover; /**< do crossover when sampling*/
			int keypose_skip; /**< keypose is ever n frames*/

			int lmExistenceProb_;
			int numIterations_; /**< number of iterations of main algorithm */
			double initTemp_;
			double tempFactor_;
			Eigen::Matrix3d anchorInfo_;		   /** information for anchor edges, should be low*/
			Eigen::Matrix3d stereoInfo_;		   /** information for stereo uvu edges */
			Eigen::Matrix<double, 6, 6> odomInfo_; /** information for odometry edges */


			gtsam::noiseModel::Diagonal::shared_ptr odom_noise; /** covariance of odometry*/
			
			gtsam::noiseModel::Robust::shared_ptr stereo_noise; /** covariance of stereo measurements*/


			std::string finalStateFile_;

			gtsam::Cal3_S2Stereo::shared_ptr  cam_params;
			double viewingCosLimit_;

			gtsam::ISAM2Params isam2_parameters;


			std::string eurocFolder_, eurocTimestampsFilename_ , resultFolder;

			bool use_gui_;


			Eigen::MatrixXd base_link_to_cam0;
			gtsam::Pose3 base_link_to_cam0_se3;
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

		void loadEstimate(VectorGLMBComponent6D &c);
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
		 * rebuild isam2 from scratch.
		 * @param c the GLMB component
		 */
		void rebuildIsam(VectorGLMBComponent6D &c);

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
		VectorGLMBComponent6D::BimapType sexyTime(
			const VectorGLMBComponent6D::BimapType &map1, const VectorGLMBComponent6D::BimapType &map2);

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
		 * Check optimizer graph is consistent
		 * @param c the GLMB component
		 */
		void checkGraph(VectorGLMBComponent6D &c, std::ostream &s = std::cout);
		/**
		 * Check optimizer number of edges matches number of detections
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
		 * Use the new sampled data association to update the optimizer graph
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
		 * @param stereo_point
		 * @param trans_xyz
		 */
		bool cam_unproject(const gtsam::StereoPoint2 &stereo_point, Eigen::Vector3d &trans_xyz);


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
			EstimateWeight>
			visited_;
		
		EstimateWeight gt_est;
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

		transformStamped.transform.translation.x = c.poses_[c.maxpose_-1].pose.translation()[0];
		transformStamped.transform.translation.y = c.poses_[c.maxpose_-1].pose.translation()[1];
		transformStamped.transform.translation.z = c.poses_[c.maxpose_-1].pose.translation()[2];

		auto q = c.poses_[c.maxpose_-1].pose.rotation().toQuaternion();
		transformStamped.transform.rotation.x = q.x();
		transformStamped.transform.rotation.y = q.y();
		transformStamped.transform.rotation.z = q.z();
		transformStamped.transform.rotation.w = q.w();

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
		// 		auto point_world_frame = c.poses_[k].pose.transformFrom(c.poses_[k].point_camera_frame[i]);
				
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
				auto pos =  c.poses_[0].pose.transformFrom( c.landmarks_[i].position );
				map_cloud->at(nmap).x = pos[0];
				map_cloud->at(nmap).y = pos[1];
				map_cloud->at(nmap).z = pos[2];
				
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
			path.poses[i].pose.position.x = c.poses_[i].pose.translation()[0];
			path.poses[i].pose.position.y = c.poses_[i].pose.translation()[1];
			path.poses[i].pose.position.z = c.poses_[i].pose.translation()[2];
			auto q =  c.poses_[i].pose.rotation().toQuaternion();
			path.poses[i].pose.orientation.x = q.x();
			path.poses[i].pose.orientation.y = q.y();
			path.poses[i].pose.orientation.z = q.z();
			path.poses[i].pose.orientation.w = q.w();
		}
		trajectory_pub.publish(path);

	// gt trajectory
		nav_msgs::Path gtpath;
		gtpath.header.stamp = now;
		gtpath.header.frame_id = "map";

		gtpath.poses.resize(gt_est.trajectory.size());
		for (int i = 0; i < gt_est.trajectory.size(); i++)
		{
			gtpath.poses[i].header.stamp = now;
			gtpath.poses[i].header.frame_id = "map";
			gtpath.poses[i].pose.position.x = gt_est.trajectory[i].pose.translation()[0];
			gtpath.poses[i].pose.position.y = gt_est.trajectory[i].pose.translation()[1];
			gtpath.poses[i].pose.position.z = gt_est.trajectory[i].pose.translation()[2];
			auto q =  gt_est.trajectory[i].pose.rotation().toQuaternion();
			gtpath.poses[i].pose.orientation.x = q.x();
			gtpath.poses[i].pose.orientation.y = q.y();
			gtpath.poses[i].pose.orientation.z = q.z();
			gtpath.poses[i].pose.orientation.w = q.w();
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
		

			as_marker.points[a].x = c.landmarks_[iter->second - c.landmarks_[0].id].position[0];
			as_marker.points[a].y = c.landmarks_[iter->second - c.landmarks_[0].id].position[1];
			as_marker.points[a].z = c.landmarks_[iter->second - c.landmarks_[0].id].position[2];

			auto point_world_frame = c.poses_[k].pose.transformFrom(c.poses_[k].point_camera_frame[iter->first]);

			as_marker.points[a+1].x = point_world_frame[0];
			as_marker.points[a+1].y = point_world_frame[1];
			as_marker.points[a+1].z = point_world_frame[2];

			a+=2;
		}

		}
		std::cout << "n ass: " << c.DA_bimap_[c.maxpose_-1].size()  << "  nz: "  << c.poses_[c.maxpose_-1].Z_.size() << "\n";

		association_pub.publish(as_marker);
		ros::spinOnce();

// show last image

			cv::Mat imLeft, imRight;
			cv::Mat imLeft_rect, imRight_rect;
			// Read left and right images from file
			imLeft = cv::imread(vstrImageLeft[c.maxpose_-1], cv::IMREAD_UNCHANGED);	//,cv::IMREAD_UNCHANGED);
			imRight = cv::imread(vstrImageRight[c.maxpose_-1], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

			if (imLeft.empty())
			{
				std::cerr << std::endl
						  << "Failed to load image at: "
						  << std::string(vstrImageLeft[c.maxpose_-1]) << std::endl;
				exit(1);
			}

			if (imRight.empty())
			{
				std::cerr << std::endl
						  << "Failed to load image at: "
						  << std::string(vstrImageRight[c.maxpose_-1]) << std::endl;
				exit(1);
			}

			cv::remap(imLeft, imLeft_rect, config.camera_parameters_[0].M1,
					  config.camera_parameters_[0].M2, cv::INTER_LINEAR);
			cv::remap(imRight, imRight_rect, config.camera_parameters_[1].M1,
					  config.camera_parameters_[1].M2, cv::INTER_LINEAR);
			// plot stereo matches
			cv::Mat imLeftKeys, imRightKeys, imMatches;
			cv::Scalar kpColor = cv::Scalar(255, 0, 0);

			cv::drawMatches(imLeft_rect,
							initial_component_.poses_[c.maxpose_-1].keypoints_left, imRight_rect,
							initial_component_.poses_[c.maxpose_-1].keypoints_right,
							initial_component_.poses_[c.maxpose_-1].matches_left_to_right, imMatches);

			// cv::drawKeypoints(imRight_rect,
			// 				  initial_component_.poses_[ni].keypoints_right, imRightKeys,
			// 				  kpColor);
			// cv::drawKeypoints(imLeft_rect,
			// 				  initial_component_.poses_[ni].keypoints_left, imLeftKeys,
			// 				  kpColor);
			  cv::imshow("matches", imMatches);
			 //cv::imshow("imLeft", imLeft);
			 //cv::imshow("imLeft_rect", imLeft_rect);

			 cv::waitKey(1); // Wait for a keystroke in the window

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
			mp.is_detected.clear();
		}
		for (int k = 0 ; k < c.DA_bimap_.size() ; k++)
		{
			if (!c.poses_[k].isKeypose){
				c.DA_bimap_[k].clear();
			}
			auto &bimap = c.DA_bimap_[k];
			for (auto it = bimap.begin(), it_end = bimap.end(); it != it_end;
				 it++)
			{
				c.landmarks_[it->right - c.landmarks_[0].id].numDetections_++;
				c.landmarks_[it->right - c.landmarks_[0].id].is_detected.insert(k);
			}
		}

		checkDA(c);

	}
	void VectorGLMBSLAM6D::sampleComponents()
	{

		// do logsumexp on the components to calculate probabilities

		std::map<double, std::map<rfs::VectorGLMBComponent6D::BimapType, rfs::EstimateWeight>::iterator > probs_to_iterator;
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
		probs_to_iterator.insert({std::exp((visited_.begin()->second.weight - maxw) / temp_) , visited_.begin()});
		for (auto it = std::next(visited_.begin()), it_end = visited_.end();
			 it != it_end; i++, it++)
		{
			if (it->second.weight > maxw)
				maxw = it->second.weight;
			probs[i] = probs[i - 1] + std::exp((it->second.weight - maxw) / temp_);

			probs_to_iterator.insert({probs[i] , it});
		}
		/*
		 std:: cout << "maxw " << maxw <<  "  temp " << temp_ << "\n";
		 std::cout << "probs " ;
		 for (auto p:probs){std::cout <<"  " << p;}
		 std::cout <<"\n";*/

		boost::uniform_real<> dist(0.0,
								   probs[probs.size() - 1] );

		double r1 = dist(randomGenerators_[0]);
		double r2 = dist(randomGenerators_[0]);
		int j = 0;
		auto it = visited_.begin();
		for (int i = 0; i < components_.size(); i++)
		{
			auto &c = components_[i];
			auto  da1 = probs_to_iterator.lower_bound(r1)->second;
			auto  da2 = probs_to_iterator.lower_bound(r2)->second;

			if (config.doCrossover){
				auto da = sexyTime(da1->first , da2->first);
				// add data association j to component i
				changeDA(c, da);
			}else{
				changeDA(c, da1->first);
			}
			
			for(int k = 0 ; k < c.poses_.size() ; k++){

				c.poses_[k].pose = da1->second.trajectory[k].pose;	
				if(k < minpose_){
					bool new_is_keypose =  (k % config.keypose_skip == 0 );
					c.poses_[k].isKeypose = new_is_keypose;

					if (!c.poses_[k].isKeypose){
						
						c.poses_[k].referenceKeypose = (k / config.keypose_skip) * config.keypose_skip;
						c.poses_[k].transformFromReferenceKeypose = c.poses_[c.poses_[k].referenceKeypose].pose.transformPoseTo(c.poses_[k].pose);

						for (auto iter = c.DA_bimap_[k].left.begin(), iend = c.DA_bimap_[k].left.end(); iter != iend;
						++iter){
							auto &lm = c.landmarks_[ iter->second - c.landmarks_[0].id ];
							lm.numDetections_--;
							lm.is_detected.erase(k);
							if (lm.numDetections_ == 1){
								int k_remove = * lm.is_detected.begin();
								c.DA_bimap_[k_remove].right.erase(lm.id);
								lm.is_detected.clear();
								lm.numDetections_ = 0;
							}
							

						}
						c.DA_bimap_[k].clear();


						
					}
				}


			}
			

			


			for (auto it: it->second.estimate){
				if (it.key  < c.poses_.size()){
					c.poses_[it.key].pose =  it.value.cast<PoseType>();
				}else{
					c.landmarks_[it.key-c.landmarks_[0].id].position = it.value.cast<PointType>();
				}
			}

			// std::cout  << "sample w: " << it->second << " j " << j  << " r " << r <<" prob "  << probs[j]<< "\n";

			r1 = fmod(r1 + probs[probs.size() - 1] / components_.size() , probs[probs.size() - 1]);
			r2 = fmod(r2 + probs[probs.size() - 1] / components_.size() , probs[probs.size() - 1]);
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
			initial_component_.poses_[ni].id = ni;
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

			cv::drawMatches(imLeft_rect,
							initial_component_.poses_[ni].keypoints_left, imRight_rect,
							initial_component_.poses_[ni].keypoints_right,
							initial_component_.poses_[ni].matches_left_to_right, imMatches);

			// cv::drawKeypoints(imRight_rect,
			// 				  initial_component_.poses_[ni].keypoints_right, imRightKeys,
			// 				  kpColor);
			// cv::drawKeypoints(imLeft_rect,
			// 				  initial_component_.poses_[ni].keypoints_left, imLeftKeys,
			// 				  kpColor);
			  cv::imshow("matches", imMatches);
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
		config.doCrossover = node["doCrossover"].as<bool>();
		config.keypose_skip = node["keypose_skip"].as<int>();
		config.finalStateFile_ = node["finalStateFile"].as<std::string>();


		config.isam2_parameters.relinearizeThreshold = node["relinearizeThreshold"].as<double>();
		config.isam2_parameters.relinearizeSkip = node["relinearizeSkip"].as<int>();
		config.isam2_parameters.cacheLinearizedFactors = node["cacheLinearizedFactors"].as<bool>();
		config.isam2_parameters.enableDetailedResults = node["enableDetailedResults"].as<bool>();
		config.isam2_parameters.evaluateNonlinearError = false;
		config.isam2_parameters.findUnusedFactorSlots = true; 
		config.isam2_parameters.factorization = gtsam::ISAM2Params::Factorization::CHOLESKY ;
		config.isam2_parameters.enablePartialRelinearizationCheck = node["enablePartialRelinearizationCheck"].as<bool>();


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
		config.odom_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6(1/sqrt(config.odomInfo_(0,0)),
		 1/sqrt(config.odomInfo_(1,1)), 1/sqrt(config.odomInfo_(2,2)), 1/sqrt(config.odomInfo_(3,3)), 
		 1/sqrt(config.odomInfo_(4,4)), 1/sqrt(config.odomInfo_(5,5)) ));

		if (!YAML::convert<Eigen::Matrix3d>::decode(node["stereoInfo"],
													config.stereoInfo_))
		{
			std::cerr << "could not load stereo info matrix \n";
			exit(1);
		}
		config.stereo_noise = gtsam::noiseModel::Robust::Create(  gtsam::noiseModel::mEstimator::Huber::Create(4), gtsam::noiseModel::Isotropic::Sigma(3, 1/sqrt(config.stereoInfo_(0,0)) ));

		

		double focal_length = node["camera.focal_length"].as<double>();

		config.viewingCosLimit_ = node["viewingCosLimit"].as<double>();

		config.perturbTrans = node["perturbTrans"].as<double>();
		config.perturbRot = node["perturbRot"].as<double>();


		config.eurocFolder_ = node["eurocFolder"].as<std::string>();
		config.resultFolder = node["resultFolder"].as<std::string>();

		std::filesystem::create_directory(std::filesystem::path(config.resultFolder));
		std::filesystem::create_directory(std::filesystem::path(config.resultFolder+"/video"));

		config.use_gui_ = node["use_gui"].as<bool>();

		config.eurocTimestampsFilename_ = node["eurocTimestampsFilename"].as<std::string>();

		if (!YAML::convert<Eigen::MatrixXd>::decode(node["base_link_to_cam0"],
														config.base_link_to_cam0))
		{
			std::cerr << "could not load base_link_to_cam0 \n";
			exit(1);
		}
		config.base_link_to_cam0_se3 = PoseType( config.base_link_to_cam0);



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

		config.cam_params = boost::make_shared<gtsam::Cal3_S2Stereo>(config.camera_parameters_[0].fx ,
																	config.camera_parameters_[0].fy ,
																	0 ,                                     // 0 skew
																	config.camera_parameters_[0].cx ,
																	config.camera_parameters_[0].cy ,
																	config.stereo_baseline);


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


		return (pose->translation()-*lm).norm();
	}

	inline void VectorGLMBSLAM6D::deleteComponents(){

		components_.clear();
	}



	inline void VectorGLMBSLAM6D::initComponents()
	{
		components_.reserve(config.numComponents_);
		for (int i=0; i< config.numComponents_ ;i++){
			components_.emplace_back(config.cam_params);

			init(components_[i]);
			constructGraph(components_[i]);
			for(int j = 0 ; j< components_.size() ; j++){
				checkNumDet(components_[j]);
			}
		}
		for (auto &c:components_){
			checkNumDet(c);
		}
			
		
		
	}
	inline void VectorGLMBSLAM6D::run(int numSteps)
	{

		for (int i = 0; i < numSteps && ros::ok(); i++)
		{
			maxpose_prev_ = maxpose_;
			maxpose_ = 2 + components_[0].poses_.size() * i / (numSteps * 0.95);

			if (best_DA_max_detection_time_ + 5 < maxpose_)
			{
				maxpose_ = best_DA_max_detection_time_ +5;// config.numPosesToOptimize_/2;
			}
			if (maxpose_ > components_[0].poses_.size())
				maxpose_ = components_[0].poses_.size();

			if (maxpose_ > maxpose_prev_ && maxpose_>20){
				if (visited_.size() < 10) maxpose_ = maxpose_prev_;
			}

			
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


	void VectorGLMBSLAM6D::updateDescriptors(VectorGLMBComponent6D &c){
		
	}
	void VectorGLMBSLAM6D::perturbTraj(VectorGLMBComponent6D &c){
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif

		boost::uniform_real<> uni_dist(0, 1);
		int startk = std::max(std::max(minpose_ , config.staticframes-config.minframe ) ,1);
		
		int max_detection_time = maxpose_ - 1;
		while (max_detection_time > 0 && c.DA_bimap_[max_detection_time].size() == 0)
		{
			max_detection_time--;
		}
		gtsam::Pose3 displacement;

		if (max_detection_time>0){
			displacement = c.poses_[max_detection_time-1].pose.transformPoseTo(c.poses_[max_detection_time].pose);
			displacement = gtsam::Pose3( displacement.rotation().normalized() , displacement.translation() );
			
			auto current_pose = c.poses_[max_detection_time].pose;

			// std::cout << "displacement  :  " << displacement << "\n";
			// std::cout << "scale  :  " << displacement.rotation().toQuaternion().norm() << "\n";
			// std::cout << "last  :  " << c.poses_[max_detection_time].pose << "\n";
			// std::cout << "scale  :  " << c.poses_[max_detection_time].pose.rotation().toQuaternion().norm() << "\n";
			// std::cout << "prev  :  " << c.poses_[max_detection_time-1].pose << "\n";
			// std::cout << "scale  :  " << c.poses_[max_detection_time-1].pose.rotation().toQuaternion().norm() << "\n";
			// std::cout << "max_detection_time  :  " << max_detection_time  << "\n";

			for(int k = max_detection_time+1 ; k < maxpose_ ; k++){
				current_pose = current_pose.transformPoseFrom(displacement);
				current_pose = gtsam::Pose3( current_pose.rotation().normalized() , current_pose.translation());
				
				c.poses_[k].pose = current_pose;

			}
		}

		displacement = c.poses_[startk].pose.transformPoseTo(c.poses_[startk+1].pose);
		displacement = gtsam::Pose3( displacement.rotation().normalized() , displacement.translation() );
			


		double dist = displacement.translation().norm();

		for(int k = startk ; k < maxpose_ ; k++){
			int numdet = c.DA_bimap_[k].size();
			
			double d = uni_dist(randomGenerators_[threadnum]);


			




			if( k < c.poses_.size()-1 ){
				displacement = c.poses_[k].pose.transformPoseTo(c.poses_[k+1].pose);
				displacement = gtsam::Pose3( displacement.rotation().normalized() , displacement.translation() );
				dist = displacement.translation().norm();
			}	
			if (numdet > 10) {
				gtsam::Point3 translation_noise(gaussianGenerators_[threadnum](randomGenerators_[threadnum])*dist*config.perturbTrans/2
				, gaussianGenerators_[threadnum](randomGenerators_[threadnum])*dist*config.perturbTrans/2
				, gaussianGenerators_[threadnum](randomGenerators_[threadnum])*dist*config.perturbTrans/2);

				gtsam::Pose3 noise(gtsam::Rot3::Rodrigues(gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbRot,
				gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbRot,
				gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbRot), translation_noise);

				c.poses_[k].pose = c.poses_[k].pose  * noise;

			}else{
				gtsam::Point3 translation_noise(gaussianGenerators_[threadnum](randomGenerators_[threadnum])*dist*config.perturbTrans
				, gaussianGenerators_[threadnum](randomGenerators_[threadnum])*dist*config.perturbTrans  
				, gaussianGenerators_[threadnum](randomGenerators_[threadnum])*dist*config.perturbTrans);

				gtsam::Pose3 noise(gtsam::Rot3::Rodrigues(gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbRot,
				gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbRot,
				gaussianGenerators_[threadnum](randomGenerators_[threadnum])*config.perturbRot), translation_noise);

				c.poses_[k].pose = c.poses_[k].pose  * noise;
				// c.poses_[k].pose = c.poses_[k-1].pose * displacement * noise;

			}


		}



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

			c.isam_result = c.isam->update(c.new_edges, c.new_nodes, c.removed_edges);
			c.current_estimate = c.isam->calculateBestEstimate();
			loadEstimate(c);


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
						if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id].numDetections_ == 1)
						{
							likelihood += config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id].numFoV_) * std::log(1 - config.PD_);
							// std::cout <<" single detection: increase:  " <<logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].id].numFoV_)*std::log(1-config.PD_) <<"\n";
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
							if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id].numDetections_ == 0)
							{
								likelihood +=
									config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id].numFoV_) * std::log(1 - config.PD_);
								// std::cout <<" 0 detection: increase:  " << logExistenceOdds+ (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].id].numFoV_)*std::log(1-config.PD_)<<"\n";
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
						c.landmarks_[newdai - c.landmarks_[0].id].numDetections_++;
						c.landmarks_[newdai - c.landmarks_[0].id].is_detected.insert(k);
						if (selectedDA < 0)
						{
							c.DA_bimap_[k].insert({nz, newdai});
						}
						else
						{

							c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_--;
							c.landmarks_[selectedDA - c.landmarks_[0].id].is_detected.erase(k);

							assert(c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_ >= 0);
							c.DA_bimap_[k].left.replace_data(it, newdai);
						}
					}
					else
					{ // if a change has to be made and new DA is false alarm, we need to remove the association
						c.DA_bimap_[k].left.erase(it);
					if(selectedDA>=0){
						c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_--;
						c.landmarks_[selectedDA - c.landmarks_[0].id].is_detected.erase(k);
						assert(c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_ >= 0);
					}
					}
				}
			}
		}
	}

	VectorGLMBComponent6D::BimapType VectorGLMBSLAM6D::sexyTime(
		const VectorGLMBComponent6D::BimapType &map1, const VectorGLMBComponent6D::BimapType &map2)
	{

		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif
		if (maxpose_ == 0)
		{
			std::vector<boost::bimap<int, int, boost::container::allocator<int>>> out(
				map1);
			return out;
		}
		boost::uniform_int<> random_merge_point(minpose_, maxpose_-1);
		std::vector<boost::bimap<int, int, boost::container::allocator<int>>> out;
		out.resize(map1.size());
		int merge_point = random_merge_point(rfs::randomGenerators_[threadnum]);


		// first half from da1 second from da2
		for (int i = merge_point; i < map2.size(); i++)
		{
			out[i] = map2[i];
		}
		for (int i = 0; i < merge_point; i++)
		{
			out[i] = map1[i];
		}

		return out;
	}
	inline void VectorGLMBSLAM6D::optimize(int ni)
	{

		std::cout << "visited  " << visited_.size() << "\n";
		// if (iteration_ % 15 == 0)
		if (visited_.size() > 0)
		{
			sampleComponents();
			//visited_.clear();
			std::cout << "sampling compos \n";
		}


		#pragma omp parallel for
		for (int i = 0; i < components_.size(); i++)
		{


			
			auto &c = components_[i];

			// if (iteration_ % 15 == 0){
			// 	rebuildIsam(c);
			// }

			int threadnum = 0;
			#ifdef _OPENMP
			threadnum = omp_get_thread_num();
			#endif


		





			bool inserted;
			std::map<
			std::vector<boost::bimap<int, int, boost::container::allocator<int>>>,
			EstimateWeight>::iterator it;


			perturbTraj(c);
			moveBirth(c);

			updateMetaStates(c);

			updateFoV(c);

			if (!c.reverted_)
				updateDAProbs(c, minpose_, maxpose_);
			for (int p = 0; p < maxpose_; p++)
			{
				c.prevDA_bimap_[p] = c.DA_bimap_[p];
			}
			double expectedChange = 0;


			{
				// do{
				// checkGraph(c);
				 //selectNN(c);
				// checkGraph(c);
				expectedChange += sampleDA(c);


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
						 expectedChange += sampleLMBirth(c);
						 //expectedChange += mergeLM(c);
						break;
					}


					checkGraph(c);
					updateGraph(c);
					checkGraph(c);

					try{
					c.isam_result = c.isam->update(c.new_edges, c.new_nodes, c.removed_edges);
					}catch(const std::exception& e){
						std::cout << termcolor::red <<  e.what();
						//std::cout << "caught exception saving graph to graph.isam \n";
						//gtsam::writeG2o 	( c.isam->getFactorsUnsafe() ,c.isam->calculateBestEstimate(), "graph.isam"	);
						//c.isam->getFactorsUnsafe().saveGraph("graph.dot");
						std::cout << "threadnum: "<<  threadnum << " \n";
						std::cout << termcolor::reset ;
						c.logweight_ = -std::numeric_limits<double>::infinity();
						// throw e;
						rebuildIsam(c);
						continue;
						
					}
					c.current_estimate = c.isam->calculateBestEstimate();
					loadEstimate(c);

					moveBirth(c);
					updateMetaStates(c);

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



				}

				
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




			updateGraph(c);
			checkGraph(c);

			try{
				c.isam_result = c.isam->update(c.new_edges, c.new_nodes, c.removed_edges);
				for(int isam_i = 0; isam_i < config.numLevenbergIterations_; isam_i++){
					c.isam->update();
				}
			
			}catch(const std::exception& e){
					std::cout << termcolor::red <<  e.what();
					std::cout << "caught exception saving graph to graph.isam \n";
					//gtsam::writeG2o 	( c.isam->getFactorsUnsafe() ,c.isam->calculateBestEstimate(), "graph.isam"	);
					//c.isam->getFactorsUnsafe().saveGraph("graph.dot");
					std::cout << "threadnum: "<<  threadnum << " \n";
					std::cout << termcolor::reset ;
					c.logweight_ = -std::numeric_limits<double>::infinity();
					// continue;
					// throw e;
					rebuildIsam(c);
					continue;
			}
			
			c.current_estimate = c.isam->calculateBestEstimate();
			
			
			loadEstimate(c);
			moveBirth(c);
			updateMetaStates(c);
			updateFoV(c);

			calculateWeight(c);

			EstimateWeight to_insert;
			to_insert.weight = c.logweight_;
			to_insert.trajectory.resize(c.poses_.size() );
			to_insert.estimate = c.current_estimate;
			int max_detection_time_ = std::max(maxpose_ - 1, 0);
			while (max_detection_time_ > 0 && c.DA_bimap_[max_detection_time_].size() == 0)
			{
				max_detection_time_--;
			}

			for(int numpose =0; numpose <= max_detection_time_; numpose++){
				to_insert.trajectory[numpose].pose=c.poses_[numpose].pose;
			}
			for(int numpose = max_detection_time_+1; numpose< c.poses_.size(); numpose++){
				to_insert.trajectory[numpose].pose = c.poses_[max_detection_time_].pose;
				c.poses_[numpose].pose = c.poses_[max_detection_time_].pose;
			}
			auto pair = std::make_pair(c.DA_bimap_, to_insert);


			for ( int numpose =minpose_+1 ; numpose < maxpose_; numpose++){
				double dist = (c.poses_[numpose].pose.translation()-c.poses_[numpose-1].pose.translation()).norm();
				if (dist>0.2){
					std::cout << termcolor::red << "dist to high setting w to -inf"  << "\n";
					// std::cout << "loglikelihood " << c.odometries_[numpose-1]->error(c.current_estimate) << "\n";
					std::cout << "dist " << dist << "\n" << termcolor::reset ;
				
					c.logweight_ = -std::numeric_limits<double>::infinity();
					rebuildIsam(c);
					break;
				}
				

				//assert (dist<0.1);
			}
			if ( c.logweight_ == -std::numeric_limits<double>::infinity()){
				continue;
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
					std::stringstream filename_g2o, filename_dot;

					best_DA_max_detection_time_ = std::max(maxpose_ - 1, 0);
					while (best_DA_max_detection_time_ > 0 && best_DA_[best_DA_max_detection_time_].size() == 0)
					{
						best_DA_max_detection_time_--;
					}
					filename_g2o << config.resultFolder << "/video/beststate_" << std::setfill('0')
							 << std::setw(5) << iterationBest_++ << ".g2o";

					filename_dot << config.resultFolder << "/video/beststate_" << std::setfill('0')
							 << std::setw(5) << iterationBest_++ << ".dot";
					// c.optimizer_->save(filename.str().c_str(), 0);
					// std::cout << termcolor::yellow << "========== newbest:"
					// 		  << bestWeight_ << " ============\n"
					// 		  << termcolor::reset;

					gtsam::writeG2o 	( c.isam->getFactorsUnsafe() ,c.current_estimate, filename_g2o.str()	);
					c.isam->saveGraph(filename_dot.str() );
					// std::cout << "globalchi2 " << c.isam_result.getErrorAfter() << "\n";
					std::cout <<"  determinant not implemented: " << 0.0 << "\n";

					std::stringstream name;
					static int numbest=0;
					name  << config.resultFolder <<  "/best__" << numbest++ << ".tum"; 

					c.saveAsTUM(name.str(),config.base_link_to_cam0_se3);

					if (config.use_gui_)
					{
						std::cout << termcolor::yellow << "========== piblishingmarkers:"
								  << bestWeight_ << " ============\n"
								  << termcolor::reset;
						// perturbTraj(c);
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

		}

		iteration_++;



		std::cout << "insertionp: " << insertionP_ << " temp: " << temp_ << "\n";


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
			if(!c.poses_[k].isKeypose){
				continue;
			}
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
						-0.5 * (3 * std::log(2 * M_PI) - std::log(config.stereoInfo_.determinant() ));
					


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
							// for (int kk:lm.is_in_fov_){
							// 	auto it = c.DA_bimap_[kk].right.find(lm.id);
							// 	if (it != c.DA_bimap_[kk].right.end()){
							// 		std::cout << "k " << kk << " nz  " <<it->second << "\n";
							// 	}
							// }
							std::cout << "bad descriptor likelihood   \n";
							// assert(0);
							// exit(1);
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
					bool exists = c.landmarks_[c.poses_[k].fov_[lm] - c.landmarks_[0].id].numDetections_ > 0;
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
		double chi_logw=- c.isam->getFactorsUnsafe().error(c.current_estimate);
		double det_logw = 0.0 ;// DET NOT IMPLEMENTED -0.5* c.linearSolver_->_determinant;

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
		
		c.new_nodes.clear();
		c.removed_edges.clear();
		c.new_edges.resize (0);



		int max_detection_time = maxpose_ - 1;
		while (max_detection_time > 0 && c.DA_bimap_[max_detection_time].size() == 0)
		{
			max_detection_time--;
		}
		for (int k = 1; k <= max_detection_time; k++)
		{
			auto it = c.odometry_indices.left.find(k-1);
			if(c.poses_[k-1].isKeypose && c.poses_[k].isKeypose){
				if (it == c.odometry_indices.left.end() ){
					c.new_edges.push_back(c.odometries_[k-1]);
				}
			}else{
				if (it != c.odometry_indices.left.end() ){
					c.removed_edges.push_back(it->second);
				}
			}
			
		}


		for (int k = 0; k <= max_detection_time; k++)
		{
			// odometries
			
			if (!c.poses_[k].isKeypose){
				continue;
			}

			for (auto iter = c.DA_bimap_[k].left.begin(), iend = c.DA_bimap_[k].left.end(); iter != iend;
			 ++iter)
			{
			
				if (c.poses_[k].Z_[iter->first]){
					if (c.poses_[k].Z_[iter->first]->key2() == iter->second){
						// measurement found , it should have an index

						std::array<int , 3> association = {k , iter->first , iter->second };

						auto it_index_left  = c.landmark_edge_indices.left.find(association);
						auto found_index = it_index_left->second;
						assert(it_index_left !=c.landmark_edge_indices.left.end() );
						
					}else{

						
						c.poses_[k].Z_[iter->first] = boost::make_shared<StereoMeasurementEdge>(
							c.poses_[k].stereo_points[iter->first], config.stereo_noise, k, iter->second , config.cam_params);
						c.new_edges.push_back(c.poses_[k].Z_[iter->first]);

						//measurement not found , it should not have an index
						std::array<int , 3> association = {k , iter->first , iter->second };

						auto it_index_left  = c.landmark_edge_indices.left.find(association);
						auto found_index = it_index_left->second;
						assert(it_index_left ==c.landmark_edge_indices.left.end() );
						
					}
				}else{


					//measurement not found , it should not have an index
					std::array<int , 3> association = {k , iter->first , iter->second };

					auto it_index_left  = c.landmark_edge_indices.left.find(association);
					auto found_index = it_index_left->second;
					if (it_index_left != c.landmark_edge_indices.left.end()){
						c.poses_[k].Z_[iter->first] = boost::dynamic_pointer_cast<StereoMeasurementEdge>(c.isam->getFactorsUnsafe()[found_index] );
					}else{
						c.poses_[k].Z_[iter->first] = boost::make_shared<StereoMeasurementEdge>(
							c.poses_[k].stereo_points[iter->first], config.stereo_noise, k, iter->second , config.cam_params);
						c.new_edges.push_back(c.poses_[k].Z_[iter->first]);

					}
					

				}
				

			}
		}

		for (auto it: c.landmark_edge_indices.left){
			
			int k = it.first[0];
			int nz = it.first[1];
			int lmid = it.first[2];

			auto bimap_it = c.DA_bimap_[k].left.find(nz);

			if (bimap_it ==  c.DA_bimap_[k].left.end() || !c.poses_[k].isKeypose ){
				c.poses_[k].Z_[nz] = NULL; 
				c.removed_edges.push_back(it.second);
			}else{
				if(bimap_it->second != lmid){
					if (c.poses_[k].Z_[nz]->keys()[1] == lmid){
						c.poses_[k].Z_[nz] = NULL; 

					}else{
						assert(c.poses_[k].Z_[nz]->keys()[1] == bimap_it->second);
					}
					c.removed_edges.push_back(it.second);
				}
			}
			
		}



		
		
		for (int k = 0; k <= max_detection_time; k++)
		{
			if (!c.current_estimate.exists(k) ){
				c.new_nodes.insert(k,c.poses_[k].pose);
			}

			
		}

		for(auto &lm: c.landmarks_){
			if (lm.numDetections_ > 0 ){
				
				if (!c.current_estimate.exists(lm.id)){
					c.new_nodes.insert(lm.id, lm.position);
				}
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

		checkNumDet(c,s);

	}
	inline void VectorGLMBSLAM6D::checkNumDet(VectorGLMBComponent6D &c,
										  std::ostream &s)
	{
		for (auto &lm:c.landmarks_){
			bool found = false;
			for (auto &p:c.poses_){
				if (lm.birthPose == &p) {
					found = true;
					break;
				}
			}
			assert(found);
			int numdet=0;
			for(int k=0;k< maxpose_;k++){
				auto it = c.DA_bimap_[k].right.find(lm.id);
				if (it != c.DA_bimap_[k].right.end()){
					numdet++;
					assert(lm.is_detected.count(k) == 1);
				}else{
					assert(lm.is_detected.count(k)==0);
				}
			}
			assert(numdet==lm.numDetections_);
		}
	}	
	
	inline void VectorGLMBSLAM6D::checkGraph(VectorGLMBComponent6D &c,
										  std::ostream &s)
	{

		checkNumDet(c,s);
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
			double numdet = c.landmarksInitProb_[i];

			if ( lm.numDetections_ > 0 || lm.numFoV_ == 0 || lm.birthTime_>c.poses_[maxpose_-1].stamp || lm.birthTime_<c.poses_[minpose_].stamp)
			{
				continue;
			}
			auto birth_it = c.DA_bimap_[lm.birthPose->id].left.find(lm.birthMatch);
			if (birth_it != c.DA_bimap_[lm.birthPose->id].left.end()){
				// birth measurement is associated, cannot spawn
				continue;
			}
			//
			//c.landmarksInitProb_[i] = c.landmarksInitProb_[i] / (c.landmarks_[i].numFoV_ * config.PD_);
			//c.landmarksInitProb_[i] =-(numdet)*config.logKappa_ +c.landmarksInitProb_[i]+ config.logExistenceOdds;
			c.landmarksInitProb_[i] += std::log(1 - config.PD_) *lm.numFoV_; 
			double aux= exp(c.landmarksInitProb_[i] );
			double p = aux/(1+aux);
			if (p > uni_dist(randomGenerators_[threadnum]))
			{
				// reset all associations to false alarms
				expectedWeightChange += (config.logKappa_ + (1 - config.PD_)) * c.landmarks_[i].numFoV_;
				//int numdet = 0;
				for (int k = minpose_; k < maxpose_; k++)
				{
					if ( k == lm.birthPose->id ){
						continue;
					}
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
							if (c.DAProbs_[k][nz].i[a] == c.landmarks_[i].id)
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
						c.DA_bimap_[k].insert( {max_nz, c.landmarks_[i].id});
						expectedWeightChange += maxl - config.logKappa_;
						c.landmarks_[i].numDetections_++;
						c.landmarks_[i].is_detected.insert(k);
					}

				}


				int kbirth = lm.birthPose->id;
				
				auto birth_it2 = c.DA_bimap_[kbirth].right.find(lm.id); 

				if (birth_it2 == c.DA_bimap_[kbirth].right.end()){
					if(lm.numDetections_== lm.numFoV_){
						std::cout << termcolor::red << "adding imposssible det_:\n" << termcolor::reset;
						printDA(c);
						assert(0);
					}
					auto result = c.DA_bimap_[kbirth].insert({lm.birthMatch, lm.id });
					assert(result.second);
					lm.numDetections_++;
					lm.is_detected.insert(kbirth);
					assert(lm.numDetections_ == lm.is_detected.size() );
				}else{
					assert(0);
				}
				



				expectedWeightChange += config.logExistenceOdds;
				
				//  std::cout << termcolor::green << "LANDMARK BORN "
				//  << termcolor::reset << " initprob: "
				//  << c.landmarksInitProb_[i] << " numDet "
				//  << c.landmarks_[i].numDetections_ << " nd: "
				//  << numdet << " numfov: "
				//  << c.landmarks_[i].numFoV_ << "  expectedChange "
				//  << expectedWeightChange << "\n";
				 

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

		

		if (c.landmarks_[toAddMeasurements - c.landmarks_[0].id].numDetections_ <  c.landmarks_[todelete - c.landmarks_[0].id].numDetections_ ){
			todelete = c.tomerge_[rp].second;
			toAddMeasurements = c.tomerge_[rp].first;
		}
		 auto &del_lm = c.landmarks_[todelete - c.landmarks_[0].id];
		 auto &add_lm = c.landmarks_[toAddMeasurements - c.landmarks_[0].id];
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
					c.landmarks_[todelete - c.landmarks_[0].id].numDetections_--;
					c.landmarks_[todelete - c.landmarks_[0].id].is_detected.erase(k);
					
					assert(c.landmarks_[todelete - c.landmarks_[0].id].numDetections_ >= 0);
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
				c.landmarks_[todelete - c.landmarks_[0].id].numDetections_--;
				c.landmarks_[todelete - c.landmarks_[0].id].is_detected.erase(k);
				assert(c.landmarks_[todelete - c.landmarks_[0].id].numDetections_ == c.landmarks_[todelete - c.landmarks_[0].id].is_detected.size());
				assert(c.landmarks_[todelete - c.landmarks_[0].id].numDetections_ >= 0);
				c.landmarks_[toAddMeasurements - c.landmarks_[0].id].numDetections_++;
				c.landmarks_[toAddMeasurements - c.landmarks_[0].id].is_detected.insert(k);

				assert( c.landmarks_[toAddMeasurements - c.landmarks_[0].id].numDetections_ == c.landmarks_[toAddMeasurements - c.landmarks_[0].id].is_detected.size());


				bool result = c.DA_bimap_[k].right.replace_key(it, toAddMeasurements);
				if (!result){

					std::cerr << termcolor::red  << "key not replaced\n"<< termcolor::reset;
				}
			}
		}
		if (c.landmarks_[todelete - c.landmarks_[0].id].numDetections_ != 0)
		{
			std::cerr << "landmark  numDetections_ not zero"
					  << c.landmarks_[todelete - c.landmarks_[0].id].numDetections_
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
						c.landmarks_[i].id);
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
				
				//  std::cout << termcolor::red << "KILL LANDMARK\n" << termcolor::reset
				//  << c.landmarksResetProb_[i] << " n "
				//  << c.landmarks_[i].numDetections_ << " nfov:"
				//  << c.landmarks_[i].numFoV_ << "  expectedChange "
				//  << expectedWeightChange << "\n";
				 

				c.landmarks_[i].numDetections_ = 0;
				c.landmarks_[i].is_detected.clear();
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
			// break;
			// std::cout  << "testk " << k  << " maxpose " << maxpose_ << "\n";
			if(k>=maxpose_){
				break;
			}
			// std::cout  << "testk " << k << "\n";
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
						int kbirth = lm.birthPose->id;
						auto it = c.DA_bimap_[kbirth].left.find(lm.birthMatch);
						if (it != c.DA_bimap_[kbirth].left.end()){
							continue;
							// initial measurement is associated skip
						}
						
					}



					int prevDA = -5;
					auto it = c.DA_bimap_[k].right.find(lm.id);
					if (it != c.DA_bimap_[k].right.end())
					{
						prevDA = it->second;
					}
					
					if(&c.poses_[k] == lm.birthPose){
						if(lm.numDetections_>0){
							assert(prevDA == lm.birthMatch);
						}
						continue;
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
							lm.is_detected.insert(k);
							auto result = c.DA_bimap_[k].insert({ probs.i[sample] ,  lmidx});

							assert(result.second);
							assert(lm.is_detected.size() == lm.numDetections_);
						}
						else
						{
							bool result = c.DA_bimap_[k].right.replace_data(it, probs.i[sample]);
							assert(result);

						}
						if (lm.numDetections_>0 ){
							int kbirth = lm.birthPose->id;
							
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
									lm.is_detected.insert(kbirth);
									assert(lm.numDetections_ == lm.is_detected.size() );
								}else{
									assert(0);
								}
							}else{
								assert(birth_it->second == lm.id);
							}

						}
					}
					else
					{ // if a change has to be made and new DA is false alarm, we need to remove the association
						c.DA_bimap_[k].right.erase(it);
						lm.numDetections_--;
						lm.is_detected.erase(k);
						assert(lm.numDetections_ == lm.is_detected.size() );
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
					else{
						auto &lm  = c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id] ;
						if (c.DAProbs_[k][nz].i[a] == selectedDA)
						{
							probs.i.push_back(c.DAProbs_[k][nz].i[a]);
							if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id].numDetections_ == 1)
							{
								likelihood += config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id].numFoV_) * std::log(1 - config.PD_);
								// std::cout <<" single detection: increase:  " << config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].id].numFoV_)*std::log(1-config.PD_) <<"\n";
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
								
								if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id].numDetections_ == 0)
								{
									int kbirth  = lm.birthPose->id;
									auto birth_it = c.DA_bimap_[kbirth].left.find(lm.birthMatch);
									if (birth_it == c.DA_bimap_[kbirth].left.end() ){ // if lm spawn measurement is associated then not sample
										if (lm.birthTime_ < c.poses_[k].stamp
										&& lm.birthTime_ > c.poses_[k].stamp-1.0){
											likelihood +=
												config.logExistenceOdds + (lm.numFoV_) * std::log(1 - config.PD_);
											// std::cout <<" 0 detection: increase:  " << config.logExistenceOdds + (c.landmarks_[c.DAProbs_[k][nz].i[a]-c.landmarks_[0].id].numFoV_)*std::log(1-config.PD_)<<"\n";
											probs.i.push_back(c.DAProbs_[k][nz].i[a]);
											probs.l.push_back(likelihood);
											if (likelihood > maxprob)
											{
												maxprob = likelihood;
												maxprobi = a;
											}
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
					// c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0].id] *= (P[ P.size()-1] )/P[sample];
					/*
					 if(probs.i[sample] != c.DAProbs_[k][nz].i[maxprobi]){
					 c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0].id] +=  c.DAProbs_[k][nz].l[maxprobi] - probs.l[sample]; //(1 )/alternativeprob;

					 }else{
					 c.landmarksResetProb_[probs.i[sample] -c.landmarks_[0].id] += probs.l[probs.l.size()-1] - probs.l[sample] ;
					 }*/

					c.landmarksResetProb_[probs.i[sample] - c.landmarks_[0].id] += std::log(
						P[sample] / alternativeprob);
				}
				else
				{

					if (c.DAProbs_[k][nz].i[maxlikelihoodi] >= 0)
					{
						// std::cout << "increasing init prob of lm " <<c.DAProbs_[k][nz].i[maxlikelihoodi] << "  by " <<maxlikelihood  << "- " << probs.l[sample]<< "\n";
						c.landmarksInitProb_[c.DAProbs_[k][nz].i[maxlikelihoodi] - c.landmarks_[0].id] += maxlikelihood-config.logKappa_;
					}
				}

				if (probs.i[sample] != selectedDA)
				{ // if selected association, change bimap

					if (probs.i[sample] >= 0)
					{
						auto &lm = c.landmarks_[probs.i[sample] - c.landmarks_[0].id];
						lm.numDetections_++;
						lm.is_detected.insert(k);
						if (selectedDA < 0)
						{
							auto result = c.DA_bimap_[k].insert({nz, probs.i[sample]});
							assert(result.second);
						}
						else
						{
							auto &prev_lm = c.landmarks_[selectedDA - c.landmarks_[0].id];

							prev_lm.numDetections_--;
							prev_lm.is_detected.erase(k);
							assert(prev_lm.numDetections_ >= 0);
							auto result = c.DA_bimap_[k].left.replace_data(it, probs.i[sample]);
							assert(result);

							// add an log for possible landmark merge
							c.tomerge_.push_back(
								std::make_pair(probs.i[sample], selectedDA));
						}
						if (lm.numDetections_>0 ){
							int kbirth = lm.birthPose->id;
							
							auto birth_it = c.DA_bimap_[kbirth].left.find(lm.birthMatch);
							if (birth_it == c.DA_bimap_[kbirth].left.end()){
								auto birth_it2 = c.DA_bimap_[kbirth].right.find(probs.i[sample]); 
								if (birth_it2 == c.DA_bimap_[kbirth].right.end()){
									if(lm.numDetections_== lm.numFoV_){
										std::cout << termcolor::red << " sampling: adding imposssible det_:\n" << termcolor::reset;
										printDA(c);
										assert(0);
									}
									auto result = c.DA_bimap_[kbirth].insert({c.landmarks_[probs.i[sample] - c.landmarks_[0].id].birthMatch, probs.i[sample] });
									assert(result.second);
									lm.numDetections_++;
									lm.is_detected.insert(kbirth);
								}else{
									assert(birth_it2->second  == lm.birthMatch);
								}
							}else{
								assert(birth_it->second == lm.id);
							}
						}
					}
					else
					{ // if a change has to be made and new DA is false alarm, we need to remove the association
						c.DA_bimap_[k].left.erase(it);
						c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_--;
						c.landmarks_[selectedDA - c.landmarks_[0].id].is_detected.erase(k);
						assert(c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_ >= 0);
						assert(c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_ == c.landmarks_[selectedDA - c.landmarks_[0].id].is_detected.size());
					}
				}

			}
		}
		//printDA(c);

		checkNumDet(c);
		return expectedWeightChange;
	}

	inline void VectorGLMBSLAM6D::rebuildIsam(VectorGLMBComponent6D &c){
		//std::cout << "rebuilding isam\n";
		c.isam = boost::make_shared<gtsam::ISAM2>(config.isam2_parameters);
		c.landmark_edge_indices.clear();
		c.removed_edges.clear();
		c.odometry_indices.clear();
		c.new_edges.resize (0);
		c.new_nodes.clear();


		for (auto &pose: c.poses_){
			for (auto &z : pose.Z_){
				z.reset();
			}
		}
		for (int k=0; k< c.poses_.size(); k++){
			c.DA_bimap_[k].clear();
		}
		//updateGraph(c);

		//c.new_edges.push_back(c.odometries_[0]);
		c.new_edges.addPrior(0 , c.initial_pose , config.odom_noise );
		c.new_nodes.insert(0 , c.initial_pose);

		

		c.isam_result = c.isam->update(c.new_edges, c.new_nodes, c.removed_edges);
		c.current_estimate = c.isam->calculateBestEstimate();
		loadEstimate(c);


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
					int max_dist_bef = 0 ;

					std::vector<ORBDescriptor*> descriptors;
					std::vector<int> distances_bef, distances_after;


					for (int k : map_point.is_detected){

						
						
						auto it = c.DA_bimap_[k].right.find(map_point.id);
						assert ( it != c.DA_bimap_[k].right.end() );
						int nz = it->second;
						int nl = c.poses_[k].matches_left_to_right[nz].queryIdx;
						int nr = c.poses_[k].matches_left_to_right[nz].trainIdx;
						auto &desc_left = c.poses_[k].descriptors_left[nl];
						auto &desc_right = c.poses_[k].descriptors_right[nr];
						distances_bef.push_back(ORBDescriptor::distance(map_point.descriptor , desc_left));
						distances_bef.push_back(ORBDescriptor::distance(map_point.descriptor , desc_right));
						int d =ORBDescriptor::distance(map_point.descriptor , desc_left) + ORBDescriptor::distance(map_point.descriptor , desc_right);

						if (max_dist_bef < d){
							max_dist_bef = d;
						}
						// assert(d <=200);


						descriptors.push_back(&c.poses_[k].descriptors_left[nl]);
						descriptors.push_back(&c.poses_[k].descriptors_right[nr]);
						for(int i=0;i<256;i++){
							if (desc_left.desc[i])
								avg_desc[i]++;
							if (desc_right.desc[i])
								avg_desc[i]++;
						}
						Eigen::Vector3d lmpos = map_point.position;
						Eigen::Vector3d poset = c.poses_[k].pose.translation();
						int level = c.poses_[k].keypoints_left[nl].octave;
						double depth = (lmpos-poset).norm();
						if (max_depth<depth){
							max_depth = depth;
						}
						if (min_depth > depth){
							min_depth = depth;
						}
						avg_depth +=depth*c.poses_[k].mvScaleFactors[level];
						map_point.normalVector += (lmpos-poset).normalized();

					}
					avg_depth = avg_depth/map_point.numDetections_;
					map_point.mfMaxDistance = std::max( 1.8*avg_depth , 1.3*max_depth);
					//assert(map_point.mfMaxDistance<10.0);
					map_point.mfMinDistance = std::min(map_point.mfMaxDistance / c.poses_[0].mvScaleFactors[ c.poses_[0].mnScaleLevels - 1] * 0.4 , 0.6*min_depth);

					assert(max_depth < map_point.mfMaxDistance);
					assert(min_depth > map_point.mfMinDistance);

					ORBDescriptor newDesc ;

					
						
					for(int i=0;i<256;i++){
						newDesc.desc[i] = avg_desc[i]>map_point.numDetections_;
					}
					
					int ii=0;
					int max_dist = 0;
					for (int ndesc=1; ndesc < descriptors.size() ; ndesc+=2){
						int d = ORBDescriptor::distance(newDesc , *descriptors[ndesc])+ORBDescriptor::distance(newDesc , *descriptors[ndesc-1]);
						if (d>max_dist){
							max_dist = d;
						}
						// distances_after.push_back(ORBDescriptor::distance(map_point.descriptor , *desc));
						// ii++;
						// if(ii%2 ==0 ){
						// 	assert(distances_after[ii-1]+distances_after[ii-2] <=200);
						// }
						
					}
					if (max_dist_bef>200 && max_dist >200){
						std::cout << termcolor::red <<  " dist is impossible\n"  << termcolor::reset; 
						assert(0);
					}

					if (max_dist <=200){
						map_point.descriptor = newDesc;
					}
					map_point.normalVector.normalize();



				}else{
					map_point.normalVector =  (map_point.position-map_point.birthPose->pose.translation()).normalized();
					
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


			if ( c.poses_[k].Z_.size() > 0 && c.poses_[k].isKeypose)
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
													config.viewingCosLimit_, c.camera, &predScale))
						{

							c.poses_[k].fov_.push_back(c.landmarks_[lm].id);
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


				for(int lmFovIdx =0; lmFovIdx <c.poses_[k].fov_.size();lmFovIdx++){
					int lmidx = c.poses_[k].fov_[lmFovIdx];
					auto &lm = c.landmarks_[lmidx - c.landmarks_[0].id];

					auto point_in_camera_frame = c.poses_[k].pose.transformTo(lm.position);
					if (point_in_camera_frame[2]<0 ){

						break;
					}

					gtsam::StereoPoint2 predictedZ = c.camera.project(c.poses_[k].pose.transformTo(lm.position));

					for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
					{

						double p=0.0;





						int predictedScale = c.poses_[k].predicted_scales[lmFovIdx];
						int scalediff = abs(predictedScale-c.poses_[k].keypoints_left[c.poses_[k].matches_left_to_right[nz].queryIdx].octave );
						if(scalediff >2){
							continue;
						}
						p += 20-20.0*scalediff;

						double error_scalar = (c.poses_[k].stereo_points[nz]-predictedZ).vector().norm();

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
						
						{ 
							 p += -0.5 * (error_scalar*error_scalar* config.stereoInfo_(0,0)); //chi2, this assumes information is a*Identity(3,3)
							assert(!isnan(p));
							p += -0.5 * 3 * std::log(2 * M_PI); // 3 is dimension of measurement
							assert(!isnan(p));





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







			}
		}
	}

	void VectorGLMBSLAM6D::initStereoEdge(OrbslamPose &pose, int numMatch){


		int nl = pose.matches_left_to_right[numMatch].queryIdx;



		gtsam::StereoPoint2 stereo_point( pose.keypoints_left[nl].pt.x, pose.uRight[nl] , pose.keypoints_left[nl].pt.y);


		pose.stereo_points[numMatch] = stereo_point;

		pose.initial_lm_id[numMatch] = -1;
	}

	bool VectorGLMBSLAM6D::initMapPoint(OrbslamPose &pose, int numMatch, OrbslamMapPoint &lm, int newId)
	{

		int nl = pose.matches_left_to_right[numMatch].queryIdx;

		lm.id = newId;

		lm.birthMatch = numMatch;
		lm.birthPose = &pose;

		Eigen::Vector3d point_world_frame;

		lm.numDetections_ = 0;
		lm.numFoV_ = 0;
		int level = pose.keypoints_left[nl].octave;
		const int nLevels = pose.mnScaleLevels;

		if (!cam_unproject(pose.stereo_points[numMatch], pose.point_camera_frame[numMatch]))
		{
			// std::cerr << "stereofail\n";

			return false;
		}

		lm.mfMaxDistance = pose.point_camera_frame[numMatch].norm() * pose.mvScaleFactors[level] * 1.8;
		lm.mfMinDistance = lm.mfMaxDistance / pose.mvScaleFactors[nLevels - 1] * 0.3;
		pose.initial_lm_id[numMatch] = newId;
		//pose.Z_[numMatch]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(lm.pPoint));

		// with only 2 descriptors (left,right) we can pick either one to represent the point
		// allocating new
		lm.descriptor = pose.descriptors_left[nl];
		//pose.descriptors_left.row(nl).copyTo(lm.descriptor);

		point_world_frame = pose.pose.transformFrom(pose.point_camera_frame[numMatch]);
		lm.position = point_world_frame;
		
		lm.birthTime_ = pose.stamp;


		lm.normalVector = (point_world_frame - pose.pose.translation()).normalized();
		

		return true;
	}
	void VectorGLMBSLAM6D::loadEstimate(VectorGLMBComponent6D &c)
	{



		for (auto it:c.current_estimate){
			if (it.key  < c.poses_.size()){
				c.poses_[it.key].pose =  it.value.cast<PoseType>();
			}else{
				c.landmarks_[it.key-c.landmarks_[0].id].position = it.value.cast<PointType>();
			}
		}
		for (auto &pose:c.poses_){
			if (!pose.isKeypose){
				pose.pose = c.poses_[pose.referenceKeypose].pose.transformPoseFrom(pose.transformFromReferenceKeypose);
			}
		}


		for (auto index: c.removed_edges){
			auto result_lm = c.landmark_edge_indices.right.erase(index);
			auto result_odo = c.odometry_indices.right.erase(index) ;
			assert(result_lm || result_odo );
		}
		c.removed_edges.clear();

		auto indices = c.isam_result.getNewFactorsIndices() ;
		for(int i = 0 ; i < indices.size() ;  i++){
			if (c.new_edges.at(i)->keys().size() == 1){
				// this is the prior
				continue;
			}
			auto k0 = c.new_edges.at(i)->keys()[0];
			auto k1 = c.new_edges.at(i)->keys()[1];
			if (k1 < c.poses_.size()){
				// second key is a pose , so this is odometry

				c.odometry_indices.insert( {k0 , indices[i]});

			}else{
				auto it = c.DA_bimap_[k0].right.find(k1);
				assert(it != c.DA_bimap_[k0].right.end());
				

				std::array<int , 3> association = {k0 , it->second , k1 };

				auto it_index_left  = c.landmark_edge_indices.left.find(association);
				auto found_index = it_index_left->second;
				auto it_index_right  = c.landmark_edge_indices.right.find(indices[i]);
				auto found_association = it_index_right->second;

				assert( it_index_right == c.landmark_edge_indices.right.end() );
				assert( it_index_left == c.landmark_edge_indices.left.end() );

				auto result = c.landmark_edge_indices.insert({association , indices[i]});
				
				assert(result.second);
			}
		}
		
	}


	void VectorGLMBSLAM6D::moveBirth(VectorGLMBComponent6D &c)
	{

		for( OrbslamMapPoint &lm:c.landmarks_){
			
			if ( lm.numDetections_ == 0){
				

				Eigen::Vector3d point_world_frame;
				point_world_frame = lm.birthPose->pose.transformFrom(lm.birthPose->point_camera_frame[lm.birthMatch]);
				lm.position = point_world_frame;

			}
		}



	}

	inline void VectorGLMBSLAM6D::constructGraph(VectorGLMBComponent6D &c)
	{
		c.numPoses_ = initial_component_.numPoses_;

		c.poses_.resize(initial_component_.poses_.size());
		

		int edgeid = c.numPoses_;

		for (int k = 0; k < c.poses_.size(); k++)
		{
			c.poses_[k].stamp = initial_component_.poses_[k].stamp;
			c.poses_[k].id = k;
			c.poses_[k].mvInvLevelSigma2 = initial_component_.poses_[k].mvInvLevelSigma2;


			c.poses_[k].mnScaleLevels = initial_component_.poses_[k].mnScaleLevels;
			c.poses_[k].mfScaleFactor = initial_component_.poses_[k].mfScaleFactor;
			c.poses_[k].mfLogScaleFactor = initial_component_.poses_[k].mfLogScaleFactor;
			c.poses_[k].mvScaleFactors = initial_component_.poses_[k].mvScaleFactors;
			c.poses_[k].mvLevelSigma2 = initial_component_.poses_[k].mvLevelSigma2;
			c.poses_[k].mvInvLevelSigma2 = initial_component_.poses_[k].mvInvLevelSigma2;
			c.poses_[k].isKeypose = true;
			c.poses_[k].mnMinX = initial_component_.poses_[k].mnMinX;
			c.poses_[k].mnMaxX = initial_component_.poses_[k].mnMaxX;
			c.poses_[k].mnMinY = initial_component_.poses_[k].mnMinY;
			c.poses_[k].mnMaxY = initial_component_.poses_[k].mnMaxY;
			c.poses_[k].keypoints_left = initial_component_.poses_[k].keypoints_left;
			c.poses_[k].keypoints_right = initial_component_.poses_[k].keypoints_right;
			c.poses_[k].descriptors_left = initial_component_.poses_[k].descriptors_left;
			c.poses_[k].descriptors_right = initial_component_.poses_[k].descriptors_right;


			c.poses_[k].uRight = initial_component_.poses_[k].uRight;
			c.poses_[k].depth = initial_component_.poses_[k].depth;
			c.poses_[k].matches_left_to_right = initial_component_.poses_[k].matches_left_to_right;
			c.poses_[k].predicted_scales = initial_component_.poses_[k].predicted_scales;
			c.poses_[k].initial_lm_id = initial_component_.poses_[k].initial_lm_id;
			// create graph pose
			gtsam::Pose3 pose_estimate;


			
			if (k > 0)
			{
				PoseType odometry_pose; // this is just zero odom , meaning brownian motion
				OdometryEdge::shared_ptr odo = boost::make_shared<OdometryEdge>(k-1 , k , odometry_pose , config.odom_noise);

				c.odometries_.push_back(odo);

			}

			c.poses_[k].Z_.resize(c.poses_[k].matches_left_to_right.size());

			c.poses_[k].stereo_points.resize(c.poses_[k].matches_left_to_right.size());
			c.poses_[k].initial_lm_id.resize(c.poses_[k].matches_left_to_right.size());
			c.poses_[k].point_camera_frame.resize(c.poses_[k].matches_left_to_right.size());
			
			for (int nz = 0; nz < c.poses_[k].matches_left_to_right.size(); nz++)
			{
				OrbslamMapPoint lm;
				initStereoEdge(c.poses_[k], nz);
				cam_unproject(c.poses_[k].stereo_points[nz], c.poses_[k].point_camera_frame[nz]);
		
				if (k%config.keypose_skip==0){
					if (initMapPoint(c.poses_[k], nz, lm, edgeid))
					{
						edgeid++;
						//c.optimizer_->addVertex(lm.pPoint);
						c.landmarks_.push_back(lm);
						assert(c.landmarks_.back().birthPose == &c.poses_[k]);

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
			c.landmarks_[i].is_detected.clear();
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

		c.new_nodes.clear();
		c.removed_edges.clear();
		c.new_edges.resize(0);

		c.new_edges.push_back(c.odometries_[0]);
		c.new_edges.addPrior(0 , c.initial_pose , config.odom_noise );

		c.new_nodes.insert(0,c.poses_[0].pose);
		c.new_nodes.insert(1,c.poses_[1].pose);

		c.isam_result = c.isam->update(c.new_edges, c.new_nodes, c.removed_edges);
		c.current_estimate = c.isam->calculateBestEstimate();
		loadEstimate(c);


	}

	bool VectorGLMBSLAM6D::cam_unproject(const gtsam::StereoPoint2 &stereo_point,
										 Eigen::Vector3d &trans_xyz)
	{

		trans_xyz[2] = config.stereo_baseline_f / (stereo_point.uL() -stereo_point.uR() );
		trans_xyz[0] = (stereo_point.uL() - config.camera_parameters_[0].cx) * trans_xyz[2] / config.camera_parameters_[0].fx;
		trans_xyz[1] = (stereo_point.v() - config.camera_parameters_[0].cy) * trans_xyz[2] / config.camera_parameters_[0].fy;


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
		c.isam = boost::make_shared<gtsam::ISAM2>(config.isam2_parameters);

	}

}
#endif
