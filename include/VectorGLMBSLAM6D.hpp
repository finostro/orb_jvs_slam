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
#include <atomic>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <stdio.h>
#include <filesystem>
#include <unordered_map>
#include <math.h>
#include "GaussianGenerators.hpp"
#include "AssociationSampler.hpp"
#include "OrbslamMapPoint.hpp"
#include "OrbslamPose.hpp"
#include "VectorGLMBComponent6D.hpp"
#include "MotionPriorFactor.hpp"



#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>

#include <boost/bimap.hpp>
#include <boost/container/allocator.hpp>
#include <yaml-cpp/yaml.h>

#include "misc/EigenYamlSerialization.hpp"
#include <misc/termcolor.hpp>

#include <external/ORBextractor.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "sophus/se3.hpp"

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>


#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/StereoFactor.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/inference/Symbol.h>
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
			const VectorGLMBComponent6D::FrameBimap x,
			const VectorGLMBComponent6D::FrameBimap y) const
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
	inline uint64_t daHash(const VectorGLMBComponent6D::BimapType &da);
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
			long seed_ = -1;


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
			double stereo_init_min_disparity;

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
			double visitedPruneLogRatio_ = 0; /**< >0: drop visited_ entries with (w - max w)/temp below -this */
			int tbbMaxThreads_ = 0; /**< >0: cap TBB (iSAM2's internal parallelism) at this many threads */
			double fovCacheTol_ = 0; /**< >0: updateFoV reuses pose/landmark visibility while they move less than this (m, and rad-ish) */
			bool fovCacheVerify_ = false; /**< check every cached updateFoV against the full computation and abort on mismatch */
			bool predictScale_ = false; /**< predict landmark scale from distance; otherwise every prediction is 0 (mfLogScaleFactor unset, associated landmarks skipped) */
			bool predictScaleNoMargin_ = false; /**< with predictScale: predict from distance*scale without the 1.8 margin of mfMaxDistance */
			std::string motionModel_ = "brownian"; /**< odometry prior: brownian, cv, ca or cj. odomInfo is the information of the chosen model's one-frame residual */
			double accelerationDampingTime_ = 0; /**< with ca: acceleration decays as exp(-t/this); <=0 is undamped */
			bool imu_ = false; /**< CombinedImuFactor between consecutive keyposes (spanning demoted frames) instead of the motion prior */
			std::vector<double> imuNoise_ = {1.6968e-4, 2.0e-3, 1.9393e-5, 3.0e-3}; /**< gyro noise density, accel noise density, gyro random walk, accel random walk */
			double imuInitTime_ = 1.0; /**< IMU seconds from the first frame, assumed static: gravity direction and gyro bias */
			std::vector<double> imuPriorSigmas_ = {0.05, 0.2, 0.01}; /**< frame 0 priors: velocity, accel bias, gyro bias */
			size_t visitedMaxEntries_ = 0; /**< >0: keep at most this many visited_ entries, dropping the lowest weights */
			double isamDogleg_ = 0; /**< >0: iSAM2 uses Powell's dogleg with this initial trust-region radius instead of Gauss-Newton */
			bool rebuildIsamOnResample_ = false; /**< when a component is given a hypothesis its iSAM2 does not hold, rebuild the iSAM2 from that hypothesis instead of updating the old one */
			double maxDistanceFactor_ = 0; /**< >0: a landmark's maxD is this times its level-0 distance (max with 1.05 x farthest detection) instead of 1.8 times; minD unchanged */
			std::vector<double> descOctaveScale_; /**< 4 values: descriptor per-bit weight times value |D| (cutoff divided by it), D = measured octave - predicted octave, clamped to 3; the cutoff then only gates candidates, the weight has none */
			bool scaleDescriptors_ = false; /**< a landmark keeps one descriptor per level-0 distance bin round(log(dist*1.2^octave)/log 1.2); a measurement is scored against the nearest bin, and refD is the mean level-0 distance of the most detected bin */
			std::vector<double> descScaleCost_ = {0, 1, 0, 1, 0, 1, 0, 1}; /**< scaleDescriptors: per |bin offset| 0..3 (clamped), descriptor log likelihood = c + f * descriptorLogWeight * bits, as pairs c,f; candidates need it >= 200 * descriptorLogWeight, the weight has no cutoff */
			std::vector<double> visHMM_; /**< [a, b, eps]: per-landmark visible/occluded Markov chain over its FoV keyposes, a = P(vis->vis), b = P(occ->vis) per keypose step, eps = slot PD while occluded; needs pdOctaveRatio */
			double visInit_ = -1; /**< P(visible) at a landmark's first FoV keypose; <0: stationary b/(1-a+b) */
			std::vector<double> visPDAngles_; /**< upper edges (deg) of the viewing-angle bins of visPD; the last bin is open */
			std::vector<double> visPD_; /**< visible-state slot PD, row per angle bin, columns octave offset -3..3; empty: slotPD(base PD, D) */
			bool mergeTwinAll_ = false; /**< mergeTwins merges every twin pair of the sampled keypose, each with probability mergeTwinProb, instead of one */
			double mergeTwinProb_ = 1;
			bool twinDiag_ = false; /**< write twin_diag.csv */
			std::vector<double> mergeTwinMaxBits_; /**< 3 values: if set, every birth/death step tries mergeTwins, with this max L+R distance between the twin measurements at octave difference 1, 2, 3+ */
			bool mergeLM_ = false; /**< every birth/death step, merge one pair of landmarks between which sampleDA switched a measurement */
			bool splitLM_ = false; /**< every birth/death step, move one landmark's detections at one octave to the landmark spawned by one of them */
			std::vector<double> pdOctaveRatio_; /**< 7 values: if set, a landmark is detected independently at each octave predicted+D, D=-3..3, with PD times value D+3; the per-octave scale penalty is dropped (scaleLogWeight becomes a constant bonus) */
			double pdAngleTau_ = 0; /**< >0: PD = pdAngleA exp(-theta/pdAngleTau) + pdAngleB, theta in degrees between the viewing ray and the landmark normal; PD is then unused */
			double pdAngleA_ = 0;
			double pdAngleB_ = 0;
			double maxRelDepthSigma_ = 0; /**< >0: a landmark gets stereo factors in iSAM2 only once its associated measurements fix its position to this fraction of its depth; until then it rides on its birth pose and is weighted outside iSAM2 */
			bool fullWindowAtEnd_ = true; /**< once the last pose is covered, sample data association over the whole trajectory instead of the sliding window */
			size_t isamCacheSize_ = 0; /**< >0: keep the iSAM2 state of this many highest-weight hypotheses and hand it to components that are given one of them */
			size_t dumpDACandidates_ = 0; /**< >0: every this many iterations write the association candidates of component 0 to resultFolder/dacand.csv */
			double scaleLogWeight_ = 20; /**< an association scores this much for a measurement at the predicted octave, and loses it per octave of difference */
			size_t dumpDetections_ = 0; /**< >0: every this many iterations write, for the best component, which landmarks in each keypose's field of view are detected, to resultFolder/detections.csv */
			size_t dumpIndeterminate_ = 0; /**< >0: write the factors around the variable of this many iSAM2 indeterminate-system exceptions to resultFolder/indet.txt */
			size_t dumpStallTop_ = 0; /**< >0: at the end of run() write the trajectories of this many highest-weight hypotheses evaluated since coverage last advanced */
			bool isamUnconstrainedOrdering_ = false; /**< plain COLAMD: drop iSAM2's default constraint that puts newly observed variables last */
			bool eliminateLandmarksFirst_ = false; /**< constrain iSAM2 to eliminate landmarks before poses */
			bool useRectifiedIntrinsics_ = false; /**< use the rectified camera matrix P1, not the raw calibration, for projection and unprojection */
			bool fovFreezeOld_ = false; /**< with fovCacheTol: stop re-testing visibility from keyposes before minpose (changes results) */
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

		struct TraceRow
		{
			double logweight = -std::numeric_limits<double>::infinity();
			int max_detection_time = -1;
			int frontier_k = -1, frontier_nZ = 0, frontier_n_assoc = 0, frontier_n_candidates_nonclutter = 0;
			double frontier_stamp = 0;
			gtsam::Pose3 frontier_pose;
			int seeded = 0, ransac_inliers = 0;
			double ransac_ms = 0, pose_jump_trans = 0, pose_jump_rot = 0;
		};
		std::vector<TraceRow> traceRows_;
		std::ofstream trace_;
		std::chrono::steady_clock::time_point startTime_;
		int maxDetectionTime(const VectorGLMBComponent6D &c) const;
		void writeTrace();

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

		std::vector<double> imuT_; /**< imu: sample times (s) */
		std::vector<Eigen::Vector3d> imuAcc_, imuGyro_;
		std::shared_ptr<gtsam::PreintegrationCombinedParams> imuParams_;
		gtsam::imuBias::ConstantBias imuBias0_; /**< prior mean and preintegration point: zero accel bias, static mean gyro */
		static gtsam::Key velKey(int k) { return gtsam::Symbol('v', k); }
		static gtsam::Key biasKey(int k) { return gtsam::Symbol('b', k); }
		static bool isNavKey(gtsam::Key key) { return gtsam::Symbol(key).chr() != 0; }

		/** read mav0/imu0, take gravity and gyro bias from the first imuInitTime seconds; the world frame is the first camera frame */
		void loadImu()
		{
			std::ifstream f(config.eurocFolder_ + "/mav0/imu0/data.csv");
			std::string line;
			while (std::getline(f, line))
			{
				if (line.empty() || line[0] == '#')
					continue;
				std::replace(line.begin(), line.end(), ',', ' ');
				std::istringstream ss(line);
				long long t;
				Eigen::Vector3d w, a;
				ss >> t >> w[0] >> w[1] >> w[2] >> a[0] >> a[1] >> a[2];
				imuT_.push_back(t / 1e9);
				imuGyro_.push_back(w);
				imuAcc_.push_back(a);
			}
			Eigen::Vector3d am = Eigen::Vector3d::Zero(), wm = Eigen::Vector3d::Zero();
			int n = 0;
			for (size_t i = 0; i < imuT_.size(); i++)
				if (imuT_[i] >= vTimestampsCam[0] && imuT_[i] < vTimestampsCam[0] + config.imuInitTime_)
				{
					am += imuAcc_[i];
					wm += imuGyro_[i];
					n++;
				}
			if (n == 0)
			{
				std::cerr << "no IMU samples in the first imuInitTime seconds\n";
				exit(1);
			}
			am /= n;
			wm /= n;
			gtsam::Pose3 camImu = config.base_link_to_cam0_se3.inverse(); // base_link is the IMU frame
			imuParams_ = gtsam::PreintegrationCombinedParams::MakeSharedU(9.81);
			imuParams_->n_gravity = -(camImu.rotation() * am).normalized() * 9.81; // at rest the accelerometer reads -gravity
			imuParams_->body_P_sensor = camImu;
			const auto &q = config.imuNoise_;
			imuParams_->gyroscopeCovariance = gtsam::I_3x3 * q[0] * q[0];
			imuParams_->accelerometerCovariance = gtsam::I_3x3 * q[1] * q[1];
			imuParams_->biasOmegaCovariance = gtsam::I_3x3 * q[2] * q[2];
			imuParams_->biasAccCovariance = gtsam::I_3x3 * q[3] * q[3];
			imuParams_->integrationCovariance = gtsam::I_3x3 * 1e-8;
			imuBias0_ = gtsam::imuBias::ConstantBias(gtsam::Vector3::Zero(), wm);
			std::cout << "imu: " << imuT_.size() << " samples, gravity " << imuParams_->n_gravity.transpose() << " gyro bias " << wm.transpose() << "\n";
		}

		/** preintegrated IMU from the stamp of keypose a to that of keypose b, zero-order hold */
		gtsam::NonlinearFactor::shared_ptr imuFactor(VectorGLMBComponent6D &c, int a, int b)
		{
			gtsam::PreintegratedCombinedMeasurements pim(imuParams_, imuBias0_);
			double t = c.poses_[a].stamp, tb = c.poses_[b].stamp;
			size_t i = std::max<size_t>(std::upper_bound(imuT_.begin(), imuT_.end(), t) - imuT_.begin(), 1);
			for (; t < tb; i++)
			{
				double tn = i < imuT_.size() ? std::min(imuT_[i], tb) : tb;
				if (tn > t)
					pim.integrateMeasurement(imuAcc_[i - 1], imuGyro_[i - 1], tn - t);
				t = tn;
			}
			return std::make_shared<gtsam::CombinedImuFactor>(a, velKey(a), b, velKey(b), biasKey(a), biasKey(b), pim);
		}

		/** imu: velocity and bias priors at frame 0 */
		void addImuPriors(VectorGLMBComponent6D &c)
		{
			if (!config.imu_)
				return;
			const auto &s = config.imuPriorSigmas_;
			c.new_edges.addPrior<gtsam::Vector3>(velKey(0), gtsam::Vector3::Zero(), gtsam::noiseModel::Isotropic::Sigma(3, s[0]));
			c.new_edges.addPrior(biasKey(0), imuBias0_, gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector6() << s[1], s[1], s[1], s[2], s[2], s[2]).finished()));
			c.new_nodes.insert(velKey(0), gtsam::Vector3(gtsam::Vector3::Zero()));
			c.new_nodes.insert(biasKey(0), imuBias0_);
		}

		/** imu: one factor between each pair of consecutive keyposes up to max_detection_time, replacing those whose pair changed */
		void updateImuGraph(VectorGLMBComponent6D &c, int max_detection_time)
		{
			std::vector<int> K;
			for (int k = 0; k <= max_detection_time; k++)
				if (c.poses_[k].isKeypose)
					K.push_back(k);
			std::set<std::pair<int, int>> want;
			for (size_t i = 1; i < K.size(); i++)
				want.insert({K[i - 1], K[i]});
			for (auto &e : c.imu_indices.left)
				if (!want.count(e.first))
					c.removed_edges.push_back(e.second);
			for (auto &p : want)
				if (c.imu_indices.left.find(p) == c.imu_indices.left.end())
					c.new_edges.push_back(imuFactor(c, p.first, p.second));
			for (size_t i = 1; i < K.size(); i++)
			{
				int k = K[i], kp = K[i - 1];
				if (!c.current_estimate.exists(velKey(k)))
				{
					double dt = c.poses_[k].stamp - c.poses_[kp].stamp;
					gtsam::Vector3 v = (c.poses_[k].pose.translation() - c.poses_[kp].pose.translation()) / dt;
					c.new_nodes.insert(velKey(k), v);
				}
				if (!c.current_estimate.exists(biasKey(k)))
					c.new_nodes.insert(biasKey(k), c.current_estimate.exists(biasKey(kp)) ? c.current_estimate.at<gtsam::imuBias::ConstantBias>(biasKey(kp)) : imuBias0_);
			}
		}

		/** dumpIndeterminate: describe the variable named by an indeterminate-system exception and every factor on it */
		void dumpIndeterminate(VectorGLMBComponent6D &c, const std::exception &e)
		{
			auto ie = dynamic_cast<const gtsam::IndeterminantLinearSystemException *>(&e);
			if (!ie || config.dumpIndeterminate_ == 0)
				return;
			#pragma omp critical(dumpIndet)
			if (indetDumped_ < config.dumpIndeterminate_)
			{
				indetDumped_++;
				std::ofstream f(config.resultFolder + "/indet.txt", std::ios::app);
				f << std::setprecision(6);
				gtsam::Key key = ie->nearbyVariable();
				const auto &lin = c.isam->getLinearizationPoint();
				const auto &factors = c.isam->getFactorsUnsafe();
				std::map<gtsam::Key, std::pair<int, int>> count; // per variable: stereo factors, other factors
				for (const auto &fac : factors)
					if (fac)
						for (gtsam::Key q : fac->keys())
							(std::dynamic_pointer_cast<StereoMeasurementEdge>(fac) ? count[q].first : count[q].second)++;
				f << "iteration " << iteration_ << " window " << minpose_ << " " << maxpose_ << " key " << key << " numPoses " << c.poses_.size()
				  << " inLin " << lin.exists(key) << " stereo " << count[key].first << " other " << count[key].second
				  << " newEdges " << c.new_edges.size() << " removed " << c.removed_edges.size() << "\n";
				bool isLm = key >= c.poses_.size();
				if (isNavKey(key))
					f << "  nav " << gtsam::Symbol(key).chr() << " " << gtsam::Symbol(key).index() << "\n";
				else if (lin.exists(key))
					f << "  value " << (isLm ? lin.at<gtsam::Point3>(key).transpose() : lin.at<gtsam::Pose3>(key).translation().transpose()) << "\n";
				for (const auto &fac : factors)
				{
					if (!fac || std::find(fac->keys().begin(), fac->keys().end(), key) == fac->keys().end())
						continue;
					auto st = std::dynamic_pointer_cast<StereoMeasurementEdge>(fac);
					if (!st)
					{
						f << "  other keys";
						for (gtsam::Key q : fac->keys())
							f << " " << q;
						f << "\n";
						continue;
					}
					gtsam::Key k = st->keys()[0], l = st->keys()[1];
					f << "  stereo pose " << k << " lm " << l << " z " << st->measured().uL() << " " << st->measured().uR() << " " << st->measured().v()
					  << " poseFactors " << count[k].first << "+" << count[k].second << " lmFactors " << count[l].first;
					if (lin.exists(k) && lin.exists(l))
						f << " pCam " << lin.at<gtsam::Pose3>(k).transformTo(lin.at<gtsam::Point3>(l)).transpose() << " err " << st->error(lin);
					// state before this update: what the sampler used (poses_/landmarks_) and the last iSAM2 estimate
					const auto &lm = c.landmarks_[l - c.landmarks_[0].id];
					bool isNew = false;
					for (const auto &ne : c.new_edges)
						isNew |= ne == fac;
					f << " newEdge " << isNew << " newLm " << c.new_nodes.exists(l) << " newPose " << c.new_nodes.exists(k) << " birth " << lm.birthTime_
					  << " zSampler " << c.poses_[k].pose.transformTo(lm.position).z();
					if (c.current_estimate.exists(k) && c.current_estimate.exists(l))
						f << " zPrevEst " << c.current_estimate.at<gtsam::Pose3>(k).transformTo(c.current_estimate.at<gtsam::Point3>(l)).z();
					f << "\n";
				}
			}
		}

		/** with dogleg the estimate must use iSAM2's trust-region step; calculateBestEstimate always takes the full Gauss-Newton step */
		gtsam::Values isamEstimate(VectorGLMBComponent6D &c)
		{
			return config.isamDogleg_ > 0 ? c.isam->calculateEstimate() : c.isam->calculateBestEstimate();
		}

		/** information on a landmark's position from its associated stereo measurements, poses held fixed; minDepth is its smallest depth in those cameras */
		Eigen::Matrix3d landmarkInfo(VectorGLMBComponent6D &c, const OrbslamMapPoint &lm, double &minDepth)
		{
			Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
			minDepth = std::numeric_limits<double>::infinity();
			for (int k : lm.is_detected)
			{
				gtsam::Matrix36 Hpose;
				gtsam::Matrix3 J;
				try
				{
					gtsam::StereoCamera(c.poses_[k].pose, config.cam_params).project2(lm.position, Hpose, J);
				}
				catch (const gtsam::StereoCheiralityException &)
				{
					continue;
				}
				info += J.transpose() * config.stereoInfo_ * J;
				minDepth = std::min(minDepth, c.poses_[k].pose.transformTo(lm.position).z());
			}
			return info;
		}

		double pdFor(const OrbslamPose &pose, const OrbslamMapPoint &lm) const
		{
			Eigen::Vector3d ray = (lm.position - pose.pose.translation()).normalized();
			double deg = std::acos(std::clamp(ray.dot(lm.normalVector), -1.0, 1.0)) * 180.0 / M_PI;
			return config.pdAngleA_ * std::exp(-deg / config.pdAngleTau_) + config.pdAngleB_;
		}
		/** pdOctaveRatio: a landmark is detected independently at each octave within +-3 of its predicted one */
		bool multiOctave() const { return !config.pdOctaveRatio_.empty(); }
		double descScale(int octaveDiff) const { return config.descOctaveScale_.empty() ? 1.0 : config.descOctaveScale_[std::min(std::abs(octaveDiff), 3)]; }

		/** scaleDescriptors: level-0 distance bin of a detection at this octave and distance */
		int scaleBin(const OrbslamPose &pose, double dist, int octave) const
		{
			return (int)std::lround(std::log(dist * pose.mvScaleFactors[octave]) / pose.mfLogScaleFactor);
		}

		/** scaleDescriptors: the landmark's only bin is its birth detection */
		void resetScaleDesc(OrbslamMapPoint &lm) const
		{
			auto &pose = *lm.birthPose;
			int nl = pose.matches_left_to_right[lm.birthMatch].queryIdx;
			lm.scaleDesc.assign(1, {scaleBin(pose, pose.point_camera_frame[lm.birthMatch].norm(), pose.keypoints_left[nl].octave), pose.descriptors_left[nl]});
		}

		/** descriptor log likelihood of measurement nz for lm; gate: -inf past the cutoff (candidate selection) */
		double descLogLik(const OrbslamPose &pose, int nz, OrbslamMapPoint &lm, int scalediff, bool gate) const
		{
			int nl = pose.matches_left_to_right[nz].queryIdx;
			auto &dl = pose.descriptors_left[nl];
			auto &dr = pose.descriptors_right[pose.matches_left_to_right[nz].trainIdx];
			if (!config.scaleDescriptors_ || lm.scaleDesc.empty())
				return lm.descriptor.likelihood(dl, dr, descScale(scalediff), gate || config.descOctaveScale_.empty());
			int bin = scaleBin(pose, (lm.position - pose.pose.translation()).norm(), pose.keypoints_left[nl].octave);
			const std::pair<int, ORBDescriptor> *best = &lm.scaleDesc[0];
			for (auto &s : lm.scaleDesc)
				if (std::abs(s.first - bin) < std::abs(best->first - bin))
					best = &s;
			int dS = std::min(std::abs(best->first - bin), 3);
			double d = ORBDescriptor::distance(best->second, dl) + ORBDescriptor::distance(best->second, dr);
			double ll = config.descScaleCost_[2 * dS] + config.descScaleCost_[2 * dS + 1] * ORBDescriptor::logWeightPerBit * d;
			if (gate && ll < 200 * ORBDescriptor::logWeightPerBit)
				return -std::numeric_limits<double>::infinity();
			return ll;
		}
		/** PD is per FoV entry (fov_pd, sumLogMiss_) */
		bool pdPerEntry() const { return config.pdAngleTau_ > 0 || multiOctave(); }
		double basePD(const OrbslamPose &pose, const OrbslamMapPoint &lm) const
		{
			return config.pdAngleTau_ > 0 ? pdFor(pose, lm) : config.PD_;
		}
		// PD at octave predicted + D; an association that drifted beyond +-3 keeps the +-3 value
		double slotPD(double base, int D) const
		{
			return base * config.pdOctaveRatio_[std::clamp(D, -3, 3) + 3];
		}
		// sum of log(1-PD) over the octaves a landmark predicted at octave s can be detected at
		double frameLogMiss(const OrbslamPose &pose, double base, int s) const
		{
			double m = 0;
			for (int o = std::max(0, s - 3); o <= std::min(pose.mnScaleLevels - 1, s + 3); o++)
				m += std::log(1 - slotPD(base, o - s));
			return m;
		}
		int measOctave(const OrbslamPose &pose, int nz) const
		{
			return pose.keypoints_left[pose.matches_left_to_right[nz].queryIdx].octave;
		}
		int geomScale(const OrbslamPose &pose, const OrbslamMapPoint &lm) const
		{
			double dist = (lm.position - pose.pose.translation()).norm();
			return std::clamp(int(std::ceil(std::log(lm.mfScaleRefDistance / dist) / pose.mfLogScaleFactor)), 0, pose.mnScaleLevels - 1);
		}
		// log(PD/(1-PD)) of a detection of lm in pose at octave
		double detLogOdds(const OrbslamPose &pose, const OrbslamMapPoint &lm, int octave) const
		{
			double pd = basePD(pose, lm);
			if (multiOctave())
				pd = slotPD(pd, octave - geomScale(pose, lm));
			return std::log(pd / (1 - pd));
		}
		double fovPD(const OrbslamPose &pose, int fovIdx) const
		{
			return config.pdAngleTau_ > 0 ? pose.fov_pd[fovIdx] : config.PD_;
		}
		// sum of log(1-PD) over the keyposes that have lm in their FoV
		double lmLogMiss(const OrbslamMapPoint &lm) const
		{
			if (visHMM())
				return lm.visLogMiss_;
			return pdPerEntry() ? lm.sumLogMiss_ : lm.numFoV_ * std::log(1 - config.PD_);
		}
		// same, leaving out pose
		double lmLogMissExcept(const OrbslamPose &pose, const OrbslamMapPoint &lm) const
		{
			if (multiOctave())
				return lm.sumLogMiss_ - frameLogMiss(pose, basePD(pose, lm), geomScale(pose, lm));
			return config.pdAngleTau_ > 0 ? lm.sumLogMiss_ - std::log(1 - pdFor(pose, lm)) : (lm.numFoV_ - 1) * std::log(1 - config.PD_);
		}
		// call after pushing the entry's predicted scale
		void addToFoV(OrbslamPose &pose, OrbslamMapPoint &lm)
		{
			if (!pdPerEntry())
				return;
			double pd = basePD(pose, lm);
			pose.fov_pd.push_back(pd);
			lm.sumLogMiss_ += multiOctave() ? frameLogMiss(pose, pd, pose.predicted_scales.back()) : std::log(1 - pd);
		}

		// visHMM: a landmark is visible or occluded at each keypose of its FoV track (keyposes with it in fov_, in order),
		// a Markov chain with per-keypose-step transitions; octave slots are detected independently given the state
		bool visHMM() const { return !config.visHMM_.empty(); }
		/** visHMM lookup tables, built once: everything below works in the probability domain without per-entry log/exp */
		struct VisTables
		{
			int L = 0;						 // octave levels
			std::vector<double> cosEdges;	 // cos of visPDAngles, descending
			std::vector<double> pdV, oddsV;	 // [abin * 7 + D + 3]: visible slot PD and PD / (1 - PD)
			std::vector<double> missV, missO; // [abin * L + s], [s]: P(no slot detected) given visible / occluded, slots s-3..s+3 within 0..L-1
			double oddsO = 0;
			std::vector<double> lamPow; // (a - b)^(dk / keypose_skip), dk frames (frontier frames are all keyposes, so steps can be fractional)
		} vis_;
		std::once_flag visOnce_;
		void visBuildTables(int L, int maxFrames)
		{
			const double a = config.visHMM_[0], b = config.visHMM_[1], eps = config.visHMM_[2];
			const int nb = config.visPD_.empty() ? 1 : config.visPDAngles_.size() + 1;
			vis_.L = L;
			for (double e : config.visPDAngles_)
				vis_.cosEdges.push_back(std::cos(e * M_PI / 180.0));
			vis_.pdV.resize(nb * 7);
			vis_.oddsV.resize(nb * 7);
			for (int i = 0; i < nb * 7; i++)
			{
				vis_.pdV[i] = config.visPD_.empty() ? 0 : config.visPD_[i];
				vis_.oddsV[i] = vis_.pdV[i] / (1 - vis_.pdV[i]);
			}
			vis_.missV.assign(nb * L, 1);
			vis_.missO.assign(L, 1);
			for (int s = 0; s < L; s++)
				for (int o = std::max(0, s - 3); o <= std::min(L - 1, s + 3); o++)
				{
					vis_.missO[s] *= 1 - eps;
					for (int ab = 0; ab < nb; ab++)
						vis_.missV[ab * L + s] *= 1 - vis_.pdV[ab * 7 + o - s + 3];
				}
			vis_.oddsO = eps / (1 - eps);
			vis_.lamPow.resize(maxFrames + 1);
			for (int dk = 0; dk <= maxFrames; dk++)
				vis_.lamPow[dk] = std::pow(a - b, double(dk) / config.keypose_skip);
		}
		int visAngleBin(const OrbslamPose &pose, const OrbslamMapPoint &lm) const
		{
			if (config.visPD_.empty())
				return 0;
			Eigen::Vector3d d = lm.position - pose.pose.translation();
			double cosv = d.dot(lm.normalVector) / d.norm();
			int i = 0;
			while (i < (int)vis_.cosEdges.size() && cosv < vis_.cosEdges[i])
				i++;
			return i;
		}
		/** visible slot PD at octave offset D (clamped to +-3) */
		double visSlotPD(const OrbslamPose &pose, int fovIdx, int abin, int D) const
		{
			D = std::clamp(D, -3, 3);
			return config.visPD_.empty() ? slotPD(pose.fov_pd[fovIdx], D) : vis_.pdV[abin * 7 + D + 3];
		}
		double visMissV(const OrbslamPose &pose, int fovIdx, int abin, int s) const
		{
			if (!config.visPD_.empty())
				return vis_.missV[abin * vis_.L + s];
			double m = 1;
			for (int o = std::max(0, s - 3); o <= std::min(vis_.L - 1, s + 3); o++)
				m *= 1 - slotPD(pose.fov_pd[fovIdx], o - s);
			return m;
		}
		/** P(visible at k+dk | visible at k) and P(visible at k+dk | occluded at k), dk frames */
		std::pair<double, double> visTrans(int dk) const
		{
			const double a = config.visHMM_[0], b = config.visHMM_[1], pi = b / (1 - a + b);
			const double lam = dk < (int)vis_.lamPow.size() ? vis_.lamPow[dk] : std::pow(a - b, double(dk) / config.keypose_skip);
			return {pi + (1 - pi) * lam, pi * (1 - lam)};
		}
		/** emission probabilities of FoV entry fovIdx of pose k given visible (ev) and occluded (eo); with allMiss every slot is a miss */
		void visEmission(const VectorGLMBComponent6D &c, int k, int fovIdx, int abin, bool allMiss, double &ev, double &eo) const
		{
			auto &pose = c.poses_[k];
			const int s = pose.predicted_scales[fovIdx];
			ev = visMissV(pose, fovIdx, abin, s);
			eo = vis_.missO[s];
			if (allMiss)
				return;
			for (auto r = c.DA_bimap_[k].right.equal_range(pose.fov_[fovIdx]); r.first != r.second; ++r.first)
			{
				const int D = measOctave(pose, r.first->second) - s;
				const double pd = visSlotPD(pose, fovIdx, abin, D);
				// a detection outside the +-3 slots (predicted scale moved) adds its PD without removing a miss
				ev *= std::abs(D) <= 3 ? pd / (1 - pd) : pd;
				eo *= std::abs(D) <= 3 ? vis_.oddsO : config.visHMM_[2];
			}
		}
		/** association log-odds of a single detection of fov entry fovIdx at octave o against none, given the rest of the track (fov_rho) */
		double visSlotLogOdds(const OrbslamPose &pose, int fovIdx, int o) const
		{
			const int s = pose.predicted_scales[fovIdx], abin = pose.fov_abin[fovIdx];
			const double rho = pose.fov_rho[fovIdx], pd = visSlotPD(pose, fovIdx, abin, o - s);
			const double mV = rho * visMissV(pose, fovIdx, abin, s), mO = (1 - rho) * vis_.missO[s];
			return std::log((mV * pd / (1 - pd) + mO * vis_.oddsO) / (mV + mO));
		}
		/** landmark track log-likelihoods (visLogLik_, visLogMiss_); with wmin >= 0 also fov_rho, fov_abin of poses wmin..maxpose_-1 */
		void visPass(VectorGLMBComponent6D &c, int wmin);
		/** maxRelDepthSigma: true while lm is associated but kept out of iSAM2 */
		bool lmGated(const VectorGLMBComponent6D &c, const OrbslamMapPoint &lm) const
		{
			if (config.maxRelDepthSigma_ <= 0 || lm.numDetections_ == 0)
				return false;
			size_t i = lm.id - c.landmarks_[0].id;
			return i >= c.lmInGraph.size() || !c.lmInGraph[i];
		}

		/** maxRelDepthSigma: admit landmarks whose position is determined, and drop admitted ones the optimiser has moved to where it is not (twice the threshold, to avoid flipping) */
		void updateLmGate(VectorGLMBComponent6D &c)
		{
			c.lmInGraph.resize(c.landmarks_.size(), 0);
			const int id0 = c.landmarks_[0].id;
			for (const auto &edge : c.landmark_edge_indices.left)
				c.lmInGraph[edge.first[2] - id0] = 1;
			for (auto &lm : c.landmarks_)
			{
				char &in = c.lmInGraph[lm.id - id0];
				if (lm.numDetections_ == 0)
				{
					in = 0;
					continue;
				}
				double minDepth;
				Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(landmarkInfo(c, lm, minDepth), Eigen::EigenvaluesOnly);
				double minInfo = eig.eigenvalues()[0];
				bool was = in;
				in = minInfo > 0 && 1 / std::sqrt(minInfo) < (in ? 2 : 1) * config.maxRelDepthSigma_ * minDepth;
				if (was && !in)
					lmDemoted_++;
			}
		}

		/** empty the iSAM2 of c but keep its data association, so the next updateGraph adds every factor and value again */
		void resetIsam(VectorGLMBComponent6D &c)
		{
			c.isam = std::make_shared<gtsam::ISAM2>(config.isam2_parameters);
			c.landmark_edge_indices.clear();
			c.removed_edges.clear();
			c.odometry_indices.clear();
			c.imu_indices.clear();
			for (auto &pose : c.poses_)
				for (auto &z : pose.Z_)
					z.reset();
			c.new_edges.resize(0);
			c.new_nodes.clear();
			c.new_edges.addPrior(0, c.initial_pose, config.odom_noise);
			c.new_nodes.insert(0, c.initial_pose);
			addImuPriors(c);
			c.isam_result = isamUpdate(c);
			c.current_estimate = c.new_nodes;
		}

		/** make the iSAM2 of c match the hypothesis it was just given; heldHash is the daHash c had before */
		void syncIsamOnResample(VectorGLMBComponent6D &c, uint64_t heldHash)
		{
			uint64_t hash = daHash(c.DA_bimap_);
			if (hash == heldHash)
			{
				resampleHeld_++;
				return;
			}
			auto cached = isamCache_.find(hash);
			if (cached != isamCache_.end())
			{
				resampleCached_++;
				c.isam = std::make_shared<gtsam::ISAM2>(*cached->second.isam);
				c.current_estimate = cached->second.estimate;
				c.odometry_indices = cached->second.odometry_indices;
				c.imu_indices = cached->second.imu_indices;
				c.landmark_edge_indices = cached->second.landmark_edge_indices;
				c.removed_edges.clear();
				// updateGraph looks the factors up again through landmark_edge_indices
				for (auto &pose : c.poses_)
					for (auto &z : pose.Z_)
						z.reset();
			}
			else if (config.rebuildIsamOnResample_)
			{
				resampleRebuilt_++;
				resetIsam(c);
			}
			else
				resampleStale_++;
		}

		/** isam update with the pending edges and nodes of c */
		gtsam::ISAM2Result isamUpdate(VectorGLMBComponent6D &c)
		{
			if (config.isamUnconstrainedOrdering_)
				return c.isam->update(c.new_edges, c.new_nodes, c.removed_edges, gtsam::FastMap<gtsam::Key, int>());
			if (!config.eliminateLandmarksFirst_)
				return c.isam->update(c.new_edges, c.new_nodes, c.removed_edges);
			// poses go last in the elimination order, so a pose seen with many landmarks does not fill in a dense block over them
			// only poses that still have factors after this update may be constrained
			std::vector<int> numFactors(c.poses_.size(), 0);
			const auto &index = c.isam->getVariableIndex();
			for (int k = 0; k < c.poses_.size(); k++)
			{
				auto it = index.find(k);
				if (it != index.end())
					numFactors[k] = it->second.size();
			}
			const auto &factors = c.isam->getFactorsUnsafe();
			for (auto i : c.removed_edges)
				for (auto key : factors[i]->keys())
					if (key < c.poses_.size())
						numFactors[key]--;
			for (auto &f : c.new_edges)
				for (auto key : f->keys())
					if (key < c.poses_.size())
						numFactors[key]++;
			gtsam::FastMap<gtsam::Key, int> groups;
			for (int k = 0; k < c.poses_.size(); k++)
				if (numFactors[k] > 0)
					groups[k] = 1;
			return c.isam->update(c.new_edges, c.new_nodes, c.removed_edges, groups);
		}

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
		void updateFoVFull(VectorGLMBComponent6D &c);
		void updateFoVCached(VectorGLMBComponent6D &c);
		int associatedScale(VectorGLMBComponent6D &c, int k, int lm);

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
		double splitLM(VectorGLMBComponent6D &c);

		/** merge two landmarks holding a twin pair: measurements of one keypose at different octaves within 3 px whose descriptors are within mergeTwinMaxBits */
		double mergeTwins(VectorGLMBComponent6D &c);

		/** calculateWeight terms that depend on one landmark; det is -0.5 log det of its info with poses fixed */
		struct LmTerms
		{
			double info = 0, desc = 0, scale = 0, pd = 0, miss = 0, exist = 0, chi = 0, det = 0;
			int n = 0;
			LmTerms &operator+=(const LmTerms &o)
			{
				info += o.info; desc += o.desc; scale += o.scale; pd += o.pd; miss += o.miss;
				exist += o.exist; chi += o.chi; det += o.det; n += o.n;
				return *this;
			}
		};
		LmTerms lmTerms(VectorGLMBComponent6D &c, OrbslamMapPoint &lm);
		/** twin_diag.csv: per applied twin merge, terms of src+dst before the move and after the re-optimization that follows it */
		struct TwinDiag
		{
			int src, dst, nMoves, nClash;
			LmTerms pre, post, opt; /**< post: right after the move; opt: after the re-optimization */
		};
		std::vector<std::vector<TwinDiag>> twinDiag_; /**< per component, this iteration */

		/** scaleDescriptors: per-bin descriptors and refD of lm from its current associations */
		void rebuildScaleDesc(VectorGLMBComponent6D &c, OrbslamMapPoint &lm);

		/** true if landmark lmId is a gated candidate of measurement nz at pose k (only within [minpose_, maxpose_)) */
		bool isCandidate(VectorGLMBComponent6D &c, int k, int nz, int lmId);

		/**
		 * Reassign the measurements (k, nz) of src to dst. A measurement colliding with one dst already holds
		 * in that frame (same octave when multiOctave) becomes clutter. Does nothing and returns false if
		 * dst is not a candidate of some other measurement; relaxed: dst only needs to be in that keypose's field of view.
		 */
		bool moveDetections(VectorGLMBComponent6D &c, OrbslamMapPoint &src, OrbslamMapPoint &dst, const std::vector<std::pair<int, int>> &moves, bool relaxed = false);

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
						  VectorGLMBComponent6D::FrameBimap> &da);

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
		std::vector<VectorGLMBComponent6D::FrameBimap> best_DA_;
		int best_DA_max_detection_time_ = 0; /**< last association time */

		std::map<
			std::vector<VectorGLMBComponent6D::FrameBimap>,
			EstimateWeight>
			visited_;
		// hashes and weights of entries dropped by visitedPruneLogRatio that the maxWeightDifference
		// rule would still keep; re-inserting one is refused as if the entry were still in visited_
		std::unordered_map<uint64_t, double> visitedTombstones_;
		std::map<double, std::vector<StampedPose>> stallTop_; /**< dumpStallTop: best hypotheses since coverage last advanced, by weight */
		int stallTopCover_ = -1;
		size_t indetDumped_ = 0;
		/** iSAM2 and its factor-index bookkeeping for one hypothesis */
		struct IsamState
		{
			double weight;
			std::shared_ptr<gtsam::ISAM2> isam;
			gtsam::Values estimate;
			boost::bimap<int, int> odometry_indices;
			boost::bimap<std::array<int, 3>, int> landmark_edge_indices;
			boost::bimap<std::pair<int, int>, int> imu_indices;
		};
		std::unordered_map<uint64_t, IsamState> isamCache_; /**< isamCacheSize: by daHash of the hypothesis */
		size_t resampleHeld_ = 0, resampleCached_ = 0, resampleRebuilt_ = 0, resampleStale_ = 0;
		std::atomic<size_t> lmDemoted_{0}, lmGatedLast_{0}; /**< gate_stats.csv: landmarks dropped from iSAM2 so far, gated landmarks in the last weighted component */
		
		EstimateWeight gt_est;
		double startingStamp;
		double temp_;
		int minpose_ = 0; /**< sample data association from this pose  onwards*/
		int maxpose_ = 2; /**< optimize only up to this pose */
		int maxpose_prev_ = 2;
		int iteration_ = 0;
		long mergeCount_ = 0, splitCount_ = 0, mergeTwinCount_ = 0; /**< applied mergeLM / splitLM / mergeTwins moves, all components */
		int iterationBest_ = 0;
		int lastGraphWriteIteration_ = -1000000; /**< iteration of the last video/beststate_* write */
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

		rclcpp::Node::SharedPtr node_;
		rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
	};

	//////////////////////////////// Implementation ////////////////////////

	VectorGLMBSLAM6D::VectorGLMBSLAM6D() : node_(std::make_shared<rclcpp::Node>("hglmbslam"))
	{
		nThreads_ = 1;

#ifdef _OPENMP
		nThreads_ = omp_get_max_threads();
#endif
		marker_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>("hglmbslam/marker", 10);
	}

	VectorGLMBSLAM6D::~VectorGLMBSLAM6D()
	{
	}

	void VectorGLMBSLAM6D::publishMarkers(VectorGLMBComponent6D &c)
	{

		auto makeMarker = [](const std::string &ns, int type, double scale, float r, float g, float b)
		{
			visualization_msgs::msg::Marker m;
			m.header.frame_id = "map";
			m.header.stamp = rclcpp::Time(0);
			m.ns = ns;
			m.id = 0;
			m.type = type;
			m.action = visualization_msgs::msg::Marker::ADD;
			m.pose.orientation.w = 1.0;
			m.scale.x = scale;
			m.scale.y = scale;
			m.scale.z = scale;
			m.color.a = 1.0;
			m.color.r = r;
			m.color.g = g;
			m.color.b = b;
			return m;
		};
		auto toPoint = [](const gtsam::Point3 &p)
		{
			geometry_msgs::msg::Point pt;
			pt.x = p[0];
			pt.y = p[1];
			pt.z = p[2];
			return pt;
		};

		visualization_msgs::msg::MarkerArray markers;

		// current camera pose
		auto camera = makeMarker("camera", visualization_msgs::msg::Marker::ARROW, 0.39, 1.0, 0.0, 0.0);
		camera.scale.y = camera.scale.z = 0.039;
		const auto &current_pose = c.poses_[c.maxpose_ - 1].pose;
		camera.pose.position = toPoint(current_pose.translation());
		auto q = (current_pose.rotation() * gtsam::Rot3::Ry(-M_PI / 2)).toQuaternion();
		camera.pose.orientation.x = q.x();
		camera.pose.orientation.y = q.y();
		camera.pose.orientation.z = q.z();
		camera.pose.orientation.w = q.w();
		markers.markers.push_back(camera);

		// map, colored by detection ratio
		auto map = makeMarker("map", visualization_msgs::msg::Marker::POINTS, 0.03, 0.0, 0.0, 0.0);
		for (auto &lm : c.landmarks_)
		{
			if (lm.numDetections_ > 0)
			{
				map.points.push_back(toPoint(c.poses_[0].pose.transformFrom(lm.position)));
				float ratio = std::clamp(lm.numDetections_ / (float)lm.numFoV_, 0.0f, 1.0f);
				std_msgs::msg::ColorRGBA color;
				color.r = 1.0 - ratio;
				color.g = ratio;
				color.b = 0.0;
				color.a = 1.0;
				map.colors.push_back(color);
			}
		}
		std::cout << termcolor::green << " = nlandmarks: " << map.points.size() << "\n" << termcolor::reset;
		markers.markers.push_back(map);

		// estimated trajectory
		auto trajectory = makeMarker("est_traj", visualization_msgs::msg::Marker::LINE_STRIP, 0.03, 1.0, 0.5, 0.0);
		for (auto &pose : c.poses_)
			trajectory.points.push_back(toPoint(pose.pose.translation()));
		markers.markers.push_back(trajectory);

		// ground truth trajectory
		auto gt_trajectory = makeMarker("gt_traj", visualization_msgs::msg::Marker::LINE_STRIP, 0.03, 0.0, 1.0, 0.0);
		for (auto &pose : gt_est.trajectory)
			gt_trajectory.points.push_back(toPoint(pose.pose.translation()));
		markers.markers.push_back(gt_trajectory);

		// associations
		auto associations = makeMarker("association", visualization_msgs::msg::Marker::LINE_LIST, 0.02, 0.0, 1.0, 0.0);
		for (int k = 0; k < c.maxpose_; k++)
		{
			for (auto iter = c.DA_bimap_[k].left.begin(), iend = c.DA_bimap_[k].left.end(); iter != iend; ++iter)
			{
				associations.points.push_back(toPoint(c.landmarks_[iter->second - c.landmarks_[0].id].position));
				associations.points.push_back(toPoint(c.poses_[k].pose.transformFrom(c.poses_[k].point_camera_frame[iter->first])));
			}
		}
		std::cout << "n ass: " << c.DA_bimap_[c.maxpose_-1].size()  << "  nz: "  << c.poses_[c.maxpose_-1].Z_.size() << "\n";
		markers.markers.push_back(associations);

		marker_pub_->publish(markers);

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
										VectorGLMBComponent6D::FrameBimap> &da)
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
	inline uint64_t daHash(const VectorGLMBComponent6D::BimapType &da)
	{
		uint64_t h = 1469598103934665603ULL;
		auto mix = [&h](uint64_t v)
		{
			h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
			h *= 1099511628211ULL;
		};
		for (size_t k = 0; k < da.size(); k++)
		{
			mix(da[k].size());
			for (auto &a : da[k].left)
				mix((uint64_t(uint32_t(a.first)) << 32) | uint32_t(a.second));
		}
		return h;
	}

	void VectorGLMBSLAM6D::sampleComponents()
	{
		if (config.visitedPruneLogRatio_ > 0)
		{
			double maxw = -std::numeric_limits<double>::infinity();
			for (auto &v : visited_)
				maxw = std::max(maxw, v.second.weight);
			for (auto it = visited_.begin(); it != visited_.end();)
			{
				if ((it->second.weight - maxw) / temp_ < -config.visitedPruneLogRatio_)
				{
					visitedTombstones_[daHash(it->first)] = it->second.weight;
					it = visited_.erase(it);
				}
				else
				{
					++it;
				}
			}
		}

		if (config.visitedMaxEntries_ > 0 && visited_.size() > config.visitedMaxEntries_)
		{
			// drop the lowest weights down to the cap
			std::vector<double> weights;
			weights.reserve(visited_.size());
			for (auto &v : visited_)
				weights.push_back(v.second.weight);
			size_t numDrop = visited_.size() - config.visitedMaxEntries_;
			std::nth_element(weights.begin(), weights.begin() + numDrop - 1, weights.end());
			double threshold = weights[numDrop - 1];
			for (auto it = visited_.begin(); it != visited_.end() && numDrop > 0;)
			{
				if (it->second.weight <= threshold)
				{
					visitedTombstones_[daHash(it->first)] = it->second.weight;
					it = visited_.erase(it);
					numDrop--;
				}
				else
					++it;
			}
		}

		// do logsumexp on the components to calculate probabilities

		std::map<double, std::map<rfs::VectorGLMBComponent6D::BimapType, rfs::EstimateWeight>::iterator > probs_to_iterator;
		std::vector<double> probs(visited_.size(), 0);
		double maxw = -std::numeric_limits<double>::infinity();
		for (auto entry = visited_.begin(), entry_end = visited_.end(); entry != entry_end;
			 entry++)
		{
			if (entry->second.weight > maxw)
				maxw = entry->second.weight;
		}
		int i = 1;
		probs[0] = std::exp((visited_.begin()->second.weight - maxw) / temp_);
		probs_to_iterator.insert({std::exp((visited_.begin()->second.weight - maxw) / temp_) , visited_.begin()});
		for (auto entry = std::next(visited_.begin()), entry_end = visited_.end();
			 entry != entry_end; i++, entry++)
		{
			if (entry->second.weight > maxw)
				maxw = entry->second.weight;
			probs[i] = probs[i - 1] + std::exp((entry->second.weight - maxw) / temp_);

			probs_to_iterator.insert({probs[i] , entry});
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
		std::vector<uint64_t> heldHash; // diagnostic: hypotheses the components hold before this resample
		if (config.dumpIndeterminate_ > 0)
			for (auto &c : components_)
				heldHash.push_back(daHash(c.DA_bimap_));
		for (int i = 0; i < components_.size(); i++)
		{
			auto &c = components_[i];
			auto  da1 = probs_to_iterator.lower_bound(r1)->second;
			auto  da2 = probs_to_iterator.lower_bound(r2)->second;
			if (config.dumpIndeterminate_ > 0)
			{
				// diagnostic: associations that differ between the component's DA (what its iSAM2 holds) and the one it is given
				int removed = 0, added = 0, kept = 0;
				for (int k = 0; k < c.DA_bimap_.size() && k < da1->first.size(); k++)
				{
					for (const auto &a : c.DA_bimap_[k].left)
					{
						auto f = da1->first[k].left.find(a.first);
						(f != da1->first[k].left.end() && f->second == a.second) ? kept++ : removed++;
					}
					added += da1->first[k].size();
				}
				added -= kept;
				std::ofstream(config.resultFolder + "/resample.csv", std::ios::app)
					<< iteration_ << "," << i << "," << removed << "," << added << "," << kept << ","
					<< (std::find(heldHash.begin(), heldHash.end(), daHash(da1->first)) != heldHash.end()) << "\n";
			}

			bool syncIsam = config.rebuildIsamOnResample_ || config.isamCacheSize_ > 0;
			uint64_t heldBefore = syncIsam ? daHash(c.DA_bimap_) : 0;
			if (config.doCrossover){
				auto da = sexyTime(da1->first , da2->first);
				// add data association j to component i
				changeDA(c, da);
			}else{
				changeDA(c, da1->first);
			}
			if (syncIsam)
				syncIsamOnResample(c, heldBefore);
			
			for(int k = 0 ; k < c.poses_.size() ; k++){

				c.poses_[k].pose = da1->second.trajectory[k].pose;	
				if(k < minpose_){
					bool new_is_keypose =  (k % config.keypose_skip == 0 );
					c.poses_[k].isKeypose = new_is_keypose;

					if (!c.poses_[k].isKeypose){
						
						c.poses_[k].referenceKeypose = (k / config.keypose_skip) * config.keypose_skip;
						c.poses_[k].transformFromReferenceKeypose = c.poses_[c.poses_[k].referenceKeypose].pose.transformPoseTo(c.poses_[k].pose);

						for (auto assoc = c.DA_bimap_[k].left.begin(), assoc_end = c.DA_bimap_[k].left.end(); assoc != assoc_end;
						++assoc){
							auto &lm = c.landmarks_[ assoc->second - c.landmarks_[0].id ];
							lm.numDetections_--;
							lm.eraseDetection(k);
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
			

			


			for (auto keyValue : da1->second.estimate){
				if (isNavKey(keyValue.key))
					continue;
				if (keyValue.key  < c.poses_.size()){
					c.poses_[keyValue.key].pose =  keyValue.value.cast<PoseType>();
				}else{
					c.landmarks_[keyValue.key-c.landmarks_[0].id].position = keyValue.value.cast<PointType>();
				}
			}
			if (config.maxRelDepthSigma_ > 0)
			{
				// the hypothesis stores a position only for landmarks that were in its iSAM2
				c.lmInGraph.assign(c.landmarks_.size(), 0);
				for (auto &lm : c.landmarks_)
					c.lmInGraph[lm.id - c.landmarks_[0].id] = da1->second.estimate.exists(lm.id);
			}

			// std::cout  << "sample w: " << da1->second.weight << " j " << j  << " r " << r1 <<" prob "  << probs[j]<< "\n";

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


			if (config.use_gui_)
			{
				cv::Mat imMatches;
				cv::drawMatches(imLeft_rect,
								initial_component_.poses_[ni].keypoints_left, imRight_rect,
								initial_component_.poses_[ni].keypoints_right,
								initial_component_.poses_[ni].matches_left_to_right, imMatches);
				cv::imshow("matches", imMatches);
				cv::waitKey(1);
			}
			std::cout << ni + 1 << "/" << nImages << "                                   \r";
		}
		std::cout << "\n";
		if (config.imu_)
			loadImu();
	}

	void VectorGLMBSLAM6D::computeStereoMatches(OrbslamPose &pose)
	{

		pose.uRight = std::vector<float>(pose.keypoints_left.size(), -1.0f);
		pose.depth = std::vector<float>(pose.keypoints_left.size(), -1.0f);

		pose.mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
		pose.mnScaleLevels = mpORBextractorLeft->GetLevels();
		if (config.predictScale_)
		{
			pose.mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
			pose.mfLogScaleFactor = std::log(pose.mfScaleFactor);
		}

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
		if (node["visitedPruneLogRatio"])
			config.visitedPruneLogRatio_ = node["visitedPruneLogRatio"].as<double>();
		if (node["tbbMaxThreads"])
			config.tbbMaxThreads_ = node["tbbMaxThreads"].as<int>();
		if (node["fovCacheTol"])
			config.fovCacheTol_ = node["fovCacheTol"].as<double>();
		if (node["fovCacheVerify"])
			config.fovCacheVerify_ = node["fovCacheVerify"].as<bool>();
		if (node["predictScale"])
			config.predictScale_ = node["predictScale"].as<bool>();
		if (node["visitedMaxEntries"])
			config.visitedMaxEntries_ = node["visitedMaxEntries"].as<size_t>();
		if (node["rebuildIsamOnResample"])
			config.rebuildIsamOnResample_ = node["rebuildIsamOnResample"].as<bool>();
		if (node["maxRelDepthSigma"])
			config.maxRelDepthSigma_ = node["maxRelDepthSigma"].as<double>();
		if (node["maxDistanceFactor"])
			config.maxDistanceFactor_ = node["maxDistanceFactor"].as<double>();
		if (node["descOctaveScale"])
		{
			config.descOctaveScale_ = node["descOctaveScale"].as<std::vector<double>>();
			if (config.descOctaveScale_.size() != 4 || !config.predictScale_)
				throw std::runtime_error("descOctaveScale needs 4 values and predictScale");
		}
		if (node["scaleDescriptors"])
			config.scaleDescriptors_ = node["scaleDescriptors"].as<bool>();
		if (node["descScaleCost"])
		{
			config.descScaleCost_ = node["descScaleCost"].as<std::vector<double>>();
			if (config.descScaleCost_.size() != 8)
				throw std::runtime_error("descScaleCost needs 8 values");
		}
		if (node["mergeTwinMaxBits"])
		{
			config.mergeTwinMaxBits_ = node["mergeTwinMaxBits"].as<std::vector<double>>();
			if (config.mergeTwinMaxBits_.size() != 3)
				throw std::runtime_error("mergeTwinMaxBits needs 3 values");
		}
		if (node["visHMM"])
		{
			config.visHMM_ = node["visHMM"].as<std::vector<double>>();
			if (config.visHMM_.size() != 3)
				throw std::runtime_error("visHMM needs 3 values");
			if (!node["pdOctaveRatio"])
				throw std::runtime_error("visHMM needs pdOctaveRatio");
		}
		if (node["visInit"])
			config.visInit_ = node["visInit"].as<double>();
		if (node["visPDAngles"])
			config.visPDAngles_ = node["visPDAngles"].as<std::vector<double>>();
		if (node["visPD"])
		{
			config.visPD_ = node["visPD"].as<std::vector<double>>();
			if (config.visPD_.size() != 7 * (config.visPDAngles_.size() + 1))
				throw std::runtime_error("visPD needs 7 values per visPDAngles bin, plus 7 for the open bin");
		}
		if (node["mergeTwinAll"])
			config.mergeTwinAll_ = node["mergeTwinAll"].as<bool>();
		if (node["mergeTwinProb"])
			config.mergeTwinProb_ = node["mergeTwinProb"].as<double>();
		if (node["twinDiag"])
			config.twinDiag_ = node["twinDiag"].as<bool>();
		if (node["mergeLM"])
			config.mergeLM_ = node["mergeLM"].as<bool>();
		if (node["splitLM"])
			config.splitLM_ = node["splitLM"].as<bool>();
		if (node["pdOctaveRatio"])
		{
			config.pdOctaveRatio_ = node["pdOctaveRatio"].as<std::vector<double>>();
			if (config.pdOctaveRatio_.size() != 7 || !config.predictScale_)
				throw std::runtime_error("pdOctaveRatio needs 7 values and predictScale");
		}
		if (node["pdAngleTau"])
		{
			config.pdAngleTau_ = node["pdAngleTau"].as<double>();
			config.pdAngleA_ = node["pdAngleA"].as<double>();
			config.pdAngleB_ = node["pdAngleB"].as<double>();
		}
		if (node["fullWindowAtEnd"])
			config.fullWindowAtEnd_ = node["fullWindowAtEnd"].as<bool>();
		if (node["isamCacheSize"])
			config.isamCacheSize_ = node["isamCacheSize"].as<size_t>();
		if (node["dumpDACandidates"])
			config.dumpDACandidates_ = node["dumpDACandidates"].as<size_t>();
		if (node["scaleLogWeight"])
			config.scaleLogWeight_ = node["scaleLogWeight"].as<double>();
		if (node["descriptorLogWeight"])
			ORBDescriptor::logWeightPerBit = node["descriptorLogWeight"].as<double>();
		if (node["dumpDetections"])
			config.dumpDetections_ = node["dumpDetections"].as<size_t>();
		if (node["dumpIndeterminate"])
			config.dumpIndeterminate_ = node["dumpIndeterminate"].as<size_t>();
		if (node["dumpStallTop"])
			config.dumpStallTop_ = node["dumpStallTop"].as<size_t>();
		if (node["isamUnconstrainedOrdering"])
			config.isamUnconstrainedOrdering_ = node["isamUnconstrainedOrdering"].as<bool>();
		if (node["eliminateLandmarksFirst"])
			config.eliminateLandmarksFirst_ = node["eliminateLandmarksFirst"].as<bool>();
		if (node["motionModel"])
			config.motionModel_ = node["motionModel"].as<std::string>();
		if (node["imu"])
			config.imu_ = node["imu"].as<bool>();
		if (node["imuNoise"])
			config.imuNoise_ = node["imuNoise"].as<std::vector<double>>();
		if (node["imuInitTime"])
			config.imuInitTime_ = node["imuInitTime"].as<double>();
		if (node["imuPriorSigmas"])
			config.imuPriorSigmas_ = node["imuPriorSigmas"].as<std::vector<double>>();
		if (config.imuNoise_.size() != 4 || config.imuPriorSigmas_.size() != 3)
		{
			std::cerr << "imuNoise needs 4 values and imuPriorSigmas 3\n";
			exit(1);
		}
		if (config.motionModel_ != "brownian" && config.motionModel_ != "cv" && config.motionModel_ != "ca" && config.motionModel_ != "cj")
		{
			std::cerr << "unknown motionModel " << config.motionModel_ << "\n";
			exit(1);
		}
		if (node["accelerationDampingTime"])
			config.accelerationDampingTime_ = node["accelerationDampingTime"].as<double>();
		if (node["predictScaleNoMargin"])
			config.predictScaleNoMargin_ = node["predictScaleNoMargin"].as<bool>();
		if (node["useRectifiedIntrinsics"])
			config.useRectifiedIntrinsics_ = node["useRectifiedIntrinsics"].as<bool>();
		if (node["fovFreezeOld"])
			config.fovFreezeOld_ = node["fovFreezeOld"].as<bool>();

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
		if (node["isamDogleg"])
			config.isamDogleg_ = node["isamDogleg"].as<double>();
		if (config.isamDogleg_ > 0)
			config.isam2_parameters.optimizationParams = gtsam::ISAM2DoglegParams(config.isamDogleg_);


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
		if (node["seed"])
		{
			config.seed_ = node["seed"].as<long>();
			srand48(config.seed_);
		}
		trace_.open(config.resultFolder + "/trace.csv");
		trace_ << std::setprecision(17);
		trace_ << "iteration,wall_s,component,logweight,best_logweight,maxpose,minpose,max_detection_time,"
				  "best_DA_max_detection_time,frontier_k,frontier_nZ,frontier_n_assoc,frontier_n_candidates_nonclutter,"
				  "seeded,ransac_inliers,ransac_ms,pose_jump_trans,pose_jump_rot,"
				  "frontier_stamp,frontier_tx,frontier_ty,frontier_tz,frontier_qw,frontier_qx,frontier_qy,frontier_qz\n";

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

		config.cam_params = std::make_shared<gtsam::Cal3_S2Stereo>(config.camera_parameters_[0].fx ,
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

		if (config.useRectifiedIntrinsics_)
		{
			config.camera_parameters_[0].fx = P1.at<double>(0, 0);
			config.camera_parameters_[0].fy = P1.at<double>(1, 1);
			config.camera_parameters_[0].cx = P1.at<double>(0, 2);
			config.camera_parameters_[0].cy = P1.at<double>(1, 2);
			config.stereo_baseline_f = config.stereo_baseline * config.camera_parameters_[0].fx;
			config.cam_params = std::make_shared<gtsam::Cal3_S2Stereo>(config.camera_parameters_[0].fx,
																		config.camera_parameters_[0].fy,
																		0,
																		config.camera_parameters_[0].cx,
																		config.camera_parameters_[0].cy,
																		config.stereo_baseline);
		}

        
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
		config.stereo_init_min_disparity = node["stereo_init_min_disparity"].as<double>();

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
	inline int VectorGLMBSLAM6D::maxDetectionTime(const VectorGLMBComponent6D &c) const
	{
		int max_detection_time = std::max(maxpose_ - 1, 0);
		while (max_detection_time > 0 && c.DA_bimap_[max_detection_time].size() == 0)
			max_detection_time--;
		return max_detection_time;
	}

	inline void VectorGLMBSLAM6D::writeTrace()
	{
		double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime_).count();
		for (int i = 0; i < traceRows_.size(); i++)
		{
			auto &r = traceRows_[i];
			auto q = r.frontier_pose.rotation().toQuaternion();
			auto t = r.frontier_pose.translation();
			trace_ << iteration_ << "," << wall_s << "," << i << "," << r.logweight << "," << bestWeight_ << ","
				   << maxpose_ << "," << minpose_ << "," << r.max_detection_time << "," << best_DA_max_detection_time_ << ","
				   << r.frontier_k << "," << r.frontier_nZ << "," << r.frontier_n_assoc << "," << r.frontier_n_candidates_nonclutter << ","
				   << r.seeded << "," << r.ransac_inliers << "," << r.ransac_ms << "," << r.pose_jump_trans << "," << r.pose_jump_rot << ","
				   << r.frontier_stamp << ","
				   << t.x() << "," << t.y() << "," << t.z() << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "\n";
		}
		trace_.flush();
	}

	inline void VectorGLMBSLAM6D::run(int numSteps)
	{
		startTime_ = std::chrono::steady_clock::now();

		for (int i = 0; i < numSteps && rclcpp::ok(); i++)
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
				if (visited_.size() + visitedTombstones_.size() < 10) maxpose_ = maxpose_prev_;
			}

			
			if (config.fullWindowAtEnd_ && best_DA_max_detection_time_ == components_[0].poses_.size() - 1)
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
			if (config.rebuildIsamOnResample_ || config.isamCacheSize_ > 0)
				std::ofstream(config.resultFolder + "/resample_stats.txt")
					<< "iteration " << iteration_ << " held " << resampleHeld_ << " cached " << resampleCached_
					<< " rebuilt " << resampleRebuilt_ << " stale " << resampleStale_ << " cacheSize " << isamCache_.size() << "\n";
			if (config.mergeLM_ || config.splitLM_ || !config.mergeTwinMaxBits_.empty())
				std::ofstream(config.resultFolder + "/merge_split.csv", std::ios::app)
					<< iteration_ << "," << mergeCount_ << "," << splitCount_ << "," << mergeTwinCount_ << "\n";
			if (config.maxRelDepthSigma_ > 0)
				std::ofstream(config.resultFolder + "/gate_stats.csv", std::ios::app)
					<< iteration_ << "," << lmDemoted_ << "," << lmGatedLast_ << "\n";
			if (config.dumpDetections_ > 0 && iteration_ % config.dumpDetections_ == 0)
			{
				// one row per existing landmark in the field of view of a keypose, for the highest-weight component
				auto best = std::max_element(components_.begin(), components_.end(),
					[](const VectorGLMBComponent6D &a, const VectorGLMBComponent6D &b) { return a.logweight_ < b.logweight_; });
				std::ofstream f(config.resultFolder + "/detections.csv");
				// map, poses and raw stereo detections, for analysis that does not use the associations
				auto hex = [](const ORBDescriptor &d) {
					std::ostringstream s;
					for (int i = 0; i < 4; i++)
						s << std::hex << std::setw(16) << std::setfill('0') << ((d.desc >> (64 * i)) & std::bitset<256>(~0ULL)).to_ullong();
					return s.str();
				};
				auto poseRow = [](const gtsam::Pose3 &p) {
					std::ostringstream s;
					auto q = p.rotation().toQuaternion();
					s << std::setprecision(10) << p.x() << "," << p.y() << "," << p.z() << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z();
					return s.str();
				};
				std::ofstream fm(config.resultFolder + "/map.csv");
				fm << "lm,numDet,birthK,x,y,z,nx,ny,nz,minD,maxD,desc\n";
				for (auto &lm : best->landmarks_)
					if (lm.numDetections_ > 0)
						fm << lm.id << "," << lm.numDetections_ << "," << lm.birthPose->id << "," << std::setprecision(10) << lm.position.x() << "," << lm.position.y() << ","
						   << lm.position.z() << "," << lm.normalVector.x() << "," << lm.normalVector.y() << "," << lm.normalVector.z() << ","
						   << lm.mfMinDistance << "," << lm.mfMaxDistance << "," << hex(lm.descriptor) << "\n";
				std::ofstream fp(config.resultFolder + "/poses.csv");
				fp << "k,isKeypose,cx,cy,cz,cqw,cqx,cqy,cqz,bx,by,bz,bqw,bqx,bqy,bqz\n";
				for (int k = 0; k < maxpose_; k++)
					fp << k << "," << best->poses_[k].isKeypose << "," << poseRow(best->poses_[k].pose) << ","
					   << poseRow(best->poses_[k].pose * config.base_link_to_cam0_se3.inverse()) << "\n";
				if (!std::ifstream(config.resultFolder + "/frames.csv").good())
				{
					std::ofstream ff(config.resultFolder + "/frames.csv");
					ff << "k,nz,uL,uR,v,octave,descL,descR\n";
					for (int k = 0; k < best->poses_.size(); k++)
					{
						auto &pose = best->poses_[k];
						for (int nz = 0; nz < pose.stereo_points.size(); nz++)
						{
							auto &match = pose.matches_left_to_right[nz];
							ff << k << "," << nz << "," << pose.stereo_points[nz].uL() << "," << pose.stereo_points[nz].uR() << "," << pose.stereo_points[nz].v() << ","
							   << pose.keypoints_left[match.queryIdx].octave << "," << hex(pose.descriptors_left[match.queryIdx]) << "," << hex(pose.descriptors_right[match.trainIdx]) << "\n";
						}
					}
				}
				f << "k,nZ,lm,detected,numDet,numFoV,birthK,depth,cosView,predScale,octave\n";
				// one row per stereo measurement of a keypose; lm is -1 when it is unassociated (clutter)
				std::ofstream fz(config.resultFolder + "/measurements.csv");
				fz << "k,nz,uL,uR,v,octave,lm,descDist,errPx,scaleDiff\n";
				for (int k = 0; k < maxpose_; k++)
				{
					auto &pose = best->poses_[k];
					if (!pose.isKeypose)
						continue;
					std::unordered_map<int, int> fovIndex;
					for (int n = 0; n < pose.fov_.size(); n++)
						fovIndex[pose.fov_[n]] = n;
					for (int nz = 0; nz < pose.Z_.size(); nz++)
					{
						int octave = pose.keypoints_left[pose.matches_left_to_right[nz].queryIdx].octave;
						auto &z = pose.stereo_points[nz];
						fz << k << "," << nz << "," << z.uL() << "," << z.uR() << "," << z.v() << "," << octave;
						auto assoc = best->DA_bimap_[k].left.find(nz);
						if (assoc == best->DA_bimap_[k].left.end())
						{
							fz << ",-1,,,\n";
							continue;
						}
						auto &lm = best->landmarks_[assoc->second - best->landmarks_[0].id];
						Eigen::Vector3d pCam = pose.pose.transformTo(lm.position);
						auto fovIt = fovIndex.find(lm.id);
						fz << "," << lm.id << ","
						   << lm.descriptor.likelihood(pose.descriptors_left[pose.matches_left_to_right[nz].queryIdx], pose.descriptors_right[pose.matches_left_to_right[nz].trainIdx]) / ORBDescriptor::logWeightPerBit << ",";
						if (pCam.z() > 0)
							fz << (z - best->camera.project(pCam)).vector().norm();
						fz << ",";
						if (fovIt != fovIndex.end())
							fz << std::abs(pose.predicted_scales[fovIt->second] - octave);
						fz << "\n";
					}
					for (int n = 0; n < pose.fov_.size(); n++)
					{
						auto &lm = best->landmarks_[pose.fov_[n] - best->landmarks_[0].id];
						if (lm.numDetections_ == 0)
							continue;
						auto assoc = best->DA_bimap_[k].right.find(lm.id);
						bool detected = assoc != best->DA_bimap_[k].right.end();
						Eigen::Vector3d pCam = pose.pose.transformTo(lm.position);
						Eigen::Vector3d pBirth = lm.birthPose->pose.transformTo(lm.position);
						f << k << "," << pose.Z_.size() << "," << lm.id << "," << detected << "," << lm.numDetections_ << "," << lm.numFoV_ << ","
						  << lm.birthPose->id << "," << pCam.norm() << ","
						  << (pose.pose.rotation() * pCam).normalized().dot((lm.birthPose->pose.rotation() * pBirth).normalized()) << ","
						  << pose.predicted_scales[n] << ","
						  << (detected ? pose.keypoints_left[pose.matches_left_to_right[assoc->second].queryIdx].octave : -1) << "\n";
					}
				}
			}
		}
		if (config.dumpStallTop_ > 0)
		{
			std::ofstream weights(config.resultFolder + "/visited_weights.txt");
			weights << std::setprecision(17);
			size_t r = 0;
			for (auto top = stallTop_.rbegin(); top != stallTop_.rend(); ++top, ++r)
			{
				weights << r << " " << top->first << "\n";
				std::ofstream file(config.resultFolder + "/visited_" + std::to_string(r) + ".tum");
				file << std::setprecision(20);
				for (int k = 0; k < maxpose_ && k < (int)top->second.size(); k++)
				{
					PoseType p = top->second[k].pose * config.base_link_to_cam0_se3.inverse();
					auto q = p.rotation().toQuaternion();
					file << components_[0].poses_[k].stamp << " " << p.translation().x() << " " << p.translation().y() << " "
						 << p.translation().z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
				}
			}
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
		std::vector<VectorGLMBComponent6D::FrameBimap> out;
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

			c.isam_result = isamUpdate(c);
			c.current_estimate = isamEstimate(c);
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
							likelihood += config.logExistenceOdds + lmLogMiss(c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id]);
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
									config.logExistenceOdds + lmLogMiss(c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id]);
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
							c.landmarks_[selectedDA - c.landmarks_[0].id].eraseDetection(k);

							assert(c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_ >= 0);
							c.DA_bimap_[k].left.replace_data(it, newdai);
						}
					}
					else
					{ // if a change has to be made and new DA is false alarm, we need to remove the association
						c.DA_bimap_[k].left.erase(it);
					if(selectedDA>=0){
						c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_--;
						c.landmarks_[selectedDA - c.landmarks_[0].id].eraseDetection(k);
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
			std::vector<VectorGLMBComponent6D::FrameBimap> out(
				map1);
			return out;
		}
		boost::uniform_int<> random_merge_point(minpose_, maxpose_-1);
		std::vector<VectorGLMBComponent6D::FrameBimap> out;
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


		traceRows_.assign(components_.size(), TraceRow());
		twinDiag_.assign(components_.size(), {});
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < components_.size(); i++)
		{


			
			auto &c = components_[i];
			auto &trace_row = traceRows_[i];

			// if (iteration_ % 15 == 0){
			// 	rebuildIsam(c);
			// }

			int threadnum = 0;
			#ifdef _OPENMP
			threadnum = omp_get_thread_num();
			#endif


		





			bool inserted;
			std::map<
			std::vector<VectorGLMBComponent6D::FrameBimap>,
			EstimateWeight>::iterator it;


			perturbTraj(c);
			moveBirth(c);

			updateMetaStates(c);

			updateFoV(c);

			if (!c.reverted_)
				updateDAProbs(c, minpose_, maxpose_);

			if (maxDetectionTime(c) + 1 < maxpose_)
			{
				int fk = maxDetectionTime(c) + 1;
				trace_row.frontier_k = fk;
				trace_row.frontier_nZ = c.poses_[fk].Z_.size();
				trace_row.frontier_stamp = c.poses_[fk].stamp;
				trace_row.frontier_pose = c.poses_[fk].pose;
				for (auto &probs : c.DAProbs_[fk])
					if (probs.i.size() > 1)
						trace_row.frontier_n_candidates_nonclutter++;
			}
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
					if (config.mergeLM_)
						mergeLM(c);
					if (config.splitLM_)
						splitLM(c);
					if (!config.mergeTwinMaxBits_.empty())
						mergeTwins(c);


					checkGraph(c);
					updateGraph(c);
					checkGraph(c);

					try{
					c.isam_result = isamUpdate(c);
					}catch(const std::exception& e){
						dumpIndeterminate(c, e);
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
					c.current_estimate = isamEstimate(c);
					loadEstimate(c);

					moveBirth(c);
					updateMetaStates(c);

					checkGraph(c);
					updateFoV(c);
					checkGraph(c);
					for (auto &d : twinDiag_[i])
					{
						d.opt = lmTerms(c, c.landmarks_[d.src - c.landmarks_[0].id]);
						d.opt += lmTerms(c, c.landmarks_[d.dst - c.landmarks_[0].id]);
					}
					updateDAProbs(c, minpose_, maxpose_);

				}

				for (int ng = 1; ng < config.numGibbs_; ng++)
				{
					if(ng%2==0){
						expectedChange += sampleDA(c);
						//reverseSampleDA(c);
					}else if (multiOctave()){
						// reverseSampleDA gives a landmark at most one measurement per frame
						sampleDA(c);
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
				c.isam_result = isamUpdate(c);
				for(int isam_i = 0; isam_i < config.numLevenbergIterations_; isam_i++){
					c.isam->update();
				}
			
			}catch(const std::exception& e){
					dumpIndeterminate(c, e);
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
			
			c.current_estimate = isamEstimate(c);
			
			
			loadEstimate(c);
			moveBirth(c);
			updateMetaStates(c);
			updateFoV(c);

			calculateWeight(c);
			if (!twinDiag_[i].empty())
#pragma omp critical(twindiag)
			{
				std::ofstream f(config.resultFolder + "/twin_diag.csv", std::ios::app);
				auto put = [&](const LmTerms &t)
				{ f << "," << t.n << "," << t.info << "," << t.desc << "," << t.scale << "," << t.pd << "," << t.miss << "," << t.exist << "," << t.chi << "," << t.det; };
				for (auto &d : twinDiag_[i])
				{
					f << iteration_ << "," << i << "," << d.src << "," << d.dst << "," << d.nMoves << "," << d.nClash;
					put(d.pre);
					put(d.post);
					put(d.opt);
					f << "," << c.landmarks_[d.src - c.landmarks_[0].id].numDetections_ << "," << c.landmarks_[d.dst - c.landmarks_[0].id].numDetections_ << "," << c.logweight_ << "\n";
				}
			}

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
			uint64_t pairHash = (config.visitedPruneLogRatio_ > 0 || config.visitedMaxEntries_ > 0 || config.isamCacheSize_ > 0) ? daHash(pair.first) : 0;


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
			trace_row.logweight = c.logweight_;
			trace_row.max_detection_time = max_detection_time_;
			if (trace_row.frontier_k >= 0)
				trace_row.frontier_n_assoc = c.DA_bimap_[trace_row.frontier_k].size();
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

			// isamCacheSize: copy the solver outside the critical section, only when it will enter the cache
			auto cacheMin = [this]() {
				auto m = isamCache_.begin();
				for (auto e = isamCache_.begin(); e != isamCache_.end(); ++e)
					if (e->second.weight < m->second.weight)
						m = e;
				return m;
			};
			bool cacheState = false;
			IsamState state;
			if (config.isamCacheSize_ > 0)
			{
				#pragma omp critical(bestweight)
				cacheState = !isamCache_.count(pairHash) && (isamCache_.size() < config.isamCacheSize_ || c.logweight_ > cacheMin()->second.weight);
				if (cacheState)
					state = {c.logweight_, std::make_shared<gtsam::ISAM2>(*c.isam), c.current_estimate, c.odometry_indices, c.landmark_edge_indices, c.imu_indices};
			}

			#pragma omp critical(bestweight)
			{
				if (cacheState && isamCache_.emplace(pairHash, std::move(state)).second && isamCache_.size() > config.isamCacheSize_)
					isamCache_.erase(cacheMin());
				if (config.dumpStallTop_ > 0)
				{
					if (best_DA_max_detection_time_ != stallTopCover_)
					{
						stallTop_.clear();
						stallTopCover_ = best_DA_max_detection_time_;
					}
					stallTop_.emplace(c.logweight_, to_insert.trajectory);
					if (stallTop_.size() > config.dumpStallTop_)
						stallTop_.erase(stallTop_.begin());
				}
				if (c.logweight_ > bestWeight_-config.maxWeightDifference){
					if (!visitedTombstones_.empty() && visitedTombstones_.count(pairHash))
						inserted = false;
					else
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

					// the full-graph dumps grow with the trajectory, so write at most once per 10 iterations
					if (iteration_ - lastGraphWriteIteration_ >= 10)
					{
						lastGraphWriteIteration_ = iteration_;
						gtsam::writeG2o 	( c.isam->getFactorsUnsafe() ,c.current_estimate, filename_g2o.str()	);
						c.isam->saveGraph(filename_dot.str() );
					}
					// std::cout << "globalchi2 " << c.isam_result.getErrorAfter() << "\n";
					std::cout <<"  log det H: " << c.logDetH_ << "\n";

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

					for (auto i = visitedTombstones_.begin(); i != visitedTombstones_.end();)
					{
						if (i->second < bestWeight_ - config.maxWeightDifference)
							i = visitedTombstones_.erase(i);
						else
							++i;
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

		writeTrace();
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
						}
						int scalediff = 0;
						if (lmFovIdx >= 0)
						{
							int predictedScale = c.poses_[k].predicted_scales[lmFovIdx];
							scalediff = abs(predictedScale-c.poses_[k].keypoints_left[c.poses_[k].matches_left_to_right[nz].queryIdx].octave );

							scale_logw += multiOctave() ? config.scaleLogWeight_ : config.scaleLogWeight_ * (1 - scalediff);
						}
						//int dist = ORBDescriptor::distance(lm.descriptor, c.poses_[k].descriptors_left[c.poses_[k].matches_left_to_right[nz].queryIdx]);
						double desclikelihood = descLogLik(c.poses_[k], nz, lm, scalediff, false);

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

			if (visHMM())
				continue;
			if (multiOctave())
			{
				for (int lm = 0; lm < c.poses_[k].fov_.size(); lm++)
				{
					const double base = c.poses_[k].fov_pd[lm];
					const int s = c.poses_[k].predicted_scales[lm];
					int detected = 0;
					for (auto r = c.DA_bimap_[k].right.equal_range(c.poses_[k].fov_[lm]); r.first != r.second; ++r.first)
					{
						int o = measOctave(c.poses_[k], r.first->second);
						detected |= 1 << o;
						pd_logw += std::log(slotPD(base, o - s));
					}
					if (c.landmarks_[c.poses_[k].fov_[lm] - c.landmarks_[0].id].numDetections_ > 0)
						for (int o = std::max(0, s - 3); o <= std::min(c.poses_[k].mnScaleLevels - 1, s + 3); o++)
							if (!(detected & (1 << o)))
								pd_logw += std::log(1 - slotPD(base, o - s));
				}
				continue;
			}
			for (int lm = 0; lm < c.poses_[k].fov_.size(); lm++)
			{

				if (c.DA_bimap_[k].right.count(c.poses_[k].fov_[lm]) > 0)
				{
					pd_logw += std::log(fovPD(c.poses_[k], lm));
				}
				else
				{
					bool exists = c.landmarks_[c.poses_[k].fov_[lm] - c.landmarks_[0].id].numDetections_ > 0;
					if (exists)
					{
						pd_logw += std::log(1 - fovPD(c.poses_[k], lm));
					}
				}
			}
		}
		if (visHMM())
		{
			visPass(c, -1);
			for (auto &lm : c.landmarks_)
				if (lm.numDetections_ > 0)
					pd_logw += lm.visLogLik_;
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
		// Laplace term -0.5*log det(H). The Bayes tree is whitened, so H = R^T R and
		// log det(H) = 2 * sum over cliques of log det(R_clique).
		double logDetR = 0;
		std::vector<gtsam::ISAM2::sharedClique> cliques(c.isam->roots().begin(), c.isam->roots().end());
		while (!cliques.empty())
		{
			auto clique = cliques.back();
			cliques.pop_back();
			logDetR += clique->conditional()->logDeterminant();
			cliques.insert(cliques.end(), clique->children.begin(), clique->children.end());
		}
		c.logDetH_ = 2 * logDetR;
		// maxRelDepthSigma: gated landmarks are not in iSAM2, so add their residuals and Laplace term here, poses held fixed
		int numGated = 0;
		if (config.maxRelDepthSigma_ > 0)
			for (auto &lm : c.landmarks_)
			{
				if (!lmGated(c, lm))
					continue;
				numGated++;
				for (auto kIt = lm.is_detected.begin(); kIt != lm.is_detected.end(); kIt = lm.is_detected.upper_bound(*kIt))
				{
					int k = *kIt;
					for (auto r = c.DA_bimap_[k].right.equal_range(lm.id); r.first != r.second; ++r.first)
					{
						int nz = r.first->second;
						StereoMeasurementEdge f(c.poses_[k].stereo_points[nz], config.stereo_noise, k, lm.id, config.cam_params);
						gtsam::Vector e = f.evaluateError(c.poses_[k].pose, lm.position, nullptr, nullptr);
						chi_logw -= config.stereo_noise->loss(config.stereo_noise->squaredMahalanobisDistance(e));
					}
				}
				double minDepth;
				double det = landmarkInfo(c, lm, minDepth).determinant();
				if (det > 0)
					c.logDetH_ += std::log(det);
			}
		lmGatedLast_ = numGated;
		double det_logw = -0.5 * c.logDetH_;

		logw += chi_logw+det_logw;
		 std::cout << termcolor::blue << "weight: " <<logw << " gated: " << numGated << " det_logw: " <<     det_logw
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
		if (config.imu_)
			updateImuGraph(c, max_detection_time);
		for (int k = 1; k <= max_detection_time && !config.imu_; k++)
		{
			auto it = c.odometry_indices.left.find(k-1);
			bool allKeyposes = true;
			for (auto key : c.odometries_[k-1]->keys())
				allKeyposes = allKeyposes && c.poses_[key].isKeypose;
			if(allKeyposes){
				if (it == c.odometry_indices.left.end() ){
					c.new_edges.push_back(c.odometries_[k-1]);
				}
			}else{
				if (it != c.odometry_indices.left.end() ){
					c.removed_edges.push_back(it->second);
				}
			}
			
		}


		const bool gate = config.maxRelDepthSigma_ > 0;
		if (gate)
			updateLmGate(c);

		for (int k = 0; k <= max_detection_time; k++)
		{
			// odometries

			if (!c.poses_[k].isKeypose){
				continue;
			}

			for (auto iter = c.DA_bimap_[k].left.begin(), iend = c.DA_bimap_[k].left.end(); iter != iend;
			 ++iter)
			{
				if (gate && !c.lmInGraph[iter->second - c.landmarks_[0].id])
					continue;

				if (c.poses_[k].Z_[iter->first]){
					if (c.poses_[k].Z_[iter->first]->key2() == iter->second){
						// measurement found , it should have an index

						std::array<int , 3> association = {k , iter->first , iter->second };

						auto it_index_left  = c.landmark_edge_indices.left.find(association);
						auto found_index = it_index_left->second;
						assert(it_index_left !=c.landmark_edge_indices.left.end() );
						
					}else{

						
						c.poses_[k].Z_[iter->first] = std::make_shared<StereoMeasurementEdge>(
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
						c.poses_[k].Z_[iter->first] = std::dynamic_pointer_cast<StereoMeasurementEdge>(c.isam->getFactorsUnsafe()[found_index] );
					}else{
						c.poses_[k].Z_[iter->first] = std::make_shared<StereoMeasurementEdge>(
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

			if (bimap_it ==  c.DA_bimap_[k].left.end() || !c.poses_[k].isKeypose || (gate && !c.lmInGraph[lmid - c.landmarks_[0].id])){
				c.poses_[k].Z_[nz] = NULL; 
				c.removed_edges.push_back(it.second);
			}else{
				if(bimap_it->second != lmid){
					// Z_ is empty when the measurement moved to a gated landmark after Z_ was reset
					if (!c.poses_[k].Z_[nz] || c.poses_[k].Z_[nz]->keys()[1] == lmid){
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
			if (lm.numDetections_ > 0 && !lmGated(c, lm)){

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
				int n = c.DA_bimap_[k].right.count(lm.id);
				numdet += n;
				assert(lm.is_detected.count(k) == n);
				assert(n <= 1 || multiOctave());
				if (n > 1)
				{
					// at most one measurement per octave
					int seen = 0;
					for (auto r = c.DA_bimap_[k].right.equal_range(lm.id); r.first != r.second; ++r.first)
					{
						int bit = 1 << measOctave(c.poses_[k], r.first->second);
						assert(!(seen & bit));
						seen |= bit;
					}
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

		// Candidate (k, nz) entries of each landmark in DAProbs_, in (k, nz) order. Built on the
		// first birth so each birth reads its own entries instead of scanning every candidate list.
		struct BirthCandidate { int k, nz; double l, l0; };
		std::vector<int> candidateStart;
		std::vector<BirthCandidate> candidates;
		auto buildCandidates = [&]() {
			const int lmId0 = c.landmarks_[0].id;
			candidateStart.assign(c.landmarks_.size() + 1, 0);
			for (int k = minpose_; k < maxpose_; k++)
				for (auto &probs : c.DAProbs_[k])
					for (int a = 0; a < probs.i.size(); a++)
						if (probs.i[a] >= 0)
							candidateStart[probs.i[a] - lmId0 + 1]++;
			for (int i = 0; i < c.landmarks_.size(); i++)
				candidateStart[i + 1] += candidateStart[i];
			candidates.resize(candidateStart.back());
			std::vector<int> fill(candidateStart.begin(), candidateStart.end() - 1);
			for (int k = minpose_; k < maxpose_; k++)
				for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
				{
					auto &probs = c.DAProbs_[k][nz];
					for (int a = 0; a < probs.i.size(); a++)
						if (probs.i[a] >= 0)
							candidates[fill[probs.i[a] - lmId0]++] = {k, nz, probs.l[a], probs.l[0]};
				}
		};

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
			c.landmarksInitProb_[i] += lmLogMiss(lm);
			double aux= exp(c.landmarksInitProb_[i] );
			double p = aux/(1+aux);
			if (p > uni_dist(randomGenerators_[threadnum]))
			{
				// reset all associations to false alarms
				expectedWeightChange += lmLogMiss(lm);
				//int numdet = 0;
				if (candidateStart.empty())
					buildCandidates();
				double birthL = std::numeric_limits<double>::quiet_NaN();
				for (int e = candidateStart[i], eEnd = candidateStart[i + 1]; e < eEnd;)
				{
					int k = candidates[e].k;
					if ( k == lm.birthPose->id ){
						for (; e < eEnd && candidates[e].k == k; e++)
							if (candidates[e].nz == lm.birthMatch)
								birthL = candidates[e].l;
						continue;
					}
					// best free measurement per octave slot (a single slot unless multiOctave)
					const int numSlots = multiOctave() ? c.poses_[k].mnScaleLevels : 1;
					double maxl[16];
					int max_nz[16];
					std::fill(maxl, maxl + numSlots, -std::numeric_limits<double>::infinity());
					std::fill(max_nz, max_nz + numSlots, -5);
					for (; e < eEnd && candidates[e].k == k; e++)
					{
						auto &cand = candidates[e];
						// if measurement is associated, continue
						auto it = c.DA_bimap_[k].left.find(cand.nz);
						if (it != c.DA_bimap_[k].left.end())
						{
							continue;
						}
						int slot = multiOctave() ? measOctave(c.poses_[k], cand.nz) : 0;
						if (cand.l > cand.l0 && cand.l > maxl[slot])
						{
							maxl[slot] = cand.l;
							max_nz[slot] = cand.nz;
						}
					}
					for (int slot = 0; slot < numSlots; slot++)
						if (max_nz[slot]>=0){
							c.DA_bimap_[k].insert( {max_nz[slot], c.landmarks_[i].id});
							expectedWeightChange += maxl[slot] - config.logKappa_;
							c.landmarks_[i].numDetections_++;
							c.landmarks_[i].is_detected.insert(k);
						}

				}


				int kbirth = lm.birthPose->id;
				
				auto birth_it2 = c.DA_bimap_[kbirth].right.find(lm.id); 

				if (birth_it2 == c.DA_bimap_[kbirth].right.end()){
					if(!multiOctave() && lm.numDetections_== lm.numFoV_){
						std::cout << termcolor::red << "adding imposssible det_:\n" << termcolor::reset;
						printDA(c);
						assert(0);
					}
					auto result = c.DA_bimap_[kbirth].insert({lm.birthMatch, lm.id });
					assert(result.second);
					lm.numDetections_++;
					lm.is_detected.insert(kbirth);
					assert(lm.numDetections_ == lm.is_detected.size() );
					if (std::isfinite(birthL))
						expectedWeightChange += birthL - config.logKappa_;
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
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif
		if (c.tomerge_.empty())
			return 0;
		boost::uniform_int<> random_pair(0, c.tomerge_.size() - 1);
		auto pair = c.tomerge_[random_pair(rfs::randomGenerators_[threadnum])];
		c.tomerge_.clear();
		auto *del = &c.landmarks_[pair.first - c.landmarks_[0].id];
		auto *add = &c.landmarks_[pair.second - c.landmarks_[0].id];
		if (add->numDetections_ < del->numDetections_)
			std::swap(del, add);
		if (del->numDetections_ == 0)
			return 0;
		std::vector<std::pair<int, int>> moves;
		for (auto kIt = del->is_detected.begin(); kIt != del->is_detected.end(); kIt = del->is_detected.upper_bound(*kIt))
			for (auto r = c.DA_bimap_[*kIt].right.equal_range(del->id); r.first != r.second; ++r.first)
				moves.push_back({*kIt, r.first->second});
		if (moveDetections(c, *del, *add, moves))
		{
#pragma omp atomic
			mergeCount_++;
		}
		checkNumDet(c);
		return 0;
	}

	inline double VectorGLMBSLAM6D::splitLM(VectorGLMBComponent6D &c)
	{
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif
		const int lmId0 = c.landmarks_[0].id;
		boost::uniform_int<> random_k(minpose_, maxpose_ - 1);
		for (int tries = 0; tries < 20; tries++)
		{
			int k = random_k(rfs::randomGenerators_[threadnum]);
			auto &bm = c.DA_bimap_[k];
			if (bm.empty())
				continue;
			boost::uniform_int<> random_nz(0, bm.size() - 1);
			auto it = bm.left.begin();
			std::advance(it, random_nz(rfs::randomGenerators_[threadnum]));
			int nz = it->first;
			auto &src = c.landmarks_[it->second - lmId0];
			int dstId = c.poses_[k].initial_lm_id[nz];
			if (dstId < 0 || dstId == src.id || c.landmarks_[dstId - lmId0].numDetections_ > 0)
				continue;
			int octave = measOctave(c.poses_[k], nz);
			std::vector<std::pair<int, int>> moves;
			for (auto kIt = src.is_detected.begin(); kIt != src.is_detected.end(); kIt = src.is_detected.upper_bound(*kIt))
				for (auto r = c.DA_bimap_[*kIt].right.equal_range(src.id); r.first != r.second; ++r.first)
					if (measOctave(c.poses_[*kIt], r.first->second) == octave)
						moves.push_back({*kIt, r.first->second});
			if (moves.size() == src.numDetections_)
				continue;
			if (moveDetections(c, src, c.landmarks_[dstId - lmId0], moves))
			{
#pragma omp atomic
				splitCount_++;
				break;
			}
		}
		checkNumDet(c);
		return 0;
	}

	inline double VectorGLMBSLAM6D::mergeTwins(VectorGLMBComponent6D &c)
	{
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif
		const int lmId0 = c.landmarks_[0].id;
		boost::uniform_int<> random_k(minpose_, maxpose_ - 1);
		boost::uniform_real<> uni(0, 1);
		std::vector<std::pair<int, int>> twins, moves; // twins: measurement pairs (nz, nz)
		for (int tries = 0; tries < 20; tries++)
		{
			int k = random_k(rfs::randomGenerators_[threadnum]);
			auto &pose = c.poses_[k];
			auto &bm = c.DA_bimap_[k];
			twins.clear();
			for (auto a = bm.left.begin(); a != bm.left.end(); ++a)
				for (auto b = std::next(a); b != bm.left.end(); ++b)
				{
					int dO = std::abs(measOctave(pose, a->first) - measOctave(pose, b->first));
					if (a->second == b->second || dO == 0 || (pose.stereo_points[a->first] - pose.stereo_points[b->first]).vector().norm() > 3)
						continue;
					auto &ma = pose.matches_left_to_right[a->first], &mb = pose.matches_left_to_right[b->first];
					double d = ORBDescriptor::distance(pose.descriptors_left[ma.queryIdx], pose.descriptors_left[mb.queryIdx]) + ORBDescriptor::distance(pose.descriptors_right[ma.trainIdx], pose.descriptors_right[mb.trainIdx]);
					if (d <= config.mergeTwinMaxBits_[std::min(dO, 3) - 1])
						twins.push_back({a->first, b->first});
				}
			if (twins.empty())
				continue;
			if (config.mergeTwinAll_)
				std::shuffle(twins.begin(), twins.end(), rfs::randomGenerators_[threadnum]);
			else
			{
				boost::uniform_int<> random_pair(0, twins.size() - 1);
				twins = {twins[random_pair(rfs::randomGenerators_[threadnum])]};
			}
			bool merged = false;
			for (auto [nzA, nzB] : twins)
			{
				if (config.mergeTwinAll_ && uni(rfs::randomGenerators_[threadnum]) >= config.mergeTwinProb_)
					continue;
				// earlier merges in this frame may have moved or dropped either measurement
				auto ia = bm.left.find(nzA), ib = bm.left.find(nzB);
				if (ia == bm.left.end() || ib == bm.left.end() || ia->second == ib->second)
					continue;
				auto *src = &c.landmarks_[ia->second - lmId0];
				auto *dst = &c.landmarks_[ib->second - lmId0];
				if (dst->numDetections_ < src->numDetections_)
					std::swap(src, dst);
				moves.clear();
				for (auto kIt = src->is_detected.begin(); kIt != src->is_detected.end(); kIt = src->is_detected.upper_bound(*kIt))
					for (auto r = c.DA_bimap_[*kIt].right.equal_range(src->id); r.first != r.second; ++r.first)
						moves.push_back({*kIt, r.first->second});
				TwinDiag diag{src->id, dst->id, (int)moves.size(), dst->numDetections_};
				if (config.twinDiag_)
				{
					diag.pre = lmTerms(c, *src);
					diag.pre += lmTerms(c, *dst);
				}
				if (!moveDetections(c, *src, *dst, moves, true))
					continue;
				merged = true;
				diag.nClash = diag.nMoves - (dst->numDetections_ - diag.nClash);
				if (config.scaleDescriptors_)
					rebuildScaleDesc(c, *dst);
				if (config.twinDiag_)
				{
					diag.post = lmTerms(c, *src);
					diag.post += lmTerms(c, *dst);
					twinDiag_[&c - components_.data()].push_back(diag);
				}
#pragma omp atomic
				mergeTwinCount_++;
			}
			if (merged)
				break;
		}
		checkNumDet(c);
		return 0;
	}

	inline void VectorGLMBSLAM6D::visPass(VectorGLMBComponent6D &c, int wmin)
	{
		const int nlm = c.landmarks_.size(), lmId0 = c.landmarks_[0].id;
		const double a = config.visHMM_[0], b = config.visHMM_[1];
		const double init = config.visInit_ >= 0 ? config.visInit_ : b / (1 - a + b);
		const bool window = wmin >= 0;
		std::call_once(visOnce_, [&] { visBuildTables(c.poses_[0].mnScaleLevels, c.poses_.size()); });
		// forward: filtered P(visible) of each landmark at its latest FoV entry, as associated (D) and with all misses (M);
		// track likelihoods are running products of the normalizers, folded into logs before they underflow
		thread_local std::vector<double> alphaD, alphaM, prodD, prodM, muV, muO;
		thread_local std::vector<int> lastK;
		thread_local std::vector<std::vector<std::array<double, 3>>> win; // window entries: predicted P(visible), emissions
		alphaD.assign(nlm, 0);
		alphaM.assign(nlm, 0);
		prodD.assign(nlm, 1);
		prodM.assign(nlm, 1);
		lastK.assign(nlm, -1);
		for (auto &lm : c.landmarks_)
			lm.visLogLik_ = lm.visLogMiss_ = 0;
		if (window && win.size() < maxpose_ - wmin)
			win.resize(maxpose_ - wmin);
		constexpr double fold = 1e-250;
		for (int k = 0; k < maxpose_; k++)
		{
			auto &pose = c.poses_[k];
			const bool inWin = window && k >= wmin;
			if (inWin)
			{
				pose.fov_abin.resize(pose.fov_.size());
				pose.fov_rho.resize(pose.fov_.size());
				win[k - wmin].resize(pose.fov_.size());
			}
			for (int f = 0; f < pose.fov_.size(); f++)
			{
				const int i = pose.fov_[f] - lmId0;
				auto &lm = c.landmarks_[i];
				const int abin = visAngleBin(pose, lm);
				double ev, eo, mv, mo;
				visEmission(c, k, f, abin, true, mv, mo);
				if (lm.numDetections_ > 0 && lm.is_detected.find(k) != lm.is_detected.end())
					visEmission(c, k, f, abin, false, ev, eo);
				else
					ev = mv, eo = mo;
				double pD = init, pM = init;
				if (lastK[i] >= 0)
				{
					auto [vv, ov] = visTrans(k - lastK[i]);
					pD = alphaD[i] * vv + (1 - alphaD[i]) * ov;
					pM = alphaM[i] * vv + (1 - alphaM[i]) * ov;
				}
				double nv = pD * ev, z = nv + (1 - pD) * eo;
				alphaD[i] = nv / z;
				if ((prodD[i] *= z) < fold)
					lm.visLogLik_ += std::log(prodD[i]), prodD[i] = 1;
				nv = pM * mv, z = nv + (1 - pM) * mo;
				alphaM[i] = nv / z;
				if ((prodM[i] *= z) < fold)
					lm.visLogMiss_ += std::log(prodM[i]), prodM[i] = 1;
				lastK[i] = k;
				if (inWin)
				{
					pose.fov_abin[f] = abin;
					win[k - wmin][f] = {pD, ev, eo};
				}
			}
		}
		for (int i = 0; i < nlm; i++)
		{
			c.landmarks_[i].visLogLik_ += std::log(prodD[i]);
			c.landmarks_[i].visLogMiss_ += std::log(prodM[i]);
		}
		if (!window)
			return;
		// backward over the window: mu = emission times backward message at the landmark's next FoV entry, rescaled
		muV.assign(nlm, 1);
		muO.assign(nlm, 1);
		lastK.assign(nlm, -1);
		for (int k = maxpose_ - 1; k >= wmin; k--)
		{
			auto &pose = c.poses_[k];
			for (int f = 0; f < pose.fov_.size(); f++)
			{
				const int i = pose.fov_[f] - lmId0;
				double bv = 1, bo = 1;
				if (lastK[i] >= 0)
				{
					auto [vv, ov] = visTrans(lastK[i] - k);
					bv = vv * muV[i] + (1 - vv) * muO[i];
					bo = ov * muV[i] + (1 - ov) * muO[i];
				}
				auto &w = win[k - wmin][f];
				pose.fov_rho[f] = w[0] * bv / (w[0] * bv + (1 - w[0]) * bo);
				double v = w[1] * bv, o = w[2] * bo, s = v + o;
				muV[i] = v / s;
				muO[i] = o / s;
				lastK[i] = k;
			}
		}
	}

	inline VectorGLMBSLAM6D::LmTerms VectorGLMBSLAM6D::lmTerms(VectorGLMBComponent6D &c, OrbslamMapPoint &lm)
	{
		LmTerms t;
		const double info = -0.5 * (3 * std::log(2 * M_PI) - std::log(config.stereoInfo_.determinant()));
		// refit the position by IRLS with poses fixed, so pre and post are compared at their own optimum
		const gtsam::Point3 saved = lm.position;
		for (int iter = 0; iter < 10 && lm.numDetections_ > 0; iter++)
		{
			Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
			Eigen::Vector3d g = Eigen::Vector3d::Zero();
			for (auto kIt = lm.is_detected.begin(); kIt != lm.is_detected.end(); kIt = lm.is_detected.upper_bound(*kIt))
				for (auto r = c.DA_bimap_[*kIt].right.equal_range(lm.id); r.first != r.second; ++r.first)
				{
					StereoMeasurementEdge e(c.poses_[*kIt].stereo_points[r.first->second], config.stereo_noise, *kIt, lm.id, config.cam_params);
					gtsam::Matrix J;
					gtsam::Vector res;
					try
					{
						res = e.evaluateError(c.poses_[*kIt].pose, lm.position, nullptr, &J);
					}
					catch (const gtsam::StereoCheiralityException &)
					{
						continue;
					}
					double w = config.stereo_noise->robust()->weight(std::sqrt(res.dot(config.stereoInfo_ * res)));
					H += w * J.transpose() * config.stereoInfo_ * J;
					g += w * J.transpose() * config.stereoInfo_ * res;
				}
			if (H.determinant() <= 0)
				break;
			lm.position = lm.position - H.ldlt().solve(g);
		}
		for (int k = 0; k < c.poses_.size(); k++)
		{
			auto &pose = c.poses_[k];
			if (!pose.isKeypose)
				continue;
			auto f = std::find(pose.fov_.begin(), pose.fov_.end(), lm.id);
			if (f == pose.fov_.end())
				continue;
			int fi = f - pose.fov_.begin();
			int s = pose.predicted_scales[fi];
			int detected = 0;
			for (auto r = c.DA_bimap_[k].right.equal_range(lm.id); r.first != r.second; ++r.first)
			{
				int nz = r.first->second;
				int o = measOctave(pose, nz);
				int scalediff = std::abs(s - o);
				t.n++;
				t.info += info;
				t.scale += multiOctave() ? config.scaleLogWeight_ : config.scaleLogWeight_ * (1 - scalediff);
				t.desc += descLogLik(pose, nz, lm, scalediff, false);
				StereoMeasurementEdge e(pose.stereo_points[nz], config.stereo_noise, k, lm.id, config.cam_params);
				t.chi -= config.stereo_noise->loss(config.stereo_noise->squaredMahalanobisDistance(e.evaluateError(pose.pose, lm.position, nullptr, nullptr)));
				if (multiOctave())
				{
					detected |= 1 << o;
					t.pd += std::log(slotPD(pose.fov_pd[fi], o - s));
				}
			}
			if (lm.numDetections_ == 0)
				continue;
			if (multiOctave())
			{
				for (int o = std::max(0, s - 3); o <= std::min(pose.mnScaleLevels - 1, s + 3); o++)
					if (!(detected & (1 << o)))
						t.miss += std::log(1 - slotPD(pose.fov_pd[fi], o - s));
			}
			else if (c.DA_bimap_[k].right.count(lm.id) > 0)
				t.pd += std::log(fovPD(pose, fi));
			else
				t.miss += std::log(1 - fovPD(pose, fi));
		}
		if (lm.numDetections_ > 0)
		{
			t.exist = config.logExistenceOdds;
			double minDepth;
			double det = landmarkInfo(c, lm, minDepth).determinant();
			if (det > 0)
				t.det = -0.5 * std::log(det);
		}
		lm.position = saved;
		return t;
	}

	inline void VectorGLMBSLAM6D::rebuildScaleDesc(VectorGLMBComponent6D &c, OrbslamMapPoint &lm)
	{
		struct BinAcc
		{
			int bin;
			int n = 0;
			double level0 = 0;
			std::array<int, 256> cnt{};
		};
		thread_local std::vector<BinAcc> acc;
		acc.clear();
		for (auto kIt = lm.is_detected.begin(); kIt != lm.is_detected.end(); kIt = lm.is_detected.upper_bound(*kIt))
			for (auto r = c.DA_bimap_[*kIt].right.equal_range(lm.id); r.first != r.second; ++r.first)
			{
				auto &pose = c.poses_[*kIt];
				auto &m = pose.matches_left_to_right[r.first->second];
				int level = pose.keypoints_left[m.queryIdx].octave;
				double depth = (lm.position - pose.pose.translation()).norm();
				int bin = scaleBin(pose, depth, level);
				auto a = std::find_if(acc.begin(), acc.end(), [&](const BinAcc &x) { return x.bin == bin; });
				if (a == acc.end())
					a = acc.insert(acc.end(), BinAcc{bin});
				a->n++;
				a->level0 += depth * pose.mvScaleFactors[level];
				auto &dl = pose.descriptors_left[m.queryIdx].desc, &dr = pose.descriptors_right[m.trainIdx].desc;
				for (int i = 0; i < 256; i++)
					a->cnt[i] += dl[i] + dr[i];
			}
		if (acc.empty())
		{
			resetScaleDesc(lm);
			return;
		}
		std::sort(acc.begin(), acc.end(), [](const BinAcc &a, const BinAcc &b) { return a.bin < b.bin; });
		lm.scaleDesc.clear();
		const BinAcc *mode = &acc[0];
		for (auto &a : acc)
		{
			ORBDescriptor d;
			for (int i = 0; i < 256; i++)
				d.desc[i] = a.cnt[i] > a.n;
			lm.scaleDesc.push_back({a.bin, d});
			if (a.n > mode->n)
				mode = &a;
		}
		if (config.predictScaleNoMargin_)
			lm.mfScaleRefDistance = mode->level0 / mode->n;
	}

	inline bool VectorGLMBSLAM6D::isCandidate(VectorGLMBComponent6D &c, int k, int nz, int lmId)
	{
		if (k < minpose_ || k >= maxpose_ || nz >= c.DAProbs_[k].size())
			return false;
		auto &i = c.DAProbs_[k][nz].i;
		return std::find(i.begin(), i.end(), lmId) != i.end();
	}

	inline bool VectorGLMBSLAM6D::moveDetections(VectorGLMBComponent6D &c, OrbslamMapPoint &src, OrbslamMapPoint &dst, const std::vector<std::pair<int, int>> &moves, bool relaxed)
	{
		auto inFoV = [&](int k)
		{
			auto &fov = c.poses_[k].fov_;
			return k >= minpose_ && k < maxpose_ && std::find(fov.begin(), fov.end(), dst.id) != fov.end();
		};
		auto clashes = [&](int k, int nz)
		{
			for (auto r = c.DA_bimap_[k].right.equal_range(dst.id); r.first != r.second; ++r.first)
				if (!multiOctave() || measOctave(c.poses_[k], r.first->second) == measOctave(c.poses_[k], nz))
					return true;
			return false;
		};
		for (auto [k, nz] : moves)
			if (!clashes(k, nz) && !(relaxed ? inFoV(k) : isCandidate(c, k, nz, dst.id)))
				return false;
		for (auto [k, nz] : moves)
		{
			auto it = c.DA_bimap_[k].left.find(nz);
			src.numDetections_--;
			src.eraseDetection(k);
			if (clashes(k, nz))
			{
				c.DA_bimap_[k].left.erase(it);
				continue;
			}
			c.DA_bimap_[k].left.replace_data(it, dst.id);
			dst.numDetections_++;
			dst.is_detected.insert(k);
		}
		return true;
	}

	inline double VectorGLMBSLAM6D::sampleLMDeath(VectorGLMBComponent6D &c)
	{
		double expectedWeightChange = 0;
		boost::uniform_real<> uni_dist(0, 1);
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif
		if (visHMM())
			visPass(c, -1);

		for (int i = 0; i < c.landmarks_.size(); i++)
		{
			if (c.landmarks_[i].numDetections_ < 1)
			{
				continue;
			}
			if (c.landmarks_[i].birthTime_< c.poses_[minpose_].stamp){
				continue;
			}

			if (visHMM())
			{
				auto &lm = c.landmarks_[i];
				c.landmarksResetProb_[i] = lm.numDetections_ * config.logKappa_ - lm.visLogLik_ - config.logExistenceOdds;
			}
			else if (multiOctave())
			{
				auto &lm = c.landmarks_[i];
				double logPDOddsDet = 0;
				for (auto kIt = lm.is_detected.begin(); kIt != lm.is_detected.end(); kIt = lm.is_detected.upper_bound(*kIt))
					for (auto r = c.DA_bimap_[*kIt].right.equal_range(lm.id); r.first != r.second; ++r.first)
						logPDOddsDet += detLogOdds(c.poses_[*kIt], lm, measOctave(c.poses_[*kIt], r.first->second));
				c.landmarksResetProb_[i] = lm.numDetections_ * config.logKappa_ - logPDOddsDet - lmLogMiss(lm) - config.logExistenceOdds;
			}
			else if (config.pdAngleTau_ > 0)
			{
				double logPDOddsDet = 0;
				for (int k : c.landmarks_[i].is_detected)
				{
					double pd = pdFor(c.poses_[k], c.landmarks_[i]);
					logPDOddsDet += std::log(pd / (1 - pd));
				}
				c.landmarksResetProb_[i] = c.landmarks_[i].numDetections_ * config.logKappa_ - logPDOddsDet - lmLogMiss(c.landmarks_[i]) - config.logExistenceOdds;
			}
			else
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
						c.DA_bimap_[k].right.erase(c.landmarks_[i].id);
					}
				}
				expectedWeightChange +=-config.logExistenceOdds;
				expectedWeightChange += -lmLogMiss(c.landmarks_[i]);
				
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
	/**
	 * Per-thread scratch marks indexed by landmark. Each call reserves `span` fresh stamp values
	 * starting at the returned base + 1, so stale marks from earlier calls never match and the
	 * buffer is never cleared; 0 is never a valid stamp.
	 */
	inline std::vector<int> &landmarkMarks(size_t size, int span, int &base)
	{
		thread_local std::vector<int> marks;
		thread_local int next = 0;
		if (marks.size() < size)
			marks.resize(size, 0);
		if (next > std::numeric_limits<int>::max() - span - 1)
		{
			std::fill(marks.begin(), marks.end(), 0);
			next = 0;
		}
		base = next;
		next += span + 1;
		return marks;
	}

	/**
	 * Cheap frustum test on flat copies of landmark positions and distance limits. It rejects only
	 * pairs OrbslamPose::isInFrustum would also reject (margins far above rounding error), so calling
	 * isInFrustum only when mayBeVisible() holds gives the same result.
	 */
	struct FrustumPrefilter
	{
		std::vector<Eigen::Vector3d> pos;
		std::vector<double> maxD2, minD2;
		double fx, fy, skew, cx, cy, fxb;
		Eigen::Matrix3d R;
		Eigen::Vector3d t;
		double minX, maxX, minY, maxY;

		FrustumPrefilter(const VectorGLMBComponent6D &c, const std::vector<int> &lms)
		{
			const auto &K = c.camera.calibration();
			fx = K.fx(), fy = K.fy(), skew = K.skew(), cx = K.px(), cy = K.py(), fxb = K.fx() * K.baseline();
			pos.resize(lms.size());
			maxD2.resize(lms.size());
			minD2.resize(lms.size());
			for (size_t i = 0; i < lms.size(); i++)
			{
				auto &lm = c.landmarks_[lms[i]];
				pos[i] = lm.position;
				double maxD = lm.mfMaxDistance * (1 + 1e-4) + 1e-3;
				double minD = lm.mfMinDistance * (1 - 1e-4) - 1e-3;
				maxD2[i] = maxD * maxD;
				minD2[i] = minD > 0 ? minD * minD : 0;
			}
		}
		void setPose(const OrbslamPose &p)
		{
			R = p.pose.rotation().matrix();
			t = p.pose.translation();
			minX = p.mnMinX - 1, maxX = p.mnMaxX + 1, minY = p.mnMinY - 1, maxY = p.mnMaxY + 1;
		}
		bool mayBeVisible(size_t i) const
		{
			Eigen::Vector3d d = pos[i] - t;
			double d2 = d.squaredNorm();
			if (d2 > maxD2[i] || d2 < minD2[i])
				return false;
			double z = R.col(2).dot(d);
			if (z < -1e-9)
				return false;
			if (z > 1e-6)
			{
				double x = R.col(0).dot(d), y = R.col(1).dot(d);
				double u = (fx * x + skew * y) / z + cx;
				double v = fy * y / z + cy;
				if (u < minX || u > maxX || v < minY || v > maxY || u - fxb / z < minX)
					return false;
			}
			return true;
		}
	};

	/**
	 * Bounds OrbslamPose::isInFrustum over every pose within tol of (R0, t0) (|dt| and Frobenius |dR|)
	 * and every landmark within tol of its Ref (position, normal, min/max distance):
	 * OUT if it is false for all of them, IN if it is true with the same predicted scale for all,
	 * FRAGILE otherwise. Margins also cover isInFrustum's float rounding.
	 */
	struct FovClassifier
	{
		enum Result { OUT, IN, FRAGILE };
		double fx, fy, skew, cx, cy, fxb, tol, cosLimit;

		FovClassifier(const VectorGLMBComponent6D &c, double tol, float viewingCosLimit) : tol(tol), cosLimit(viewingCosLimit)
		{
			const auto &K = c.camera.calibration();
			fx = K.fx(), fy = K.fy(), skew = K.skew(), cx = K.px(), cy = K.py(), fxb = K.fx() * K.baseline();
		}
		Result operator()(const Eigen::Matrix3d &R0, const Eigen::Vector3d &t0, const OrbslamPose &pose, const FovCache::Ref &r) const
		{
			Eigen::Vector3d a = r.p0 - t0;
			double dist = a.norm();
			if (dist < 1e-6)
				return FRAGILE;
			double ea = 2 * tol;			   // bound on |change of p - t|
			double ed = ea + tol * dist;	   // bound on |change of the camera-frame point|
			double eD = tol + 1e-5 + 1e-6 * dist; // distance-limit change plus float rounding
			double minD = r.minD0, maxD = r.maxD0;
			if (dist + ea + eD < minD || dist - ea - eD > maxD)
				return OUT;
			bool in = dist - ea - eD >= minD && dist + ea + eD <= maxD;

			double cos0 = a.dot(r.n0) / dist;
			double ecos = 2 * ea / dist * (r.n0.norm() + tol) + tol + 1e-6;
			if (cos0 + ecos < cosLimit)
				return OUT;
			in = in && cos0 - ecos >= cosLimit;

			Eigen::Vector3d d = R0.transpose() * a;
			double x = d.x(), y = d.y(), z = d.z();
			if (z + ed < 0)
				return OUT;
			double zmin = z - ed;
			if (zmin <= 0)
				return FRAGILE;
			double bx = ed * (z + std::abs(x)) / (z * zmin), by = ed * (z + std::abs(y)) / (z * zmin), b1 = ed / (z * zmin);
			double u = (fx * x + skew * y) / z + cx, v = fy * y / z + cy, uR = u - fxb / z;
			double du = std::abs(fx) * bx + std::abs(skew) * by + 1e-6 * (1 + std::abs(u));
			double dv = std::abs(fy) * by + 1e-6 * (1 + std::abs(v));
			double duR = du + std::abs(fxb) * b1 + 1e-6 * (1 + std::abs(uR));
			if (u + du < pose.mnMinX || u - du > pose.mnMaxX || v + dv < pose.mnMinY || v - dv > pose.mnMaxY
				|| uR + duR < pose.mnMinX || uR - duR > pose.mnMaxX)
				return OUT;
			in = in && u - du >= pose.mnMinX && u + du <= pose.mnMaxX && v - dv >= pose.mnMinY && v + dv <= pose.mnMaxY
				&& uR - duR >= pose.mnMinX && uR + duR <= pose.mnMaxX;
			if (!in)
				return FRAGILE;

			double lo = dist - ea - eD, hi = dist + ea + eD;
			if (lo <= 0 || maxD - tol <= 0)
				return FRAGILE;
			double lsf = pose.mfLogScaleFactor;
			auto level = [&](double q)
			{
				int n = std::ceil(q);
				return n < 0 ? 0 : n >= pose.mnScaleLevels ? pose.mnScaleLevels - 1 : n;
			};
			if (r.refD0 - tol <= 0)
				return FRAGILE;
			if (level(std::log((r.refD0 - tol) / hi) / lsf - 1e-6) != level(std::log((r.refD0 + tol) / lo) / lsf + 1e-6))
				return FRAGILE;
			return IN;
		}
	};

	/**
	 * Flat mirror of which measurements are associated in frames [minpose, maxpose); frames outside
	 * the window are answered from the bimap. Callers must call set() whenever they change DA_bimap_.
	 */
	struct MeasurementAssociationFlags
	{
		int k0 = 0;
		std::vector<std::vector<char>> flags;

		void build(const VectorGLMBComponent6D &c, int minpose, int maxpose)
		{
			k0 = minpose;
			flags.resize(std::max(maxpose - minpose, 0));
			for (int k = minpose; k < maxpose; k++)
			{
				auto &f = flags[k - k0];
				f.assign(c.poses_[k].Z_.size(), 0);
				for (auto &assoc : c.DA_bimap_[k].left)
					f[assoc.first] = 1;
			}
		}
		bool associated(const VectorGLMBComponent6D &c, int k, int nz) const
		{
			if (k < k0 || k - k0 >= flags.size())
				return c.DA_bimap_[k].left.count(nz) > 0;
			return flags[k - k0][nz];
		}
		void set(int k, int nz, bool value)
		{
			if (k >= k0 && k - k0 < flags.size())
				flags[k - k0][nz] = value;
		}
	};

	inline double VectorGLMBSLAM6D::reverseSampleDA(VectorGLMBComponent6D &c){
		//checkNumDet(c);
		int threadnum = 0;
#ifdef _OPENMP
		threadnum = omp_get_thread_num();
#endif

		//c.tomerge_.clear();
		//AssociationProbabilities probs;
		double expectedWeightChange = 0;
		std::vector<double> P;
		AssociationProbabilities probs;
		MeasurementAssociationFlags measAssoc;
		measAssoc.build(c, minpose_, maxpose_);
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





					probs.i.clear();
					probs.l.clear();
					probs.i.push_back(-5);
					probs.l.push_back(config.logKappa_);


					if (lm.numDetections_==0 ){
						if (lm.birthTime_ < c.poses_[k].stamp && lm.birthTime_ > c.poses_[k].stamp-1.0){
							probs.l[0] += -config.logExistenceOdds - lmLogMissExcept(c.poses_[k], lm);
						}else{
							continue;
						}
						int kbirth = lm.birthPose->id;
						if (measAssoc.associated(c, kbirth, lm.birthMatch)){
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
						probs.l[0] += -config.logExistenceOdds - lmLogMissExcept(c.poses_[k], lm);
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
							if (!measAssoc.associated(c, k, nz))
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
							if (result.second)
								measAssoc.set(k, probs.i[sample], true);
							assert(lm.is_detected.size() == lm.numDetections_);
						}
						else
						{
							bool result = c.DA_bimap_[k].right.replace_data(it, probs.i[sample]);
							assert(result);
							if (result)
							{
								measAssoc.set(k, prevDA, false);
								measAssoc.set(k, probs.i[sample], true);
							}

						}
						if (lm.numDetections_>0 ){
							int kbirth = lm.birthPose->id;
							
							auto birth_it = c.DA_bimap_[kbirth].left.find(lm.birthMatch);
							if (birth_it == c.DA_bimap_[kbirth].left.end()){
								auto birth_it2 = c.DA_bimap_[kbirth].right.find(lmidx); 

								if (birth_it2 == c.DA_bimap_[kbirth].right.end()){
									if(!multiOctave() && lm.numDetections_== lm.numFoV_){
										std::cout << termcolor::red << "adding imposssible det_:\n" << termcolor::reset;
										printDA(c);
										assert(0);
									}
									auto result = c.DA_bimap_[kbirth].insert({lm.birthMatch, lmidx });
									assert(result.second);
									if (result.second)
										measAssoc.set(kbirth, lm.birthMatch, true);
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
						measAssoc.set(k, prevDA, false);
						lm.numDetections_--;
						lm.eraseDetection(k);
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
		// lmAssocStamp[slot] == stampBase+k+1  <=>  in frame k the landmark holds a measurement in that slot;
		// one slot per landmark, or one per landmark and octave with multiOctave
		const int numSlots = multiOctave() ? c.poses_[0].mnScaleLevels : 1;
		auto slot = [&](int lmIndex, int k, int nz) { return lmIndex * numSlots + (numSlots > 1 ? measOctave(c.poses_[k], nz) : 0); };
		int stampBase;
		std::vector<int> &lmAssocStamp = landmarkMarks(c.landmarks_.size() * numSlots, maxpose_, stampBase);
		const int lmId0 = c.landmarks_[0].id;
		MeasurementAssociationFlags measAssoc;
		measAssoc.build(c, minpose_, maxpose_);
		for (int k = minpose_; k < maxpose_; k++)
		{
			for (auto &assoc : c.DA_bimap_[k].left)
				lmAssocStamp[slot(assoc.second - lmId0, k, assoc.first)] = stampBase + k + 1;

			for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
			{
				probs.i.clear();
				probs.l.clear();
				double maxprob = -std::numeric_limits<double>::infinity();
				int maxprobi = 0;
				auto it = measAssoc.associated(c, k, nz) ? c.DA_bimap_[k].left.find(nz) : c.DA_bimap_[k].left.end();
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
								likelihood += config.logExistenceOdds + lmLogMiss(c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id]);
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
							// unborn landmark outside the birth window: reject before the bimap lookups
							if (lm.numDetections_ == 0 && !(lm.birthTime_ < c.poses_[k].stamp && lm.birthTime_ > c.poses_[k].stamp - 1.0))
								continue;
							if (lmAssocStamp[slot(c.DAProbs_[k][nz].i[a] - lmId0, k, nz)] != stampBase + k + 1)
							{ // landmark is not already associated to another measurement

								if (c.landmarks_[c.DAProbs_[k][nz].i[a] - c.landmarks_[0].id].numDetections_ == 0)
								{
									int kbirth  = lm.birthPose->id;
									// multiOctave: the birth measurement, added below, needs its octave slot
									bool birthSlotTaken = numSlots > 1 && k == kbirth && nz != lm.birthMatch
										&& measOctave(c.poses_[k], nz) == measOctave(c.poses_[k], lm.birthMatch);
									if (!birthSlotTaken && !measAssoc.associated(c, kbirth, lm.birthMatch)){ // if lm spawn measurement is associated then not sample
										if (lm.birthTime_ < c.poses_[k].stamp
										&& lm.birthTime_ > c.poses_[k].stamp-1.0){
											likelihood +=
												config.logExistenceOdds + lmLogMiss(lm);
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
							if (result.second)
							{
								lmAssocStamp[slot(probs.i[sample] - lmId0, k, nz)] = stampBase + k + 1;
								measAssoc.set(k, nz, true);
							}
						}
						else
						{
							auto &prev_lm = c.landmarks_[selectedDA - c.landmarks_[0].id];

							prev_lm.numDetections_--;
							prev_lm.eraseDetection(k);
							assert(prev_lm.numDetections_ >= 0);
							auto result = c.DA_bimap_[k].left.replace_data(it, probs.i[sample]);
							assert(result);
							if (result)
							{
								lmAssocStamp[slot(selectedDA - lmId0, k, nz)] = 0;
								lmAssocStamp[slot(probs.i[sample] - lmId0, k, nz)] = stampBase + k + 1;
							}

							// add an log for possible landmark merge
							c.tomerge_.push_back(
								std::make_pair(probs.i[sample], selectedDA));
						}
						if (lm.numDetections_>0 ){
							int kbirth = lm.birthPose->id;
							
							auto birth_it = c.DA_bimap_[kbirth].left.find(lm.birthMatch);
							if (birth_it == c.DA_bimap_[kbirth].left.end()){
								auto birth_it2 = c.DA_bimap_[kbirth].right.find(probs.i[sample]);
								// multiOctave: lm may hold another octave in the birth frame; its birth slot is free (birthSlotTaken)
								if (numSlots > 1 || birth_it2 == c.DA_bimap_[kbirth].right.end()){
									if(!multiOctave() && lm.numDetections_== lm.numFoV_){
										std::cout << termcolor::red << " sampling: adding imposssible det_:\n" << termcolor::reset;
										printDA(c);
										assert(0);
									}
									auto result = c.DA_bimap_[kbirth].insert({c.landmarks_[probs.i[sample] - c.landmarks_[0].id].birthMatch, probs.i[sample] });
									assert(result.second);
									if (result.second && kbirth == k)
										lmAssocStamp[slot(probs.i[sample] - lmId0, k, lm.birthMatch)] = stampBase + k + 1;
									if (result.second)
										measAssoc.set(kbirth, c.landmarks_[probs.i[sample] - c.landmarks_[0].id].birthMatch, true);
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
						lmAssocStamp[slot(selectedDA - lmId0, k, nz)] = 0;
						measAssoc.set(k, nz, false);
						c.landmarks_[selectedDA - c.landmarks_[0].id].numDetections_--;
						c.landmarks_[selectedDA - c.landmarks_[0].id].eraseDetection(k);
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
		c.isam = std::make_shared<gtsam::ISAM2>(config.isam2_parameters);
		c.landmark_edge_indices.clear();
		c.removed_edges.clear();
		c.odometry_indices.clear();
		c.imu_indices.clear();
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
		addImuPriors(c);

		

		c.isam_result = isamUpdate(c);
		c.current_estimate = isamEstimate(c);
		loadEstimate(c);


	}
	inline void VectorGLMBSLAM6D::updateMetaStates(VectorGLMBComponent6D &c)
	{
		const std::bitset<256> lowWord(~0ULL);
		std::vector<int> avg_desc;
		std::vector<ORBDescriptor*> descriptors;
		std::vector<int> distances_bef, distances_after;

		for (auto &map_point : c.landmarks_)
			{

				if (map_point.numDetections_ >0){
					
					avg_desc.assign(256, 0);
					map_point.normalVector.setZero();
					double avg_depth=0;
					double max_depth=0;
					double min_depth= std::numeric_limits<double>::infinity();
					int max_dist_bef = 0 ;

					descriptors.clear();
					distances_bef.clear();
					distances_after.clear();


					// each associated measurement once
					for (auto kIt = map_point.is_detected.begin(); kIt != map_point.is_detected.end(); kIt = map_point.is_detected.upper_bound(*kIt))
					for (auto r = c.DA_bimap_[*kIt].right.equal_range(map_point.id); r.first != r.second; ++r.first){
						int k = *kIt;
						int nz = r.first->second;
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
						for (int w = 0; w < 4; w++)
						{
							unsigned long long left_bits = ((desc_left.desc >> (64 * w)) & lowWord).to_ullong();
							unsigned long long right_bits = ((desc_right.desc >> (64 * w)) & lowWord).to_ullong();
							for (int b = 0; b < 64; b++)
								avg_desc[64 * w + b] += ((left_bits >> b) & 1) + ((right_bits >> b) & 1);
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
					map_point.mfScaleRefDistance = config.predictScaleNoMargin_ ? avg_depth : map_point.mfMaxDistance;
					//assert(map_point.mfMaxDistance<10.0);
					map_point.mfMinDistance = std::min(map_point.mfMaxDistance / c.poses_[0].mvScaleFactors[ c.poses_[0].mnScaleLevels - 1] * 0.4 , 0.6*min_depth);
					if (config.maxDistanceFactor_ > 0)
						map_point.mfMaxDistance = std::max(config.maxDistanceFactor_ * avg_depth, 1.05 * max_depth);

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
					if (!config.scaleDescriptors_ && max_dist_bef>200 && max_dist >200){
						std::cout << termcolor::red <<  " dist is impossible\n"  << termcolor::reset; 
						assert(0);
					}

					if (max_dist <=200){
						map_point.descriptor = newDesc;
					}
					if (config.scaleDescriptors_)
						rebuildScaleDesc(c, map_point);
					map_point.normalVector.normalize();



				}else{
					map_point.normalVector =  (map_point.position-map_point.birthPose->pose.translation()).normalized();
					
					map_point.descriptor = map_point.birthPose->descriptors_left[map_point.birthPose->matches_left_to_right[map_point.birthMatch].queryIdx];
					if (config.scaleDescriptors_)
						resetScaleDesc(map_point);

				}

			}

		return;
	}

	/** landmarks that can enter any FoV, in index order (stops where the scan used to break) */
	inline std::vector<int> fovCandidates(const VectorGLMBComponent6D &c, int minpose, int maxpose)
	{
		std::vector<int> candidates;
		for (int lm = 0; lm < c.landmarks_.size(); lm++)
		{
			if(c.landmarks_[lm].birthTime_ > c.poses_[maxpose-1].stamp){
				break;
			}
			if (c.landmarks_[lm].numDetections_ > 0 || (c.landmarks_[lm].birthTime_ <= c.poses_[maxpose-1].stamp && c.landmarks_[lm].birthTime_ >= c.poses_[minpose].stamp))
				candidates.push_back(lm);
		}
		return candidates;
	}

	inline void VectorGLMBSLAM6D::updateFoV(VectorGLMBComponent6D &c)
	{
		if (config.fovCacheTol_ <= 0)
		{
			updateFoVFull(c);
			return;
		}
		updateFoVCached(c);
		if (!config.fovCacheVerify_)
			return;
		std::vector<std::vector<int>> fov(maxpose_), scales(maxpose_);
		for (int k = 0; k < maxpose_; k++)
			fov[k] = c.poses_[k].fov_, scales[k] = c.poses_[k].predicted_scales;
		std::vector<int> numFoV(c.landmarks_.size());
		for (int lm = 0; lm < c.landmarks_.size(); lm++)
			numFoV[lm] = c.landmarks_[lm].numFoV_;
		updateFoVFull(c);
		for (int k = 0; k < maxpose_; k++)
		{
			if (fov[k] != c.poses_[k].fov_ || scales[k] != c.poses_[k].predicted_scales)
			{
				std::cerr << "fovCache mismatch at pose " << k << ": " << fov[k].size() << " vs " << c.poses_[k].fov_.size() << " landmarks\n";
				std::abort();
			}
		}
		for (int lm = 0; lm < c.landmarks_.size(); lm++)
		{
			if (numFoV[lm] != c.landmarks_[lm].numFoV_)
			{
				std::cerr << "fovCache numFoV mismatch at landmark " << lm << "\n";
				std::abort();
			}
		}
	}

	inline int VectorGLMBSLAM6D::associatedScale(VectorGLMBComponent6D &c, int k, int lm)
	{
		if (!config.predictScale_)
			return 0;
		const float dist = c.poses_[k].pose.transformTo(c.landmarks_[lm].position).norm();
		return c.landmarks_[lm].predictScale(dist, &c.poses_[k]);
	}

	inline void VectorGLMBSLAM6D::updateFoVFull(VectorGLMBComponent6D &c)
	{
		for (auto &map_point : c.landmarks_)
			map_point.numFoV_ = 0, map_point.sumLogMiss_ = 0;

		std::vector<int> candidates = fovCandidates(c, minpose_, maxpose_);
		// lmAssocStamp[lm index] == stampBase+k+1  <=>  DA_bimap_[k].right contains the landmark
		int stampBase;
		std::vector<int> &lmAssocStamp = landmarkMarks(c.landmarks_.size(), maxpose_, stampBase);
		const int lmId0 = c.landmarks_.empty() ? 0 : c.landmarks_[0].id;
		FrustumPrefilter prefilter(c, candidates);

		for (int k = 0; k < maxpose_; k++)
		{

			c.poses_[k].fov_.clear();
			c.poses_[k].predicted_scales.clear();
			c.poses_[k].fov_pd.clear();


			if ( c.poses_[k].Z_.size() > 0 && c.poses_[k].isKeypose)
			{ // if no measurements we set FoV to empty ,
				for (auto &assoc : c.DA_bimap_[k].left)
					lmAssocStamp[assoc.second - lmId0] = stampBase + k + 1;

				prefilter.setPose(c.poses_[k]);
				for (size_t ci = 0; ci < candidates.size(); ci++)
				{
					int lm = candidates[ci];
					bool associated = lmAssocStamp[lm] == stampBase + k + 1;
					bool isInFov = false;
					{
						double predScale=0;
						if (associated || (prefilter.mayBeVisible(ci) && c.poses_[k].isInFrustum(&c.landmarks_[lm],
													config.viewingCosLimit_, c.camera, &predScale)))
						{

							c.poses_[k].fov_.push_back(c.landmarks_[lm].id);
							c.poses_[k].predicted_scales.push_back(associated ? associatedScale(c, k, lm) : predScale);
							c.landmarks_[lm].numFoV_++;
							addToFoV(c.poses_[k], c.landmarks_[lm]);
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

	/**
	 * Same result as updateFoVFull. A keypose row is rebuilt only when the pose moved more than
	 * fovCacheTol from the row's reference; otherwise only landmarks that are new, moved past the
	 * tolerance or no longer candidates are re-tested, plus the row's fragile entries.
	 */
	inline void VectorGLMBSLAM6D::updateFoVCached(VectorGLMBComponent6D &c)
	{
		for (auto &map_point : c.landmarks_)
			map_point.numFoV_ = 0, map_point.sumLogMiss_ = 0;

		auto &fc = c.fovCache_;
		const double tol = config.fovCacheTol_;
		const float cosLimit = config.viewingCosLimit_;
		const int nlm = c.landmarks_.size();
		const int lmId0 = c.landmarks_.empty() ? 0 : c.landmarks_[0].id;
		std::vector<int> candidates = fovCandidates(c, minpose_, maxpose_);
		fc.lms.resize(nlm);
		fc.candMark.resize(nlm, 0);
		fc.changedMark.resize(nlm, 0);
		if (fc.rows.size() < c.poses_.size())
			fc.rows.resize(c.poses_.size());
		const int s = ++fc.stamp;
		for (int lm : candidates)
			fc.candMark[lm] = s;

		std::vector<int> refreshed;
		for (int lm : fc.tracked)
		{
			if (fc.candMark[lm] != s)
			{
				fc.lms[lm].valid = false;
				fc.changedMark[lm] = s;
			}
		}
		for (int lm : candidates)
		{
			auto &ref = fc.lms[lm];
			auto &mp = c.landmarks_[lm];
			if (ref.valid && (mp.position - ref.p0).norm() <= tol && (mp.normalVector - ref.n0).norm() <= tol &&
				std::abs(mp.mfMinDistance - ref.minD0) <= tol && std::abs(mp.mfMaxDistance - ref.maxD0) <= tol &&
				std::abs(mp.mfScaleRefDistance - ref.refD0) <= tol)
				continue;
			ref.valid = true;
			ref.p0 = mp.position;
			ref.n0 = mp.normalVector;
			ref.minD0 = mp.mfMinDistance;
			ref.maxD0 = mp.mfMaxDistance;
			ref.refD0 = mp.mfScaleRefDistance;
			fc.changedMark[lm] = s;
			refreshed.push_back(lm);
		}
		fc.tracked = candidates;
		fc.lmsRefreshed += refreshed.size();

		FovClassifier classify(c, tol, cosLimit);
		auto test = [&](int k, const FovCache::Row &row, int lm, std::vector<FovCache::Entry> &out)
		{
			auto cls = classify(row.R0, row.t0, c.poses_[k], fc.lms[lm]);
			if (cls == FovClassifier::OUT)
				return;
			double predScale = 0;
			bool vis = c.poses_[k].isInFrustum(&c.landmarks_[lm], cosLimit, c.camera, &predScale);
			bool miss = cls == FovClassifier::IN && !vis;
			fc.pairTests++;
			fc.boundMisses += miss;
			out.push_back({lm, (int)predScale, vis, cls == FovClassifier::FRAGILE || miss});
		};

		for (int k = maxpose_; k <= fc.maxValidRow && k < (int)fc.rows.size(); k++)
		{
			fc.rows[k].valid = false;
			fc.rows[k].e.clear();
		}
		thread_local std::vector<FovCache::Entry> fresh, merged;
		std::vector<int> assoc;
		int maxValid = -1;
		for (int k = 0; k < maxpose_; k++)
		{
			auto &pose = c.poses_[k];
			pose.fov_.clear();
			pose.predicted_scales.clear();
			pose.fov_pd.clear();
			auto &row = fc.rows[k];
			if (!(pose.Z_.size() > 0 && pose.isKeypose))
			{
				row.valid = false;
				row.e.clear();
				continue;
			}
			fc.rowsSeen++;
			Eigen::Matrix3d R = pose.pose.rotation().matrix();
			Eigen::Vector3d t = pose.pose.translation();
			if (config.fovFreezeOld_ && k < minpose_ && row.valid)
			{
				// stale geometry; candidates and associations are still applied below
				row.frozen = true;
				fc.rowsFrozen++;
			}
			else if (!row.valid || row.frozen || (t - row.t0).norm() > tol || (R - row.R0).norm() > tol)
			{
				fc.rowsRebuilt++;
				row.valid = true;
				row.frozen = false;
				row.R0 = R;
				row.t0 = t;
				row.e.clear();
				for (int lm : candidates)
					test(k, row, lm, row.e);
			}
			else
			{
				fresh.clear();
				for (int lm : refreshed)
					test(k, row, lm, fresh);
				merged.clear();
				size_t j = 0;
				for (auto &e : row.e)
				{
					if (fc.changedMark[e.lm] == s)
						continue;
					while (j < fresh.size() && fresh[j].lm < e.lm)
						merged.push_back(fresh[j++]);
					merged.push_back(e);
					if (e.fragile)
					{
						double predScale = 0;
						merged.back().visible = pose.isInFrustum(&c.landmarks_[e.lm], cosLimit, c.camera, &predScale);
						merged.back().scale = predScale;
						fc.fragileTests++;
					}
				}
				while (j < fresh.size())
					merged.push_back(fresh[j++]);
				row.e.swap(merged);
			}
			maxValid = k;

			// associated landmarks are always in the FoV, as in updateFoVFull
			assoc.clear();
			for (auto &a : c.DA_bimap_[k].left)
			{
				int lm = a.second - lmId0;
				if (fc.candMark[lm] == s)
					assoc.push_back(lm);
			}
			std::sort(assoc.begin(), assoc.end());
			assoc.erase(std::unique(assoc.begin(), assoc.end()), assoc.end());
			auto emit = [&](int lm, int scale)
			{
				pose.fov_.push_back(c.landmarks_[lm].id);
				pose.predicted_scales.push_back(scale);
				c.landmarks_[lm].numFoV_++;
				addToFoV(pose, c.landmarks_[lm]);
			};
			size_t j = 0;
			for (auto &e : row.e)
			{
				while (j < assoc.size() && assoc[j] < e.lm)
				{
					emit(assoc[j], associatedScale(c, k, assoc[j]));
					j++;
				}
				if (j < assoc.size() && assoc[j] == e.lm)
				{
					emit(assoc[j], associatedScale(c, k, assoc[j]));
					j++;
				}
				else if (e.visible && fc.candMark[e.lm] == s)
					emit(e.lm, e.scale);
			}
			for (; j < assoc.size(); j++)
				emit(assoc[j], associatedScale(c, k, assoc[j]));
		}
		fc.maxValidRow = maxValid;

		if (++fc.calls % 2000 == 0)
		{
#pragma omp critical(fovstats)
			std::cout << "fovCache calls " << fc.calls << " rowsRebuilt/frozen/seen " << fc.rowsRebuilt << "/" << fc.rowsFrozen << "/" << fc.rowsSeen
					  << " lmsRefreshed " << fc.lmsRefreshed << " pairTests " << fc.pairTests << " fragileTests " << fc.fragileTests
					  << " boundMisses " << fc.boundMisses << "\n";
		}
	}
	

	inline void VectorGLMBSLAM6D::updateDAProbs(VectorGLMBComponent6D &c,
												int minpose, int maxpose)
	{
		const double stereoGate = 50;
		const double logPDOdds = std::log(config.PD_) - std::log(1 - config.PD_);
		const double logGaussNorm = -0.5 * 3 * std::log(2 * M_PI); // 3 is dimension of measurement
		// dumpDACandidates: every N iterations write each candidate of component 0 as iteration,k,nz,lm,numDet,errPx,scaleDiff,descDist,logp
		const bool dumpCandidates = config.dumpDACandidates_ > 0 && iteration_ % config.dumpDACandidates_ == 0 && &c == &components_[0];
		std::ofstream candidateDump;
		if (dumpCandidates)
			candidateDump.open(config.resultFolder + "/dacand.csv", std::ios::app);


		if (visHMM())
			visPass(c, minpose);

		// swap with empty vectors to release the buffers; clear() keeps their capacity for every past pose
		for (int k =0; k< minpose;k++){
			std::vector<AssociationProbabilities>().swap(c.reverseDAProbs_[k]);
			std::vector<AssociationProbabilities>().swap(c.DAProbs_[k]);
		}

		for (int k = minpose; k < maxpose; k++)
		{

			for (auto &probs : c.reverseDAProbs_[k])
			{
				probs.i.clear();
				probs.l.clear();
			}
			c.reverseDAProbs_[k].resize(c.poses_[k].fov_.size());
			c.DAProbs_[k].resize(c.poses_[k].Z_.size());

			for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
			{
				c.DAProbs_[k][nz].i.clear();
				c.DAProbs_[k][nz].l.clear();
				c.DAProbs_[k][nz].i.push_back(-5);
				c.DAProbs_[k][nz].l.push_back(config.logKappa_);
			}

			// Bucket measurements by (uL, v). A pair passing the stereo gate has |duL| <= gate and
			// |dv| <= gate, so only the cells around the prediction can contain it.
			const double cellSize = stereoGate / 2;
			double umin = std::numeric_limits<double>::infinity(), vmin = umin;
			double umax = -umin, vmax = -umin;
			for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
			{
				auto &z = c.poses_[k].stereo_points[nz];
				if (!std::isfinite(z.uL()) || !std::isfinite(z.v()))
					continue;
				umin = std::min(umin, z.uL());
				umax = std::max(umax, z.uL());
				vmin = std::min(vmin, z.v());
				vmax = std::max(vmax, z.v());
			}
			int ncols = umax >= umin ? int((umax - umin) / cellSize) + 1 : 0;
			int nrows = vmax >= vmin ? int((vmax - vmin) / cellSize) + 1 : 0;
			std::vector<std::vector<int>> cells(ncols * nrows);
			std::vector<int> octave(c.DAProbs_[k].size());
			for (int nz = 0; nz < c.DAProbs_[k].size(); nz++)
			{
				octave[nz] = c.poses_[k].keypoints_left[c.poses_[k].matches_left_to_right[nz].queryIdx].octave;
				auto &z = c.poses_[k].stereo_points[nz];
				if (!std::isfinite(z.uL()) || !std::isfinite(z.v()))
					continue;
				cells[int((z.v() - vmin) / cellSize) * ncols + int((z.uL() - umin) / cellSize)].push_back(nz);
			}

				for(int lmFovIdx =0; lmFovIdx <c.poses_[k].fov_.size();lmFovIdx++){
					int lmidx = c.poses_[k].fov_[lmFovIdx];
					auto &lm = c.landmarks_[lmidx - c.landmarks_[0].id];

					auto point_in_camera_frame = c.poses_[k].pose.transformTo(lm.position);
					if (point_in_camera_frame[2]<0 ){

						continue;
					}

					gtsam::StereoPoint2 predictedZ = c.camera.project(point_in_camera_frame);
					if (!std::isfinite(predictedZ.uL()) || !std::isfinite(predictedZ.v()))
						continue;

					int c0 = std::clamp(std::floor((predictedZ.uL() - stereoGate - umin) / cellSize), 0.0, double(ncols));
					int c1 = std::clamp(std::floor((predictedZ.uL() + stereoGate - umin) / cellSize), -1.0, ncols - 1.0);
					int r0 = std::clamp(std::floor((predictedZ.v() - stereoGate - vmin) / cellSize), 0.0, double(nrows));
					int r1 = std::clamp(std::floor((predictedZ.v() + stereoGate - vmin) / cellSize), -1.0, nrows - 1.0);
					int predictedScale = c.poses_[k].predicted_scales[lmFovIdx];
					for (int r = r0; r <= r1; r++)
					for (int col = c0; col <= c1; col++)
					for (int nz : cells[r * ncols + col])
					{
						auto &z = c.poses_[k].stereo_points[nz];
						if (std::abs(z.uL() - predictedZ.uL()) > stereoGate || std::abs(z.v() - predictedZ.v()) > stereoGate)
							continue;

						double p=0.0;

						int scalediff = abs(predictedScale-octave[nz]);
						if (multiOctave())
						{
							if (scalediff > 3)
								continue;
							p += config.scaleLogWeight_;
						}
						else
						{
							if(scalediff >2){
								continue;
							}
							p += config.scaleLogWeight_ * (1 - scalediff);
						}

						double error_scalar = (c.poses_[k].stereo_points[nz]-predictedZ).vector().norm();

						if (error_scalar > stereoGate){
							continue;
						}

						//int dist = ORBDescriptor::distance(lm.descriptor, c.poses_[k].descriptors_left[c.poses_[k].matches_left_to_right[nz].queryIdx]);
						double desclikelihood = descLogLik(c.poses_[k], nz, lm, scalediff, true);
						if (!std::isfinite(desclikelihood) ){
							continue;
						}
						p += desclikelihood;

						if (visHMM())
							p += visSlotLogOdds(c.poses_[k], lmFovIdx, octave[nz]);
						else if (multiOctave())
						{
							double pd = slotPD(c.poses_[k].fov_pd[lmFovIdx], octave[nz] - predictedScale);
							p += std::log(pd / (1 - pd));
						}
						else if (config.pdAngleTau_ > 0)
						{
							double pd = c.poses_[k].fov_pd[lmFovIdx];
							p += std::log(pd / (1 - pd));
						}
						else
							p += logPDOdds;
						
						{ 
							 p += -0.5 * (error_scalar*error_scalar* config.stereoInfo_(0,0)); //chi2, this assumes information is a*Identity(3,3)
							assert(!isnan(p));
							p += logGaussNorm;
							assert(!isnan(p));





						}
					
					assert(!isnan(p));
					if (dumpCandidates)
						candidateDump << iteration_ << "," << k << "," << nz << "," << lmidx << "," << lm.numDetections_ << ","
									  << error_scalar << "," << scalediff << "," << desclikelihood / ORBDescriptor::logWeightPerBit << "," << p << "\n";
					if (p  > config.logKappa_ - 1000){
						c.DAProbs_[k][nz].l.push_back(p);
						c.DAProbs_[k][nz].i.push_back(lmidx);
						c.reverseDAProbs_[k][lmFovIdx].i.push_back(nz);
						c.reverseDAProbs_[k][lmFovIdx].l.push_back(p);

						// avg_desc_likelihood += desclikelihood;
					}
				}

					// cells are visited out of nz order; restore ascending nz in the reverse list
					auto &rev = c.reverseDAProbs_[k][lmFovIdx];
					for (int x = 1; x < rev.i.size(); x++)
						for (int y = x; y > 0 && rev.i[y - 1] > rev.i[y]; y--)
						{
							std::swap(rev.i[y - 1], rev.i[y]);
							std::swap(rev.l[y - 1], rev.l[y]);
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
		lm.sumLogMiss_ = 0;
		int level = pose.keypoints_left[nl].octave;
		const int nLevels = pose.mnScaleLevels;

		if (!cam_unproject(pose.stereo_points[numMatch], pose.point_camera_frame[numMatch]))
		{
			// std::cerr << "stereofail\n";

			return false;
		}

		lm.mfMaxDistance = pose.point_camera_frame[numMatch].norm() * pose.mvScaleFactors[level] * 1.8;
		lm.mfScaleRefDistance = config.predictScaleNoMargin_ ? lm.mfMaxDistance / 1.8 : lm.mfMaxDistance;
		lm.mfMinDistance = lm.mfMaxDistance / pose.mvScaleFactors[nLevels - 1] * 0.3;
		if (config.maxDistanceFactor_ > 0)
			lm.mfMaxDistance = pose.point_camera_frame[numMatch].norm() * pose.mvScaleFactors[level] * config.maxDistanceFactor_;
		pose.initial_lm_id[numMatch] = newId;
		//pose.Z_[numMatch]->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(lm.pPoint));

		// with only 2 descriptors (left,right) we can pick either one to represent the point
		// allocating new
		lm.descriptor = pose.descriptors_left[nl];
		if (config.scaleDescriptors_)
			resetScaleDesc(lm);
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
			if (isNavKey(it.key))
				continue;
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
			auto result_imu = c.imu_indices.right.erase(index);
			assert(result_lm || result_odo || result_imu);
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
			if (std::dynamic_pointer_cast<gtsam::CombinedImuFactor>(c.new_edges.at(i))){
				c.imu_indices.insert({{(int)k0, (int)c.new_edges.at(i)->keys()[2]}, indices[i]});
			}else if (k1 < c.poses_.size()){
				// second key is a pose , so this is odometry

				c.odometry_indices.insert( {k0 , indices[i]});

			}else{
				// the measurement whose factor this is (several with multiOctave)
				auto range = c.DA_bimap_[k0].right.equal_range(k1);
				auto it = range.first;
				assert(it != range.second);
				for (auto r = range.first; r != range.second; ++r)
					if (c.poses_[k0].Z_[r->second] == c.new_edges.at(i))
					{
						it = r;
						break;
					}

				std::array<int , 3> association = {(int)k0 , it->second , (int)k1 };

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
			
			if ( lm.numDetections_ == 0 || lmGated(c, lm)){


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
			c.poses_[k].fov_pd = initial_component_.poses_[k].fov_pd;
			c.poses_[k].initial_lm_id = initial_component_.poses_[k].initial_lm_id;
			// create graph pose
			gtsam::Pose3 pose_estimate;


			
			if (k > 0)
			{
				// lower-order model for the first frames, which lack the history
				std::vector<double> coeffs;
				if (config.motionModel_ == "cv" && k >= 2)
					coeffs = {1};
				else if (config.motionModel_ == "ca" && k >= 2)
				{
					double e = 1;
					if (config.accelerationDampingTime_ > 0)
						e = std::exp(-(c.poses_[k].stamp - c.poses_[k - 1].stamp) / config.accelerationDampingTime_);
					if (k >= 3)
						coeffs = {1 + e, -e};
					else
						coeffs = {1};
				}
				else if (config.motionModel_ == "cj" && k >= 2)
				{
					if (k >= 4)
						coeffs = {3, -3, 1};
					else if (k == 3)
						coeffs = {2, -1};
					else
						coeffs = {1};
				}
				if (coeffs.empty())
				{
					PoseType odometry_pose; // this is just zero odom , meaning brownian motion
					c.odometries_.push_back(std::make_shared<OdometryEdge>(k-1 , k , odometry_pose , config.odom_noise));
				}
				else
					c.odometries_.push_back(std::make_shared<MotionPriorFactor>(k, coeffs, config.odom_noise));

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
		VectorGLMBComponent6D::FrameBimap empty_bimap;
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

		if (!config.imu_)
			c.new_edges.push_back(c.odometries_[0]);
		c.new_edges.addPrior(0 , c.initial_pose , config.odom_noise );

		c.new_nodes.insert(0,c.poses_[0].pose);
		if (!config.imu_)
			c.new_nodes.insert(1,c.poses_[1].pose);
		addImuPriors(c);

		c.isam_result = isamUpdate(c);
		c.current_estimate = isamEstimate(c);
		loadEstimate(c);


	}

	bool VectorGLMBSLAM6D::cam_unproject(const gtsam::StereoPoint2 &stereo_point,
										 Eigen::Vector3d &trans_xyz)
	{

		double disparity = stereo_point.uL() - stereo_point.uR();
		trans_xyz[2] = config.stereo_baseline_f / disparity;
		trans_xyz[0] = (stereo_point.uL() - config.camera_parameters_[0].cx) * trans_xyz[2] / config.camera_parameters_[0].fx;
		trans_xyz[1] = (stereo_point.v() - config.camera_parameters_[0].cy) * trans_xyz[2] / config.camera_parameters_[0].fy;


		if (!(disparity >= config.stereo_init_min_disparity))
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
		c.isam = std::make_shared<gtsam::ISAM2>(config.isam2_parameters);

	}

}
#endif
