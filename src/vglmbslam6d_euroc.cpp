#include "VectorGLMBSLAM6D.hpp"
#include <boost/program_options.hpp>
#include <tbb/global_control.h>
#include <memory>


int main(int argc, char* argv[]){



  std::string cfgFileName;
  boost::program_options::options_description desc("Options");
  desc.add_options()
    ("help,h", "produce this help message")
    ("cfg,c", boost::program_options::value<std::string>(&cfgFileName)->default_value("cfg/vglmbslam6d.yaml"), "configuration yaml file");
  boost::program_options::variables_map vm;
  rclcpp::init(argc, argv);
  auto args = rclcpp::remove_ros_arguments(argc, argv);
  std::vector<const char*> argv_no_ros;
  for (auto &a : args) argv_no_ros.push_back(a.c_str());
  boost::program_options::store( boost::program_options::parse_command_line(argv_no_ros.size(), argv_no_ros.data(), desc), vm);
  boost::program_options::notify(vm);

  if( vm.count("help") ){
    std::cout << desc << "\n";
    return 1;
  }


  rfs::VectorGLMBSLAM6D vglmb;

  vglmb.loadConfig(cfgFileName);
  // the 12 components already run in parallel under OpenMP; iSAM2's own TBB workers mostly spin in sched_yield
  std::unique_ptr<tbb::global_control> tbbLimit;
  if (vglmb.config.tbbMaxThreads_ > 0)
    tbbLimit = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, vglmb.config.tbbMaxThreads_);
  rfs::initializeGaussianGenerators();
  vglmb.loadEuroc();
  // OpenCV's parallel_for (TBB backend) is only needed for loading; its worker pool otherwise keeps spinning in sched_yield and competes with the OpenMP component threads
  cv::setNumThreads(0);
  vglmb.gt_est.loadTUM(vglmb.config.eurocFolder_ + "/mav0/state_groundtruth_estimate0/data.tum",vglmb.config.base_link_to_cam0_se3, vglmb.startingStamp);
  //vglmb.calculateWeight(vglmb.gt_graph);
  //std::cout << "GROUND TRUTH WEIGHT:             " <<vglmb.gt_graph.logweight_ << "\n";
  //std::cout << "weight: " << vglmb.gt_graph.logweight_ << "   chi2:  " <<vglmb.gt_graph.optimizer_->chi2() << "  determinant: " << vglmb.gt_graph.linearSolver_->_determinant<< "\n";

  vglmb.initComponents();



  

#ifdef _PERFTOOLS_CPU
		std::string perfCPU_file =  "vglmb.prof";
		ProfilerStart(perfCPU_file.data());
#endif
#ifdef _PERFTOOLS_HEAP
		std::string perfHEAP_file = "vglmb_heap.prof";
		HeapProfilerStart(perfHEAP_file.data());
#endif
  vglmb.run(vglmb.config.numIterations_);
  vglmb.deleteComponents();
  #ifdef _PERFTOOLS_HEAP
		HeapProfilerStop();
#endif
#ifdef _PERFTOOLS_CPU
		ProfilerStop();
#endif

  // vglmb.components_[0].optimizer_->save(vglmb.config.finalStateFile_.c_str() , 0);
  // vglmb.components_[0].DA_bimap_ = vglmb.best_DA_;
  // vglmb.updateGraph(vglmb.components_[0]);
  // vglmb.components_[0].optimizer_->initializeOptimization();
  // vglmb.components_[0].optimizer_->optimize(50);
  // vglmb.components_[0].optimizer_->save("beststate.g2o" , 0);


  vglmb.waitForGuiClose();
  rclcpp::shutdown();
}
