#include <algorithm>
#include <iostream>

#include <amsfilter_model/model.hpp>
#include <amsfilter/amsfilter_lite.hpp>
#include <thread>

//===----------------------------------------------------------------------===//
/// The code snippet below shows how to parameterize a filter for performance
/// optimality using the amsfilter::Model.
int32_t main() {

  // Obtain a model instance. - Note: The calibration tool needs to be executed
  // before.
  amsfilter::Model model;

  // Specify the environment in which the filter probes are executed.
  // For instance, on the CPU using one thread per core:
  const auto thread_cnt = std::thread::hardware_concurrency() / 2;
  const auto cpu_env = amsfilter::model::Env::cpu(thread_cnt);
  // When the probes are executed on a GPU, we also need to specify whether the
  // probe keys are located in host/pageable, in host/pinned, or in device
  // memory.
  const auto device_no = 0u; // first CUDA device == 0
  const auto gpu_env = amsfilter::model::Env::gpu(device_no,
      amsfilter::model::Memory::HOST_PINNED);

  // Obtain the parameters for a (close to) performance-optimal filter.
  // The model needs the following two values to find the optimal parameters:
  //   build size (n):  The number of keys that will be inserted in the filter.
  //   work time (tw):  The execution time in nanoseconds that is saved when an
  //                    element is filtered out.
  const auto n  = 10000000ull;
  const auto tw = 7.0; // [ns]

  const auto cpu_params = model.determine_filter_params(cpu_env, n, tw);
  const auto gpu_params = model.determine_filter_params(gpu_env, n, tw);

  std::cout
      << "Host-side filter:   m=" << cpu_params.get_filter_size()
      << ", config=" << cpu_params.get_filter_config() << std::endl;
  std::cout
      << "Device-side filter: m=" << gpu_params.get_filter_size()
      << ", config=" << gpu_params.get_filter_config() << std::endl;


  // The returned object contains all the necessary information to instantiate
  // the filter.  However, before we do so, we need to check whether a filter
  // should be installed altogether.
  // To make this decision, we need to know the selectivity (sel), which is the
  // probability of a true hit.
  const auto sel = 0.1;
  if (sel >= cpu_params.get_max_selectivity()) {
    // Filtering is not beneficial. - No filter should be used.
  }
  if (sel >= gpu_params.get_max_selectivity()) {
    // The condition needs to be checked for each execution environment.
  }

  // Construct the filter(s).
  amsfilter::AmsFilterLite cpu_filter(cpu_params.get_filter_config(),
      cpu_params.get_filter_size());
  amsfilter::AmsFilterLite gpu_filter(gpu_params.get_filter_config(),
      gpu_params.get_filter_size());

  // Populate the filter(s).
  // ...

  // Obtain a probe instance.
  const auto max_cpu_batch_size = 1ull << 10;
  auto cpu_probe = cpu_filter.batch_probe(max_cpu_batch_size,
      cpu_params.get_tuning_params()); // Notice the tuning parameters.

  const auto max_gpu_batch_size = 1ull << 20;
  auto gpu_probe = gpu_filter.batch_probe_cuda(max_gpu_batch_size,
      gpu_env.get_device());

  // Probe the filters(s).
  // ...

}
//===----------------------------------------------------------------------===//
