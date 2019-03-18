#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

#include <amsfilter/cuda/internal/cuda_api_helper.hpp>
#include <amsfilter/cuda/probe.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include <amsfilter/amsfilter.hpp>
#include <amsfilter/bitmap_view.hpp>
#include <boost/tokenizer.hpp>
#include <dtl/env.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "util.hpp"

//===----------------------------------------------------------------------===//
void __attribute__((noinline))
benchmark(
    const amsfilter::Config config,
    const std::size_t m,
    const amsfilter::key_t* to_insert, const std::size_t to_insert_cnt,
    const amsfilter::key_t* to_lookup, const std::size_t to_lookup_cnt,
    const uint32_t cuda_device_no) {

  cudaSetDevice(cuda_device_no);

  //===--------------------------------------------------------------------===//
  // Construct the filter.
  amsfilter::AmsFilter filter(config, m);
  std::vector<amsfilter::word_t> filter_data(filter.size(), 0);
  for (std::size_t i = 0; i < to_insert_cnt; ++i) {
    const auto key = to_insert[i];
    filter.insert(filter_data.data(), key);
  }

  thrust::device_vector<amsfilter::word_t> device_filter_data(
      filter_data.begin(), filter_data.end());

  std::size_t bitmap_word_cnt =
      (to_lookup_cnt + (bitwidth<amsfilter::word_t> - 1)) / bitwidth<amsfilter::word_t>;
  thrust::device_vector<amsfilter::word_t> device_bitmap(bitmap_word_cnt, 0);


  thrust::device_vector<amsfilter::key_t> device_keys(
      to_lookup, to_lookup + to_lookup_cnt);

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream,
      cudaStreamNonBlocking & cudaEventDisableTiming);
  cuda_check_error();

  cudaEvent_t done_event;
  cudaEventCreate(&done_event);
  cuda_check_error();

  if (dtl::env<int32_t>::get("VALIDATE", 1) != 0) {

    amsfilter::cuda::Probe probe(filter);
    probe(
        thrust::raw_pointer_cast(device_filter_data.data()),
        thrust::raw_pointer_cast(device_keys.data()), to_lookup_cnt,
        thrust::raw_pointer_cast(device_bitmap.data()),
        stream);
    cudaEventRecord(done_event, stream);
    cuda_check_error();
    cudaEventSynchronize(done_event);
    cuda_check_error();

    thrust::host_vector<amsfilter::word_t> host_bitmap = device_bitmap;
    amsfilter::bitmap_view<amsfilter::word_t> device_result {
        host_bitmap.data(),
        host_bitmap.data() + host_bitmap.size()
    };

    // Validation.
    for (std::size_t i = 0; i < to_lookup_cnt; ++i) {
      const auto key = to_lookup[i];
      u1 is_host_match = filter.contains(filter_data.data(), key);
      u1 is_device_match = device_result[i];
      if (is_host_match != is_device_match) {
        std::cerr << "Validation failed." << std::endl;
        std::exit(1);
      }
    }
  }

  // Performance measurement.
  {
    amsfilter::cuda::Probe probe(filter);

    auto time_start = std::chrono::high_resolution_clock::now();

    probe(
        thrust::raw_pointer_cast(device_filter_data.data()),
        thrust::raw_pointer_cast(device_keys.data()), to_lookup_cnt,
        thrust::raw_pointer_cast(device_bitmap.data()),
        stream);
    cudaEventRecord(done_event, stream);
    cuda_check_error();
    cudaEventSynchronize(done_event);
    cuda_check_error();

    auto time_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = time_end - time_start;
    auto probes_per_second =
        static_cast<u64>(to_lookup_cnt / duration.count());

    std::cout
        << config.word_cnt_per_block
        << "," << config.sector_cnt
        << "," << config.zone_cnt
        << "," << config.k
        << "," << m
        << "," << probes_per_second
        << "," << amsfilter::internal::get_word_access_cnt(config)
        << std::endl;
  }
};
//===----------------------------------------------------------------------===//
int32_t main() {
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  // The filter parameters.
  std::string bbf_config_str = dtl::env<std::string>::get("CONFIG", "1,1,1,4");
  amsfilter::Config config = parse_filter_config(bbf_config_str);

  std::cout << "Filter parameters: w=" << config.word_cnt_per_block
      << ", s=" << config.sector_cnt << ", z=" << config.zone_cnt << ", k="
      << config.k << std::endl;

  int32_t cuda_device_no = dtl::env<int32_t>::get("DEVICE_NO", 0);
  std::cout << "using CUDA device no. " << cuda_device_no
      << " (" << get_cuda_device_name(cuda_device_no) << ")" << std::endl;

  std::size_t to_insert_cnt_log2 = dtl::env<std::size_t>::get("INSERT_CNT_LOG2", 24);
  std::size_t to_insert_cnt = 1ull << to_insert_cnt_log2;
  std::size_t to_lookup_cnt_log2 = dtl::env<std::size_t>::get("LOOKUP_CNT_LOG2", 28);
  std::size_t to_lookup_cnt = 1ull << to_lookup_cnt_log2;

  std::vector<amsfilter::key_t> to_insert(to_insert_cnt);
  std::vector<amsfilter::key_t> to_lookup(to_lookup_cnt);

  std::string data_gen_config_str = dtl::env<std::string>::get("GEN", "uniform");
  auto data_gen_config = parse_datagen_config(data_gen_config_str);
  gen_data_uniform(to_insert.begin(), to_insert.end());
  gen_data(to_lookup.begin(), to_lookup.end(), data_gen_config);

  amsfilter::key_t* to_lookup_ptr = nullptr;
  to_lookup_ptr = to_lookup.data();

  // The filter size in bits.
  std::vector<std::size_t> filter_sizes {
       1ull * 1024 * 1024 * 8,
       2ull * 1024 * 1024 * 8,
       4ull * 1024 * 1024 * 8,
       8ull * 1024 * 1024 * 8,
      16ull * 1024 * 1024 * 8,
      32ull * 1024 * 1024 * 8,
      64ull * 1024 * 1024 * 8,
     128ull * 1024 * 1024 * 8,
     256ull * 1024 * 1024 * 8,
     512ull * 1024 * 1024 * 8
  };

  std::cout << "word_cnt,sector_cnt,zone_cnt,k,m,"
      "probes_per_second,word_access_cnt_per_lookup"
      << std::endl;

  auto run_benchmark = [&](const std::size_t m) {
    benchmark(
        config,
        m,
        to_insert.data(), to_insert.size(),
        to_lookup_ptr, to_lookup_cnt,
        cuda_device_no
    );
  };

  for (std::size_t m : filter_sizes) {
    run_benchmark(m);
  }
  return 0;
}