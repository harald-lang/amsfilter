#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

#include <amsfilter/amsfilter_lite.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include <boost/tokenizer.hpp>
#include <dtl/env.hpp>

#include "util.hpp"

//===----------------------------------------------------------------------===//
void __attribute__((noinline))
benchmark(
    const amsfilter::Config config,
    const std::size_t m,
    const amsfilter::key_t* to_insert, const std::size_t to_insert_cnt,
    const amsfilter::key_t* to_lookup, const std::size_t to_lookup_cnt,
    const std::size_t batch_size,
    const std::size_t cuda_stream_cnt,
    const uint32_t cuda_device_no) {

  //===--------------------------------------------------------------------===//
  // Construct the filter.
  amsfilter::AmsFilterLite filter(config, m);
  for (std::size_t i = 0; i < to_insert_cnt; ++i) {
    const auto key = to_insert[i];
    filter.insert(key);
  }

  if (dtl::env<int32_t>::get("VALIDATE", 1) != 0) {

    // Validation (scalar code).
    for (std::size_t i = 0; i < to_insert_cnt; ++i) {
      const auto key = to_insert[i];
      if (!filter.contains(key)) {
        std::cerr << "Validation failed (scalar code)." << std::endl;
        std::exit(1);
      }
    }

    // Validation (CUDA code).
    {
      auto probe = filter.batch_probe_cuda(batch_size, cuda_device_no);
      const auto probe_results = probe.get_results();

      // Probe the CUDA filter and validate the results.
      std::size_t cpu_match_cnt = 0;
      std::size_t gpu_match_cnt = 0;
      for (std::size_t i = 0; i < to_lookup_cnt; i += batch_size) {
        const std::size_t key_cnt = std::min(batch_size, (to_lookup_cnt - i));
        probe(&to_lookup[i], key_cnt);
        probe.wait();

        // Validate the results.
        for (std::size_t j = 0; j < key_cnt; ++j) {
          u1 is_cpu_match = filter.contains(to_lookup[i + j]);
          u1 is_gpu_match = probe_results[j];
          if (is_cpu_match != is_gpu_match) {
            std::cerr << "Validation failed (CUDA code)." << std::endl;
            std::exit(1);
          }
          cpu_match_cnt += is_cpu_match;
          gpu_match_cnt += is_gpu_match;
        }
      }
      if (cpu_match_cnt != gpu_match_cnt) {
        std::cerr << "Result mismatch. Expected " << cpu_match_cnt << " matches"
            << " but got " << gpu_match_cnt << std::endl;
        std::exit(1);
      }
    }
  }

  // Performance measurement.
  {
    using cuda_probe_t = amsfilter::cuda::ProbeLite;
    std::vector<cuda_probe_t> probes;

    using cuda_probe_results_t = amsfilter::cuda::ProbeLite::result_type;
    std::vector<cuda_probe_results_t> results;

    // Instantiate the CUDA probe instance(s).
    for (std::size_t i = 0; i < cuda_stream_cnt; ++i) {
      probes.emplace_back(filter.batch_probe_cuda(batch_size, cuda_device_no));
      results.emplace_back(probes.back().get_results());
    }

    auto time_start = std::chrono::high_resolution_clock::now();
    const std::size_t input_cnt = to_lookup_cnt;
    std::size_t dispatched_cntr = 0;
    std::size_t batch_cntr = 0;
    while (dispatched_cntr < input_cnt) {
      const std::size_t key_cnt =
          std::min(batch_size, (input_cnt - dispatched_cntr));

      // Find an idle probe instance.
      $u1 dispatched = false;
      while (!dispatched) {
        for (auto& probe : probes) {
          if (probe.is_done()) {
            probe(&to_lookup[dispatched_cntr], key_cnt);
            dispatched = true;
            break;
          }
        }
        // All probe instances are busy.
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
      dispatched_cntr += key_cnt;
      batch_cntr++;
    }

    // Wait for in-flight probes.
    for (auto& probe_instance : probes) {
      probe_instance.wait();
    }

    auto time_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = time_end - time_start;
    auto probes_per_second =
        static_cast<u64>(to_lookup_cnt / duration.count());

    // Free resources.
    probes.clear();
    results.clear();

    std::cout
        << config.word_cnt_per_block
        << "," << config.sector_cnt
        << "," << config.zone_cnt
        << "," << config.k
        << "," << m
        << "," << batch_size
        << "," << cuda_stream_cnt
        << "," << batch_cntr
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

  std::size_t batch_size_log2 = dtl::env<std::size_t>::get("BATCH_SIZE_LOG2", 20);
  std::size_t batch_size = 1ull << batch_size_log2;
  std::cout << "Batch size: 2^" << batch_size_log2 << std::endl;

  int32_t cuda_device_no = dtl::env<int32_t>::get("DEVICE_NO", 0);
  std::cout << "using CUDA device no. " << cuda_device_no
      << " (" << get_cuda_device_name(cuda_device_no) << ")" << std::endl;

  std::size_t to_insert_cnt_log2 = dtl::env<std::size_t>::get("INSERT_CNT_LOG2", 24);
  std::size_t to_insert_cnt = 1ull << to_insert_cnt_log2;
  std::size_t to_lookup_cnt_log2 = dtl::env<std::size_t>::get("LOOKUP_CNT_LOG2", 28);
  std::size_t to_lookup_cnt = 1ull << to_lookup_cnt_log2;

  bool keys_in_pinned_memory = dtl::env<$i32>::get("KEYS_PINNED", 1) != 0;

  std::vector<amsfilter::key_t> to_insert(to_insert_cnt);
  std::vector<amsfilter::key_t> to_lookup(to_lookup_cnt);

  gen_data(to_insert.begin(), to_insert.end());
  gen_data(to_lookup.begin(), to_lookup.end());

  amsfilter::key_t* to_lookup_ptr = nullptr;

  if (keys_in_pinned_memory) {
    std::cout << "Keys in pinned memory" << std::endl;
    amsfilter::key_t* to_lookup_pinned;
    std::size_t size = to_lookup.size() * sizeof(amsfilter::key_t);
    cudaMallocHost(&to_lookup_pinned, size, cudaHostAllocPortable);
    std::copy(to_lookup.begin(), to_lookup.end(), to_lookup_pinned);
    to_lookup_ptr = to_lookup_pinned;
    to_lookup.clear();
  }
  else {
    std::cout << "Keys in pageable memory" << std::endl;
    to_lookup_ptr = to_lookup.data();
  }

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

  std::cout << "word_cnt,sector_cnt,zone_cnt,k,m,probe_cnt,batch_size,"
      "cuda_stream_cnt,batch_cntr,probes_per_second,word_access_cnt_per_lookup"
      << std::endl;

  auto run_end_to_end_benchmark = [&](
      const std::size_t m,
      const std::size_t cuda_stream_cnt) {
    benchmark(
        config,
        m,
        to_insert.data(), to_insert.size(),
        to_lookup_ptr, to_lookup_cnt,
        batch_size,
        cuda_stream_cnt,
        cuda_device_no
    );
  };

  for (std::size_t m : filter_sizes) {
    run_end_to_end_benchmark(m, 4);
  }
  return 0;
}