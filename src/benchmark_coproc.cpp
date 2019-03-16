#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

#include <amsfilter/amsfilter_lite.hpp>
#include <amsfilter/internal/blocked_bloomfilter_template.hpp>
#include <boost/tokenizer.hpp>
#include <dtl/env.hpp>
#include <dtl/thread.hpp>

#include "util.hpp"

//===----------------------------------------------------------------------===//
// Read the CPU affinity of this process.
static const auto cpu_mask = dtl::this_thread::get_cpu_affinity();
//===----------------------------------------------------------------------===//
void __attribute__((noinline))
benchmark(
    const amsfilter::Config config,
    const std::size_t m,
    const amsfilter::key_t* to_insert, const std::size_t to_insert_cnt,
    const amsfilter::key_t* to_lookup, const std::size_t to_lookup_cnt,
    const std::size_t host_batch_size,
    const std::size_t device_batch_size,
    const std::vector<$u32> cuda_devices,
    const std::size_t thread_cnt,
    const std::size_t coproc_thread_cnt) {

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
  }

  //===--------------------------------------------------------------------===//
  // Performance measurement.
  {
    using cuda_probe_t = amsfilter::cuda::ProbeLite;
    std::vector<cuda_probe_t> probes;

    using cuda_probe_results_t = amsfilter::cuda::ProbeLite::result_type;
    std::vector<cuda_probe_results_t> results;

    // Instantiate the CUDA probe instance(s).
    for (auto cuda_device : cuda_devices) {
      for (std::size_t i = 0; i < 4; ++i) { // four streams per device
        probes.emplace_back(filter.batch_probe_cuda(device_batch_size, cuda_device));
        results.emplace_back(probes.back().get_results());
      }
    }

    // Assign the CUDA probe instances to the host threads;
    std::vector<$u32> probe_to_thread_id_map;
    {
      if (coproc_thread_cnt > 0) {
        for (std::size_t i = 0; i < probes.size(); ++i) {
          probe_to_thread_id_map.push_back(u32(i % coproc_thread_cnt));
        }
      }
      else {
        for (std::size_t i = 0; i < probes.size(); ++i) {
          probe_to_thread_id_map.push_back(u32(-1));
        }
      }
    }

    const std::size_t input_cnt = to_lookup_cnt;
    std::atomic<std::size_t> cntr { 0 };

    std::atomic<std::size_t> total_processed_key_cntr_host { 0 };
    std::atomic<std::size_t> total_processed_key_cntr_device { 0 };

    //===------------------------------------------------------------------===//
    auto thread_fn = [&](u32 thread_id) {
      std::size_t processed_key_cntr_host = 0;
      std::size_t processed_key_cntr_device = 0;

      // Determine the CUDA probe instance for the current thread.
      std::vector<cuda_probe_t*> device_probe_ptrs;
      for (std::size_t i = 0; i < probes.size(); ++i) {
        if (probe_to_thread_id_map[i] == thread_id) {
          device_probe_ptrs.push_back(&probes[i]);
        }
      }

      // The current batch of keys.
      std::size_t input_idx_begin = 0;
      std::size_t input_idx_end = 0;
      // Get a batch of keys.
      auto get_batch = [&](std::size_t size) {
        input_idx_begin = cntr.fetch_add(size);
        input_idx_end = std::min(input_idx_begin + size, input_cnt);
      };

      // Dispatch work to device(s).
      auto device_dispatch = [&]() {
        bool dispatched = false;
        for (auto probe_ptr : device_probe_ptrs) {
          if (probe_ptr->is_done()) {
            get_batch(device_batch_size);
            const std::size_t key_cnt = input_idx_end - input_idx_begin;
            if (input_idx_begin >= input_cnt || key_cnt == 0) return true;
            probe_ptr->operator()(&to_lookup[input_idx_begin], key_cnt);
            processed_key_cntr_device += key_cnt;
            dispatched = true;
          }
        }
        return dispatched;
      };

      // Each thread obtains a host-side probe instance.
      auto host_probe = filter.batch_probe(host_batch_size);
//      auto host_results = host_probe.get_results();

      // The main loop.
      while (true) {
        if (input_idx_begin >= input_cnt) break;

        u1 dispatched_to_device = device_dispatch();

        if (!dispatched_to_device) {
          get_batch(host_batch_size);
          const std::size_t key_cnt = input_idx_end - input_idx_begin;
          if (input_idx_begin >= input_cnt || key_cnt == 0) break;
          host_probe(&to_lookup[input_idx_begin], input_idx_end - input_idx_begin);
          processed_key_cntr_host += key_cnt;
        }
      }

      // Wait for in-flight device probes.
      for (auto probe_ptr : device_probe_ptrs) {
        probe_ptr->wait();
      }

      total_processed_key_cntr_host += processed_key_cntr_host;
      total_processed_key_cntr_device += processed_key_cntr_device;
    };
    //===------------------------------------------------------------------===//

    auto time_start = std::chrono::high_resolution_clock::now();
    dtl::run_in_parallel(thread_fn, cpu_mask, thread_cnt);
    auto time_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = time_end - time_start;
    auto probes_per_second_host =
        static_cast<u64>(total_processed_key_cntr_host / duration.count());
    auto probes_per_second_device =
        static_cast<u64>(total_processed_key_cntr_device / duration.count());
    auto probes_per_second_total =
        static_cast<u64>(to_lookup_cnt / duration.count());

    std::cout
        << config.word_cnt_per_block
        << "," << config.sector_cnt
        << "," << config.zone_cnt
        << "," << config.k
        << "," << m
        << "," << to_lookup_cnt
        << "," << host_batch_size
        << "," << device_batch_size
        << "," << thread_cnt
        << "," << coproc_thread_cnt
        << "," << total_processed_key_cntr_host
        << "," << total_processed_key_cntr_device
        << "," << probes_per_second_host
        << "," << probes_per_second_device
        << "," << probes_per_second_total
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

  std::size_t host_batch_size_log2 =
      dtl::env<std::size_t>::get("HOST_BATCH_SIZE_LOG2", 10);
  std::size_t host_batch_size = 1ull << host_batch_size_log2;

  std::size_t device_batch_size_log2 =
      dtl::env<std::size_t>::get("DEVICE_BATCH_SIZE_LOG2", 20);
  std::size_t device_batch_size = 1ull << device_batch_size_log2;
  std::cout << "batch size: 2^" << host_batch_size_log2
      << " (host), 2^" << device_batch_size_log2 << " (device)"
      << std::endl;

  std::string devices_str = dtl::env<std::string>::get("DEVICE_NO", "0");
  typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
  boost::char_separator<char> sep{","};
  tokenizer devices_tok{devices_str, sep};
  std::vector<$u32> devices;
  std::cout << "Devices:";
  for (auto& device_no : devices_tok) {
    devices.push_back(u32(std::stoul(device_no)));
    std::cout << " " << devices.back();
    std::cout << " (" << get_cuda_device_name(devices.back()) << ") ";
  }
  std::cout << std::endl;

  int32_t thread_cnt = dtl::env<int32_t>::get("THREAD_CNT", cpu_mask.count());
  std::cout << "using " << thread_cnt << " thread(s) " << std::endl;

  int32_t coproc_thread_cnt =
      devices.empty()
          ? 0
          : std::min(
              thread_cnt,
              dtl::env<int32_t>::get("CO_THREAD_CNT", cpu_mask.count()));
  std::cout << "using " << coproc_thread_cnt << " co-processing thread(s) " << std::endl;

  std::size_t to_insert_cnt = 1ull << dtl::env<std::size_t>::get("INSERT_CNT_LOG2", 24);
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

  std::cout << "word_cnt,sector_cnt,zone_cnt,k,m,probe_cnt,batch_size_host,"
      "batch_size_device,thread_cnt,co_thread_cnt,keys_processed_host,"
      "keys_processed_device,probes_per_second_host,probes_per_second_device,"
      "probes_per_second_total,word_access_cnt_per_lookup"
      << std::endl;

  auto run_benchmark = [&](
      const std::size_t m,
      const std::size_t thread_cnt) {
    benchmark(
        config,
        m,
        to_insert.data(), to_insert.size(),
        to_lookup_ptr, to_lookup_cnt,
        host_batch_size,
        device_batch_size,
        devices,
        thread_cnt,
        coproc_thread_cnt
    );
  };

  for (std::size_t m : filter_sizes) {
    run_benchmark(m, thread_cnt);
  }
  return 0;
}