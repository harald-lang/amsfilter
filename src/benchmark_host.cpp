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
    const std::size_t batch_size,
    const std::size_t thread_cnt) {

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

  // Performance measurement.
  {
    const std::size_t input_cnt = to_lookup_cnt;
    std::atomic<std::size_t> cntr { 0 };
    std::size_t batch_cntr = 0;

    auto thread_fn = [&](u32 thread_id) {
      // Each thread obtains a probe instance.
      auto probe = filter.batch_probe(batch_size);
      auto results = probe.get_results();

      while (true) {
        // Fetch a batch of keys.
        const std::size_t input_idx_begin = cntr.fetch_add(batch_size);
        const std::size_t input_idx_end =
            std::min(input_idx_begin + batch_size, input_cnt);
        if (input_idx_begin >= input_cnt) break;
        probe(&to_lookup[input_idx_begin], input_idx_end - input_idx_begin);
      }
    };

    auto time_start = std::chrono::high_resolution_clock::now();
    dtl::run_in_parallel(thread_fn, cpu_mask, thread_cnt);
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
        << "," << batch_size
        << "," << thread_cnt
        << "," << batch_cntr
        << "," << probes_per_second
        << "," << amsfilter::internal::get_word_access_cnt(config)
        << std::endl;
  }
};
//===----------------------------------------------------------------------===//
int32_t main() {
  // The filter parameters.
  std::string bbf_config_str = dtl::env<std::string>::get("CONFIG", "1,1,1,4");
  amsfilter::Config config = parse_filter_config(bbf_config_str);

  std::cout << "Filter parameters: w=" << config.word_cnt_per_block
      << ", s=" << config.sector_cnt << ", z=" << config.zone_cnt << ", k="
      << config.k << std::endl;

  std::size_t batch_size_log2 = dtl::env<std::size_t>::get("BATCH_SIZE_LOG2", 10);
  std::size_t batch_size = 1ull << batch_size_log2;
  std::cout << "Batch size: 2^" << batch_size_log2 << std::endl;

  int32_t thread_cnt = dtl::env<int32_t>::get("THREAD_CNT", cpu_mask.count());
  std::cout << "using " << thread_cnt << " thread(s) " << std::endl;

  std::size_t to_insert_cnt_log2 = dtl::env<std::size_t>::get("INSERT_CNT_LOG2", 24);
  std::size_t to_insert_cnt = 1ull << to_insert_cnt_log2;
  std::size_t to_lookup_cnt_log2 = dtl::env<std::size_t>::get("LOOKUP_CNT_LOG2", 28);
  std::size_t to_lookup_cnt = 1ull << to_lookup_cnt_log2;

  bool keys_in_pinned_memory = dtl::env<$i32>::get("KEYS_PINNED", 1) != 0;

  std::vector<amsfilter::key_t> to_insert(to_insert_cnt);
  std::vector<amsfilter::key_t> to_lookup(to_lookup_cnt);

  std::string data_gen_config_str = dtl::env<std::string>::get("GEN", "uniform");
  auto data_gen_config = parse_datagen_config(data_gen_config_str);
  gen_data_uniform(to_insert.begin(), to_insert.end());
  gen_data(to_lookup.begin(), to_lookup.end(), data_gen_config);

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

  auto run_benchmark = [&](
      const std::size_t m,
      const std::size_t thread_cnt) {
    benchmark(
        config,
        m,
        to_insert.data(), to_insert.size(),
        to_lookup_ptr, to_lookup_cnt,
        batch_size,
        thread_cnt
    );
  };

  for (std::size_t m : filter_sizes) {
    run_benchmark(m, thread_cnt);
  }
  return 0;
}