#include <algorithm>
#include <iostream>

#include <amsfilter/amsfilter_lite.hpp>

/// The code snippet below shows the basic usage of the AMS-Filter Lite.
int32_t main() {

  // Bloom filter parameters.
  amsfilter::Config config;
  config.word_cnt_per_block = 8;
  config.sector_cnt = 8;
  config.zone_cnt = 2;
  config.k = 4;

  // The filter size in bits.
  std::size_t m = 1024;

  // Construct the filter.
  amsfilter::AmsFilterLite af(config, m);

  // Populate the filter.
  std::cout << "contains 13: " << af.contains(13) << std::endl;
  std::cout << "contains 37: " << af.contains(37) << std::endl;
  af.insert(13);
  af.insert(37);
  std::cout << "contains 13: " << af.contains(13) << std::endl;
  std::cout << "contains 37: " << af.contains(37) << std::endl;

  // Generate some keys (for probing the filter).
  const std::size_t key_cnt = 1024;
  std::vector<amsfilter::AmsFilter::key_t> keys(key_cnt, 0);
  std::generate(keys.begin(), keys.end(), [n = 0] () mutable { return n++; });

  // The filter is probed in batches. The maximum batch size (number of keys)
  // needs to be specified in advance to enable pre-allocation of memory and
  // re-use of that memory.
  u32 max_batch_size = 1024;

  {
    // Probe the filter using GPU acceleration (CUDA).
    std::cout << "device:" << std::endl;

    // Which CUDA device to use?
    u32 cuda_device = 0;

    // The function batch_probe_cuda returns a probe instance. While the
    // probe instance is created, the Bloom filter is copied to the device
    // memory. Thus, the insert function must not be called afterwards.
    auto cuda_probe = af.batch_probe_cuda(max_batch_size, cuda_device);

    // Start probing (asynchronously).
    cuda_probe(keys.data(), key_cnt);

    // Wait for the GPU to finish.
    cuda_probe.wait();

    // Get the results (a bitmap).
    auto result_bitmap = cuda_probe.get_results();
    std::cout << "contains 13: " << result_bitmap[13] << std::endl;
    std::cout << "contains 37: " << result_bitmap[37] << std::endl;
  }

  {
    // Probe the filter (on the host side).
    std::cout << "host:" << std::endl;
    auto probe = af.batch_probe(max_batch_size);
    probe(keys.data(), key_cnt);
    // The wait function always returns true, because the batch probe on a CPU
    // is executed synchronously.
    probe.wait();
    auto result_bitmap = probe.get_results();
    std::cout << "contains 13: " << result_bitmap[13] << std::endl;
    std::cout << "contains 37: " << result_bitmap[37] << std::endl;
  }

}