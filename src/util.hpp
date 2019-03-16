#pragma once

//===----------------------------------------------------------------------===//
/// Populate the given range with random integers (uniform).
template<typename It>
void
gen_data(It begin, It end) {
  std::random_device rnd_device;
  std::mt19937 gen(rnd_device());
  std::uniform_int_distribution<amsfilter::key_t> dis;

  for (auto it = begin; it != end; ++it) {
    *it = dis(gen);
  }
}
//===----------------------------------------------------------------------===//
/// Parse the filter configuration.
amsfilter::Config
parse_filter_config(const std::string config_str) {
  typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
  boost::char_separator<char> sep{","};
  tokenizer tok{config_str, sep};
  auto tok_it = tok.begin();

  // The filter parameters.
  amsfilter::Config config;
  config.word_cnt_per_block = u32(std::stoul(*tok_it)); tok_it++;
  config.sector_cnt = u32(std::stoul(*tok_it)); tok_it++;
  config.zone_cnt = u32(std::stoul(*tok_it)); tok_it++;
  config.k = u32(std::stoul(*tok_it)); tok_it++;
  return config;
}
//===----------------------------------------------------------------------===//
#if defined(HAVE_CUDA) || defined(__CUDACC__)
static std::string
get_cuda_device_name(u32 cuda_device_no) {
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, cuda_device_no);
  return std::string(device_prop.name);
}
#endif
//===----------------------------------------------------------------------===//
