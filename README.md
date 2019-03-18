# Bloom Filter Benchmark for Heterogeneous Hardware

This repository contains benchmarks for the AMS-Filter, a high-performance
 blocked Bloom filter based on the [implementation](https://github.com/peterboncz/bloomfilter-bsd) 
 presented in the paper [*Performance-Optimal Filtering: Bloom Overtakes Cuckoo at High Throughput*](http://www.vldb.org/pvldb/vol12/p502-lang.pdf).


## Prerequisites
- A CPU with AVX2 support (or AVX-512). E.g, AMD Ryzen, Intel Haswell (or later)
- [CUDA](https://developer.nvidia.com/cuda-toolkit) v9.0 or later.
- A C++14-capable compiler compatible with your version of CUDA.


## Building
```
git clone git@github.com:harald-lang/amsfilter.git
cd amsfilter
git submodule update --remote --recursive --init
mkdir build
cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 6
```

## Benchmarks
- **host**: runs a CPU-only benchmark
- **device**: runs the benchmark on a (single) GPU
- **device_raw**: similar to **device**, but the performance 
  measurements do NOT include any data transfers between host
   and device.
- **coproc**: runs the CPU-GPU co-processing benchmark 

### Options

The benchmarks are parameterized through environment variables which are described
 in the following.

#### Common options
-  `CONFIG`: used to configure the Bloom filter. The configuration consists of
   a comma-separated list of four integer values:
   1) The number of words per block (*w*), which defines the size of a block 
      within the Bloom filter. Each block consists of one or more 32-bit words.
   2) The number of sectors per block (*s*).
   3) The number of sector groups (*z*).
   4) The number of hash functions (*k*).
   
   By default, a register-blocked Bloom filter is used (=1,1,1,4). 
  
- `GEN`: defines how the data is generated. (default: uniform)
   1) **uniform**: generates random integers,uniformly distributed.
   2) **markov,*f***: generates random integers using a Markov process. 
     The parameter *f* refers to the *clustering factor* which is the average 
     number of consecutive identical integers. For instance, the clustering
     factor of the integer sequence *1, 1, 1, 2, 2, 3, 3, 3, 4, 4* is *2.5*, as
     there are 4 *runs* and the total length of the sequence is 10.
     The Markov process generates a uniformly distributed integer sequence if
     *f = 1*. - Note, that the actual *f* is at most 2% off.
- `INSERT_CNT_LOG2`: the log2 of the number of keys to insert during the build phase (default: 24)
- `LOOKUP_CNT_LOG2`: the log2 of the number of keys to lookup during the probe phase (default: 28)
- `KEYS_PINNED`: 0 = keys are located in pageable memory, 1 = keys are located in pinned memory (default: 1)
- `VALIDATE`: 0 = do not validate the results (default: 1)

#### Host benchmark options

- `BATCH_SIZE_LOG2`: the log2 of the number of keys to lookup in one go (default: 10)
- `THREAD_CNT`: the number of threads that concurrently probe the filter

#### Device benchmark options
- `BATCH_SIZE_LOG2`: the log2 of the number of keys to lookup in one go (default: 10)
- `DEVICE_NO`: specifies the CUDA device to use (default: 0)

#### Co-processing benchmark options
- `HOST_BATCH_SIZE_LOG2`: the log2 of the number of keys to lookup in one go on the host side (default: 10)
- `DEVICE_BATCH_SIZE_LOG2`: the log2 of the number of keys to lookup in one go on the device side (default: 20)
- `THREAD_CNT`: the number of CPU threads that concurrently probe the filter on the host side (default: # of available cores)
- `CO_THREAD_CNT`: the number of CPU threads that dispatch work to a GPU (default: THREAD_CNT)
- `DEVICE_NO`: specifies the CUDA device(s) to use. Multiple devices can be specified as a comma-separated list (default: 0)

### Example

The following examples shows how to benchmark a cache-sectorized Bloom filter
 using all CPU cores and two GPU devices:
```
CONFIG=32,32,2,8 DEVICE_NO=0,1 VALIDATE=0 ./benchmark_coproc 
```