//! Memory usage and access pattern benchmarks.
//!
//! Measures memory efficiency, bytes per element, and cache effects
//! across different filter variants.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::prelude::*;
use std::mem;

fn generate_data(count: usize) -> Vec<u64> {
    (0..count as u64).collect()
}

/// Measure bytes per element for different filter sizes.
fn bench_bytes_per_element(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bytes_per_element");

    let fp_rate = 0.01;
    let sizes = [1_000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        group.bench_with_input(
            BenchmarkId::new("standard", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let filter = StandardBloomFilter::<u64>::new(size, fp_rate);

                    // Estimate memory usage
                    let bit_count = filter.size();
                    let bytes = (bit_count + 7) / 8;
                    let bytes_per_item = bytes as f64 / size as f64;

                    black_box((filter, bytes_per_item))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("counting", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let filter = CountingBloomFilter::<u64>::with_counter_size(size, fp_rate, 4);

                    // 4-bit counters
                    let counter_count = filter.size();
                    let bytes = (counter_count + 1) / 2; // 2 counters per byte
                    let bytes_per_item = bytes as f64 / size as f64;

                    black_box((filter, bytes_per_item))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory access patterns (sequential vs random).
fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access_patterns");

    let filter_size = 1_000_000;
    let fp_rate = 0.01;
    let query_count = 10_000;

    // Sequential access
    group.bench_function("sequential", |b| {
        let mut filter = StandardBloomFilter::new(filter_size, fp_rate);
        let data = generate_data(filter_size);

        // Pre-populate
        for item in &data {
            filter.insert(item);
        }

        b.iter(|| {
            let mut hits = 0;
            // Access in order - cache friendly
            for i in 0..query_count {
                if filter.contains(black_box(&data[i])) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    // Random access
    group.bench_function("random", |b| {
        let mut filter = StandardBloomFilter::new(filter_size, fp_rate);
        let data = generate_data(filter_size);

        for item in &data {
            filter.insert(item);
        }

        // Pre-generate random indices
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.shuffle(&mut rng);

        b.iter(|| {
            let mut hits = 0;
            // Random access - cache unfriendly
            for i in 0..query_count {
                let idx = indices[i % indices.len()];
                if filter.contains(black_box(&data[idx])) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    // Strided access (every Nth element)
    group.bench_function("strided", |b| {
        let mut filter = StandardBloomFilter::new(filter_size, fp_rate);
        let data = generate_data(filter_size);

        for item in &data {
            filter.insert(item);
        }

        let stride = 64; // Cache line size

        b.iter(|| {
            let mut hits = 0;
            let mut idx = 0;
            for _ in 0..query_count {
                if filter.contains(black_box(&data[idx % data.len()])) {
                    hits += 1;
                }
                idx += stride;
            }
            black_box(hits)
        });
    });

    group.finish();
}

/// Benchmark cache effects with different filter sizes.
fn bench_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_cache_effects");

    let fp_rate = 0.01;
    let query_count = 10_000;

    // Sizes chosen to test different cache levels
    let sizes = vec![
        ("L1_cache", 1_000),        // ~10KB (fits in L1)
        ("L2_cache", 10_000),       // ~100KB (fits in L2)
        ("L3_cache", 100_000),      // ~1MB (fits in L3)
        ("main_memory", 10_000_000), // ~100MB (spills to RAM)
    ];

    for (name, size) in sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &size,
            |b, &size| {
                let mut filter = StandardBloomFilter::new(size, fp_rate);
                let data = generate_data(size.min(100_000)); // Limit data generation

                for item in &data {
                    filter.insert(item);
                }

                let mut idx = 0;
                b.iter(|| {
                    let mut hits = 0;
                    for _ in 0..query_count {
                        if filter.contains(black_box(&data[idx % data.len()])) {
                            hits += 1;
                        }
                        idx += 1;
                    }
                    black_box(hits)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation overhead.
fn bench_allocation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    let fp_rate = 0.01;
    let sizes = [1_000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        // Standard filter allocation
        group.bench_with_input(
            BenchmarkId::new("standard_alloc", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let filter = StandardBloomFilter::<u64>::new(size, fp_rate);
                    black_box(filter)
                });
            },
        );

        // Counting filter allocation (more memory)
        group.bench_with_input(
            BenchmarkId::new("counting_alloc", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let filter = CountingBloomFilter::<u64>::with_counter_size(size, fp_rate, 4);
                    black_box(filter)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory footprint comparison.
fn bench_memory_footprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_footprint");

    let size = 100_000;
    let fp_rate = 0.01;

    group.bench_function("standard", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<u64>::new(size, fp_rate);
            let footprint = mem::size_of_val(&filter) + 
                (filter.size() / 8); // Approximate bit vector size
            black_box((filter, footprint))
        });
    });

    group.bench_function("counting", |b| {
        b.iter(|| {
            let filter = CountingBloomFilter::<u64>::with_counter_size(size, fp_rate, 4);
            let footprint = mem::size_of_val(&filter) + 
                (filter.size() / 2); // 4-bit counters
            black_box((filter, footprint))
        });
    });

    group.bench_function("partitioned", |b| {
        b.iter(|| {
            let filter = PartitionedBloomFilter::<u64>::new(size, fp_rate);
            let footprint = mem::size_of_val(&filter) + 
                (filter.size() / 8);
            black_box((filter, footprint))
        });
    });

    group.finish();
}

/// Benchmark false sharing effects in concurrent access.
fn bench_false_sharing(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_false_sharing");

    let filter_size = 100_000;
    let fp_rate = 0.01;
    let ops_per_thread = 10_000;

    use std::sync::Arc;
    use std::thread;

    // Sharded filter (designed to minimize false sharing)
    group.bench_function("sharded_optimized", |b| {
        b.iter_batched(
            || Arc::new(ShardedBloomFilter::new(filter_size, fp_rate)),
            |filter| {
                let mut handles = vec![];

                for t in 0..4 {
                    let filter_clone = Arc::clone(&filter);
                    let start = t * ops_per_thread;
                    let end = start + ops_per_thread;

                    let handle = thread::spawn(move || {
                        for i in start..end {
                            filter_clone.insert(black_box(&(i as u64)));
                        }
                    });

                    handles.push(handle);
                }

                for handle in handles {
                    handle.join().unwrap();
                }

                black_box(filter)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Benchmark memory bandwidth utilization.
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth");

    let filter_size = 1_000_000;
    let fp_rate = 0.01;
    let batch_size = 100_000;

    group.throughput(Throughput::Bytes((batch_size * 8) as u64)); // 8 bytes per u64

    group.bench_function("insert_bandwidth", |b| {
        b.iter_batched(
            || {
                (
                    StandardBloomFilter::new(filter_size, fp_rate),
                    generate_data(batch_size),
                )
            },
            |(mut filter, data)| {
                for item in &data {
                    filter.insert(black_box(item));
                }
                black_box(filter)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.bench_function("query_bandwidth", |b| {
        let mut filter = StandardBloomFilter::new(filter_size, fp_rate);
        let data = generate_data(batch_size);

        for item in &data {
            filter.insert(item);
        }

        b.iter(|| {
            let mut hits = 0;
            for item in &data {
                if filter.contains(black_box(item)) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    group.finish();
}

/// Benchmark memory locality with different partition sizes.
fn bench_partition_locality(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_partition_locality");

    let size = 100_000;
    let fp_rate = 0.01;

    // Standard filter (single partition)
    group.bench_function("standard_single", |b| {
        let mut filter = StandardBloomFilter::new(size, fp_rate);
        let data = generate_data(10_000);

        for item in &data {
            filter.insert(item);
        }

        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&data[idx % data.len()]));
            idx += 1;
            black_box(result)
        });
    });

    // Partitioned filter (multiple partitions)
    group.bench_function("partitioned_multiple", |b| {
        let mut filter = PartitionedBloomFilter::new(size, fp_rate);
        let data = generate_data(10_000);

        for item in &data {
            filter.insert(item);
        }

        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&data[idx % data.len()]));
            idx += 1;
            black_box(result)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bytes_per_element,
    bench_access_patterns,
    bench_cache_effects,
    bench_allocation_overhead,
    bench_memory_footprint,
    bench_false_sharing,
    bench_memory_bandwidth,
    bench_partition_locality,
);

criterion_main!(benches);
