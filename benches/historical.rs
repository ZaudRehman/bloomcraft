//! Historical comparison: 1970 methods vs modern implementations.
//!
//! Compares classic Bloom filter approaches (Burton Bloom's original 1970 paper)
//! with modern optimizations (partitioned, cache-optimized, SIMD-ready).

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::prelude::*;

fn generate_data(count: usize) -> Vec<u64> {
    (0..count as u64).collect()
}

/// Compare insert performance: classic vs modern.
fn bench_insert_classic_vs_modern(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_insert");

    let size = 100_000;
    let fp_rate = 0.01;
    let batch_size = 10_000;

    group.throughput(Throughput::Elements(batch_size as u64));

    // Classic implementation (simple bit array)
    group.bench_function("classic_1970", |b| {
        b.iter_batched(
            || {
                (
                    StandardBloomFilter::new(size, fp_rate),
                    generate_data(batch_size),
                )
            },
            |(mut filter, data)| {
                for item in &data {
                    filter.insert(black_box(item));
                }
                black_box(filter)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Modern partitioned (cache-friendly)
    group.bench_function("modern_partitioned", |b| {
        b.iter_batched(
            || {
                (
                    PartitionedBloomFilter::new(size, fp_rate),
                    generate_data(batch_size),
                )
            },
            |(mut filter, data)| {
                for item in &data {
                    filter.insert(black_box(item));
                }
                black_box(filter)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Modern hierarchical (multi-level)
    group.bench_function("modern_hierarchical", |b| {
        b.iter_batched(
            || {
                (
                    HierarchicalBloomFilter::new(
                        vec![3],
                        size / 3,
                        fp_rate,
                    ),
                    generate_data(batch_size),
                )
            },
            |(mut filter, data)| {
                for item in &data {
                    filter.insert(black_box(item));
                }
                black_box(filter)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Compare query performance: classic vs modern.
fn bench_query_classic_vs_modern(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_query");

    let size = 100_000;
    let fp_rate = 0.01;

    // Classic
    group.bench_function("classic_1970", |b| {
        let mut filter = StandardBloomFilter::new(size, fp_rate);
        let data = generate_data(size);

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

    // Modern partitioned
    group.bench_function("modern_partitioned", |b| {
        let mut filter = PartitionedBloomFilter::new(size, fp_rate);
        let data = generate_data(size);

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

/// Compare hash function evolution.
fn bench_hash_evolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_hash_functions");

    let count = 10_000;
    let data = generate_data(count);

    // Single hash (early approach - not truly Bloom filter)
    group.bench_function("single_hash_early", |b| {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        b.iter(|| {
            for item in &data {
                let mut hasher = DefaultHasher::new();
                item.hash(&mut hasher);
                let _ = black_box(hasher.finish());
            }
        });
    });

    // Enhanced double (modern optimization)
    group.bench_function("enhanced_double_modern", |b| {
        use bloomcraft::hash::StdHasher;

        let size = 100_000;
        let fp_rate = 0.01;
        let bit_count = bloomcraft::core::params::optimal_bit_count(size, fp_rate).unwrap();
        let hash_count = bloomcraft::core::params::optimal_hash_count(bit_count, size).unwrap();

        let mut filter = StandardBloomFilter::<u64, StdHasher>::with_params(
            bit_count,
            hash_count,
            StdHasher::new(),
        );

        let mut idx = 0;
        b.iter(|| {
            filter.insert(black_box(&data[idx % data.len()]));
            idx += 1;
        });
    });

    group.finish();
}

/// Compare memory efficiency improvements.
fn bench_memory_evolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_memory_efficiency");

    let size = 100_000;
    let fp_rate = 0.01;

    // Classic: standard bit array
    group.bench_function("classic_bit_array", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<u64>::new(size, fp_rate);
            let memory = filter.size() / 8; // bytes
            black_box((filter, memory))
        });
    });

    // Modern: compressed/partitioned
    group.bench_function("modern_partitioned", |b| {
        b.iter(|| {
            let filter = PartitionedBloomFilter::<u64>::new(size, fp_rate);
            let memory = filter.size() / 8;
            black_box((filter, memory))
        });
    });

    // Modern: scalable (grows as needed)
    group.bench_function("modern_scalable", |b| {
        b.iter(|| {
            let filter = ScalableBloomFilter::<u64>::new(size / 10, fp_rate);
            let memory = filter.size() / 8;
            black_box((filter, memory))
        });
    });

    group.finish();
}

/// Compare false positive rate control.
fn bench_fp_rate_control(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_fp_control");
    group.sample_size(10);

    let size = 10_000;
    let fp_rate = 0.01;
    let test_size = 100_000;

    // Classic: fixed FP rate
    group.bench_function("classic_fixed", |b| {
        b.iter_batched(
            || {
                let mut filter = StandardBloomFilter::new(size, fp_rate);
                let present = generate_data(size);
                let absent: Vec<u64> = (size as u64..(size + test_size) as u64).collect();

                for item in &present {
                    filter.insert(item);
                }

                (filter, absent)
            },
            |(filter, absent)| {
                let mut false_positives = 0;
                for item in &absent {
                    if filter.contains(black_box(item)) {
                        false_positives += 1;
                    }
                }
                let observed_fp = false_positives as f64 / absent.len() as f64;
                black_box(observed_fp)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Modern: scalable (adaptive FP rate)
    group.bench_function("modern_scalable", |b| {
        b.iter_batched(
            || {
                let mut filter = ScalableBloomFilter::new(size / 10, fp_rate);
                let present = generate_data(size);
                let absent: Vec<u64> = (size as u64..(size + test_size) as u64).collect();

                for item in &present {
                    filter.insert(item);
                }

                (filter, absent)
            },
            |(filter, absent)| {
                let mut false_positives = 0;
                for item in &absent {
                    if filter.contains(black_box(item)) {
                        false_positives += 1;
                    }
                }
                let observed_fp = false_positives as f64 / absent.len() as f64;
                black_box(observed_fp)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Compare deletion support evolution.
fn bench_deletion_support(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_deletion");

    let size = 100_000;
    let fp_rate = 0.01;
    let batch_size = 1_000;

    // Classic: no deletion support
    group.bench_function("classic_no_deletion", |b| {
        b.iter_batched(
            || {
                let mut filter = StandardBloomFilter::new(size, fp_rate);
                let data = generate_data(batch_size);

                for item in &data {
                    filter.insert(item);
                }

                (filter, data)
            },
            |(mut filter, data)| {
                // Can only clear entire filter
                filter.clear();
                black_box(filter)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Modern: counting filter with deletion
    group.bench_function("modern_counting", |b| {
        b.iter_batched(
            || {
                let mut filter = CountingBloomFilter::with_counter_size(size, fp_rate, 4);
                let data = generate_data(batch_size);

                for item in &data {
                    filter.insert(item);
                }

                (filter, data)
            },
            |(mut filter, data)| {
                // Can remove individual items
                for item in &data {
                    filter.delete(black_box(item));
                }
                black_box(filter)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Compare scalability improvements.
fn bench_scalability_evolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_scalability");

    let fp_rate = 0.01;
    let sizes = [10_000, 100_000, 1_000_000];

    for &size in &sizes {
        // Classic: must pre-allocate
        group.bench_with_input(
            BenchmarkId::new("classic_preallocated", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let filter = StandardBloomFilter::<u64>::new(size, fp_rate);
                    black_box(filter)
                });
            },
        );

        // Modern: grows dynamically
        group.bench_with_input(
            BenchmarkId::new("modern_scalable", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    // Start small, can grow
                    let filter = ScalableBloomFilter::<u64>::new(size / 10, fp_rate);
                    black_box(filter)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_classic_vs_modern,
    bench_query_classic_vs_modern,
    bench_hash_evolution,
    bench_memory_evolution,
    bench_fp_rate_control,
    bench_deletion_support,
    bench_scalability_evolution,
);

criterion_main!(benches);
