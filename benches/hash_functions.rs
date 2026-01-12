//! Hash function benchmarks.
//!
//! Compares different hash functions (WyHash, XXH3, SipHash, AHash)
//! and hashing strategies (single, double, triple hashing).

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::prelude::*;
use bloomcraft::hash::{HashStrategy, DefaultHasher};

fn generate_data(count: usize) -> Vec<u64> {
    (0..count as u64).collect()
}

fn generate_strings(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("user_{:010}", i)).collect()
}

/// Benchmark different hash strategies.
fn bench_hash_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_strategies");

    let size = 100_000;
    let fp_rate = 0.01;
    let strategies = vec![
        ("double", HashStrategy::Double),
        ("enhanced_double", HashStrategy::EnhancedDouble),
        ("triple", HashStrategy::Triple),
    ];

    for (name, strategy) in strategies {
        // Insert benchmark
        group.bench_with_input(
            BenchmarkId::new("insert", name),
            &strategy,
            |b, &strategy| {
                b.iter_batched(
                    || {
                        // Calculate optimal parameters
                        let bit_count = bloomcraft::core::params::optimal_bit_count(size, fp_rate).unwrap();
                        let hash_count = bloomcraft::core::params::optimal_hash_count(bit_count, size).unwrap();

                        (
                            StandardBloomFilter::<u64, DefaultHasher>::with_strategy(
                                bit_count, hash_count, strategy
                            ),
                            generate_data(1000),
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
            },
        );

        // Query benchmark
        group.bench_with_input(
            BenchmarkId::new("query", name),
            &strategy,
            |b, &strategy| {
                let bit_count = bloomcraft::core::params::optimal_bit_count(size, fp_rate).unwrap();
                let hash_count = bloomcraft::core::params::optimal_hash_count(bit_count, size).unwrap();

                let mut filter = StandardBloomFilter::<u64, DefaultHasher>::with_strategy(
                    bit_count, hash_count, strategy
                );
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
            },
        );
    }

    group.finish();
}

/// Benchmark hash function quality (distribution).
fn bench_hash_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_distribution");
    group.sample_size(10);

    let size = 10_000;
    let fp_rate = 0.01;

    group.bench_function("measure_collisions", |b| {
        b.iter_batched(
            || generate_data(size),
            |data| {
                use std::collections::HashSet;
                let mut hasher = DefaultHasher::default();
                let mut hashes = HashSet::new();

                for item in &data {
                    use std::hash::{Hash, Hasher};
                    item.hash(&mut hasher);
                    let hash = hasher.finish();
                    hashes.insert(hash);
                }

                let collision_rate = 1.0 - (hashes.len() as f64 / data.len() as f64);
                black_box(collision_rate)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark hash performance with different data types.
fn bench_hash_data_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_data_types");

    let count = 10_000;

    // u64 integers
    group.bench_function("u64", |b| {
        let data = generate_data(count);
        let mut filter = StandardBloomFilter::new(count, 0.01);

        let mut idx = 0;
        b.iter(|| {
            filter.insert(black_box(&data[idx % data.len()]));
            idx += 1;
        });
    });

    // Strings
    group.bench_function("string", |b| {
        let data = generate_strings(count);
        let mut filter = StandardBloomFilter::new(count, 0.01);

        let mut idx = 0;
        b.iter(|| {
            filter.insert(black_box(&data[idx % data.len()]));
            idx += 1;
        });
    });

    // Byte slices
    group.bench_function("bytes", |b| {
        let data: Vec<Vec<u8>> = (0..count)
            .map(|i| i.to_le_bytes().to_vec())
            .collect();
        let mut filter = StandardBloomFilter::new(count, 0.01);

        let mut idx = 0;
        b.iter(|| {
            filter.insert(black_box(&data[idx % data.len()]));
            idx += 1;
        });
    });

    group.finish();
}

/// Benchmark hash count impact on performance.
fn bench_hash_count_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_count_impact");

    let size = 100_000;
    let hash_counts = [1, 3, 5, 7, 10, 15];

    for &hash_count in &hash_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(hash_count),
            &hash_count,
            |b, &hash_count| {
                let bit_count = size * 10; // Fixed bit count

                let mut filter = StandardBloomFilter::<u64, DefaultHasher>::with_strategy(
                    bit_count,
                    hash_count,
                    HashStrategy::EnhancedDouble,
                );

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
            },
        );
    }

    group.finish();
}

/// Benchmark hash function throughput.
fn bench_hash_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_throughput");

    let batch_size = 100_000;
    group.throughput(Throughput::Elements(batch_size as u64));

    group.bench_function("hash_only", |b| {
        let data = generate_data(batch_size);

        b.iter(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::default();

            for item in &data {
                item.hash(&mut hasher);
                let _ = black_box(hasher.finish());
            }
        });
    });

    group.bench_function("hash_with_modulo", |b| {
        let data = generate_data(batch_size);
        let filter_size = 1_000_000;

        b.iter(|| {
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::default();

            for item in &data {
                item.hash(&mut hasher);
                let hash = hasher.finish();
                let _ = black_box(hash % filter_size);
            }
        });
    });

    group.finish();
}

/// Benchmark double hashing implementation.
fn bench_double_hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group("double_hashing");

    let filter_size = 1_000_000;
    let hash_count = 7;

    group.bench_function("generate_positions", |b| {
        use std::hash::{Hash, Hasher};

        let data = generate_data(1000);
        let mut hasher = DefaultHasher::default();

        b.iter(|| {
            for item in &data {
                item.hash(&mut hasher);
                let h1 = hasher.finish();
                let h2 = hasher.finish().wrapping_add(1);

                let mut positions = Vec::with_capacity(hash_count);
                for i in 0..hash_count {
                    let combined = h1.wrapping_add(i as u64 * h2);
                    positions.push(black_box(combined % filter_size));
                }

                black_box(positions);
            }
        });
    });

    group.finish();
}

/// Benchmark cache effects on hash computation.
fn bench_hash_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_cache_effects");

    let count = 10_000;

    // Small data (fits in cache)
    group.bench_function("small_keys", |b| {
        let data: Vec<u32> = (0..count).map(|i| i as u32).collect();
        let mut filter = StandardBloomFilter::new(count, 0.01);

        let mut idx = 0;
        b.iter(|| {
            filter.insert(black_box(&data[idx % data.len()]));
            idx += 1;
        });
    });

    // Large data (doesn't fit in cache)
    group.bench_function("large_keys", |b| {
        let data: Vec<Vec<u8>> = (0..count)
            .map(|i| vec![i as u8; 1024]) // 1KB per key
            .collect();
        let mut filter = StandardBloomFilter::new(count, 0.01);

        let mut idx = 0;
        b.iter(|| {
            filter.insert(black_box(&data[idx % data.len()]));
            idx += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hash_strategies,
    bench_hash_distribution,
    bench_hash_data_types,
    bench_hash_count_impact,
    bench_hash_throughput,
    bench_double_hashing,
    bench_hash_cache_effects,
);

criterion_main!(benches);
