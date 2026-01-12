//! Concurrent operations benchmarks.
//!
//! Measures performance of concurrent inserts and queries across
//! multiple threads with different contention levels.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::prelude::*;
use bloomcraft::sync::{ShardedBloomFilter, StripedBloomFilter};
use std::sync::Arc;
use std::thread;

fn generate_data(count: usize) -> Vec<u64> {
    (0..count as u64).collect()
}

/// Benchmark concurrent inserts with different thread counts.
fn bench_concurrent_inserts(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_inserts");

    let filter_size = 1_000_000;
    let fp_rate = 0.01;
    let ops_per_thread = 10_000;
    let thread_counts = [1, 2, 4, 8];

    for &threads in &thread_counts {
        group.throughput(Throughput::Elements((ops_per_thread * threads) as u64));

        // Sharded filter
        group.bench_with_input(
            BenchmarkId::new("sharded", threads),
            &threads,
            |b, &threads| {
                b.iter_batched(
                    || Arc::new(ShardedBloomFilter::<u64>::new(filter_size, fp_rate)),
                    |filter| {
                        let mut handles = vec![];

                        for t in 0..threads {
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
            },
        );

        // Striped filter
        group.bench_with_input(
            BenchmarkId::new("striped", threads),
            &threads,
            |b, &threads| {
                b.iter_batched(
                    || Arc::new(StripedBloomFilter::<u64>::new(filter_size, fp_rate)),
                    |filter| {
                        let mut handles = vec![];

                        for t in 0..threads {
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
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent queries with different thread counts.
fn bench_concurrent_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_queries");

    let filter_size = 1_000_000;
    let fp_rate = 0.01;
    let queries_per_thread = 100_000;
    let thread_counts = [1, 2, 4, 8];

    for &threads in &thread_counts {
        group.throughput(Throughput::Elements((queries_per_thread * threads) as u64));

        group.bench_with_input(
            BenchmarkId::new("sharded", threads),
            &threads,
            |b, &threads| {
                let filter = Arc::new(ShardedBloomFilter::<u64>::new(filter_size, fp_rate));
                let data = Arc::new(generate_data(filter_size));

                // Pre-populate
                for item in data.iter().take(filter_size / 2) {
                    filter.insert(item);
                }

                b.iter(|| {
                    let mut handles = vec![];

                    for _ in 0..threads {
                        let filter_clone = Arc::clone(&filter);
                        let data_clone = Arc::clone(&data);

                        let handle = thread::spawn(move || {
                            let mut hits = 0;
                            for i in 0..queries_per_thread {
                                if filter_clone.contains(black_box(&data_clone[i % data_clone.len()])) {
                                    hits += 1;
                                }
                            }
                            hits
                        });

                        handles.push(handle);
                    }

                    let total: usize = handles.into_iter()
                        .map(|h| h.join().unwrap())
                        .sum();

                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mixed concurrent operations (reads + writes).
fn bench_concurrent_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_mixed");

    let filter_size = 1_000_000;
    let fp_rate = 0.01;
    let ops_per_thread = 10_000;
    let read_ratios = [0.5, 0.9, 0.99]; // Percentage of reads

    for &read_ratio in &read_ratios {
        group.bench_with_input(
            BenchmarkId::new("sharded", format!("{:.0}%_reads", read_ratio * 100.0)),
            &read_ratio,
            |b, &read_ratio| {
                b.iter_batched(
                    || {
                        let filter = Arc::new(ShardedBloomFilter::<u64>::new(filter_size, fp_rate));
                        let data = Arc::new(generate_data(filter_size));

                        // Pre-populate
                        for item in data.iter().take(filter_size / 10) {
                            filter.insert(item);
                        }

                        (filter, data)
                    },
                    |(filter, data)| {
                        let mut handles = vec![];

                        for t in 0..4 {
                            let filter_clone = Arc::clone(&filter);
                            let data_clone = Arc::clone(&data);

                            let handle = thread::spawn(move || {
                                use rand::{Rng, SeedableRng};
                                let mut rng = rand::rngs::StdRng::seed_from_u64(t as u64);

                                for i in 0..ops_per_thread {
                                    if rng.gen::<f64>() < read_ratio {
                                        // Read operation
                                        let _ = filter_clone.contains(
                                            black_box(&data_clone[i % data_clone.len()])
                                        );
                                    } else {
                                        // Write operation
                                        filter_clone.insert(
                                            black_box(&data_clone[i % data_clone.len()])
                                        );
                                    }
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
            },
        );
    }

    group.finish();
}

/// Benchmark contention levels.
fn bench_contention_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_contention");

    let filter_size = 100_000;
    let fp_rate = 0.01;
    let ops_per_thread = 10_000;

    // Low contention: each thread works on different data
    group.bench_function("low_contention", |b| {
        b.iter_batched(
            || Arc::new(ShardedBloomFilter::<u64>::new(filter_size, fp_rate)),
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

    // High contention: all threads work on same data
    group.bench_function("high_contention", |b| {
        b.iter_batched(
            || {
                (
                    Arc::new(ShardedBloomFilter::<u64>::new(filter_size, fp_rate)),
                    Arc::new(generate_data(1000)), // Small shared dataset
                )
            },
            |(filter, data)| {
                let mut handles = vec![];

                for _ in 0..4 {
                    let filter_clone = Arc::clone(&filter);
                    let data_clone = Arc::clone(&data);

                    let handle = thread::spawn(move || {
                        for i in 0..ops_per_thread {
                            filter_clone.insert(black_box(&data_clone[i % data_clone.len()]));
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

/// Benchmark scalability with increasing core count.
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_scalability");

    let filter_size = 1_000_000;
    let fp_rate = 0.01;
    let total_ops = 1_000_000;

    let thread_counts = [1, 2, 4, 8, 16];

    for &threads in &thread_counts {
        let ops_per_thread = total_ops / threads;
        group.throughput(Throughput::Elements(total_ops as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &threads| {
                let filter = Arc::new(ShardedBloomFilter::<u64>::new(filter_size, fp_rate));
                let data = Arc::new(generate_data(filter_size));

                // Pre-populate
                for item in data.iter().take(filter_size / 2) {
                    filter.insert(item);
                }

                b.iter(|| {
                    let mut handles = vec![];

                    for t in 0..threads {
                        let filter_clone = Arc::clone(&filter);
                        let data_clone = Arc::clone(&data);
                        let start = t * ops_per_thread;
                        let end = start + ops_per_thread;

                        let handle = thread::spawn(move || {
                            let mut hits = 0;
                            for i in start..end {
                                if filter_clone.contains(black_box(&data_clone[i % data_clone.len()])) {
                                    hits += 1;
                                }
                            }
                            hits
                        });

                        handles.push(handle);
                    }

                    let total: usize = handles.into_iter()
                        .map(|h| h.join().unwrap())
                        .sum();

                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_concurrent_inserts,
    bench_concurrent_queries,
    bench_concurrent_mixed,
    bench_contention_levels,
    bench_scalability,
);

criterion_main!(benches);
