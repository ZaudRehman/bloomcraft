//! Comparison benchmarks across filter variants.
//!
//! Head-to-head comparison of different Bloom filter implementations
//! including classic (1970), standard (modern), partitioned, and counting variants.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::prelude::*;

fn generate_data(count: usize) -> Vec<u64> {
    (0..count as u64).collect()
}

/// Compare insert performance across all variants.
fn bench_insert_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_insert");

    let size = 100_000;
    let fp_rate = 0.01;
    let batch_size = 10_000;

    group.throughput(Throughput::Elements(batch_size as u64));

    let variants = vec![
        "standard",
        "counting",
        "partitioned",
        "hierarchical",
        "scalable",
    ];

    for variant in variants {
        group.bench_with_input(
            BenchmarkId::from_parameter(variant),
            &variant,
            |b, &variant| {
                b.iter_batched(
                    || match variant {
                        "standard" => {
                            (Box::new(StandardBloomFilter::new(size, fp_rate)) as Box<dyn BloomFilter<u64>>, generate_data(batch_size))
                        }
                        "counting" => {
                            (Box::new(CountingBloomFilter::new(size, fp_rate, 15)) as Box<dyn BloomFilter<u64>>, generate_data(batch_size))
                        }
                        "partitioned" => {
                            (Box::new(PartitionedBloomFilter::new(size, fp_rate)) as Box<dyn BloomFilter<u64>>, generate_data(batch_size))
                        }
                        "hierarchical" => {
                            (Box::new(HierarchicalBloomFilter::new(vec![size / 3, size / 3, size / 3], fp_rate)) as Box<dyn BloomFilter<u64>>, generate_data(batch_size))
                        }
                        "scalable" => {
                            (Box::new(ScalableBloomFilter::new(size / 10, fp_rate)) as Box<dyn BloomFilter<u64>>, generate_data(batch_size))
                        }
                        _ => unreachable!(),
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
    }

    group.finish();
}

/// Compare query performance across all variants.
fn bench_query_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_query");

    let size = 100_000;
    let fp_rate = 0.01;

    // Standard filter
    group.bench_function("standard", |b| {
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

    // Counting filter
    group.bench_function("counting", |b| {
        let mut filter = CountingBloomFilter::new(size, fp_rate, 15);
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

    // Partitioned filter (should be faster)
    group.bench_function("partitioned", |b| {
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

    // Hierarchical filter
    group.bench_function("hierarchical", |b| {
        let mut filter = HierarchicalBloomFilter::new(
            vec![size / 3, size / 3, size / 3],
            fp_rate
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
    });

    // Scalable filter
    group.bench_function("scalable", |b| {
        let mut filter = ScalableBloomFilter::new(size / 10, fp_rate);
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

/// Compare memory efficiency across variants.
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_memory_efficiency");

    let size = 100_000;
    let fp_rate = 0.01;

    group.bench_function("standard", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<u64>::new(size, fp_rate);
            black_box(filter)
        });
    });

    group.bench_function("counting", |b| {
        b.iter(|| {
            let filter = CountingBloomFilter::<u64>::new(size, fp_rate, 15);
            black_box(filter)
        });
    });

    group.bench_function("partitioned", |b| {
        b.iter(|| {
            let filter = PartitionedBloomFilter::<u64>::new(size, fp_rate);
            black_box(filter)
        });
    });

    group.finish();
}

/// Compare false positive rates in practice.
fn bench_fp_rate_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_fp_rate");
    group.sample_size(10); // Fewer samples for statistical test

    let size = 10_000;
    let fp_rate = 0.01;
    let test_queries = 100_000;

    group.bench_function("standard", |b| {
        b.iter_batched(
            || {
                let mut filter = StandardBloomFilter::new(size, fp_rate);
                let present = generate_data(size);
                let absent: Vec<u64> = (size as u64..(size + test_queries) as u64).collect();

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
                let observed_fp_rate = false_positives as f64 / absent.len() as f64;
                black_box(observed_fp_rate)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Compare scalability: how performance degrades with size.
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_scalability");

    let fp_rate = 0.01;
    let sizes = [1_000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        // Standard filter
        group.bench_with_input(
            BenchmarkId::new("standard", size),
            &size,
            |b, &size| {
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
            },
        );

        // Partitioned filter
        group.bench_with_input(
            BenchmarkId::new("partitioned", size),
            &size,
            |b, &size| {
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
            },
        );
    }

    group.finish();
}

/// Compare concurrent performance.
fn bench_concurrent_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_concurrent");

    let size = 100_000;
    let fp_rate = 0.01;
    let queries_per_thread = 1000;

    // Standard filter (single-threaded baseline)
    group.bench_function("standard_single_thread", |b| {
        let mut filter = StandardBloomFilter::new(size, fp_rate);
        let data = generate_data(size);

        for item in &data {
            filter.insert(item);
        }

        b.iter(|| {
            let mut hits = 0;
            for i in 0..queries_per_thread {
                if filter.contains(black_box(&data[i % data.len()])) {
                    hits += 1;
                }
            }
            black_box(hits)
        });
    });

    // Sharded filter (multi-threaded)
    group.bench_function("sharded_multi_thread", |b| {
        use std::sync::Arc;

        let filter = Arc::new(ShardedBloomFilter::new(size, fp_rate));
        let data = Arc::new(generate_data(size));

        // Pre-populate
        for item in data.iter() {
            filter.insert(item);
        }

        b.iter(|| {
            let mut handles = vec![];

            for _ in 0..4 {
                let filter_clone = Arc::clone(&filter);
                let data_clone = Arc::clone(&data);

                let handle = std::thread::spawn(move || {
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
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_comparison,
    bench_query_comparison,
    bench_memory_efficiency,
    bench_fp_rate_accuracy,
    bench_scalability,
    bench_concurrent_comparison,
);

criterion_main!(benches);
