#![allow(unused_imports)]

//! Batch query (contains) operation benchmarks
//!
//! Measures performance of querying multiple items at once.
//! Batch queries are critical for:
//!
//! 1. **Real-world workloads**: Applications rarely query one item at a time
//! 2. **Throughput optimization**: Amortize function call overhead
//! 3. **Cache efficiency**: Process adjacent items together
//! 4. **Vectorization**: Compilers can optimize batch loops
//!
//! # Test Scenarios
//!
//! 1. **Batch Size Scaling**: How throughput changes with batch size
//!    - Tests: 1, 10, 100, 1K, 10K queries per batch
//!    - Expected: Throughput increases, plateaus at memory bandwidth
//!
//! 2. **Batch vs Individual**: Compare batch API vs repeated single queries
//!    - Shows benefit of batch operations
//!
//! 3. **Mixed Workloads**: Batches with different hit/miss ratios
//!    - All hits, all misses, 50/50 mix
//!
//! 4. **Parallel Batch Query**: Multi-threaded batch queries
//!    - Tests read scalability (no synchronization needed)
//!
//! 5. **Early Termination**: Batch queries with stop-on-first-miss
//!    - Useful for AND queries (all must be present)
//!
//! # Expected Results
//!
//! - Single queries: ~10M ops/s
//! - Batch queries (100 items): ~15-20M ops/s (1.5-2x faster)
//! - Batch queries (1000 items): ~20-30M ops/s (2-3x faster)
//! - Parallel queries: Near-linear scaling (reads don't contend)

use bloomcraft::filters::StandardBloomFilter;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::sync::Arc;
use std::thread;

mod common;
use common::*;

// BENCHMARK 1: Batch Query Scaling by Size

/// Benchmark how query throughput scales with batch size
///
/// Expected results:
/// - Small batches (1-10): Marginal improvement over single queries
/// - Medium batches (100): 1.5-2x throughput improvement
/// - Large batches (1000+): 2-3x throughput improvement
/// - Very large batches (10K+): Plateaus at memory bandwidth limit
fn bench_batch_query_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_query_scaling");

    let filter_size = 1_000_000;
    let fpr = 0.01;
    let items = generate_strings(filter_size, 32);
    let filter = StandardBloomFilter::<String>::new(filter_size, fpr);

    // Fill to 50%
    for i in 0..(filter_size / 2) {
        filter.insert(&items[i]);
    }

    for batch_size in BATCH_SIZES {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_{}", batch_size)),
            batch_size,
            |b, &batch_size| {
                let mut idx = 0;

                b.iter(|| {
                    // Query batch_size items
                    let batch: Vec<String> = (0..batch_size)
                        .map(|_| {
                            let item = items[idx % items.len()].clone();
                            idx += 1;
                            item
                        })
                        .collect();

                    let results = filter.contains_batch(black_box(&batch));
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// BENCHMARK 2: Batch API vs Individual Queries

/// Compare batch query API vs calling contains() repeatedly
///
/// This demonstrates the value of batch operations:
/// - Individual: Multiple function calls, less optimization
/// - Batch: Single function call, compiler can optimize loop
///
/// Expected speedup: 1.5-2x for batch operations
fn bench_batch_vs_individual(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_individual");

    let filter_size = 100_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let items = generate_strings(filter_size, 32);
    let filter = StandardBloomFilter::<String>::new(filter_size, fpr);

    for i in 0..(filter_size / 2) {
        filter.insert(&items[i]);
    }

    group.throughput(Throughput::Elements(batch_size as u64));

    // Individual queries (baseline)
    group.bench_function("individual", |b| {
        let mut idx = 0;

        b.iter(|| {
            let mut results = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let result = filter.contains(black_box(&items[idx % items.len()]));
                results.push(result);
                idx += 1;
            }
            black_box(results)
        });
    });

    // Batch query (optimized)
    group.bench_function("batch", |b| {
        let mut idx = 0;

        b.iter(|| {
            let batch: Vec<String> = (0..batch_size)
                .map(|_| {
                    let item = items[idx % items.len()].clone();
                    idx += 1;
                    item
                })
                .collect();

            let results = filter.contains_batch(black_box(&batch));
            black_box(results)
        });
    });

    group.finish();
}

// BENCHMARK 3: Batch Query with Different Hit Rates

/// Benchmark batch query performance with varying hit/miss ratios
///
/// Real-world scenarios:
/// - All hits: Best case for filter value (no false negatives)
/// - All misses: Common in negative caching scenarios
/// - 50/50 mix: Typical balanced workload
/// - 90% miss: High-selectivity filtering
fn bench_batch_hit_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_hit_rate");

    let filter_size = 100_000;
    let fpr = 0.01;
    let batch_size = 1000;

    let (present_items, absent_items) = create_disjoint_sets(filter_size / 2, filter_size / 2, 32);

    let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
    for item in &present_items {
        filter.insert(item);
    }

    group.throughput(Throughput::Elements(batch_size as u64));

    // All hits (100% present in filter)
    group.bench_function("100%_hit", |b| {
        b.iter(|| {
            let batch: Vec<String> = present_items.iter().take(batch_size).cloned().collect();
            let results = filter.contains_batch(black_box(&batch));
            black_box(results)
        });
    });

    // All misses (0% present in filter)
    group.bench_function("0%_hit", |b| {
        b.iter(|| {
            let batch: Vec<String> = absent_items.iter().take(batch_size).cloned().collect();
            let results = filter.contains_batch(black_box(&batch));
            black_box(results)
        });
    });

    // 50% hit rate (balanced)
    group.bench_function("50%_hit", |b| {
        let mut idx_present = 0;
        let mut idx_absent = 0;

        b.iter(|| {
            let batch: Vec<String> = (0..batch_size)
                .map(|i| {
                    if i % 2 == 0 {
                        let item = present_items[idx_present % present_items.len()].clone();
                        idx_present += 1;
                        item
                    } else {
                        let item = absent_items[idx_absent % absent_items.len()].clone();
                        idx_absent += 1;
                        item
                    }
                })
                .collect();

            let results = filter.contains_batch(black_box(&batch));
            black_box(results)
        });
    });

    // 90% miss rate (high selectivity)
    group.bench_function("10%_hit", |b| {
        let mut idx_present = 0;
        let mut idx_absent = 0;

        b.iter(|| {
            let batch: Vec<String> = (0..batch_size)
                .map(|i| {
                    if i % 10 == 0 {
                        let item = present_items[idx_present % present_items.len()].clone();
                        idx_present += 1;
                        item
                    } else {
                        let item = absent_items[idx_absent % absent_items.len()].clone();
                        idx_absent += 1;
                        item
                    }
                })
                .collect();

            let results = filter.contains_batch(black_box(&batch));
            black_box(results)
        });
    });

    group.finish();
}

// BENCHMARK 4: Batch Query with Different Item Types

/// Benchmark batch query for different data types
fn bench_batch_by_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_by_type");

    let filter_size = 100_000;
    let fpr = 0.01;
    let batch_size = 1000;

    group.throughput(Throughput::Elements(batch_size as u64));

    // u64 (small, stack-allocated)
    group.bench_function("u64", |b| {
        let filter = StandardBloomFilter::<u64>::new(filter_size, fpr);
        let items = generate_u64s(filter_size);

        for i in 0..(filter_size / 2) {
            filter.insert(&items[i]);
        }

        b.iter(|| {
            let batch: Vec<u64> = items.iter().take(batch_size).copied().collect();
            let results = filter.contains_batch(black_box(&batch));
            black_box(results)
        });
    });

    // String (32 bytes)
    group.bench_function("string_32", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let items = generate_strings(filter_size, 32);

        for i in 0..(filter_size / 2) {
            filter.insert(&items[i]);
        }

        b.iter(|| {
            let batch: Vec<String> = items.iter().take(batch_size).cloned().collect();
            let results = filter.contains_batch(black_box(&batch));
            black_box(results)
        });
    });

    // String (256 bytes)
    group.bench_function("string_256", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let items = generate_strings(filter_size, 256);

        for i in 0..(filter_size / 2) {
            filter.insert(&items[i]);
        }

        b.iter(|| {
            let batch: Vec<String> = items.iter().take(batch_size).cloned().collect();
            let results = filter.contains_batch(black_box(&batch));
            black_box(results)
        });
    });

    group.finish();
}

// BENCHMARK 5: Parallel Batch Query

/// Benchmark concurrent batch queries from multiple threads
///
/// Bloom filter queries are read-only operations, so they should
/// scale linearly with thread count (no contention).
///
/// Expected: Near-linear scaling up to CPU core count
fn bench_parallel_batch_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_batch_query");
    group.sample_size(20); // Fewer samples for expensive benchmark

    let filter_size = 1_000_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let items = generate_strings(filter_size, 32);

    let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
    for i in 0..(filter_size / 2) {
        filter.insert(&items[i]);
    }

    let filter = Arc::new(filter);
    let items = Arc::new(items);

    for num_threads in THREAD_COUNTS_REALISTIC {
        let total_queries = num_threads * batch_size;
        group.throughput(Throughput::Elements(total_queries as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}threads", num_threads)),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let items = Arc::clone(&items);

                            thread::spawn(move || {
                                let base = tid * batch_size;
                                let batch: Vec<&String> = (0..batch_size)
                                    .map(|i| &items[(base + i) % items.len()])
                                    .collect();
                                filter.contains_batch_ref(&batch)
                            })
                        })
                        .collect();

                    let mut all_results = Vec::new();
                    for h in handles {
                        all_results.extend(h.join().unwrap());
                    }

                    black_box(all_results)
                });
            },
        );
    }

    group.finish();
}

// BENCHMARK 6: Batch Query Under Load

/// Benchmark batch query performance at different filter load factors
fn bench_batch_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_by_load");

    let filter_size = 100_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let items = generate_strings(filter_size, 32);

    for load_pct in &[10, 50, 90] {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);

        // Fill to target load
        let fill_count = (filter_size * load_pct) / 100;
        for i in 0..fill_count {
            filter.insert(&items[i]);
        }

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%_load", load_pct)),
            load_pct,
            |b, _| {
                b.iter(|| {
                    let batch: Vec<String> = items.iter().take(batch_size).cloned().collect();
                    let results = filter.contains_batch(black_box(&batch));
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// BENCHMARK 7: Batch Query with Early Termination

/// Benchmark batch query with early termination on first miss
///
/// Useful for AND queries where all items must be present.
/// If any item is missing, we can stop checking the rest.
fn bench_batch_early_termination(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_early_termination");

    let filter_size = 100_000;
    let fpr = 0.01;
    let batch_size = 1000;

    let (present_items, absent_items) = create_disjoint_sets(filter_size / 2, filter_size / 2, 32);

    let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
    for item in &present_items {
        filter.insert(item);
    }

    group.throughput(Throughput::Elements(batch_size as u64));

    // Best case: First item missing (immediate termination)
    group.bench_function("first_miss", |b| {
        b.iter(|| {
            let mut batch: Vec<String> = vec![absent_items[0].clone()];
            batch.extend(present_items.iter().take(batch_size - 1).cloned());

            // Check all until first miss
            let result = batch.iter().all(|item| filter.contains(item));
            black_box(result)
        });
    });

    // Worst case: Last item missing (must check all)
    group.bench_function("last_miss", |b| {
        b.iter(|| {
            let mut batch: Vec<String> = present_items.iter().take(batch_size - 1).cloned().collect();
            batch.push(absent_items[0].clone());

            // Check all until first miss
            let result = batch.iter().all(|item| filter.contains(item));
            black_box(result)
        });
    });

    // No early termination: All present
    group.bench_function("no_termination", |b| {
        b.iter(|| {
            let batch: Vec<String> = present_items.iter().take(batch_size).cloned().collect();

            // Check all until first miss
            let result = batch.iter().all(|item| filter.contains(item));
            black_box(result)
        });
    });

    group.finish();
}

// CRITERION CONFIGURATION

criterion_group!(
    benches,
    bench_batch_query_scaling,
    bench_batch_vs_individual,
    bench_batch_hit_rate,
    bench_batch_by_type,
    bench_parallel_batch_query,
    bench_batch_by_load,
    bench_batch_early_termination,
);

criterion_main!(benches);
