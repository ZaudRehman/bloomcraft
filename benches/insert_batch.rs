//! Batch insert operation benchmarks
//!
//! Measures performance of inserting multiple items at once.
//! Batch operations are critical for real-world performance because:
//!
//! 1. **Amortized Overhead**: Fixed costs spread across many items
//! 2. **Cache Locality**: Processing adjacent items improves cache hit rate
//! 3. **Compiler Optimization**: Loops can be vectorized/unrolled
//! 4. **Real-World Usage**: Most applications insert in batches
//!
//! # Test Scenarios
//!
//! 1. **Batch Size Scaling**: How does throughput change with batch size?
//!    - Tests: 1, 10, 100, 1K, 10K items per batch
//!    - Expected: Throughput increases, then plateaus
//!
//! 2. **Batch vs Individual**: Compare batch API vs repeated single inserts
//!    - Shows benefit of batch operations
//!
//! 3. **Parallel Batch Insert**: Multiple threads inserting batches
//!    - Tests scalability of concurrent batch operations
//!    - **LOCK-FREE**: Uses Arc<StandardBloomFilter> directly (no Mutex!)
//!
//! 4. **Batch Insert with Builder**: Pre-allocated batch insertion
//!    - Tests memory allocation overhead
//!
//! 5. **False Positive Rate Impact**: How FPR affects batch performance
//!    - Lower FPR = more hash functions = slower inserts
//!
//! 6. **Amortization Analysis**: Cost per item as batch size grows
//!    - Shows when batching stops providing benefits
//!
//! 7. **Cache Effects**: Impact of item size on batch performance
//!    - Tests L1/L2/L3 cache locality
//!
//! 8. **Sustained Throughput**: Batch insert rates across filter sizes
//!    - Tests memory bandwidth limits
//!
//! # Expected Results
//!
//! - Single inserts: ~10M ops/s
//! - Batch inserts (100 items): ~15-20M ops/s (1.5-2x faster)
//! - Batch inserts (1000 items): ~20-25M ops/s (2-2.5x faster)
//! - **Lock-free parallel**: Near-linear scaling to 8 threads (~8x throughput)

use bloomcraft::core::ConcurrentBloomFilter;
use bloomcraft::filters::StandardBloomFilter;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::sync::Arc;
use std::thread;

mod common;
use common::*;

// BENCHMARK 1: Batch Insert Scaling by Size

/// Benchmark how throughput scales with batch size
///
/// Batch operations should show:
/// - Better cache utilization (processing adjacent items)
/// - Better compiler optimization (loop unrolling/vectorization)
/// - Lower per-item overhead
///
/// We expect throughput to increase with batch size up to a point,
/// then plateau when we saturate memory bandwidth or cache capacity.
fn bench_batch_insert_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert_scaling");

    let filter_size = 1_000_000;
    let fpr = 0.01;
    let items = generate_strings(filter_size, 32);

    for batch_size in BATCH_SIZES {
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_{}", batch_size)),
            batch_size,
            |b, &batch_size| {
                let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
                let mut idx = 0;

                b.iter(|| {
                    let start = idx;
                    let end = (idx + batch_size).min(items.len());
                    let batch = &items[start..end];

                    filter.insert_batch(black_box(batch));

                    idx = (idx + batch_size) % (items.len() - batch_size.max(1));
                });
            },
        );
    }

    group.finish();
}

// BENCHMARK 2: Batch API vs Individual Inserts

/// Compare batch insert API vs calling insert() repeatedly
///
/// This shows the benefit of using batch operations:
/// - Batch API: Single function call, optimized loop
/// - Individual: Multiple function calls, less optimization
///
/// Expected speedup: 1.5-2x for batch operations
fn bench_batch_vs_individual(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_individual");

    let filter_size = 100_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let items = generate_strings(filter_size, 32);

    group.throughput(Throughput::Elements(batch_size as u64));

    // Individual inserts (baseline)
    group.bench_function("individual", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut idx = 0;

        b.iter(|| {
            for _ in 0..batch_size {
                filter.insert(black_box(&items[idx % items.len()]));
                idx += 1;
            }
        });
    });

    // Batch insert (optimized)
    group.bench_function("batch", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut idx = 0;

        b.iter(|| {
            let end = (idx + batch_size).min(items.len());
            let batch = &items[idx..end];

            filter.insert_batch(black_box(batch));

            idx = (idx + batch_size) % (items.len() - batch_size);
        });
    });

    group.finish();
}

// BENCHMARK 3: Batch Insert with Pre-allocated Vectors

/// Test batch insert when batch vector is pre-allocated vs allocated per call
///
/// This tests memory allocation overhead in batch operations.
fn bench_batch_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_allocation");

    let filter_size = 100_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let items = generate_strings(filter_size, 32);

    group.throughput(Throughput::Elements(batch_size as u64));

    // Allocate batch vector every time
    group.bench_function("allocate_each_time", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut idx = 0;

        b.iter(|| {
            let end = (idx + batch_size).min(items.len());
            let batch = &items[idx..end];

            filter.insert_batch(black_box(batch));

            idx = (idx + batch_size) % (items.len() - batch_size);
        });
    });

    // Reuse pre-allocated batch vector
    group.bench_function("preallocated", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut batch: Vec<&String> = Vec::with_capacity(batch_size);
        let mut idx = 0;

        b.iter(|| {
            batch.clear();
            for _ in 0..batch_size {
                batch.push(&items[idx % items.len()]);
                idx += 1;
            }

            filter.insert_batch_ref(black_box(&batch));
        });
    });

    group.finish();
}

// BENCHMARK 4: Batch Insert with Different Item Types

/// Benchmark batch insert for different data types
///
/// Tests how item size affects batch insert throughput
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
        let mut idx = 0;

        b.iter(|| {
            let end = (idx + batch_size).min(items.len());
            let batch = &items[idx..end];

            filter.insert_batch(black_box(batch));

            idx = (idx + batch_size) % (items.len() - batch_size);
        });
    });

    // String (32 bytes)
    group.bench_function("string_32", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let items = generate_strings(filter_size, 32);
        let mut idx = 0;

        b.iter(|| {
            let end = (idx + batch_size).min(items.len());
            let batch = &items[idx..end];

            filter.insert_batch(black_box(batch));

            idx = (idx + batch_size) % (items.len() - batch_size);
        });
    });

    // String (256 bytes)
    group.bench_function("string_256", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let items = generate_strings(filter_size, 256);
        let mut idx = 0;

        b.iter(|| {
            let end = (idx + batch_size).min(items.len());
            let batch = &items[idx..end];

            filter.insert_batch(black_box(batch));

            idx = (idx + batch_size) % (items.len() - batch_size);
        });
    });

    group.finish();
}

// BENCHMARK 5: Parallel Batch Insert (LOCK-FREE!)

/// Benchmark concurrent batch insert from multiple threads using LOCK-FREE operations
///
/// **CRITICAL CHANGE**: Uses Arc<StandardBloomFilter> directly (no Mutex!)
///
/// Tests:
/// 1. Lock-free atomic operations
/// 2. Cache contention (atomic bit flips)
/// 3. Scalability with thread count
///
/// Expected: Near-linear scaling to ~8 threads, then memory bandwidth saturation
/// This is MUCH faster than Mutex-based approach!
fn bench_parallel_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_batch_insert");
    group.sample_size(20); // Fewer samples for expensive benchmark

    let filter_size = 1_000_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let items = Arc::new(generate_strings(filter_size, 32));

    for num_threads in THREAD_COUNTS_REALISTIC {
        let total_items = num_threads * batch_size;

        group.throughput(Throughput::Elements(total_items as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}threads", num_threads)),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    // NO MUTEX! Direct Arc usage with lock-free atomic operations
                    let filter = Arc::new(
                        StandardBloomFilter::<String>::new(filter_size, fpr)
                    );

                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let items = Arc::clone(&items);

                            thread::spawn(move || {
                                let base = tid * batch_size;
                                let end = (base + batch_size).min(items.len());
                                let batch = &items[base..end];

                                // Use lock-free concurrent batch insert
                                filter.insert_batch_concurrent(batch);
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// BENCHMARK 6: Batch Insert Under Load

/// Benchmark batch insert performance at different filter load factors
///
/// Tests whether batch performance degrades as filter fills up
fn bench_batch_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_by_load");

    let filter_size = 100_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let items = generate_strings(filter_size, 32);

    for load_pct in &[10, 50, 90] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%_load", load_pct)),
            load_pct,
            |b, &load_pct| {
                let filter = StandardBloomFilter::<String>::new(filter_size, fpr);

                // Pre-fill to target load
                let prefill = (filter_size * load_pct) / 100;
                for i in 0..prefill {
                    filter.insert(&items[i]);
                }

                let mut idx = prefill;
                b.iter(|| {
                    let end = (idx + batch_size).min(items.len());
                    let batch = &items[idx..end];

                    filter.insert_batch(black_box(batch));

                    idx = (idx + batch_size) % (items.len() - batch_size);
                });
            },
        );
    }

    group.finish();
}

// BENCHMARK 7: Batch Insert with Duplicates

/// Benchmark batch insert when batch contains many duplicate items
///
/// Tests:
/// - All unique items (best case)
/// - 50% duplicates (typical)
/// - All same item (worst case)
fn bench_batch_duplicates(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_duplicates");

    let filter_size = 100_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let items = generate_strings(filter_size, 32);

    group.throughput(Throughput::Elements(batch_size as u64));

    // All unique items
    group.bench_function("all_unique", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut idx = 0;

        b.iter(|| {
            let end = (idx + batch_size).min(items.len());
            let batch = &items[idx..end];

            filter.insert_batch(black_box(batch));

            idx = (idx + batch_size) % (items.len() - batch_size);
        });
    });

    // 50% duplicates
    group.bench_function("50pct_duplicates", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut batch = Vec::with_capacity(batch_size);

        b.iter(|| {
            batch.clear();
            for i in 0..batch_size {
                let idx = if i % 2 == 0 { 0 } else { i % items.len() };
                batch.push(items[idx].clone());
            }

            filter.insert_batch(black_box(&batch));
        });
    });

    // All same item (worst case)
    group.bench_function("all_same", |b| {
        let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let batch = vec![items[0].clone(); batch_size];

        b.iter(|| {
            filter.insert_batch(black_box(&batch));
        });
    });

    group.finish();
}

// BENCHMARK 8: Batch Insert by False Positive Rate

/// Test how target FPR affects batch insert performance
///
/// Lower FPR = more hash functions = slower inserts
/// Expected: Linear scaling with hash count (k)
fn bench_batch_by_fpr(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_by_fpr");

    let items = generate_strings(10_000, 32);
    let batch_size = 100;

    for fpr in FP_RATES {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", fpr)),
            fpr,
            |b, &fpr| {
                let filter = StandardBloomFilter::<String>::new(10_000, fpr);
                let mut idx = 0;

                b.iter(|| {
                    let end = (idx + batch_size).min(items.len());
                    let batch = &items[idx..end];
                    filter.insert_batch(black_box(batch));
                    idx = (idx + batch_size) % (items.len() - batch_size);
                });
            },
        );
    }

    group.finish();
}

// BENCHMARK 9: Batch Insert Amortization Analysis

/// Measure cost per item as batch size increases
///
/// Shows when batching stops providing benefits.
/// Expected: Improvement flattens after ~1000 items
fn bench_batch_amortization(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_amortization");

    let items = generate_strings(100_000, 32);
    let fpr = 0.01;

    for batch_size in &[1, 10, 100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_items", batch_size)),
            batch_size,
            |b, &batch_size| {
                let filter = StandardBloomFilter::<String>::new(100_000, fpr);
                let mut idx = 0;

                b.iter(|| {
                    let end = (idx + batch_size).min(items.len());
                    let batch = &items[idx..end];
                    filter.insert_batch(black_box(batch));
                    idx = (idx + batch_size) % (items.len() - batch_size);
                });
            },
        );
    }

    group.finish();
}

// BENCHMARK 10: Cache Effects in Batch Insert

/// Test how item size affects cache performance during batch inserts
///
/// Smaller items = better cache locality = faster batch processing
/// Expected: 8-byte items fastest, 1024-byte slowest
fn bench_batch_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cache_effects");

    let fpr = 0.01;
    let batch_size = 100;

    // Small strings (fits in L1 cache)
    group.bench_function("small_strings_8", |b| {
        let items = generate_strings(10_000, 8);
        let filter = StandardBloomFilter::<String>::new(10_000, fpr);
        let mut idx = 0;

        b.iter(|| {
            let end = (idx + batch_size).min(items.len());
            let batch = &items[idx..end];
            filter.insert_batch(black_box(batch));
            idx = (idx + batch_size) % (items.len() - batch_size);
        });
    });

    // Medium strings (L2 cache)
    group.bench_function("medium_strings_64", |b| {
        let items = generate_strings(10_000, 64);
        let filter = StandardBloomFilter::<String>::new(10_000, fpr);
        let mut idx = 0;

        b.iter(|| {
            let end = (idx + batch_size).min(items.len());
            let batch = &items[idx..end];
            filter.insert_batch(black_box(batch));
            idx = (idx + batch_size) % (items.len() - batch_size);
        });
    });

    // Large strings (L3/RAM)
    group.bench_function("large_strings_1024", |b| {
        let items = generate_strings(10_000, 1024);
        let filter = StandardBloomFilter::<String>::new(10_000, fpr);
        let mut idx = 0;

        b.iter(|| {
            let end = (idx + batch_size).min(items.len());
            let batch = &items[idx..end];
            filter.insert_batch(black_box(batch));
            idx = (idx + batch_size) % (items.len() - batch_size);
        });
    });

    group.finish();
}

// BENCHMARK 11: Sustained Batch Insert Throughput

/// Measure sustained batch insert throughput across different filter sizes
///
/// Tests:
/// - Memory bandwidth limits
/// - Cache saturation effects
/// - Scalability to large filters
fn bench_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_throughput");

    let items = generate_strings(1_000_000, 32);
    let fpr = 0.01;
    let batch_size = 10_000;

    group.throughput(Throughput::Elements(batch_size as u64));

    for size in &[10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_capacity", size)),
            size,
            |b, &size| {
                let filter = StandardBloomFilter::<String>::new(size, fpr);
                let mut idx = 0;

                b.iter(|| {
                    let end = (idx + batch_size).min(items.len());
                    let batch = &items[idx..end];
                    filter.insert_batch(black_box(batch));
                    idx = (idx + batch_size) % (items.len() - batch_size);
                });
            },
        );
    }

    group.finish();
}

// BENCHMARK 12: Lock-Free vs Mutex Comparison

/// Direct comparison: Lock-free (Arc<Filter>) vs Mutex (Arc<Mutex<Filter>>)
///
/// This benchmark demonstrates the MASSIVE performance difference between:
/// 1. Lock-free atomic operations (StandardBloomFilter with ConcurrentBloomFilter trait)
/// 2. Traditional Mutex-based synchronization
///
/// Expected: Lock-free is 3-10x faster for concurrent workloads
fn bench_lockfree_vs_mutex(c: &mut Criterion) {
    let mut group = c.benchmark_group("lockfree_vs_mutex");
    group.sample_size(20);

    let filter_size = 1_000_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let num_threads = 4;
    let items = Arc::new(generate_strings(filter_size, 32));

    let total_items = num_threads * batch_size;
    group.throughput(Throughput::Elements(total_items as u64));

    // Lock-free approach (Arc<StandardBloomFilter>)
    group.bench_function("lockfree", |b| {
        b.iter(|| {
            let filter = Arc::new(StandardBloomFilter::<String>::new(filter_size, fpr));

            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);

                    thread::spawn(move || {
                        let base = tid * batch_size;
                        let end = (base + batch_size).min(items.len());
                        let batch = &items[base..end];

                        // Lock-free concurrent insert
                        filter.insert_batch_concurrent(batch);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    // Traditional Mutex approach (Arc<Mutex<StandardBloomFilter>>)
    group.bench_function("mutex", |b| {
        b.iter(|| {
            let filter = Arc::new(std::sync::Mutex::new(
                StandardBloomFilter::<String>::new(filter_size, fpr)
            ));

            let handles: Vec<_> = (0..num_threads)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items);

                    thread::spawn(move || {
                        let base = tid * batch_size;
                        let end = (base + batch_size).min(items.len());
                        let batch = &items[base..end];

                        // Acquire lock for each batch
                        let f = filter.lock().unwrap();
                        f.insert_batch(batch);
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    group.finish();
}

// CRITERION CONFIGURATION

criterion_group!(
    benches,
    bench_batch_insert_scaling,
    bench_batch_vs_individual,
    bench_batch_allocation,
    bench_batch_by_type,
    bench_parallel_batch_insert,
    bench_batch_by_load,
    bench_batch_duplicates,
    bench_batch_by_fpr,
    bench_batch_amortization,
    bench_batch_cache_effects,
    bench_batch_throughput,
    bench_lockfree_vs_mutex,
);

criterion_main!(benches);