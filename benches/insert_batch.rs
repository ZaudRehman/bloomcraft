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
//!
//! 4. **Batch Insert with Builder**: Pre-allocated batch insertion
//!    - Tests memory allocation overhead
//!
//! # Expected Results
//!
//! - Single inserts: ~10M ops/s
//! - Batch inserts (100 items): ~15-20M ops/s (1.5-2x faster)
//! - Batch inserts (1000 items): ~20-25M ops/s (2-2.5x faster)

use bloomcraft::core::BloomFilter;
use bloomcraft::filters::StandardBloomFilter;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::sync::{Arc, Mutex};
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
                let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
                let mut idx = 0;
                
                b.iter(|| {
                    // Insert batch_size items
                    let batch: Vec<&String> = (0..batch_size)
                        .map(|_| {
                            let item = &items[idx % items.len()];
                            idx += 1;
                            item
                        })
                        .collect();
                    
                    filter.insert_batch(black_box(batch));
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
        let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
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
        let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut idx = 0;
        
        b.iter(|| {
            let batch: Vec<&String> = (0..batch_size)
                .map(|_| {
                    let item = &items[idx % items.len()];
                    idx += 1;
                    item
                })
                .collect();
            
            filter.insert_batch(black_box(batch));
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
        let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut idx = 0;
        
        b.iter(|| {
            let batch: Vec<&String> = (0..batch_size)
                .map(|_| {
                    let item = &items[idx % items.len()];
                    idx += 1;
                    item
                })
                .collect();
            
            filter.insert_batch(black_box(batch));
        });
    });
    
    // Reuse pre-allocated batch vector
    group.bench_function("preallocated", |b| {
        let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut batch = Vec::with_capacity(batch_size);
        let mut idx = 0;
        
        b.iter(|| {
            batch.clear();
            for _ in 0..batch_size {
                batch.push(&items[idx % items.len()]);
                idx += 1;
            }
            
            filter.insert_batch(black_box(batch.iter().copied()));
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
        let mut filter = StandardBloomFilter::<u64>::new(filter_size, fpr);
        let items = generate_u64s(filter_size);
        let mut idx = 0;
        
        b.iter(|| {
            let batch: Vec<&u64> = (0..batch_size)
                .map(|_| {
                    let item = &items[idx % items.len()];
                    idx += 1;
                    item
                })
                .collect();
            
            filter.insert_batch(black_box(batch));
        });
    });
    
    // String (32 bytes)
    group.bench_function("string_32", |b| {
        let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let items = generate_strings(filter_size, 32);
        let mut idx = 0;
        
        b.iter(|| {
            let batch: Vec<&String> = (0..batch_size)
                .map(|_| {
                    let item = &items[idx % items.len()];
                    idx += 1;
                    item
                })
                .collect();
            
            filter.insert_batch(black_box(batch));
        });
    });
    
    // String (256 bytes)
    group.bench_function("string_256", |b| {
        let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let items = generate_strings(filter_size, 256);
        let mut idx = 0;
        
        b.iter(|| {
            let batch: Vec<&String> = (0..batch_size)
                .map(|_| {
                    let item = &items[idx % items.len()];
                    idx += 1;
                    item
                })
                .collect();
            
            filter.insert_batch(black_box(batch));
        });
    });
    
    group.finish();
}

 // BENCHMARK 5: Parallel Batch Insert
 
/// Benchmark concurrent batch insert from multiple threads
///
/// Tests:
/// 1. Thread synchronization overhead
/// 2. Cache contention
/// 3. Scalability with thread count
///
/// Expected: Near-linear scaling up to CPU core count, then saturation
fn bench_parallel_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_batch_insert");
    group.sample_size(20); // Fewer samples for expensive benchmark
    
    let filter_size = 1_000_000;
    let fpr = 0.01;
    let batch_size = 1000;
    let items = generate_strings(filter_size, 32);
    
    for num_threads in THREAD_COUNTS_REALISTIC {
        let total_items = num_threads * batch_size;
        
        group.throughput(Throughput::Elements(total_items as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}threads", num_threads)),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let filter = Arc::new(Mutex::new(
                        StandardBloomFilter::<String>::new(filter_size, fpr)
                    ));
                    
                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let items = items.clone();
                            
                            thread::spawn(move || {
                                let base = tid * batch_size;
                                let batch: Vec<&String> = (0..batch_size)
                                    .map(|i| &items[(base + i) % items.len()])
                                    .collect();
                                
                                let mut f = filter.lock().unwrap();
                                f.insert_batch(batch);
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
                let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
                
                // Pre-fill to target load
                let prefill = (filter_size * load_pct) / 100;
                for i in 0..prefill {
                    filter.insert(&items[i]);
                }
                
                let mut idx = prefill;
                b.iter(|| {
                    let batch: Vec<&String> = (0..batch_size)
                        .map(|_| {
                            let item = &items[idx % items.len()];
                            idx += 1;
                            item
                        })
                        .collect();
                    
                    filter.insert_batch(black_box(batch));
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
        let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let mut idx = 0;
        
        b.iter(|| {
            let batch: Vec<&String> = (0..batch_size)
                .map(|_| {
                    let item = &items[idx % items.len()];
                    idx += 1;
                    item
                })
                .collect();
            
            filter.insert_batch(black_box(batch));
        });
    });
    
    // 50% duplicates
    group.bench_function("50pct_duplicates", |b| {
        let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        
        b.iter(|| {
            let batch: Vec<&String> = (0..batch_size)
                .map(|i| {
                    // Every other item is the same
                    &items[if i % 2 == 0 { 0 } else { i % items.len() }]
                })
                .collect();
            
            filter.insert_batch(black_box(batch));
        });
    });
    
    // All same item (worst case)
    group.bench_function("all_same", |b| {
        let mut filter = StandardBloomFilter::<String>::new(filter_size, fpr);
        let item = &items[0];
        
        b.iter(|| {
            let batch: Vec<&String> = vec![item; batch_size];
            filter.insert_batch(black_box(batch));
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
);

criterion_main!(benches);
