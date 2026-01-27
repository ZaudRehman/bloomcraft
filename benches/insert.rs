//! Single-item insert operation benchmarks
//!
//! Comprehensive performance analysis of StandardBloomFilter insert operations
//! under various conditions:
//!
//! # Test Scenarios
//!
//! 1. **By Size**: How does insert latency scale with filter capacity?
//!    - Tests: 1K, 10K, 100K, 1M items
//!    - Expected: O(1) with slight increase due to cache effects
//!
//! 2. **By False Positive Rate**: How does FPR affect insert speed?
//!    - Tests: 10%, 1%, 0.1%, 0.01%
//!    - Lower FPR = more hash functions = slower inserts
//!
//! 3. **By Item Type**: How does data type affect performance?
//!    - Tests: u64 (8 bytes), UUID (16 bytes), String (32/256 bytes)
//!    - Larger items = more hashing overhead
//!
//! 4. **By Load Factor**: How does saturation affect insert speed?
//!    - Tests: 10%, 25%, 50%, 75%, 90% full
//!    - Expected: Minimal impact (bit array operations are O(1))
//!
//! 5. **By Access Pattern**: How do different insertion patterns perform?
//!    - Sequential, random, duplicate, worst-case
//!
//! 6. **Throughput**: Sustained bulk insert performance
//!    - Measures ops/second for large batches
//!
//! # Key Metrics
//!
//! - **Latency**: Time per individual insert (nanoseconds)
//! - **Throughput**: Inserts per second (millions)
//! - **Scalability**: How performance changes with size
//!
//! # Baseline Performance Targets
//!
//! - Small filters (1K): ~80ns per insert (~12.5M ops/s)
//! - Medium filters (100K): ~100ns per insert (~10M ops/s)
//! - Large filters (1M): ~150ns per insert (~6.6M ops/s)
#![allow(unused_imports)]
use bloomcraft::core::BloomFilter;
use bloomcraft::filters::StandardBloomFilter;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::time::Instant;

mod common;
use common::*;

// BENCHMARK 1: Insert Performance by Filter Size
 
/// Benchmark insert latency across different filter sizes
///
/// This tests whether insert operations remain O(1) as filter size grows.
/// We expect slight degradation due to:
/// - L1/L2/L3 cache misses on larger filters
/// - TLB misses on very large filters
///
/// But the algorithmic complexity should remain constant.
fn bench_insert_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_by_size");
    
    for size in SIZES {
        let items = generate_strings(*size, 32);
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let filter = StandardBloomFilter::<String>::new(size, 0.01);
                let mut idx = 0;
                
                b.iter(|| {
                    filter.insert(black_box(&items[idx % items.len()]));
                    idx += 1;
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 2: Insert Performance by False Positive Rate

/// Benchmark insert latency for different target FPRs
///
/// Lower FPR requires more hash functions:
/// - 10% FPR: k ≈ 3-4 hashes
/// - 1% FPR: k ≈ 7 hashes
/// - 0.1% FPR: k ≈ 10 hashes
/// - 0.01% FPR: k ≈ 13 hashes
///
/// Each insert must compute k hashes and set k bits, so
/// insert time should scale linearly with k.
fn bench_insert_by_fpr(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_by_fpr");
    
    let size = 100_000;
    let items = generate_strings(size, 32);
    
    for fpr in FP_RATES {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", fpr)),
            fpr,
            |b, &fpr| {
                let filter = StandardBloomFilter::<String>::new(size, fpr);
                let mut idx = 0;
                
                b.iter(|| {
                    filter.insert(black_box(&items[idx % items.len()]));
                    idx += 1;
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 3: Insert Performance by Item Type

/// Benchmark insert latency for different data types
///
/// Tests how hashing overhead varies with item size:
/// - u64 (8 bytes): Fast, no allocation
/// - UUID (16 bytes): Fast, no allocation
/// - String (32 bytes): Moderate, heap-allocated
/// - String (256 bytes): Slower, more data to hash
/// - String (1024 bytes): Much slower, cache misses
///
/// Hash function time is proportional to input size, so
/// we expect linear scaling with item size.
fn bench_insert_by_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_by_type");
    
    let size = 100_000;
    let fpr = 0.01;
    
    group.throughput(Throughput::Elements(1));
    
    // u64 (8 bytes, stack-allocated)
    group.bench_function("u64", |b| {
        let filter = StandardBloomFilter::<u64>::new(size, fpr);
        let mut counter = 0u64;
        
        b.iter(|| {
            filter.insert(black_box(&counter));
            counter = counter.wrapping_add(1);
        });
    });
    
    // UUID (16 bytes, stack-allocated)
    group.bench_function("uuid_16", |b| {
        let filter = StandardBloomFilter::<[u8; 16]>::new(size, fpr);
        let uuids = generate_uuids(size);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&uuids[idx % uuids.len()]));
            idx += 1;
        });
    });
    
    // String (32 bytes, heap-allocated)
    group.bench_function("string_32", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_strings(size, 32);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });
    
    // String (256 bytes, heap-allocated)
    group.bench_function("string_256", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_strings(size, 256);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });
    
    // String (1024 bytes, heap-allocated, cache-unfriendly)
    group.bench_function("string_1024", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_strings(size, 1024);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });
    
    // URLs (realistic web crawler workload)
    group.bench_function("url", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let urls = generate_urls(size);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&urls[idx % urls.len()]));
            idx += 1;
        });
    });
    
    // Email addresses (realistic user deduplication workload)
    group.bench_function("email", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let emails = generate_emails(size);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&emails[idx % emails.len()]));
            idx += 1;
        });
    });
    
    group.finish();
}

// BENCHMARK 4: Insert Performance by Load Factor

/// Benchmark insert latency at different filter saturation levels
///
/// Tests whether insert performance degrades as filter fills up.
/// 
/// Bit array operations are O(1), so we expect minimal impact.
/// However, at very high load (>90%), cache effects may appear
/// because more bits are set, potentially causing more cache
/// line writes.
fn bench_insert_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_by_load");
    
    let capacity = 100_000;
    let fpr = 0.01;
    let items = generate_strings(capacity, 32);
    
    for load_pct in LOAD_FACTORS {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%", load_pct)),
            load_pct,
            |b, &load_pct| {
                let filter = StandardBloomFilter::<String>::new(capacity, fpr);
                
                // Pre-fill to target load factor
                let prefill_count = (capacity * load_pct) / 100;
                for i in 0..prefill_count {
                    filter.insert(&items[i]);
                }
                
                let mut idx = prefill_count;
                b.iter(|| {
                    filter.insert(black_box(&items[idx % items.len()]));
                    idx += 1;
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 5: Insert Performance by Access Pattern

/// Benchmark insert latency for different access patterns
///
/// Real-world workloads have different characteristics:
/// - Sequential: Best case (cache-friendly)
/// - Random: Average case (typical workload)
/// - Duplicate: Inserting same item repeatedly
/// - Zipfian: Realistic (some items inserted more often)
/// - Worst-case: Alternating between cache-distant items
fn bench_insert_by_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_by_pattern");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    group.throughput(Throughput::Elements(1));
    
    // Sequential pattern (best case for CPU cache)
    group.bench_function("sequential", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });
    
    // Random pattern (typical workload)
    group.bench_function("random", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let pattern = generate_uniform_pattern(items.len(), 10_000);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&items[pattern[idx % pattern.len()]]));
            idx += 1;
        });
    });
    
    // Duplicate inserts (same item repeatedly)
    group.bench_function("duplicate", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let item = &items[0];
        
        b.iter(|| {
            filter.insert(black_box(item));
        });
    });
    
    // Zipfian pattern (80/20 rule: hot items accessed frequently)
    group.bench_function("zipfian", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let pattern = generate_zipfian_pattern(items.len(), 10_000, 1.5);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&items[pattern[idx % pattern.len()]]));
            idx += 1;
        });
    });
    
    // Worst-case pattern (cache-thrashing)
    group.bench_function("worst_case", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let pattern = generate_worst_case_pattern(items.len(), 10_000);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&items[pattern[idx % pattern.len()]]));
            idx += 1;
        });
    });
    
    group.finish();
}

// BENCHMARK 6: Insert Throughput (Sustained Performance)

/// Benchmark sustained insert throughput
///
/// Measures operations per second for large batches.
/// This tests whether performance remains consistent
/// under sustained load (no thermal throttling, cache effects, etc.).
fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");
    
    // Test different sizes to see throughput scaling
    for size in &[10_000, 100_000, 1_000_000] {
        let items = generate_strings(*size, 32);
        
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_items", size)),
            size,
            |b, &size| {
                let filter = StandardBloomFilter::<String>::new(size, 0.01);
                let mut idx = 0;
                
                b.iter(|| {
                    // Insert 1000 items per iteration
                    for _ in 0..1000 {
                        filter.insert(black_box(&items[idx % items.len()]));
                        idx += 1;
                    }
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 7: Cold vs Warm Cache Performance

/// Compare insert performance with cold vs warm CPU cache
///
/// Cold cache: Filter accessed for first time
/// Warm cache: Filter already in CPU cache
///
/// This shows the impact of cache locality on real-world performance.
fn bench_insert_cold_vs_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_cold_vs_warm");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    group.throughput(Throughput::Elements(1));
    
    // Warm cache: Filter pre-warmed with operations
    group.bench_function("warm_cache", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        
        // Warm up the cache
        for i in 0..1000 {
            filter.insert(&items[i]);
        }
        
        let mut idx = 1000;
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });
    
    // Cold cache: New filter for each iteration
    group.bench_function("cold_cache", |b| {
        let mut idx = 0;
        
        b.iter(|| {
            // Create fresh filter (cold cache)
            let filter = StandardBloomFilter::<String>::new(size, fpr);
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
            
            // Force filter to stay alive
            black_box(filter);
        });
    });
    
    group.finish();
}

// BENCHMARK 8: Insert with Different Hash Qualities

/// Benchmark insert with items that have different hash characteristics
///
/// - High entropy: Random strings (ideal)
/// - Low entropy: Sequential numbers (worst case for some hash functions)
/// - Common prefixes: Strings with shared prefixes
fn bench_insert_hash_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_hash_quality");
    
    let size = 100_000;
    let fpr = 0.01;
    
    group.throughput(Throughput::Elements(1));
    
    // High entropy (random strings)
    group.bench_function("high_entropy", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_strings(size, 32);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });
    
    // Low entropy (sequential numbers)
    group.bench_function("low_entropy", |b| {
        let filter = StandardBloomFilter::<u64>::new(size, fpr);
        let mut counter = 0u64;
        
        b.iter(|| {
            filter.insert(black_box(&counter));
            counter += 1;
        });
    });
    
    // Common prefixes (worst case for poor hash functions)
    group.bench_function("common_prefix", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_prefixed_strings(size);
        let mut idx = 0;
        
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });
    
    group.finish();
}

// CRITERION CONFIGURATION

criterion_group!(
    benches,
    bench_insert_by_size,
    bench_insert_by_fpr,
    bench_insert_by_type,
    bench_insert_by_load,
    bench_insert_by_pattern,
    bench_insert_throughput,
    bench_insert_cold_vs_warm,
    bench_insert_hash_quality,
);

criterion_main!(benches);
