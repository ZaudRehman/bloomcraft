//! Single-item query (contains) operation benchmarks
//!
//! Comprehensive performance analysis of StandardBloomFilter query operations.
//! Query performance is THE critical metric for Bloom filters because:
//!
//! 1. **Queries are 10-100x more frequent than inserts** in typical workloads
//! 2. **Query latency directly impacts application response time**
//! 3. **False positives waste downstream resources** (database lookups, etc.)
//!
//! # Test Scenarios
//!
//! 1. **By Size**: Query latency vs filter capacity
//!    - Tests: 1K, 10K, 100K, 1M items
//!    - Expected: O(1) with cache effects
//!
//! 2. **Hit vs Miss**: Query present vs absent items
//!    - Hit: Item is in filter (true positive)
//!    - Miss: Item not in filter (true negative or false positive)
//!    - Miss queries can short-circuit early if any bit is 0
//!
//! 3. **By False Positive Rate**: Query time vs FPR
//!    - Lower FPR = more hash functions = slower queries
//!    - But miss queries may short-circuit earlier
//!
//! 4. **By Load Factor**: Query performance vs filter saturation
//!    - Empty filter: Fast misses (bits are 0)
//!    - Full filter: Slower misses (more false positives)
//!
//! 5. **By Access Pattern**: Sequential, random, Zipfian, adversarial
//!
//! 6. **Cache Effects**: Cold vs warm cache performance
//!
//! 7. **False Positive Measurement**: Actual FPR vs theoretical
//!
//! # Baseline Performance Targets
//!
//! - Small filters (1K): ~50ns per query (~20M ops/s)
//! - Medium filters (100K): ~70ns per query (~14M ops/s)
//! - Large filters (1M): ~100ns per query (~10M ops/s)
//! - Query hit: ~10-20% slower than miss (all bits checked)
//! - Query miss: ~10-20% faster (early termination on first zero bit)

use bloomcraft::core::BloomFilter;
use bloomcraft::filters::StandardBloomFilter;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::collections::HashSet;

mod common;
use common::*;

// BENCHMARK 1: Query Performance by Filter Size

/// Benchmark query latency across different filter sizes
///
/// Tests whether query operations remain O(1) as filter size grows.
/// Expected results:
/// - Algorithmic complexity: O(1) - fixed number of hash functions
/// - Practical performance: Slight degradation due to cache misses
/// - L1 cache (~32KB): Fast
/// - L2 cache (~256KB): Moderate
/// - L3 cache (~8MB): Slower
/// - RAM (>8MB): Much slower
fn bench_query_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_by_size");
    
    for size in SIZES {
        let items = generate_strings(*size, 32);
        let mut filter = StandardBloomFilter::<String>::new(*size, 0.01);
        
        // Fill filter to 50%
        for i in 0..(*size / 2) {
            filter.insert(&items[i]);
        }
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                let mut idx = 0;
                
                b.iter(|| {
                    let result = filter.contains(black_box(&items[idx % items.len()]));
                    idx += 1;
                    result
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 2: Query Hit vs Miss Performance

/// Compare query performance for items present vs absent in filter
///
/// Key insight: Miss queries can short-circuit early!
/// - Hit query: Must check all k hash functions (all bits must be 1)
/// - Miss query: Can return false on first zero bit (early termination)
///
/// At low load factors, misses are much faster.
/// At high load factors, performance converges (most bits are 1).
fn bench_query_hit_vs_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_hit_vs_miss");
    
    let size = 100_000;
    let fpr = 0.01;
    
    // Create disjoint sets (guaranteed no overlap)
    let (present_items, absent_items) = create_disjoint_sets(size / 2, size / 2, 32);
    
    let mut filter = StandardBloomFilter::<String>::new(size, fpr);
    
    // Insert only present_items
    for item in &present_items {
        filter.insert(item);
    }
    
    group.throughput(Throughput::Elements(1));
    
    // Query present items (all bits are 1 → must check all k hashes)
    group.bench_function("hit", |b| {
        let mut idx = 0;
        
        b.iter(|| {
            let result = filter.contains(black_box(&present_items[idx % present_items.len()]));
            idx += 1;
            result
        });
    });
    
    // Query absent items (likely has zero bit → early termination possible)
    group.bench_function("miss", |b| {
        let mut idx = 0;
        
        b.iter(|| {
            let result = filter.contains(black_box(&absent_items[idx % absent_items.len()]));
            idx += 1;
            result
        });
    });
    
    group.finish();
}

// BENCHMARK 3: Query Performance by False Positive Rate

/// Benchmark query latency for different target FPRs
///
/// Lower FPR = more hash functions = more bit checks
/// - 10% FPR: k ≈ 3-4 hashes → ~3-4 bit lookups
/// - 1% FPR: k ≈ 7 hashes → ~7 bit lookups
/// - 0.1% FPR: k ≈ 10 hashes → ~10 bit lookups
/// - 0.01% FPR: k ≈ 13 hashes → ~13 bit lookups
///
/// Query time should scale linearly with k.
fn bench_query_by_fpr(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_by_fpr");
    
    let size = 100_000;
    let items = generate_strings(size, 32);
    
    for fpr in FP_RATES {
        let mut filter = StandardBloomFilter::<String>::new(size, *fpr);
        
        // Fill to 50%
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", fpr)),
            fpr,
            |b, _| {
                let mut idx = 0;
                
                b.iter(|| {
                    let result = filter.contains(black_box(&items[idx % items.len()]));
                    idx += 1;
                    result
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 4: Query Performance by Load Factor

/// Benchmark query performance at different filter saturation levels
///
/// Critical insight: Load factor dramatically affects miss query performance!
///
/// - Empty filter (0% load): Miss queries are instant (all bits are 0)
/// - Low load (10-25%): Miss queries are fast (likely to hit zero bit early)
/// - Medium load (50%): Miss queries are moderate
/// - High load (75-90%): Miss queries are slow (most bits are 1, no short-circuit)
/// - Full (99%): Miss queries as slow as hit queries
///
/// Hit query performance is independent of load factor.
fn bench_query_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_by_load");
    
    let capacity = 100_000;
    let fpr = 0.01;
    let items = generate_strings(capacity, 32);
    
    for load_pct in LOAD_FACTORS {
        let mut filter = StandardBloomFilter::<String>::new(capacity, fpr);
        
        // Fill to target load factor
        let fill_count = (capacity * load_pct) / 100;
        for i in 0..fill_count {
            filter.insert(&items[i]);
        }
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%", load_pct)),
            load_pct,
            |b, _| {
                let mut idx = 0;
                
                b.iter(|| {
                    let result = filter.contains(black_box(&items[idx % items.len()]));
                    idx += 1;
                    result
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 5: Query Performance by Item Type

/// Benchmark query latency for different data types
///
/// Tests how hashing overhead varies with item size:
/// - u64 (8 bytes): Fast hashing, no allocation
/// - UUID (16 bytes): Fast hashing, no allocation
/// - String (32 bytes): Moderate hashing, heap-allocated
/// - String (256 bytes): Slower hashing
/// - String (1024 bytes): Much slower hashing
fn bench_query_by_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_by_type");
    
    let size = 100_000;
    let fpr = 0.01;
    
    group.throughput(Throughput::Elements(1));
    
    // u64 (8 bytes, stack-allocated)
    group.bench_function("u64", |b| {
        let mut filter = StandardBloomFilter::<u64>::new(size, fpr);
        let items = generate_u64s(size);
        
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        
        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            result
        });
    });
    
    // UUID (16 bytes, stack-allocated)
    group.bench_function("uuid_16", |b| {
        let mut filter = StandardBloomFilter::<[u8; 16]>::new(size, fpr);
        let uuids = generate_uuids(size);
        
        for i in 0..(size / 2) {
            filter.insert(&uuids[i]);
        }
        
        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&uuids[idx % uuids.len()]));
            idx += 1;
            result
        });
    });
    
    // String (32 bytes, heap-allocated)
    group.bench_function("string_32", |b| {
        let mut filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_strings(size, 32);
        
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        
        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            result
        });
    });
    
    // String (256 bytes, heap-allocated)
    group.bench_function("string_256", |b| {
        let mut filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_strings(size, 256);
        
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        
        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            result
        });
    });
    
    // String (1024 bytes, cache-unfriendly)
    group.bench_function("string_1024", |b| {
        let mut filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_strings(size, 1024);
        
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        
        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            result
        });
    });
    
    // URLs (realistic web crawler workload)
    group.bench_function("url", |b| {
        let mut filter = StandardBloomFilter::<String>::new(size, fpr);
        let urls = generate_urls(size);
        
        for i in 0..(size / 2) {
            filter.insert(&urls[i]);
        }
        
        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&urls[idx % urls.len()]));
            idx += 1;
            result
        });
    });
    
    group.finish();
}

// BENCHMARK 6: Query Performance by Access Pattern

/// Benchmark query latency for different access patterns
///
/// Real-world workloads exhibit different locality patterns:
/// - Sequential: Best case (cache-friendly, prefetching works)
/// - Random: Average case (typical workload)
/// - Zipfian: Realistic (80/20 rule - hot items cached)
/// - Worst-case: Pathological (cache thrashing)
fn bench_query_by_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_by_pattern");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    let mut filter = StandardBloomFilter::<String>::new(size, fpr);
    for i in 0..(size / 2) {
        filter.insert(&items[i]);
    }
    
    group.throughput(Throughput::Elements(1));
    
    // Sequential pattern (best case for CPU cache and prefetching)
    group.bench_function("sequential", |b| {
        let mut idx = 0;
        
        b.iter(|| {
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            result
        });
    });
    
    // Random pattern (typical real-world workload)
    group.bench_function("random", |b| {
        let pattern = generate_uniform_pattern(items.len(), 10_000);
        let mut idx = 0;
        
        b.iter(|| {
            let result = filter.contains(black_box(&items[pattern[idx % pattern.len()]]));
            idx += 1;
            result
        });
    });
    
    // Zipfian pattern (realistic: some items accessed frequently)
    // Models cache/CDN workloads where 20% of items get 80% of traffic
    group.bench_function("zipfian", |b| {
        let pattern = generate_zipfian_pattern(items.len(), 10_000, 1.5);
        let mut idx = 0;
        
        b.iter(|| {
            let result = filter.contains(black_box(&items[pattern[idx % pattern.len()]]));
            idx += 1;
            result
        });
    });
    
    // Worst-case pattern (cache-thrashing: alternates between distant items)
    group.bench_function("worst_case", |b| {
        let pattern = generate_worst_case_pattern(items.len(), 10_000);
        let mut idx = 0;
        
        b.iter(|| {
            let result = filter.contains(black_box(&items[pattern[idx % pattern.len()]]));
            idx += 1;
            result
        });
    });
    
    group.finish();
}

// BENCHMARK 7: Query Throughput (Sustained Performance)

/// Benchmark sustained query throughput
///
/// Measures operations per second for large query batches.
/// Tests whether performance remains consistent under sustained load.
fn bench_query_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_throughput");
    
    // Test different sizes to see throughput scaling
    for size in &[10_000, 100_000, 1_000_000] {
        let items = generate_strings(*size, 32);
        let mut filter = StandardBloomFilter::<String>::new(*size, 0.01);
        
        // Fill to 50%
        for i in 0..(*size / 2) {
            filter.insert(&items[i]);
        }
        
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_items", size)),
            size,
            |b, _| {
                let mut idx = 0;
                
                b.iter(|| {
                    // Query 1000 items per iteration
                    for _ in 0..1000 {
                        black_box(filter.contains(&items[idx % items.len()]));
                        idx += 1;
                    }
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 8: Cold vs Warm Cache Performance

/// Compare query performance with cold vs warm CPU cache
///
/// Cold cache: Filter accessed for first time (realistic for large datasets)
/// Warm cache: Filter already in CPU cache (realistic for small/hot filters)
///
/// This shows the impact of cache locality on real-world performance.
fn bench_query_cold_vs_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_cold_vs_warm");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    group.throughput(Throughput::Elements(1));
    
    // Warm cache: Filter pre-warmed with queries
    group.bench_function("warm_cache", |b| {
        let mut filter = StandardBloomFilter::<String>::new(size, fpr);
        
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        
        // Warm up the cache
        for i in 0..1000 {
            filter.contains(&items[i]);
        }
        
        let mut idx = 1000;
        b.iter(|| {
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            result
        });
    });
    
    // Cold cache: New filter for each iteration
    group.bench_function("cold_cache", |b| {
        let mut idx = 0;
        
        b.iter(|| {
            // Create fresh filter (cold cache)
            let mut filter = StandardBloomFilter::<String>::new(size, fpr);
            for i in 0..(size / 2) {
                filter.insert(&items[i]);
            }
            
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            
            // Force filter to stay alive
            black_box(filter);
            result
        });
    });
    
    group.finish();
}

// BENCHMARK 9: False Positive Rate Measurement

/// Measure actual false positive rate vs theoretical
///
/// This benchmark verifies that the filter achieves its target FPR
/// and measures the performance cost of false positives.
fn bench_false_positive_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("false_positive_rate");
    group.sample_size(10); // Fewer samples (statistical test)
    
    let size = 100_000;
    let items = generate_strings(size * 2, 32); // Generate 2x items
    
    for fpr in FP_RATES {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("target_{}", fpr)),
            fpr,
            |b, &fpr| {
                b.iter(|| {
                    let mut filter = StandardBloomFilter::<String>::new(size, fpr);
                    
                    // Insert first half
                    for i in 0..size {
                        filter.insert(&items[i]);
                    }
                    
                    // Query second half (true negatives)
                    let mut false_positives = 0;
                    for i in size..(size + 10_000) {
                        if filter.contains(&items[i]) {
                            false_positives += 1;
                        }
                    }
                    
                    let actual_fpr = false_positives as f64 / 10_000.0;
                    black_box(actual_fpr);
                });
            },
        );
    }
    
    group.finish();
}

// BENCHMARK 10: Query with Different Hash Qualities

/// Benchmark query with items that have different hash characteristics
///
/// - High entropy: Random strings (ideal distribution)
/// - Low entropy: Sequential numbers (potential hash collisions)
/// - Common prefixes: Strings with shared prefixes (worst case for poor hashes)
fn bench_query_hash_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_hash_quality");
    
    let size = 100_000;
    let fpr = 0.01;
    
    group.throughput(Throughput::Elements(1));
    
    // High entropy (random strings)
    group.bench_function("high_entropy", |b| {
        let mut filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_strings(size, 32);
        
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        
        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            result
        });
    });
    
    // Low entropy (sequential numbers)
    group.bench_function("low_entropy", |b| {
        let mut filter = StandardBloomFilter::<u64>::new(size, fpr);
        let items = generate_sequential_u64s(size);
        
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        
        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            result
        });
    });
    
    // Common prefixes (worst case for poor hash functions)
    group.bench_function("common_prefix", |b| {
        let mut filter = StandardBloomFilter::<String>::new(size, fpr);
        let items = generate_prefixed_strings(size);
        
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        
        let mut idx = 0;
        b.iter(|| {
            let result = filter.contains(black_box(&items[idx % items.len()]));
            idx += 1;
            result
        });
    });
    
    group.finish();
}

// BENCHMARK 11: Query Hit Rate Impact

/// Benchmark query performance with different hit rates
///
/// Simulates real-world scenarios:
/// - 0% hit rate: Cache miss scenario (all queries go to backend)
/// - 25% hit rate: Moderate filtering
/// - 50% hit rate: Balanced workload
/// - 75% hit rate: High cache hit scenario
/// - 100% hit rate: All items present (best case for filter value)
fn bench_query_hit_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_hit_rate");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    let hit_rates = vec![0, 25, 50, 75, 100];
    
    for hit_rate in hit_rates {
        let mut filter = StandardBloomFilter::<String>::new(size, fpr);
        
        // Insert items according to hit rate
        let insert_count = (size * hit_rate) / 100;
        for i in 0..insert_count {
            filter.insert(&items[i]);
        }
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%_hit", hit_rate)),
            &hit_rate,
            |b, _| {
                let mut idx = 0;
                
                b.iter(|| {
                    let result = filter.contains(black_box(&items[idx % items.len()]));
                    idx += 1;
                    result
                });
            },
        );
    }
    
    group.finish();
}

// CRITERION CONFIGURATION

criterion_group!(
    benches,
    bench_query_by_size,
    bench_query_hit_vs_miss,
    bench_query_by_fpr,
    bench_query_by_load,
    bench_query_by_type,
    bench_query_by_pattern,
    bench_query_throughput,
    bench_query_cold_vs_warm,
    bench_false_positive_rate,
    bench_query_hash_quality,
    bench_query_hit_rate,
);

criterion_main!(benches);
