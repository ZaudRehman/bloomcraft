//! EXTREME Bloom Filter Benchmark Suite - God Mode Edition
//!
//! This benchmark suite is designed to TORTURE-TEST every filter implementation
//! across every conceivable dimension. NO MERCY.
//!
//! ## What This Tests
//!
//! ### 1. CORE OPERATIONS (All Filters)
//! - Single insert/query performance across 4 size scales
//! - Batch operations (10/100/1K/10K elements)
//! - Hit rate sensitivity (0%/50%/100%)
//! - Memory usage and overhead
//!
//! ### 2. FILTER-SPECIFIC TORTURE TESTS
//! - **Standard**: Raw speed baseline, union/intersect, fill patterns
//! - **Counting**: Deletion safety, overflow behavior, counter saturation
//! - **Scalable**: Growth patterns, filter accumulation, query degradation
//! - **Partitioned**: Cache alignment impact, partition distribution
//! - **Tree**: Depth scaling, branching factor impact, location tracking
//!
//! ### 3. PARAMETER SWEEPS (Systematic Torture)
//! - False positive rates: 0.001, 0.01, 0.05, 0.1, 0.2
//! - Load factors: 10%, 25%, 50%, 75%, 90%, 99%, 150% (oversaturation)
//! - Hash function counts: 1, 3, 7, 14, 21, 28
//! - Filter sizes: 1K, 10K, 100K, 1M, 10M
//!
//! ### 4. STRESS TESTS (Breaking Points)
//! - Extreme saturation (10× capacity)
//! - Concurrent insertion (1/2/4/8/16 threads)
//! - Pathological hash collisions
//! - Counter overflow (4-bit/8-bit limits)
//! - Growth limits (64+ filters in scalable)
//!
//! ### 5. CORRECTNESS VALIDATION (Under Load)
//! - False positive rate accuracy (must be ≤ 2× target)
//! - Zero false negatives guarantee
//! - Deletion safety (no corruption)
//! - Set operation correctness
//!
//! ## Performance Targets
//!
//! | Operation | Standard | Counting | Scalable | Partitioned | Tree |
//! |-----------|----------|----------|----------|-------------|------|
//! | Insert (ns) | <150 | <200 | <250 | <150 | <300 |
//! | Query (ns) | <100 | <150 | <200 | <60 | <250 |
//! | Delete (ns) | N/A | <250 | N/A | N/A | N/A |
//! | Batch 1K (µs) | <100 | <150 | <200 | <90 | <250 |
//! | Memory (bits/item) | <12 | <96 | <15 | <14 | <varies> |

use bloomcraft::core::BloomFilter;
use bloomcraft::filters::{
    CountingBloomFilter, PartitionedBloomFilter, ScalableBloomFilter,
    StandardBloomFilter, TreeBloomFilter, GrowthStrategy,
};
use bloomcraft::hash::StdHasher;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::thread;

// ============================================================================
// UTILITIES
// ============================================================================

fn generate_strings(count: usize, len: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("{:0width$}", i, width = len))
        .collect()
}

fn generate_u64s(count: usize) -> Vec<u64> {
    (0..count as u64).collect()
}

// ============================================================================
// 1. CORE OPERATIONS - INSERT
// ============================================================================

fn bench_insert_single_torture(c: &mut Criterion) {
    let mut group = c.benchmark_group("01_insert_single_extreme");
    
    // Test across 5 orders of magnitude
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let fpr = 0.01;
        let items = generate_strings(size.min(100_000), 32);
        
        group.throughput(Throughput::Elements(1));
        
        // Standard (Baseline)
        group.bench_with_input(BenchmarkId::new("standard", size), &size, |b, _| {
            let filter = StandardBloomFilter::<String>::new(size, fpr);
            let mut idx = 0;
            b.iter(|| {
                filter.insert(black_box(&items[idx % items.len()]));
                idx += 1;
            });
        });
        
        // Counting (8× memory overhead)
        group.bench_with_input(BenchmarkId::new("counting", size), &size, |b, _| {
            let mut filter = CountingBloomFilter::<String>::new(size, fpr);
            let mut idx = 0;
            b.iter(|| {
                filter.insert(black_box(&items[idx % items.len()]));
                idx += 1;
            });
        });
        
        // Scalable (Dynamic growth)
        group.bench_with_input(BenchmarkId::new("scalable", size), &size, |b, _| {
            let mut filter = ScalableBloomFilter::<String>::new(size, fpr);
            let mut idx = 0;
            b.iter(|| {
                filter.insert(black_box(&items[idx % items.len()]));
                idx += 1;
            });
        });
        
        // Partitioned (Cache-optimized) - FIXED: unwrap Result
        group.bench_with_input(BenchmarkId::new("partitioned", size), &size, |b, _| {
            let mut filter = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
            let mut idx = 0;
            b.iter(|| {
                filter.insert(black_box(&items[idx % items.len()]));
                idx += 1;
            });
        });
        
        // Tree (Hierarchical organization)
        group.bench_with_input(BenchmarkId::new("tree", size), &size, |b, _| {
            let mut filter = TreeBloomFilter::<String>::new(vec![1], size, fpr);
            let mut idx = 0;
            b.iter(|| {
                let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), &[0]);
                idx += 1;
            });
        });
    }
    
    group.finish();
}

fn bench_insert_batch_torture(c: &mut Criterion) {
    let mut group = c.benchmark_group("02_insert_batch_extreme");
    
    let filter_size = 100_000;
    let fpr = 0.01;
    
    // Batch sizes from tiny to massive
    for batch_size in [1, 10, 100, 1_000, 10_000] {
        let items = generate_strings(filter_size, 32);
        group.throughput(Throughput::Elements(batch_size as u64));
        
        group.bench_with_input(BenchmarkId::new("standard", batch_size), &batch_size, |b, _| {
            let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
            let mut idx = 0;
            b.iter(|| {
                let batch: Vec<String> = (0..batch_size)
                    .map(|_| {
                        let item = items[idx % items.len()].clone();
                        idx += 1;
                        item
                    })
                    .collect();
                black_box(filter.insert_batch(&batch));
            });
        });
        
        group.bench_with_input(BenchmarkId::new("counting", batch_size), &batch_size, |b, _| {
            let mut filter = CountingBloomFilter::<String>::new(filter_size, fpr);
            let mut idx = 0;
            b.iter(|| {
                let batch: Vec<String> = (0..batch_size)
                    .map(|_| {
                        let item = items[idx % items.len()].clone();
                        idx += 1;
                        item
                    })
                    .collect();
                black_box(filter.insert_batch(&batch));
            });
        });
        
        // FIXED: unwrap Result
        group.bench_with_input(BenchmarkId::new("partitioned", batch_size), &batch_size, |b, _| {
            let mut filter = PartitionedBloomFilter::<String>::new(filter_size, fpr).unwrap();
            let mut idx = 0;
            b.iter(|| {
                let batch: Vec<String> = (0..batch_size)
                    .map(|_| {
                        let item = items[idx % items.len()].clone();
                        idx += 1;
                        item
                    })
                    .collect();
                black_box(filter.insert_batch(&batch));
            });
        });
    }
    
    group.finish();
}

// ============================================================================
// 2. CORE OPERATIONS - QUERY
// ============================================================================

fn bench_query_hit_rate_torture(c: &mut Criterion) {
    let mut group = c.benchmark_group("03_query_hit_rate_extreme");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size * 2, 32);
    
    // Test all hit rates: 0%, 25%, 50%, 75%, 100%
    for hit_rate_pct in [0, 25, 50, 75, 100] {
        group.throughput(Throughput::Elements(1));
        
        // Standard
        group.bench_with_input(
            BenchmarkId::new("standard", hit_rate_pct),
            &hit_rate_pct,
            |b, _| {
                let filter = StandardBloomFilter::<String>::new(size, fpr);
                for i in 0..size {
                    filter.insert(&items[i]);
                }
                let mut idx = 0;
                b.iter(|| {
                    let query_idx = if (idx % 100) < hit_rate_pct {
                        idx % size // Hit
                    } else {
                        size + (idx % size) // Miss
                    };
                    let result = filter.contains(black_box(&items[query_idx]));
                    idx += 1;
                    result
                });
            },
        );
        
        // FIXED: unwrap Result
        group.bench_with_input(
            BenchmarkId::new("partitioned", hit_rate_pct),
            &hit_rate_pct,
            |b, _| {
                let mut filter = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
                for i in 0..size {
                    filter.insert(&items[i]);
                }
                let mut idx = 0;
                b.iter(|| {
                    let query_idx = if (idx % 100) < hit_rate_pct {
                        idx % size
                    } else {
                        size + (idx % size)
                    };
                    let result = filter.contains(black_box(&items[query_idx]));
                    idx += 1;
                    result
                });
            },
        );
        
        // Counting
        group.bench_with_input(
            BenchmarkId::new("counting", hit_rate_pct),
            &hit_rate_pct,
            |b, _| {
                let mut filter = CountingBloomFilter::<String>::new(size, fpr);
                for i in 0..size {
                    filter.insert(&items[i]);
                }
                let mut idx = 0;
                b.iter(|| {
                    let query_idx = if (idx % 100) < hit_rate_pct {
                        idx % size
                    } else {
                        size + (idx % size)
                    };
                    let result = filter.contains(black_box(&items[query_idx]));
                    idx += 1;
                    result
                });
            },
        );
        
        // Scalable
        group.bench_with_input(
            BenchmarkId::new("scalable", hit_rate_pct),
            &hit_rate_pct,
            |b, _| {
                let mut filter = ScalableBloomFilter::<String>::new(size, fpr);
                for i in 0..size {
                    filter.insert(&items[i]);
                }
                let mut idx = 0;
                b.iter(|| {
                    let query_idx = if (idx % 100) < hit_rate_pct {
                        idx % size
                    } else {
                        size + (idx % size)
                    };
                    let result = filter.contains(black_box(&items[query_idx]));
                    idx += 1;
                    result
                });
            },
        );
    }
    
    group.finish();
}

fn bench_query_batch_torture(c: &mut Criterion) {
    let mut group = c.benchmark_group("04_query_batch_extreme");
    
    let filter_size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(filter_size, 32);
    
    for batch_size in [1, 10, 100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(batch_size as u64));
        
        group.bench_with_input(BenchmarkId::new("standard", batch_size), &batch_size, |b, _| {
            let filter = StandardBloomFilter::<String>::new(filter_size, fpr);
            for i in 0..(filter_size / 2) {
                filter.insert(&items[i]);
            }
            b.iter(|| {
                let batch: Vec<String> = items.iter().take(batch_size).cloned().collect();
                let results = black_box(filter.contains_batch(&batch));
                black_box(results)
            });
        });
        
        // FIXED: unwrap Result
        group.bench_with_input(BenchmarkId::new("partitioned", batch_size), &batch_size, |b, _| {
            let mut filter = PartitionedBloomFilter::<String>::new(filter_size, fpr).unwrap();
            for i in 0..(filter_size / 2) {
                filter.insert(&items[i]);
            }
            b.iter(|| {
                let batch: Vec<String> = items.iter().take(batch_size).cloned().collect();
                let results = black_box(filter.contains_batch(&batch));
                black_box(results)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("counting", batch_size), &batch_size, |b, _| {
            let mut filter = CountingBloomFilter::<String>::new(filter_size, fpr);
            for i in 0..(filter_size / 2) {
                filter.insert(&items[i]);
            }
            b.iter(|| {
                let batch: Vec<String> = items.iter().take(batch_size).cloned().collect();
                let results = black_box(filter.contains_batch(&batch));
                black_box(results)
            });
        });
    }
    
    group.finish();
}

// ============================================================================
// 3. COUNTING FILTER SPECIFIC - DELETION TORTURE
// ============================================================================

fn bench_counting_delete_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("05_counting_delete_torture");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    group.throughput(Throughput::Elements(1));
    
    // Safe delete (with existence check)
    group.bench_function("delete_safe", |b| {
        let mut filter = CountingBloomFilter::<String>::new(size, fpr);
        for item in &items {
            filter.insert(item);
        }
        let mut idx = 0;
        b.iter(|| {
            let result = filter.delete(black_box(&items[idx % items.len()]));
            filter.insert(&items[idx % items.len()]);
            idx += 1;
            result
        });
    });
    
    // Unchecked delete (skip existence check)
    group.bench_function("delete_unchecked", |b| {
        let mut filter = CountingBloomFilter::<String>::new(size, fpr);
        for item in &items {
            filter.insert(item);
        }
        let mut idx = 0;
        b.iter(|| {
            let result = filter.delete_unchecked(black_box(&items[idx % items.len()]));
            filter.insert(&items[idx % items.len()]);
            idx += 1;
            result
        });
    });
    
    // Batch delete (100 items)
    group.bench_function("delete_batch_100", |b| {
        let mut filter = CountingBloomFilter::<String>::new(size, fpr);
        for item in &items {
            filter.insert(item);
        }
        b.iter(|| {
            let batch: Vec<String> = items.iter().take(100).cloned().collect();
            let count = black_box(filter.delete_batch(&batch));
            filter.insert_batch(&batch);
            count
        });
    });
    
    // Batch delete unchecked (100 items)
    group.bench_function("delete_batch_unchecked_100", |b| {
        let mut filter = CountingBloomFilter::<String>::new(size, fpr);
        for item in &items {
            filter.insert(item);
        }
        b.iter(|| {
            let batch: Vec<String> = items.iter().take(100).cloned().collect();
            let count = black_box(filter.delete_batch_unchecked(&batch));
            filter.insert_batch(&batch);
            count
        });
    });
    
    group.finish();
}

fn bench_counting_overflow_torture(c: &mut Criterion) {
    let mut group = c.benchmark_group("06_counting_overflow_extreme");
    group.sample_size(10);
    
    // 4-bit counters (max 15) - WILL overflow
    group.bench_function("4bit_saturate_20_inserts", |b| {
        b.iter(|| {
            let mut filter = CountingBloomFilter::<String>::with_counter_size(1000, 0.01, 4);
            let item = String::from("overflow_test");
            for _ in 0..20 {
                filter.insert(&item);
            }
            black_box((filter.has_overflowed(), filter.overflow_count()))
        });
    });
    
    // 4-bit counters - EXTREME overflow (100 inserts)
    group.bench_function("4bit_extreme_100_inserts", |b| {
        b.iter(|| {
            let mut filter = CountingBloomFilter::<String>::with_counter_size(1000, 0.01, 4);
            let item = String::from("overflow_test");
            for _ in 0..100 {
                filter.insert(&item);
            }
            black_box((filter.has_overflowed(), filter.overflow_count()))
        });
    });
    
    // 8-bit counters (max 255) - should NOT overflow at 20
    group.bench_function("8bit_no_overflow_20_inserts", |b| {
        b.iter(|| {
            let mut filter = CountingBloomFilter::<String>::with_counter_size(1000, 0.01, 8);
            let item = String::from("overflow_test");
            for _ in 0..20 {
                filter.insert(&item);
            }
            black_box((filter.has_overflowed(), filter.overflow_count()))
        });
    });
    
    // 8-bit counters - push to limit (300 inserts, WILL overflow)
    group.bench_function("8bit_saturate_300_inserts", |b| {
        b.iter(|| {
            let mut filter = CountingBloomFilter::<String>::with_counter_size(1000, 0.01, 8);
            let item = String::from("overflow_test");
            for _ in 0..300 {
                filter.insert(&item);
            }
            black_box((filter.has_overflowed(), filter.overflow_count()))
        });
    });
    
    group.finish();
}

fn bench_counting_health_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("07_counting_health_metrics");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_u64s(size);
    
    // Health metrics cost
    group.bench_function("health_metrics_full", |b| {
        let mut filter = CountingBloomFilter::<u64>::new(size, fpr);
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        b.iter(|| {
            let metrics = filter.health_metrics();
            black_box(metrics)
        });
    });
    
    // Counter histogram cost
    group.bench_function("counter_histogram", |b| {
        let mut filter = CountingBloomFilter::<u64>::new(size, fpr);
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        b.iter(|| {
            let histogram = filter.counter_histogram();
            black_box(histogram)
        });
    });
    
    // Individual metrics
    group.bench_function("fill_rate", |b| {
        let mut filter = CountingBloomFilter::<u64>::new(size, fpr);
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        b.iter(|| black_box(filter.fill_rate()));
    });
    
    group.bench_function("estimate_fpr", |b| {
        let mut filter = CountingBloomFilter::<u64>::new(size, fpr);
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        b.iter(|| black_box(filter.estimate_fpr()));
    });
    
    group.finish();
}

// ============================================================================
// 4. SCALABLE FILTER SPECIFIC - GROWTH TORTURE
// ============================================================================

fn bench_scalable_growth_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("08_scalable_growth_extreme");
    group.sample_size(10);
    
    let initial_capacity = 1000;
    let fpr = 0.01;
    let insert_count = 100_000; // 100× initial capacity
    let items = generate_u64s(insert_count);
    
    // Geometric 2× (doubles each time)
    group.bench_function("geometric_2x_100x_growth", |b| {
        b.iter(|| {
            let mut filter = ScalableBloomFilter::<u64>::with_strategy(
                initial_capacity,
                fpr,
                0.5,
                GrowthStrategy::Geometric(2.0),
            );
            for item in &items {
                filter.insert(item);
            }
            black_box(filter.filter_count())
        });
    });
    
    // Geometric 3× (triples each time)
    group.bench_function("geometric_3x_100x_growth", |b| {
        b.iter(|| {
            let mut filter = ScalableBloomFilter::<u64>::with_strategy(
                initial_capacity,
                fpr,
                0.5,
                GrowthStrategy::Geometric(3.0),
            );
            for item in &items {
                filter.insert(item);
            }
            black_box(filter.filter_count())
        });
    });
    
    // Constant growth (same size each time)
    group.bench_function("constant_100x_growth", |b| {
        b.iter(|| {
            let mut filter = ScalableBloomFilter::<u64>::with_strategy(
                initial_capacity,
                fpr,
                0.5,
                GrowthStrategy::Constant,
            );
            for item in &items {
                filter.insert(item);
            }
            black_box(filter.filter_count())
        });
    });
    
    group.finish();
}

fn bench_scalable_query_degradation(c: &mut Criterion) {
    let mut group = c.benchmark_group("09_scalable_query_degradation");
    
    let fpr = 0.01;
    
    // Test query performance as number of internal filters grows
    for num_filters in [1, 2, 4, 8, 16, 32, 64] {
        let items_per_filter = 1000;
        let total_items = num_filters * items_per_filter;
        let items = generate_u64s(total_items);
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_filters),
            &num_filters,
            |b, _| {
                let mut filter = ScalableBloomFilter::<u64>::new(items_per_filter, fpr);
                for item in &items {
                    filter.insert(item);
                }
                
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

// ============================================================================
// 5. TREE FILTER SPECIFIC - HIERARCHY TORTURE
// ============================================================================

fn bench_tree_depth_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("10_tree_depth_extreme");
    
    let capacity_per_bin = 1000;
    let fpr = 0.01;
    let items = generate_strings(10_000, 32);
    
    group.throughput(Throughput::Elements(1));
    
    // Depth 1 (flat)
    group.bench_function("depth_1_flat", |b| {
        let mut filter = TreeBloomFilter::<String>::new(vec![1], capacity_per_bin, fpr);
        let mut idx = 0;
        b.iter(|| {
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), &[0]);
            idx += 1;
        });
    });
    
    // Depth 2
    group.bench_function("depth_2", |b| {
        let mut filter = TreeBloomFilter::<String>::new(vec![4, 4], capacity_per_bin, fpr);
        let mut idx = 0;
        b.iter(|| {
            let bin_path = &[idx % 4, (idx / 4) % 4];
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), bin_path);
            idx += 1;
        });
    });
    
    // Depth 3
    group.bench_function("depth_3", |b| {
        let mut filter = TreeBloomFilter::<String>::new(vec![2, 4, 8], capacity_per_bin, fpr);
        let mut idx = 0;
        b.iter(|| {
            let bin_path = &[idx % 2, (idx / 2) % 4, (idx / 8) % 8];
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), bin_path);
            idx += 1;
        });
    });
    
    // Depth 4
    group.bench_function("depth_4", |b| {
        let mut filter = TreeBloomFilter::<String>::new(vec![2, 2, 2, 2], capacity_per_bin, fpr);
        let mut idx = 0;
        b.iter(|| {
            let bin_path = &[idx % 2, (idx / 2) % 2, (idx / 4) % 2, (idx / 8) % 2];
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), bin_path);
            idx += 1;
        });
    });
    
    // Depth 5
    group.bench_function("depth_5", |b| {
        let mut filter = TreeBloomFilter::<String>::new(vec![2, 2, 2, 2, 2], capacity_per_bin, fpr);
        let mut idx = 0;
        b.iter(|| {
            let bin_path = &[
                idx % 2,
                (idx / 2) % 2,
                (idx / 4) % 2,
                (idx / 8) % 2,
                (idx / 16) % 2,
            ];
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), bin_path);
            idx += 1;
        });
    });
    
    // Depth 10 (EXTREME)
    group.bench_function("depth_10_extreme", |b| {
        let mut filter = TreeBloomFilter::<String>::new(
            vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            capacity_per_bin,
            fpr,
        );
        let mut idx = 0;
        b.iter(|| {
            let bin_path = &[
                idx % 2,
                (idx / 2) % 2,
                (idx / 4) % 2,
                (idx / 8) % 2,
                (idx / 16) % 2,
                (idx / 32) % 2,
                (idx / 64) % 2,
                (idx / 128) % 2,
                (idx / 256) % 2,
                (idx / 512) % 2,
            ];
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), bin_path);
            idx += 1;
        });
    });
    
    group.finish();
}

fn bench_tree_branching_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("11_tree_branching_extreme");
    
    let capacity_per_bin = 1000;
    let fpr = 0.01;
    let items = generate_strings(10_000, 32);
    
    group.throughput(Throughput::Elements(1));
    
    // Wide (100 bins, depth 1)
    group.bench_function("wide_100", |b| {
        let mut filter = TreeBloomFilter::<String>::new(vec![100], capacity_per_bin, fpr);
        let mut idx = 0;
        b.iter(|| {
            let bin_path = &[idx % 100];
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), bin_path);
            idx += 1;
        });
    });
    
    // EXTREME wide (1000 bins, depth 1)
    group.bench_function("extreme_wide_1000", |b| {
        let mut filter = TreeBloomFilter::<String>::new(vec![1000], capacity_per_bin, fpr);
        let mut idx = 0;
        b.iter(|| {
            let bin_path = &[idx % 1000];
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), bin_path);
            idx += 1;
        });
    });
    
    // Narrow deep (2^6 = 64 bins, depth 6)
    group.bench_function("narrow_deep_2x6", |b| {
        let mut filter = TreeBloomFilter::<String>::new(vec![2, 2, 2, 2, 2, 2], capacity_per_bin, fpr);
        let mut idx = 0;
        b.iter(|| {
            let bin_path = &[
                idx % 2,
                (idx / 2) % 2,
                (idx / 4) % 2,
                (idx / 8) % 2,
                (idx / 16) % 2,
                (idx / 32) % 2,
            ];
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), bin_path);
            idx += 1;
        });
    });
    
    // Balanced (10×10 = 100 bins, depth 2)
    group.bench_function("balanced_10x10", |b| {
        let mut filter = TreeBloomFilter::<String>::new(vec![10, 10], capacity_per_bin, fpr);
        let mut idx = 0;
        b.iter(|| {
            let bin_path = &[idx % 10, (idx / 10) % 10];
            let _ = filter.insert_to_bin(black_box(&items[idx % items.len()]), bin_path);
            idx += 1;
        });
    });
    
    group.finish();
}

// ============================================================================
// 6. PARTITIONED FILTER SPECIFIC - CACHE TORTURE
// ============================================================================

fn bench_partitioned_alignment_torture(c: &mut Criterion) {
    let mut group = c.benchmark_group("12_partitioned_alignment_extreme");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    group.throughput(Throughput::Elements(1));
    
    // No explicit alignment (default) - FIXED: unwrap
    group.bench_function("no_alignment", |b| {
        let mut filter = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
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
    
    // 64-byte alignment (L1 cache line) - FIXED: unwrap
    group.bench_function("64_byte_l1", |b| {
        let mut filter = PartitionedBloomFilter::<String>::with_alignment(size, fpr, 64).unwrap();
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
    
    // 128-byte alignment - FIXED: unwrap
    group.bench_function("128_byte_l2", |b| {
        let mut filter = PartitionedBloomFilter::<String>::with_alignment(size, fpr, 128).unwrap();
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
    
    // 256-byte alignment - FIXED: unwrap
    group.bench_function("256_byte_paranoid", |b| {
        let mut filter = PartitionedBloomFilter::<String>::with_alignment(size, fpr, 256).unwrap();
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

// ============================================================================
// 7. SET OPERATIONS - UNION/INTERSECT
// ============================================================================

fn bench_set_operations_torture(c: &mut Criterion) {
    let mut group = c.benchmark_group("13_set_operations_extreme");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    // Union - Standard
    group.bench_function("union_standard", |b| {
        let filter1 = StandardBloomFilter::<String>::new(size, fpr);
        let filter2 = StandardBloomFilter::<String>::new(size, fpr);
        for i in 0..(size / 2) {
            filter1.insert(&items[i]);
        }
        for i in (size / 4)..(size * 3 / 4) {
            filter2.insert(&items[i]);
        }
        b.iter(|| {
            let union = filter1.union(&filter2).unwrap();
            black_box(union)
        });
    });
    
    // Union - Partitioned - FIXED: unwrap Result, use mutable union
    group.bench_function("union_partitioned", |b| {
        let mut filter1 = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
        let mut filter2 = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
        for i in 0..(size / 2) {
            filter1.insert(&items[i]);
        }
        for i in (size / 4)..(size * 3 / 4) {
            filter2.insert(&items[i]);
        }
        b.iter(|| {
            let mut filter1_copy = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
            filter1_copy.union(&filter1).unwrap();
            filter1_copy.union(&filter2).unwrap();
            black_box(filter1_copy)
        });
    });
    
    // Intersect - Standard
    group.bench_function("intersect_standard", |b| {
        let filter1 = StandardBloomFilter::<String>::new(size, fpr);
        let filter2 = StandardBloomFilter::<String>::new(size, fpr);
        for i in 0..(size / 2) {
            filter1.insert(&items[i]);
        }
        for i in (size / 4)..(size * 3 / 4) {
            filter2.insert(&items[i]);
        }
        b.iter(|| {
            let intersection = filter1.intersect(&filter2).unwrap();
            black_box(intersection)
        });
    });
    
    // Intersect - Partitioned - FIXED: unwrap Result, use mutable intersect
    group.bench_function("intersect_partitioned", |b| {
        let mut filter1 = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
        let mut filter2 = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
        for i in 0..(size / 2) {
            filter1.insert(&items[i]);
        }
        for i in (size / 4)..(size * 3 / 4) {
            filter2.insert(&items[i]);
        }
        b.iter(|| {
            let mut filter1_copy = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
            for i in 0..(size / 2) {
                filter1_copy.insert(&items[i]);
            }
            filter1_copy.intersect(&filter2).unwrap();
            black_box(filter1_copy)
        });
    });
    
    group.finish();
}

// ============================================================================
// 8. PARAMETER SWEEPS - SYSTEMATIC TORTURE
// ============================================================================

fn bench_fpr_sweep_extreme(c: &mut Criterion) {
    let mut group = c.benchmark_group("14_fpr_sweep_extreme");
    
    let size = 100_000;
    let items = generate_strings(size, 32);
    
    // Test 7 different FPRs from ultra-tight to loose
    for fpr in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4] {
        group.throughput(Throughput::Elements(1));
        
        group.bench_with_input(BenchmarkId::new("insert", fpr), &fpr, |b, &fpr| {
            let filter = StandardBloomFilter::<String>::new(size, fpr);
            let mut idx = 0;
            b.iter(|| {
                filter.insert(black_box(&items[idx % items.len()]));
                idx += 1;
            });
        });
        
        group.bench_with_input(BenchmarkId::new("query", fpr), &fpr, |b, &fpr| {
            let filter = StandardBloomFilter::<String>::new(size, fpr);
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
    }
    
    group.finish();
}

fn bench_load_factor_sweep_extreme(c: &mut Criterion) {
    let mut group = c.benchmark_group("15_load_factor_sweep_extreme");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size * 2, 32); // Allow oversaturation
    
    // Test from 10% to 150% capacity (oversaturation)
    for load_pct in [10, 25, 50, 75, 90, 99, 110, 125, 150] {
        let load = size * load_pct / 100;
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(load_pct), &load_pct, |b, _| {
            let filter = StandardBloomFilter::<String>::new(size, fpr);
            for i in 0..load {
                filter.insert(&items[i]);
            }
            let mut idx = 0;
            b.iter(|| {
                let result = filter.contains(black_box(&items[idx % items.len()]));
                idx += 1;
                result
            });
        });
    }
    
    group.finish();
}

fn bench_hash_count_sweep_extreme(c: &mut Criterion) {
    let mut group = c.benchmark_group("16_hash_count_sweep_extreme");
    
    let m = 1_000_000;
    let items = generate_strings(100_000, 32);
    
    // Test from 1 to 32 hash functions
    for k in [1, 2, 3, 5, 7, 10, 14, 21, 28, 32] {
        group.throughput(Throughput::Elements(1));
        
        group.bench_with_input(BenchmarkId::new("insert", k), &k, |b, &k| {
            let filter = StandardBloomFilter::<String, StdHasher>::with_params(m, k, StdHasher::new());
            let mut idx = 0;
            b.iter(|| {
                filter.insert(black_box(&items[idx % items.len()]));
                idx += 1;
            });
        });
        
        group.bench_with_input(BenchmarkId::new("query", k), &k, |b, &k| {
            let filter = StandardBloomFilter::<String, StdHasher>::with_params(m, k, StdHasher::new());
            for i in 0..50_000 {
                filter.insert(&items[i]);
            }
            let mut idx = 0;
            b.iter(|| {
                let result = filter.contains(black_box(&items[idx % items.len()]));
                idx += 1;
                result
            });
        });
    }
    
    group.finish();
}

// ============================================================================
// 9. STRESS TESTS - BREAKING POINTS
// ============================================================================

fn bench_extreme_saturation(c: &mut Criterion) {
    let mut group = c.benchmark_group("17_extreme_saturation");
    group.sample_size(10);
    
    let size = 10_000;
    let fpr = 0.01;
    let items = generate_strings(size * 20, 32); // 20× capacity
    
    // Insert 10× capacity
    group.bench_function("standard_10x_oversaturate", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<String>::new(size, fpr);
            for i in 0..(size * 10) {
                filter.insert(&items[i]);
            }
            black_box(filter.fill_rate())
        });
    });
    
    // Insert 20× capacity (EXTREME)
    group.bench_function("standard_20x_oversaturate", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<String>::new(size, fpr);
            for i in 0..(size * 20) {
                filter.insert(&items[i]);
            }
            black_box(filter.fill_rate())
        });
    });
    
    // Counting filter saturation (4-bit counters)
    group.bench_function("counting_4bit_extreme", |b| {
        b.iter(|| {
            let mut filter = CountingBloomFilter::<String>::with_counter_size(1000, 0.05, 4);
            let item = String::from("saturate");
            for _ in 0..100 {
                filter.insert(&item);
            }
            black_box(filter.saturated_counter_count())
        });
    });
    
    group.finish();
}

fn bench_concurrent_torture(c: &mut Criterion) {
    let mut group = c.benchmark_group("18_concurrent_extreme");
    group.sample_size(10);
    
    let size = 100_000;
    let fpr = 0.01;
    
    // Test from 1 to 32 threads
    for num_threads in [1, 2, 4, 8, 16, 32] {
        let items_per_thread = 10_000;
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let filter = Arc::new(StandardBloomFilter::<u64>::new(size, fpr));
                    let mut handles = vec![];
                    
                    for t in 0..num_threads {
                        let filter_clone = Arc::clone(&filter);
                        let handle = thread::spawn(move || {
                            let start = t * items_per_thread;
                            for i in start..(start + items_per_thread) {
                                filter_clone.insert(&(i as u64));
                            }
                        });
                        handles.push(handle);
                    }
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    
                    black_box(filter.len())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_pathological_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("19_pathological_extreme");
    group.sample_size(10);
    
    let size = 10_000;
    let fpr = 0.01;
    
    // All collisions (same item repeated)
    group.bench_function("all_collisions_10k", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<u64>::new(size, fpr);
            for _ in 0..size {
                filter.insert(&42u64);
            }
            black_box(filter.count_set_bits())
        });
    });
    
    // Sequential (best case)
    group.bench_function("sequential_10k", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<u64>::new(size, fpr);
            for i in 0..size {
                filter.insert(&(i as u64));
            }
            black_box(filter.count_set_bits())
        });
    });
    
    // Reverse sequential
    group.bench_function("reverse_sequential_10k", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<u64>::new(size, fpr);
            for i in (0..size).rev() {
                filter.insert(&(i as u64));
            }
            black_box(filter.count_set_bits())
        });
    });
    
    group.finish();
}

// ============================================================================
// 10. CORRECTNESS VALIDATION UNDER LOAD
// ============================================================================

fn bench_fpr_accuracy_extreme(c: &mut Criterion) {
    let mut group = c.benchmark_group("20_fpr_accuracy_extreme");
    group.sample_size(10);
    
    let size = 50_000;
    let target_fpr = 0.01;
    let items = generate_u64s(size * 2);
    
    // Standard
    group.bench_function("standard_fpr_check", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<u64>::new(size, target_fpr);
            for i in 0..size {
                filter.insert(&items[i]);
            }
            let mut false_positives = 0;
            for i in size..(size + 10_000) {
                if filter.contains(&items[i]) {
                    false_positives += 1;
                }
            }
            let actual_fpr = false_positives as f64 / 10_000.0;
            assert!(
                actual_fpr <= target_fpr * 2.0,
                "FPR {} exceeds 2× target {}",
                actual_fpr,
                target_fpr
            );
            black_box(actual_fpr)
        });
    });
    
    // Partitioned - FIXED: unwrap
    group.bench_function("partitioned_fpr_check", |b| {
        b.iter(|| {
            let mut filter = PartitionedBloomFilter::<u64>::new(size, target_fpr).unwrap();
            for i in 0..size {
                filter.insert(&items[i]);
            }
            let mut false_positives = 0;
            for i in size..(size + 10_000) {
                if filter.contains(&items[i]) {
                    false_positives += 1;
                }
            }
            let actual_fpr = false_positives as f64 / 10_000.0;
            assert!(
                actual_fpr <= target_fpr * 3.0, // Partitioned has higher FPR
                "FPR {} exceeds 3× target {}",
                actual_fpr,
                target_fpr
            );
            black_box(actual_fpr)
        });
    });
    
    // Counting
    group.bench_function("counting_fpr_check", |b| {
        b.iter(|| {
            let mut filter = CountingBloomFilter::<u64>::new(size, target_fpr);
            for i in 0..size {
                filter.insert(&items[i]);
            }
            let mut false_positives = 0;
            for i in size..(size + 10_000) {
                if filter.contains(&items[i]) {
                    false_positives += 1;
                }
            }
            let actual_fpr = false_positives as f64 / 10_000.0;
            assert!(
                actual_fpr <= target_fpr * 2.0,
                "FPR {} exceeds 2× target {}",
                actual_fpr,
                target_fpr
            );
            black_box(actual_fpr)
        });
    });
    
    group.finish();
}

fn bench_no_false_negatives_extreme(c: &mut Criterion) {
    let mut group = c.benchmark_group("21_no_false_negatives_extreme");
    group.sample_size(10);
    
    let size = 10_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    // Standard
    group.bench_function("standard_no_fn", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<String>::new(size, fpr);
            for item in &items {
                filter.insert(item);
            }
            for item in &items {
                assert!(filter.contains(item), "False negative for item: {}", item);
            }
            black_box(items.len())
        });
    });
    
    // Counting
    group.bench_function("counting_no_fn", |b| {
        b.iter(|| {
            let mut filter = CountingBloomFilter::<String>::new(size, fpr);
            for item in &items {
                filter.insert(item);
            }
            for item in &items {
                assert!(filter.contains(item), "False negative for item: {}", item);
            }
            black_box(items.len())
        });
    });
    
    // Scalable
    group.bench_function("scalable_no_fn", |b| {
        b.iter(|| {
            let mut filter = ScalableBloomFilter::<String>::new(size, fpr);
            for item in &items {
                filter.insert(item);
            }
            for item in &items {
                assert!(filter.contains(item), "False negative for item: {}", item);
            }
            black_box(items.len())
        });
    });
    
    // Partitioned - FIXED: unwrap
    group.bench_function("partitioned_no_fn", |b| {
        b.iter(|| {
            let mut filter = PartitionedBloomFilter::<String>::new(size, fpr).unwrap();
            for item in &items {
                filter.insert(item);
            }
            for item in &items {
                assert!(filter.contains(item), "False negative for item: {}", item);
            }
            black_box(items.len())
        });
    });
    
    group.finish();
}

fn bench_deletion_safety_extreme(c: &mut Criterion) {
    let mut group = c.benchmark_group("22_deletion_safety_extreme");
    group.sample_size(10);
    
    let size = 10_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    
    group.bench_function("counting_delete_safety_full", |b| {
        b.iter(|| {
            let mut filter = CountingBloomFilter::<String>::new(size, fpr);
            
            // Insert all
            for item in &items {
                filter.insert(item);
            }
            
            let initial_len = filter.len();
            
            // Delete first half
            for i in 0..(size / 2) {
                let deleted = filter.delete(&items[i]);
                assert!(
                    deleted,
                    "Failed to delete item that was inserted: {}",
                    items[i]
                );
            }
            
            // Verify len decreased
            let final_len = filter.len();
            assert!(
                final_len < initial_len,
                "Length should decrease after deletions: {} -> {}",
                initial_len,
                final_len
            );
            
            // Verify second half still present (no false negatives)
            for i in (size / 2)..size {
                assert!(
                    filter.contains(&items[i]),
                    "False negative after deletion: {}",
                    items[i]
                );
            }
            
            black_box(filter.len())
        });
    });
    
    group.finish();
}

// ============================================================================
// 11. MEMORY AND DIAGNOSTICS
// ============================================================================

fn bench_memory_usage_all_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("23_memory_usage_all_filters");
    
    let size = 1_000_000;
    let fpr = 0.01;
    
    group.bench_function("standard", |b| {
        let filter = StandardBloomFilter::<u64>::new(size, fpr);
        b.iter(|| black_box(filter.memory_usage()));
    });
    
    group.bench_function("counting", |b| {
        let filter = CountingBloomFilter::<u64>::new(size, fpr);
        b.iter(|| black_box(filter.memory_usage()));
    });
    
    group.bench_function("scalable", |b| {
        let filter = ScalableBloomFilter::<u64>::new(size, fpr);
        b.iter(|| black_box(filter.memory_usage()));
    });
    
    // FIXED: unwrap
    group.bench_function("partitioned", |b| {
        let filter = PartitionedBloomFilter::<u64>::new(size, fpr).unwrap();
        b.iter(|| black_box(filter.memory_usage()));
    });
    
    group.bench_function("tree_10bins", |b| {
        let filter = TreeBloomFilter::<u64>::new(vec![10], size / 10, fpr);
        b.iter(|| black_box(filter.memory_usage()));
    });
    
    group.finish();
}

fn bench_diagnostics_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("24_diagnostics_cost");
    
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_u64s(size);
    
    // Fill rate cost
    group.bench_function("standard_fill_rate", |b| {
        let filter = StandardBloomFilter::<u64>::new(size, fpr);
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        b.iter(|| black_box(filter.fill_rate()));
    });
    
    // FPR estimation cost
    group.bench_function("standard_estimate_fpr", |b| {
        let filter = StandardBloomFilter::<u64>::new(size, fpr);
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        b.iter(|| black_box(filter.estimate_fpr()));
    });
    
    // Scalable filter stats
    group.bench_function("scalable_filter_stats", |b| {
        let mut filter = ScalableBloomFilter::<u64>::new(1000, fpr);
        for item in &items {
            filter.insert(item);
        }
        b.iter(|| black_box(filter.filter_stats()));
    });
    
    // Partitioned partition rates - FIXED: unwrap
    group.bench_function("partitioned_partition_stats", |b| {
        let mut filter = PartitionedBloomFilter::<u64>::new(size, fpr).unwrap();
        for i in 0..(size / 2) {
            filter.insert(&items[i]);
        }
        b.iter(|| black_box(filter.partition_stats()));
    });
    
    group.finish();
}

// ============================================================================
// CRITERION GROUP REGISTRATION
// ============================================================================

criterion_group!(
    benches,
    // Core operations
    bench_insert_single_torture,
    bench_insert_batch_torture,
    bench_query_hit_rate_torture,
    bench_query_batch_torture,
    
    // Filter-specific
    bench_counting_delete_patterns,
    bench_counting_overflow_torture,
    bench_counting_health_metrics,
    bench_scalable_growth_patterns,
    bench_scalable_query_degradation,
    bench_tree_depth_scaling,
    bench_tree_branching_patterns,
    bench_partitioned_alignment_torture,
    
    // Set operations
    bench_set_operations_torture,
    
    // Parameter sweeps
    bench_fpr_sweep_extreme,
    bench_load_factor_sweep_extreme,
    bench_hash_count_sweep_extreme,
    
    // Stress tests
    bench_extreme_saturation,
    bench_concurrent_torture,
    bench_pathological_patterns,
    
    // Correctness validation
    bench_fpr_accuracy_extreme,
    bench_no_false_negatives_extreme,
    bench_deletion_safety_extreme,
    
    // Memory and diagnostics
    bench_memory_usage_all_filters,
    bench_diagnostics_cost,
);

criterion_main!(benches);
