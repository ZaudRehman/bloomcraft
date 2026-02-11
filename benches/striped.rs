//! Comprehensive Criterion benchmark suite for StripedBloomFilter
//!
//! # Benchmark Categories
//!
//! 1. **Core Operations** - Single-threaded insert/query throughput
//! 2. **Scaling Analysis** - Performance vs. filter size, item count, FPR
//! 3. **Concurrent Workloads** - Multi-threaded scaling (1-64 threads)
//! 4. **Real-World Scenarios** - Application-specific patterns
//! 5. **Stripe Count Impact** - Finding optimal stripe configuration
//! 6. **Memory & Cache Effects** - Working set size analysis
//! 7. **Contention Patterns** - Hotspot detection and load distribution
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Run all benchmarks
//! cargo bench --bench striped --features metrics
//!
//! # Run specific category
//! cargo bench --bench striped -- "concurrent"
//!
//! # Save baseline for regression detection
//! cargo bench --bench striped -- --save-baseline master
//!
//! # Compare against baseline
//! cargo bench --bench striped -- --baseline master
//! ```

use bloomcraft::core::SharedBloomFilter;
use bloomcraft::sync::StripedBloomFilter;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, BatchSize,
};
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ============================================================================
// Configuration Constants
// ============================================================================

/// Benchmark timeout to prevent hangs
const BENCH_TIMEOUT: Duration = Duration::from_secs(60);

/// Measurement time for stable results
const MEASUREMENT_TIME: Duration = Duration::from_secs(10);

/// Warm-up time
const WARMUP_TIME: Duration = Duration::from_secs(3);

/// Sample sizes for statistical significance
const SAMPLE_SIZE: usize = 100;

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate deterministic test data
fn generate_test_data(count: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count).map(|_| rng.gen()).collect()
}

/// Generate random strings (simulates real-world keys)
fn generate_string_data(count: usize, len: usize, seed: u64) -> Vec<String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            (&mut rng)
                .sample_iter(&Alphanumeric)
                .take(len)
                .map(char::from)
                .collect()
        })
        .collect()
}

/// Generate skewed workload (Zipf distribution simulation)
fn generate_zipf_data(count: usize, cardinality: usize, seed: u64) -> Vec<u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(count);
    
    // Simple Zipf approximation: 80% of accesses to 20% of keys
    let hot_keys = cardinality / 5;
    
    for _ in 0..count {
        if rng.gen::<f64>() < 0.8 {
            // Hot key (top 20%)
            data.push(rng.gen_range(0..hot_keys) as u64);
        } else {
            // Cold key (bottom 80%)
            data.push(rng.gen_range(hot_keys..cardinality) as u64);
        }
    }
    
    data
}

// ============================================================================
// 1. Core Operations - Single Threaded
// ============================================================================

fn bench_core_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/insert");
    group.measurement_time(MEASUREMENT_TIME);
    group.warm_up_time(WARMUP_TIME);
    
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_test_data(size, 0);
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("u64", size),
            &data,
            |b, data| {
                b.iter_batched(
                    || StripedBloomFilter::<u64>::new(size, 0.01).unwrap(),
                    |filter| {
                        for item in data {
                            filter.insert(black_box(item));
                        }
                        filter
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

fn bench_core_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/query");
    group.measurement_time(MEASUREMENT_TIME);
    
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        let data = generate_test_data(size, 0);
        let filter = {
            let f = StripedBloomFilter::<u64>::new(size, 0.01).unwrap();
            for item in &data {
                f.insert(item);
            }
            f
        };
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("hit", size),
            &data,
            |b, data| {
                b.iter(|| {
                    for item in data {
                        black_box(filter.contains(black_box(item)));
                    }
                });
            },
        );
        
        // Query absent keys (miss scenario)
        let absent_data = generate_test_data(size, 999);
        group.bench_with_input(
            BenchmarkId::new("miss", size),
            &absent_data,
            |b, data| {
                b.iter(|| {
                    for item in data {
                        black_box(filter.contains(black_box(item)));
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_core_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/mixed");
    group.measurement_time(MEASUREMENT_TIME);
    
    for size in [10_000, 100_000] {
        let data = generate_test_data(size, 0);
        let filter = Arc::new(StripedBloomFilter::<u64>::new(size, 0.01).unwrap());
        
        // Pre-populate 50%
        for item in data.iter().take(size / 2) {
            filter.insert(item);
        }
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("50-50_read-write", size),
            &data,
            |b, data| {
                b.iter(|| {
                    for (i, item) in data.iter().enumerate() {
                        if i % 2 == 0 {
                            filter.insert(black_box(item));
                        } else {
                            black_box(filter.contains(black_box(item)));
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// 2. Concurrent Workloads - Real Threading Patterns
// ============================================================================

fn bench_concurrent_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent/scaling");
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(50); // Fewer samples for concurrent benchmarks
    
    let filter_size = 1_000_000;
    let ops_per_thread = 100_000;
    
    for thread_count in [1, 2, 4, 8, 16, 32, 64] {
        let total_ops = thread_count * ops_per_thread;
        
        // Insert-only workload
        group.throughput(Throughput::Elements(total_ops as u64));
        group.bench_with_input(
            BenchmarkId::new("insert", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_batched(
                    || {
                        let filter = Arc::new(
                            StripedBloomFilter::<u64>::with_concurrency(
                                filter_size,
                                0.01,
                                threads,
                            )
                            .unwrap(),
                        );
                        (filter, threads)
                    },
                    |(filter, threads)| {
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                thread::spawn(move || {
                                    let data = generate_test_data(ops_per_thread, tid as u64);
                                    for item in data {
                                        f.insert(&item);
                                    }
                                })
                            })
                            .collect();
                        
                        for h in handles {
                            h.join().unwrap();
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        
        // Query-only workload (read-heavy)
        let filter = {
            let f = Arc::new(
                StripedBloomFilter::<u64>::with_concurrency(filter_size, 0.01, thread_count)
                    .unwrap(),
            );
            // Pre-populate
            let data = generate_test_data(filter_size / 2, 0);
            for item in data {
                f.insert(&item);
            }
            f
        };
        
        group.bench_with_input(
            BenchmarkId::new("query", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..threads)
                        .map(|tid| {
                            let f = Arc::clone(&filter);
                            thread::spawn(move || {
                                let data = generate_test_data(ops_per_thread, tid as u64);
                                for item in data {
                                    black_box(f.contains(&item));
                                }
                            })
                        })
                        .collect();
                    
                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
        
        // Mixed workload (90% read, 10% write) - typical cache pattern
        group.bench_with_input(
            BenchmarkId::new("mixed_90-10", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_batched(
                    || {
                        let f = Arc::new(
                            StripedBloomFilter::<u64>::with_concurrency(
                                filter_size,
                                0.01,
                                threads,
                            )
                            .unwrap(),
                        );
                        // Pre-populate
                        let data = generate_test_data(filter_size / 4, 0);
                        for item in data {
                            f.insert(&item);
                        }
                        f
                    },
                    |filter| {
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                thread::spawn(move || {
                                    let mut rng = ChaCha8Rng::seed_from_u64(tid as u64);
                                    for _ in 0..ops_per_thread {
                                        let item = rng.gen::<u64>();
                                        if rng.gen::<f64>() < 0.9 {
                                            // 90% reads
                                            black_box(f.contains(&item));
                                        } else {
                                            // 10% writes
                                            f.insert(&item);
                                        }
                                    }
                                })
                            })
                            .collect();
                        
                        for h in handles {
                            h.join().unwrap();
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// 3. Stripe Count Impact Analysis
// ============================================================================

fn bench_stripe_count_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("stripes/count_impact");
    group.measurement_time(MEASUREMENT_TIME);
    
    let filter_size = 100_000;
    let ops_per_thread = 10_000;
    let thread_count = 16;
    
    for stripe_count in [16, 64, 256, 512, 1024, 2048, 4096] {
        group.throughput(Throughput::Elements((thread_count * ops_per_thread) as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent_insert", stripe_count),
            &stripe_count,
            |b, &stripes| {
                b.iter_batched(
                    || {
                        Arc::new(
                            StripedBloomFilter::<u64>::with_stripe_count(
                                filter_size,
                                0.01,
                                stripes,
                            )
                            .unwrap(),
                        )
                    },
                    |filter| {
                        let handles: Vec<_> = (0..thread_count)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                thread::spawn(move || {
                                    let data = generate_test_data(ops_per_thread, tid as u64);
                                    for item in data {
                                        f.insert(&item);
                                    }
                                })
                            })
                            .collect();
                        
                        for h in handles {
                            h.join().unwrap();
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

// ============================================================================
// 4. Real-World Scenarios
// ============================================================================

/// Scenario: Web request deduplication (URL bloom filter)
fn bench_scenario_url_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("scenario/url_dedup");
    group.measurement_time(MEASUREMENT_TIME);
    
    // Simulate 1M URLs with 50-byte average length
    let url_count = 100_000;
    let urls = generate_string_data(url_count, 50, 42);
    let filter = Arc::new(StripedBloomFilter::<String>::new(url_count, 0.01).unwrap());
    
    // Pre-populate 70% (already seen URLs)
    for url in urls.iter().take(url_count * 7 / 10) {
        filter.insert(url);
    }
    
    group.throughput(Throughput::Elements(url_count as u64));
    group.bench_function("check_and_insert", |b| {
        b.iter(|| {
            for url in &urls {
                if !filter.contains(url) {
                    filter.insert(url);
                }
            }
        });
    });
    
    group.finish();
}

/// Scenario: Database query cache (high hit rate)
fn bench_scenario_db_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("scenario/db_cache");
    group.measurement_time(MEASUREMENT_TIME);
    
    let cache_size = 1_000_000;
    let query_count = 100_000;
    
    // Simulate skewed access pattern (realistic cache behavior)
    let queries = generate_zipf_data(query_count, cache_size, 123);
    let filter = Arc::new(StripedBloomFilter::<u64>::new(cache_size, 0.001).unwrap());
    
    // Warm cache with hot keys
    for item in queries.iter().take(query_count / 5) {
        filter.insert(item);
    }
    
    group.throughput(Throughput::Elements(query_count as u64));
    group.bench_function("zipf_queries", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(filter.contains(query));
            }
        });
    });
    
    group.finish();
}

/// Scenario: Distributed log deduplication
fn bench_scenario_log_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("scenario/log_dedup");
    group.measurement_time(MEASUREMENT_TIME);
    
    let log_size = 1_000_000;
    let workers = 8;
    let logs_per_worker = log_size / workers;
    
    group.throughput(Throughput::Elements(log_size as u64));
    group.bench_function("concurrent_dedup", |b| {
        b.iter_batched(
            || {
                Arc::new(
                    StripedBloomFilter::<String>::with_concurrency(log_size, 0.01, workers)
                        .unwrap(),
                )
            },
            |filter| {
                let handles: Vec<_> = (0..workers)
                    .map(|worker_id| {
                        let f = Arc::clone(&filter);
                        thread::spawn(move || {
                            // Simulate log entries (JSON-like strings)
                            let logs = generate_string_data(logs_per_worker, 100, worker_id as u64);
                            for log in logs {
                                if !f.contains(&log) {
                                    f.insert(&log);
                                }
                            }
                        })
                    })
                    .collect();
                
                for h in handles {
                    h.join().unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Scenario: Network packet deduplication (high throughput)
fn bench_scenario_packet_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("scenario/packet_dedup");
    group.measurement_time(Duration::from_secs(5)); // Shorter for high-throughput
    
    let packet_count = 10_000_000;
    let batch_size = 1000;
    
    // Simulate packet hashes (5-tuple: src_ip, dst_ip, src_port, dst_port, protocol)
    let packets = generate_test_data(packet_count, 777);
    let filter = Arc::new(
        StripedBloomFilter::<u64>::with_concurrency(packet_count, 0.01, 4).unwrap(),
    );
    
    group.throughput(Throughput::Elements(packet_count as u64));
    group.bench_function("streaming_dedup", |b| {
        b.iter(|| {
            for chunk in packets.chunks(batch_size) {
                for packet in chunk {
                    if !filter.contains(packet) {
                        filter.insert(packet);
                    }
                }
            }
        });
    });
    
    group.finish();
}

// ============================================================================
// 5. Parameter Sensitivity Analysis
// ============================================================================

fn bench_fpr_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("params/fpr_impact");
    group.measurement_time(MEASUREMENT_TIME);
    
    let item_count = 100_000;
    let data = generate_test_data(item_count, 0);
    
    for fpr in [0.1, 0.01, 0.001, 0.0001] {
        let filter = StripedBloomFilter::<u64>::new(item_count, fpr).unwrap();
        
        // Insert phase
        for item in &data {
            filter.insert(item);
        }
        
        group.throughput(Throughput::Elements(item_count as u64));
        group.bench_with_input(
            BenchmarkId::new("query", format!("{}", fpr)),
            &data,
            |b, data| {
                b.iter(|| {
                    for item in data {
                        black_box(filter.contains(item));
                    }
                });
            },
        );
        
        // Report actual characteristics
        println!(
            "\nFPR {}: k={}, m={}, memory={} KB",
            fpr,
            filter.hash_count(),
            filter.bit_count(),
            filter.memory_usage() / 1024
        );
    }
    
    group.finish();
}

fn bench_load_factor_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("params/load_factor");
    group.measurement_time(MEASUREMENT_TIME);
    
    let filter_capacity = 100_000;
    let query_count = 10_000;
    let query_data = generate_test_data(query_count, 999);
    
    for load_pct in [10, 25, 50, 75, 90, 95, 99] {
        let insert_count = filter_capacity * load_pct / 100;
        let insert_data = generate_test_data(insert_count, 0);
        
        let filter = StripedBloomFilter::<u64>::new(filter_capacity, 0.01).unwrap();
        for item in &insert_data {
            filter.insert(item);
        }
        
        group.throughput(Throughput::Elements(query_count as u64));
        group.bench_with_input(
            BenchmarkId::new("query", format!("{}%", load_pct)),
            &query_data,
            |b, data| {
                b.iter(|| {
                    for item in data {
                        black_box(filter.contains(item));
                    }
                });
            },
        );
        
        println!(
            "\nLoad {}%: actual_load={:.2}%, fpr={:.6}",
            load_pct,
            filter.load_factor() * 100.0,
            filter.false_positive_rate()
        );
    }
    
    group.finish();
}

// ============================================================================
// 6. Clear Operation (Edge Case Testing)
// ============================================================================

fn bench_clear_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations/clear");
    group.measurement_time(MEASUREMENT_TIME);
    
    for size in [10_000, 100_000, 1_000_000] {
        for stripe_count in [64, 256, 1024] {
            group.bench_with_input(
                BenchmarkId::new(format!("stripes_{}", stripe_count), size),
                &(size, stripe_count),
                |b, &(sz, stripes)| {
                    b.iter_batched(
                        || {
                            let f = StripedBloomFilter::<u64>::with_stripe_count(
                                sz, 0.01, stripes,
                            )
                            .unwrap();
                            // Fill to 50%
                            let data = generate_test_data(sz / 2, 0);
                            for item in data {
                                f.insert(&item);
                            }
                            f
                        },
                        |filter| {
                            black_box(filter.clear());
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    
    group.finish();
}

// ============================================================================
// 7. Metrics Collection Performance (when enabled)
// ============================================================================

#[cfg(feature = "metrics")]
fn bench_metrics_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics/overhead");
    group.measurement_time(MEASUREMENT_TIME);
    
    let size = 100_000;
    let ops = 10_000;
    let data = generate_test_data(ops, 0);
    
    let filter = Arc::new(
        StripedBloomFilter::<u64>::with_concurrency(size, 0.01, 16).unwrap(),
    );
    
    // Pre-populate
    for item in data.iter().take(ops / 2) {
        filter.insert(item);
    }
    
    group.throughput(Throughput::Elements(ops as u64));
    group.bench_function("concurrent_with_metrics", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|tid| {
                    let f = Arc::clone(&filter);
                    let d = data.clone();
                    thread::spawn(move || {
                        for (i, item) in d.iter().enumerate() {
                            if (tid + i) % 2 == 0 {
                                f.insert(item);
                            } else {
                                black_box(f.contains(item));
                            }
                        }
                    })
                })
                .collect();
            
            for h in handles {
                h.join().unwrap();
            }
            
            // Access metrics to ensure they're being collected
            let stats = filter.stripe_stats();
            black_box(stats);
        });
    });
    
    group.finish();
}

// ============================================================================
// 8. Memory Usage & Allocation Pattern
// ============================================================================

fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory/allocation");
    group.sample_size(20); // Fewer samples for memory-intensive benchmarks
    
    for size in [10_000, 100_000, 1_000_000, 10_000_000] {
        group.bench_with_input(
            BenchmarkId::new("construction", size),
            &size,
            |b, &sz| {
                b.iter(|| {
                    let filter = StripedBloomFilter::<u64>::new(sz, 0.01).unwrap();
                    black_box(filter);
                });
            },
        );
        
        // Report memory usage
        let filter = StripedBloomFilter::<u64>::new(size, 0.01).unwrap();
        println!(
            "\nFilter capacity {}: memory={} MB, bits={}, bytes_per_item={:.2}",
            size,
            filter.memory_usage() / 1_048_576,
            filter.bit_count(),
            filter.memory_usage() as f64 / size as f64
        );
    }
    
    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group! {
    name = core_ops;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARMUP_TIME)
        .sample_size(SAMPLE_SIZE);
    targets =
        bench_core_insert,
        bench_core_query,
        bench_core_mixed_workload,
}

criterion_group! {
    name = concurrent;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARMUP_TIME)
        .sample_size(50); // Fewer samples for concurrent
    targets =
        bench_concurrent_scaling,
        bench_stripe_count_impact,
}

criterion_group! {
    name = scenarios;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARMUP_TIME);
    targets =
        bench_scenario_url_dedup,
        bench_scenario_db_cache,
        bench_scenario_log_dedup,
        bench_scenario_packet_dedup,
}

criterion_group! {
    name = parameters;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARMUP_TIME);
    targets =
        bench_fpr_impact,
        bench_load_factor_impact,
}

criterion_group! {
    name = operations;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARMUP_TIME);
    targets =
        bench_clear_operation,
        bench_memory_patterns,
}

#[cfg(feature = "metrics")]
criterion_group! {
    name = metrics;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARMUP_TIME);
    targets = bench_metrics_overhead;
}

// Main entry point
#[cfg(feature = "metrics")]
criterion_main!(core_ops, concurrent, scenarios, parameters, operations, metrics);

#[cfg(not(feature = "metrics"))]
criterion_main!(core_ops, concurrent, scenarios, parameters, operations);
