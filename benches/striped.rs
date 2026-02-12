//! ═══════════════════════════════════════════════════════════════════════════
//! STRIPED BLOOM FILTER - COMPLETE PRODUCTION BENCHMARK SUITE
//! ═══════════════════════════════════════════════════════════════════════════
//!
//! **20 Comprehensive Benchmarks** covering every aspect of StripedBloomFilter
//! performance, correctness, scalability, and real-world applicability.
//!
//! ## Benchmark Categories
//!
//! ### Core Operations (3 benchmarks)
//! 1. Single-threaded insert throughput
//! 2. Single-threaded query throughput (hit/miss)
//! 3. Mixed read-write workloads (50/70/90/95% read ratios)
//!
//! ### Concurrent Scaling (3 benchmarks)
//! 4. Thread scaling analysis (1→64 threads, insert-only)
//! 5. Concurrent query scaling (read-heavy workload)
//! 6. Mixed concurrent workloads (various read/write ratios)
//!
//! ### Stripe Configuration (2 benchmarks)
//! 7. Stripe count impact on throughput (16→4096 stripes)
//! 8. Stripe distribution uniformity analysis (requires metrics)
//!
//! ### Batch Operations (1 benchmark)
//! 9. Batch vs individual insert performance
//!
//! ### Real-World Scenarios (5 benchmarks)
//! 10. Web crawler URL deduplication
//! 11. Database query cache with Zipf distribution
//! 12. Distributed log aggregation
//! 13. Network packet deduplication (high throughput)
//! 14. DDoS rate limiter simulation
//!
//! ### Correctness Validation (1 benchmark)
//! 15. Empirical false positive rate measurement
//!
//! ### Comparison (1 benchmark)
//! 16. StripedBloomFilter vs ShardedBloomFilter head-to-head
//!
//! ### Parameter Sensitivity (2 benchmarks)
//! 17. False positive rate impact on performance
//! 18. Load factor degradation analysis
//!
//! ### Latency Analysis (1 benchmark)
//! 19. Operation latency percentiles (P50/P95/P99/P999/Max)
//!
//! ### Stress Testing (1 benchmark)
//! 20. Edge cases: clear, clone, pathological contention
//!
//! ## Running Instructions
//!
//! ```bash
//! # Full suite (~12 minutes)
//! cargo bench --bench striped_complete --features metrics
//!
//! # Without metrics feature
//! cargo bench --bench striped_complete
//!
//! # Specific category
//! cargo bench --bench striped_complete -- "concurrent"
//! cargo bench --bench striped_complete -- "scenario"
//!
//! # Quick smoke test
//! cargo bench --bench striped_complete -- --quick
//!
//! # Save baseline for regression detection
//! cargo bench --bench striped_complete -- --save-baseline master
//!
//! # Compare against baseline
//! cargo bench --bench striped_complete -- --baseline master
//! ```
//!
//! ## Performance Targets
//!
//! | Benchmark                    | Target       | Good         | Acceptable   |
//! |------------------------------|--------------|--------------|--------------|
//! | Single-thread insert         | 10 M ops/s   | 8 M ops/s    | 5 M ops/s    |
//! | Single-thread query          | 15 M ops/s   | 12 M ops/s   | 8 M ops/s    |
//! | 16-thread insert             | 40 M ops/s   | 30 M ops/s   | 20 M ops/s   |
//! | 16-thread query              | 100 M ops/s  | 80 M ops/s   | 50 M ops/s   |
//! | Empirical FPR error          | < 10%        | < 20%        | < 50%        |
//!
//! ═══════════════════════════════════════════════════════════════════════════

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use bloomcraft::core::SharedBloomFilter;
use bloomcraft::sync::{ShardedBloomFilter, StripedBloomFilter};
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::distributions::Alphanumeric;
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

// Configuration
const MEASUREMENT_TIME: Duration = Duration::from_secs(10);
const WARMUP_TIME: Duration = Duration::from_secs(3);
const SAMPLE_SIZE: usize = 100;
const CONCURRENT_SAMPLE_SIZE: usize = 50;
const QUICK_SAMPLE_SIZE: usize = 20;

// ============================================================================
// Test Data Generators (using thread_rng to avoid version conflicts)
// ============================================================================

fn gen_u64_data(count: usize, _seed: u64) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..count).map(|_| rng.gen()).collect()
}

fn gen_string_data(count: usize, len_range: (usize, usize), _seed: u64) -> Vec<String> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let len = rng.gen_range(len_range.0..=len_range.1);
            (0..len).map(|_| rng.sample(Alphanumeric) as char).collect()
        })
        .collect()
}

fn gen_zipf_data(count: usize, cardinality: usize, _seed: u64) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let hot_keys = (cardinality / 5).max(1);
    (0..count)
        .map(|_| {
            if rng.gen::<f64>() < 0.8 {
                rng.gen_range(0..hot_keys) as u64
            } else {
                rng.gen_range(hot_keys..cardinality.max(hot_keys + 1)) as u64
            }
        })
        .collect()
}

fn gen_urls(count: usize, _seed: u64) -> Vec<String> {
    let mut rng = rand::thread_rng();
    let domains = ["example.com", "test.org", "demo.net", "sample.io", "site.dev"];
    let paths = ["page", "article", "post", "item", "resource"];
    
    (0..count)
        .map(|_| {
            format!(
                "https://{}/{}/{}",
                domains[rng.gen_range(0..domains.len())],
                paths[rng.gen_range(0..paths.len())],
                rng.gen::<u32>()
            )
        })
        .collect()
}

fn gen_logs(count: usize, _seed: u64) -> Vec<String> {
    let mut rng = rand::thread_rng();
    let levels = ["INFO", "WARN", "ERROR"];
    let services = ["api", "db", "cache", "queue", "worker"];
    
    (0..count)
        .map(|_| {
            format!(
                "[{}] [{}] req={} user={} ms={}",
                levels[rng.gen_range(0..levels.len())],
                services[rng.gen_range(0..services.len())],
                rng.gen::<u64>(),
                rng.gen_range(1000..10000),
                rng.gen_range(1..1000)
            )
        })
        .collect()
}

// ============================================================================
// 1-3: Core Operations
// ============================================================================

fn bench_01_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("01_core/insert");
    group.measurement_time(MEASUREMENT_TIME).warm_up_time(WARMUP_TIME);

    for size in [10_000, 100_000, 1_000_000] {
        let data = gen_u64_data(size, 0);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter_batched(
                || StripedBloomFilter::<u64>::new(size * 2, 0.01).unwrap(),
                |filter| {
                    for item in data {
                        filter.insert(black_box(item));
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_02_query_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("02_core/query");
    group.measurement_time(MEASUREMENT_TIME);

    for size in [10_000, 100_000, 1_000_000] {
        let present = gen_u64_data(size, 0);
        let absent = gen_u64_data(size, 999);
        
        let filter = {
            let f = StripedBloomFilter::<u64>::new(size * 2, 0.01).unwrap();
            for item in &present {
                f.insert(item);
            }
            f
        };

        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("hit", size), &present, |b, data| {
            b.iter(|| {
                for item in data {
                    black_box(filter.contains(black_box(item)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("miss", size), &absent, |b, data| {
            b.iter(|| {
                for item in data {
                    black_box(filter.contains(black_box(item)));
                }
            });
        });
    }
    group.finish();
}

fn bench_03_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("03_core/mixed");
    group.measurement_time(MEASUREMENT_TIME);

    let size = 100_000;
    let data = gen_u64_data(size, 0);

    for read_pct in [50, 70, 90, 95] {
        let filter = Arc::new(StripedBloomFilter::<u64>::new(size * 2, 0.01).unwrap());
        for item in data.iter().take(size * 3 / 10) {
            filter.insert(item);
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("read_{}%", read_pct), size),
            &data,
            |b, data| {
                let mut rng = rand::thread_rng();
                b.iter(|| {
                    for item in data {
                        if rng.gen::<f64>() < (read_pct as f64 / 100.0) {
                            black_box(filter.contains(black_box(item)));
                        } else {
                            filter.insert(black_box(item));
                        }
                    }
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 4-6: Concurrent Scaling
// ============================================================================

fn bench_04_concurrent_insert_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("04_concurrent/insert_scaling");
    group.measurement_time(MEASUREMENT_TIME).sample_size(CONCURRENT_SAMPLE_SIZE);

    let filter_size = 1_000_000;
    let ops_per_thread = 50_000;

    for threads in [1, 2, 4, 8, 16, 32, 64] {
        let total_ops = threads * ops_per_thread;
        group.throughput(Throughput::Elements(total_ops as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &thread_count| {
                b.iter_batched(
                    || {
                        Arc::new(
                            StripedBloomFilter::<u64>::with_concurrency(
                                filter_size,
                                0.01,
                                thread_count,
                            )
                            .unwrap(),
                        )
                    },
                    |filter| {
                        let barrier = Arc::new(Barrier::new(thread_count));
                        let handles: Vec<_> = (0..thread_count)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let b = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    let data = gen_u64_data(ops_per_thread, tid as u64);
                                    b.wait();
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
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_05_concurrent_query_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("05_concurrent/query_scaling");
    group.measurement_time(MEASUREMENT_TIME).sample_size(CONCURRENT_SAMPLE_SIZE);

    let filter_size = 1_000_000;
    let ops_per_thread = 100_000;

    for threads in [1, 2, 4, 8, 16, 32, 64] {
        let filter = {
            let f = Arc::new(
                StripedBloomFilter::<u64>::with_concurrency(filter_size, 0.01, threads).unwrap(),
            );
            let data = gen_u64_data(filter_size / 2, 0);
            for item in data {
                f.insert(&item);
            }
            f
        };

        let total_ops = threads * ops_per_thread;
        group.throughput(Throughput::Elements(total_ops as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &thread_count| {
                b.iter(|| {
                    let barrier = Arc::new(Barrier::new(thread_count));
                    let handles: Vec<_> = (0..thread_count)
                        .map(|tid| {
                            let f = Arc::clone(&filter);
                            let b = Arc::clone(&barrier);
                            thread::spawn(move || {
                                let data = gen_u64_data(ops_per_thread, tid as u64);
                                b.wait();
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
    }
    group.finish();
}

fn bench_06_concurrent_mixed_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("06_concurrent/mixed_ratios");
    group.measurement_time(MEASUREMENT_TIME).sample_size(CONCURRENT_SAMPLE_SIZE);

    let filter_size = 500_000;
    let ops_per_thread = 25_000;
    let threads = 16;

    for read_pct in [50, 70, 90, 95, 99] {
        group.throughput(Throughput::Elements((threads * ops_per_thread) as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("read_{}%", read_pct), threads),
            &read_pct,
            |b, &read_percentage| {
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
                        let data = gen_u64_data(filter_size / 4, 0);
                        for item in data {
                            f.insert(&item);
                        }
                        f
                    },
                    |filter| {
                        let barrier = Arc::new(Barrier::new(threads));
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let b = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    let mut rng = rand::thread_rng();
                                    b.wait();
                                    for _ in 0..ops_per_thread {
                                        let item = rng.gen::<u64>();
                                        if rng.gen::<f64>() < (read_percentage as f64 / 100.0) {
                                            black_box(f.contains(&item));
                                        } else {
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
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

// ============================================================================
// 7-8: Stripe Configuration
// ============================================================================

fn bench_07_stripe_count_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("07_stripes/count_impact");
    group.measurement_time(MEASUREMENT_TIME).sample_size(CONCURRENT_SAMPLE_SIZE);

    let filter_size = 200_000;
    let ops_per_thread = 10_000;
    let threads = 16;

    for stripe_count in [16, 64, 128, 256, 512, 1024, 2048, 4096] {
        group.throughput(Throughput::Elements((threads * ops_per_thread) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(stripe_count),
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
                        let barrier = Arc::new(Barrier::new(threads));
                        let handles: Vec<_> = (0..threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                let b = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    let data = gen_u64_data(ops_per_thread, tid as u64);
                                    b.wait();
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
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

#[cfg(feature = "metrics")]
fn bench_08_stripe_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("08_stripes/distribution");
    group.sample_size(QUICK_SAMPLE_SIZE);

    let filter = Arc::new(
        StripedBloomFilter::<u64>::with_stripe_count(100_000, 0.01, 256).unwrap(),
    );
    let data = gen_u64_data(50_000, 0);

    for item in &data {
        filter.insert(item);
        filter.contains(item);
    }

    group.bench_function("analyze", |b| {
        b.iter(|| {
            let stats = filter.stripe_stats();
            let total: u64 = stats.iter().map(|s| s.read_ops + s.write_ops).sum();
            let avg = if stats.is_empty() { 0.0 } else { total as f64 / stats.len() as f64 };
            let var: f64 = stats
                .iter()
                .map(|s| {
                    let ops = (s.read_ops + s.write_ops) as f64;
                    let diff = ops - avg;
                    diff * diff
                })
                .sum::<f64>()
                / stats.len().max(1) as f64;
            let std_dev = var.sqrt();
            let coeff_var = if avg > 0.0 { std_dev / avg } else { 0.0 };
            let hot = filter.most_contended_stripes(5);
            black_box((avg, std_dev, coeff_var, hot))
        });
    });
    group.finish();
}

// ============================================================================
// 9: Batch Operations
// ============================================================================

fn bench_09_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("09_batch/operations");
    group.measurement_time(MEASUREMENT_TIME);

    let filter = Arc::new(StripedBloomFilter::<u64>::new(1_000_000, 0.01).unwrap());

    for batch_size in [10, 100, 1_000, 10_000, 100_000] {
        let data = gen_u64_data(batch_size, 0);
        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_insert", batch_size),
            &data,
            |b, data| {
                b.iter(|| {
                    filter.insert_batch(data.iter());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("individual_insert", batch_size),
            &data,
            |b, data| {
                b.iter(|| {
                    for item in data {
                        filter.insert(item);
                    }
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 10-14: Real-World Scenarios
// ============================================================================

fn bench_10_web_crawler(c: &mut Criterion) {
    let mut group = c.benchmark_group("10_scenario/web_crawler");
    group.measurement_time(MEASUREMENT_TIME);

    let url_count = 100_000;
    let urls = gen_urls(url_count, 42);
    let filter = Arc::new(StripedBloomFilter::<String>::new(url_count * 2, 0.01).unwrap());

    for url in urls.iter().take(url_count * 4 / 10) {
        filter.insert(url);
    }

    group.throughput(Throughput::Elements(url_count as u64));
    group.bench_function("check_and_insert", |b| {
        b.iter(|| {
            let mut new_urls = 0;
            for url in &urls {
                if !filter.contains(url) {
                    filter.insert(url);
                    new_urls += 1;
                }
            }
            black_box(new_urls)
        });
    });
    group.finish();
}

fn bench_11_db_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("11_scenario/db_cache");
    group.measurement_time(MEASUREMENT_TIME);

    let capacity = 1_000_000;
    let query_count = 500_000;
    let queries = gen_zipf_data(query_count, capacity, 123);
    let filter = Arc::new(StripedBloomFilter::<u64>::new(capacity, 0.001).unwrap());

    for item in queries.iter().take(query_count / 5) {
        filter.insert(item);
    }

    group.throughput(Throughput::Elements(query_count as u64));
    group.bench_function("zipf_queries", |b| {
        b.iter(|| {
            let mut hits = 0;
            let mut misses = 0;
            for query in &queries {
                if filter.contains(query) {
                    hits += 1;
                } else {
                    misses += 1;
                    filter.insert(query);
                }
            }
            black_box((hits, misses))
        });
    });
    group.finish();
}

fn bench_12_log_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("12_scenario/log_aggregation");
    group.measurement_time(MEASUREMENT_TIME).sample_size(CONCURRENT_SAMPLE_SIZE);

    let log_count = 1_000_000;
    let workers = 8;
    let logs_per_worker = log_count / workers;

    group.throughput(Throughput::Elements(log_count as u64));
    group.bench_function("concurrent_dedup", |b| {
        b.iter_batched(
            || {
                Arc::new(
                    StripedBloomFilter::<String>::with_concurrency(log_count * 2, 0.01, workers)
                        .unwrap(),
                )
            },
            |filter| {
                let barrier = Arc::new(Barrier::new(workers));
                let duplicates = Arc::new(AtomicUsize::new(0));

                let handles: Vec<_> = (0..workers)
                    .map(|worker_id| {
                        let f = Arc::clone(&filter);
                        let b = Arc::clone(&barrier);
                        let dup_count = Arc::clone(&duplicates);
                        thread::spawn(move || {
                            let logs = gen_logs(logs_per_worker, worker_id as u64);
                            b.wait();
                            for log in logs {
                                if f.contains(&log) {
                                    dup_count.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    f.insert(&log);
                                }
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
                black_box(duplicates.load(Ordering::Relaxed))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_13_packet_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("13_scenario/packet_dedup");
    group.measurement_time(Duration::from_secs(5));

    let packet_count = 10_000_000;
    let batch_size = 1_000;
    let packets = gen_u64_data(packet_count, 777);
    let filter = Arc::new(
        StripedBloomFilter::<u64>::with_concurrency(packet_count * 2, 0.01, 4).unwrap(),
    );

    group.throughput(Throughput::Elements(packet_count as u64));
    group.bench_function("streaming_dedup", |b| {
        b.iter(|| {
            let mut dedup_count = 0;
            for chunk in packets.chunks(batch_size) {
                for packet in chunk {
                    if filter.contains(packet) {
                        dedup_count += 1;
                    } else {
                        filter.insert(packet);
                    }
                }
            }
            black_box(dedup_count)
        });
    });
    group.finish();
}

fn bench_14_rate_limiter(c: &mut Criterion) {
    let mut group = c.benchmark_group("14_scenario/rate_limiter");
    group.measurement_time(MEASUREMENT_TIME).sample_size(CONCURRENT_SAMPLE_SIZE);

    let ip_capacity = 10_000_000;
    let request_count = 1_000_000;
    let concurrent_clients = 16;

    group.throughput(Throughput::Elements(request_count as u64));
    group.bench_function("ddos_protection", |b| {
        b.iter_batched(
            || {
                Arc::new(
                    StripedBloomFilter::<u64>::with_concurrency(
                        ip_capacity,
                        0.001,
                        concurrent_clients,
                    )
                    .unwrap(),
                )
            },
            |filter| {
                let barrier = Arc::new(Barrier::new(concurrent_clients));
                let blocked = Arc::new(AtomicUsize::new(0));

                let handles: Vec<_> = (0..concurrent_clients)
                    .map(|tid| {
                        let f = Arc::clone(&filter);
                        let b = Arc::clone(&barrier);
                        let block_count = Arc::clone(&blocked);
                        thread::spawn(move || {
                            let ips = gen_u64_data(request_count / concurrent_clients, tid as u64);
                            b.wait();
                            for ip in ips {
                                if f.contains(&ip) {
                                    block_count.fetch_add(1, Ordering::Relaxed);
                                } else {
                                    f.insert(&ip);
                                }
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
                black_box(blocked.load(Ordering::Relaxed))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ============================================================================
// 15: Empirical FPR Validation
// ============================================================================

fn bench_15_empirical_fpr(c: &mut Criterion) {
    let mut group = c.benchmark_group("15_validation/empirical_fpr");
    group.sample_size(10);

    for target_fpr in [0.1, 0.01, 0.001] {
        let capacity = 100_000;
        let test_size = 100_000;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("fpr_{}", target_fpr)),
            &target_fpr,
            |b, &fpr| {
                b.iter_batched(
                    || {
                        let filter = StripedBloomFilter::<u64>::new(capacity, fpr).unwrap();
                        let inserted = gen_u64_data(capacity, 0);
                        for item in &inserted {
                            filter.insert(item);
                        }
                        (filter, inserted)
                    },
                    |(filter, inserted)| {
                        let mut false_negatives = 0;
                        for item in &inserted {
                            if !filter.contains(item) {
                                false_negatives += 1;
                            }
                        }

                        let absent = gen_u64_data(test_size, 999);
                        let mut false_positives = 0;
                        for item in &absent {
                            if filter.contains(item) {
                                false_positives += 1;
                            }
                        }

                        let empirical_fpr = false_positives as f64 / test_size as f64;
                        let theoretical_fpr = filter.false_positive_rate();
                        let error_pct = if theoretical_fpr > 0.0 {
                            ((empirical_fpr - theoretical_fpr).abs() / theoretical_fpr) * 100.0
                        } else {
                            0.0
                        };

                        assert_eq!(false_negatives, 0, "False negatives detected!");
                        assert!(
                            empirical_fpr < fpr * 3.0,
                            "Empirical FPR {:.6} exceeds 3× target {:.6}",
                            empirical_fpr,
                            fpr
                        );

                        black_box((false_negatives, false_positives, empirical_fpr, error_pct))
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

// ============================================================================
// 16: Striped vs Sharded Comparison
// ============================================================================

fn bench_16_striped_vs_sharded(c: &mut Criterion) {
    let mut group = c.benchmark_group("16_comparison/striped_vs_sharded");
    group.measurement_time(MEASUREMENT_TIME).sample_size(CONCURRENT_SAMPLE_SIZE);

    let size = 500_000;
    let ops_per_thread = 25_000;
    let threads = 16;

    // Striped insert
    group.bench_function("striped/insert", |b| {
        b.iter_batched(
            || {
                Arc::new(
                    StripedBloomFilter::<u64>::with_concurrency(size, 0.01, threads).unwrap(),
                )
            },
            |filter: Arc<StripedBloomFilter<u64>>| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let f = Arc::clone(&filter);
                        let b = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let data = gen_u64_data(ops_per_thread, tid as u64);
                            b.wait();
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
            criterion::BatchSize::SmallInput,
        );
    });

    // Sharded insert (no .unwrap() - ShardedBloomFilter::new returns the filter directly)
    group.bench_function("sharded/insert", |b| {
        b.iter_batched(
            || Arc::new(ShardedBloomFilter::<u64>::new(size, 0.01)),
            |filter: Arc<ShardedBloomFilter<u64>>| {
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|tid| {
                        let f = Arc::clone(&filter);
                        let b = Arc::clone(&barrier);
                        thread::spawn(move || {
                            let data = gen_u64_data(ops_per_thread, tid as u64);
                            b.wait();
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
            criterion::BatchSize::SmallInput,
        );
    });

    // Query comparison
    let striped_filter = {
        let f = Arc::new(
            StripedBloomFilter::<u64>::with_concurrency(size, 0.01, threads).unwrap(),
        );
        let data = gen_u64_data(size / 2, 0);
        for item in data {
            f.insert(&item);
        }
        f
    };

    let sharded_filter = {
        let f = Arc::new(ShardedBloomFilter::<u64>::new(size, 0.01));
        let data = gen_u64_data(size / 2, 0);
        for item in data {
            f.insert(&item);
        }
        f
    };

    group.bench_function("striped/query", |b| {
        b.iter(|| {
            let barrier = Arc::new(Barrier::new(threads));
            let handles: Vec<_> = (0..threads)
                .map(|tid| {
                    let f = Arc::clone(&striped_filter);
                    let b = Arc::clone(&barrier);
                    thread::spawn(move || {
                        let data = gen_u64_data(ops_per_thread, tid as u64);
                        b.wait();
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
    });

    group.bench_function("sharded/query", |b| {
        b.iter(|| {
            let barrier = Arc::new(Barrier::new(threads));
            let handles: Vec<_> = (0..threads)
                .map(|tid| {
                    let f = Arc::clone(&sharded_filter);
                    let b = Arc::clone(&barrier);
                    thread::spawn(move || {
                        let data = gen_u64_data(ops_per_thread, tid as u64);
                        b.wait();
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
    });

    group.finish();
}

// ============================================================================
// 17-18: Parameter Sensitivity
// ============================================================================

fn bench_17_fpr_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("17_params/fpr_impact");
    group.measurement_time(MEASUREMENT_TIME);

    let item_count = 100_000;
    let data = gen_u64_data(item_count, 0);

    for fpr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001] {
        let filter = StripedBloomFilter::<u64>::new(item_count, fpr).unwrap();
        for item in &data {
            filter.insert(item);
        }

        group.throughput(Throughput::Elements(item_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}", fpr)),
            &data,
            |b, data| {
                b.iter(|| {
                    for item in data {
                        black_box(filter.contains(item));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_18_load_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("18_params/load_factor");
    group.measurement_time(MEASUREMENT_TIME);

    let filter_capacity = 100_000;
    let query_count = 10_000;
    let query_data = gen_u64_data(query_count, 999);

    for load_pct in [10, 25, 50, 75, 90, 95, 99] {
        let insert_count = filter_capacity * load_pct / 100;
        let filter = StripedBloomFilter::<u64>::new(filter_capacity, 0.01).unwrap();

        let insert_data = gen_u64_data(insert_count, 0);
        for item in &insert_data {
            filter.insert(item);
        }

        group.throughput(Throughput::Elements(query_count as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{}%", load_pct), query_count),
            &query_data,
            |b, data| {
                b.iter(|| {
                    for item in data {
                        black_box(filter.contains(item));
                    }
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 19: Latency Percentiles
// ============================================================================

fn bench_19_latency_percentiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("19_latency/percentiles");
    group.sample_size(10);

    let filter = Arc::new(StripedBloomFilter::<u64>::new(100_000, 0.01).unwrap());
    let data = gen_u64_data(50_000, 0);

    for item in data.iter().take(25_000) {
        filter.insert(item);
    }

    group.bench_function("insert", |b| {
        b.iter_custom(|iters| {
            let mut latencies = Vec::with_capacity((iters * data.len() as u64) as usize);

            for _ in 0..iters {
                for item in &data {
                    let start = Instant::now();
                    filter.insert(item);
                    latencies.push(start.elapsed());
                }
            }

            latencies.sort_unstable();
            let len = latencies.len();

            if len > 0 {
                let p50 = latencies[len / 2];
                let p95 = latencies[len * 95 / 100];
                let p99 = latencies[len * 99 / 100];
                let p999 = latencies[len * 999 / 1000];
                let max = latencies[len - 1];

                println!(
                    "\n[Insert] P50:{:?} P95:{:?} P99:{:?} P99.9:{:?} Max:{:?}",
                    p50, p95, p99, p999, max
                );
            }

            latencies.iter().sum()
        });
    });

    group.bench_function("query", |b| {
        b.iter_custom(|iters| {
            let mut latencies = Vec::with_capacity((iters * data.len() as u64) as usize);

            for _ in 0..iters {
                for item in &data {
                    let start = Instant::now();
                    black_box(filter.contains(item));
                    latencies.push(start.elapsed());
                }
            }

            latencies.sort_unstable();
            let len = latencies.len();

            if len > 0 {
                let p50 = latencies[len / 2];
                let p95 = latencies[len * 95 / 100];
                let p99 = latencies[len * 99 / 100];
                let p999 = latencies[len * 999 / 1000];
                let max = latencies[len - 1];

                println!(
                    "\n[Query] P50:{:?} P95:{:?} P99:{:?} P99.9:{:?} Max:{:?}",
                    p50, p95, p99, p999, max
                );
            }

            latencies.iter().sum()
        });
    });

    group.finish();
}

// ============================================================================
// 20: Edge Cases & Stress Testing
// ============================================================================

fn bench_20_edge_cases_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("20_stress/edge_cases");
    group.sample_size(QUICK_SAMPLE_SIZE);

    // Clear operation
    for size in [10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::new("clear", size),
            &size,
            |b, &sz| {
                b.iter_batched(
                    || {
                        let f = StripedBloomFilter::<u64>::new(sz, 0.01).unwrap();
                        let data = gen_u64_data(sz / 2, 0);
                        for item in data {
                            f.insert(&item);
                        }
                        f
                    },
                    |filter| {
                        filter.clear();
                        black_box(&filter);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    // Clone operation
    for size in [10_000, 100_000] {
        let filter = {
            let f = StripedBloomFilter::<u64>::new(size, 0.01).unwrap();
            let data = gen_u64_data(size / 2, 0);
            for item in data {
                f.insert(&item);
            }
            f
        };

        group.bench_with_input(
            BenchmarkId::new("clone", size),
            &filter,
            |b, f| {
                b.iter(|| {
                    let cloned = f.clone();
                    black_box(cloned);
                });
            },
        );
    }

    // Pathological contention
    let threads = 16;
    let ops = 1_000;

    group.bench_function("worst_case_contention", |b| {
        b.iter_batched(
            || {
                Arc::new(
                    StripedBloomFilter::<u64>::with_stripe_count(100_000, 0.01, 4).unwrap(),
                )
            },
            |filter| {
                let key = 42u64;
                let barrier = Arc::new(Barrier::new(threads));
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        let f = Arc::clone(&filter);
                        let b = Arc::clone(&barrier);
                        thread::spawn(move || {
                            b.wait();
                            for _ in 0..ops {
                                f.insert(&key);
                                black_box(f.contains(&key));
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

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
        bench_01_insert_throughput,
        bench_02_query_throughput,
        bench_03_mixed_workload,
}

criterion_group! {
    name = concurrent;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARMUP_TIME)
        .sample_size(CONCURRENT_SAMPLE_SIZE);
    targets =
        bench_04_concurrent_insert_scaling,
        bench_05_concurrent_query_scaling,
        bench_06_concurrent_mixed_ratios,
}

criterion_group! {
    name = stripes;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARMUP_TIME);
    targets =
        bench_07_stripe_count_impact,
}

#[cfg(feature = "metrics")]
criterion_group! {
    name = stripes_metrics;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME);
    targets =
        bench_08_stripe_distribution,
}

criterion_group! {
    name = batch;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME);
    targets =
        bench_09_batch_operations,
}

criterion_group! {
    name = scenarios;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .warm_up_time(WARMUP_TIME);
    targets =
        bench_10_web_crawler,
        bench_11_db_cache,
        bench_12_log_aggregation,
        bench_13_packet_dedup,
        bench_14_rate_limiter,
}

criterion_group! {
    name = validation;
    config = Criterion::default().sample_size(10);
    targets =
        bench_15_empirical_fpr,
}

criterion_group! {
    name = comparison;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME)
        .sample_size(CONCURRENT_SAMPLE_SIZE);
    targets =
        bench_16_striped_vs_sharded,
}

criterion_group! {
    name = params;
    config = Criterion::default()
        .measurement_time(MEASUREMENT_TIME);
    targets =
        bench_17_fpr_sensitivity,
        bench_18_load_factor,
}

criterion_group! {
    name = latency;
    config = Criterion::default()
        .sample_size(10);
    targets =
        bench_19_latency_percentiles,
}

criterion_group! {
    name = stress;
    config = Criterion::default()
        .sample_size(QUICK_SAMPLE_SIZE);
    targets =
        bench_20_edge_cases_stress,
}

#[cfg(feature = "metrics")]
criterion_main!(
    core_ops,
    concurrent,
    stripes,
    stripes_metrics,
    batch,
    scenarios,
    validation,
    comparison,
    params,
    latency,
    stress,
);

#[cfg(not(feature = "metrics"))]
criterion_main!(
    core_ops,
    concurrent,
    stripes,
    batch,
    scenarios,
    validation,
    comparison,
    params,
    latency,
    stress,
);
