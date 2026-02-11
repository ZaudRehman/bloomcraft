//! Comprehensive Benchmark Suite for ShardedBloomFilter
//!
//! This benchmark suite provides exhaustive performance validation for the
//! ShardedBloomFilter implementation across multiple dimensions.
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Run all benchmarks
//! cargo bench --bench sharded_bloom_filter
//!
//! # Run specific category
//! cargo bench --bench sharded_bloom_filter core_operations
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, BatchSize,
    BenchmarkId, Criterion, Throughput,
};
use bloomcraft::core::SharedBloomFilter;
use bloomcraft::sync::ShardedBloomFilter;
use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::hash::StdHasher;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::distributions::{Distribution, Alphanumeric};
use rand_distr::Zipf;

// ============================================================================
// Constants and Configuration
// ============================================================================

const SMALL_N: usize = 1_000;
const MEDIUM_N: usize = 100_000;
const LARGE_N: usize = 1_000_000;
const TARGET_FPR: f64 = 0.01;

const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
const SHARD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];
const LOAD_FACTORS: &[f64] = &[0.0, 0.25, 0.50, 0.75, 0.90];
const BATCH_SIZES: &[usize] = &[8, 16, 32, 64, 128, 256, 512, 1024];

// ============================================================================
// Test Data Generators
// ============================================================================

fn generate_sequential_u64(count: usize) -> Vec<u64> {
    (0..count as u64).collect()
}

fn generate_random_u64(count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..count).map(|_| rng.gen()).collect()
}

fn generate_random_strings(count: usize, length: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..count)
        .map(|_| {
            (&mut rng)
                .sample_iter(&Alphanumeric)
                .take(length)
                .map(char::from)
                .collect()
        })
        .collect()
}

fn generate_zipfian_u64(count: usize, unique_count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    let zipf = Zipf::new(unique_count as u64, 1.07).unwrap();
    (0..count)
        .map(|_| zipf.sample(&mut rng) as u64)
        .collect()
}

fn generate_urls(count: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    let domains = vec![
        "example.com",
        "github.com",
        "stackoverflow.com",
        "reddit.com",
        "youtube.com",
    ];

    (0..count)
        .map(|_| {
            let domain = domains[rng.gen_range(0..domains.len())];
            let path_len = rng.gen_range(10..50);
            let path: String = (&mut rng)
                .sample_iter(&Alphanumeric)
                .take(path_len)
                .map(char::from)
                .collect();
            format!("https://{}/{}", domain, path)
        })
        .collect()
}

fn generate_uuids(count: usize) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..count)
        .map(|_| {
            format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                rng.gen::<u32>(),
                rng.gen::<u16>(),
                rng.gen::<u16>(),
                rng.gen::<u16>(),
                rng.gen::<u64>() & 0xFFFFFFFFFFFF
            )
        })
        .collect()
}

// ============================================================================
// Helper Functions
// ============================================================================

fn populate_to_load_factor<T>(filter: &ShardedBloomFilter<T>, items: &[T], load_factor: f64)
where
    T: std::hash::Hash + Sync + Send,
{
    let target_count = (filter.expected_items_configured() as f64 * load_factor) as usize;
    let target_count = target_count.min(items.len());

    for item in &items[..target_count] {
        filter.insert(item);
    }
}

// ============================================================================
// BENCHMARK 1: Core Operations
// ============================================================================

fn bench_core_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_operations");

    group.bench_function("insert_single_u64", |b| {
        let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0u64;

        b.iter(|| {
            filter.insert(black_box(&i));
            i = i.wrapping_add(1);
        });
    });

    group.bench_function("query_single_empty", |b| {
        let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0u64;

        b.iter(|| {
            black_box(filter.contains(black_box(&i)));
            i = i.wrapping_add(1);
        });
    });

    group.bench_function("query_single_full", |b| {
        let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
        for i in 0..MEDIUM_N as u64 {
            filter.insert(&i);
        }

        let mut i = 0u64;
        b.iter(|| {
            black_box(filter.contains(black_box(&i)));
            i = (i + 1) % (MEDIUM_N as u64);
        });
    });

    group.bench_function("clear", |b| {
        b.iter_batched(
            || {
                let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
                for i in 0..MEDIUM_N as u64 {
                    filter.insert(&i);
                }
                filter
            },
            |filter| {
                filter.clear();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 2: Batch Operations
// ============================================================================

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for &batch_size in BATCH_SIZES {
        let items = generate_random_u64(batch_size);

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("insert_batch_chunked", batch_size),
            &items,
            |b, items| {
                let filter = ShardedBloomFilter::<u64>::new(LARGE_N, TARGET_FPR);
                b.iter(|| {
                    filter.insert_batch_chunked(black_box(items));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("insert_individual", batch_size),
            &items,
            |b, items| {
                let filter = ShardedBloomFilter::<u64>::new(LARGE_N, TARGET_FPR);
                b.iter(|| {
                    for item in items {
                        filter.insert(black_box(item));
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 3: Concurrency Insert
// ============================================================================

fn bench_concurrency_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency_insert");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    let items = generate_random_u64(LARGE_N);

    for &thread_count in THREAD_COUNTS {
        group.throughput(Throughput::Elements(LARGE_N as u64));

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &tc| {
                b.iter(|| {
                    let filter = Arc::new(ShardedBloomFilter::<u64>::new(LARGE_N, TARGET_FPR));
                    let chunk_size = LARGE_N / tc;

                    let handles: Vec<_> = (0..tc)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let start = tid * chunk_size;
                            let end = ((tid + 1) * chunk_size).min(LARGE_N);
                            let chunk = items[start..end].to_vec();

                            thread::spawn(move || {
                                for item in chunk {
                                    filter.insert(&item);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 4: Concurrency Query
// ============================================================================

fn bench_concurrency_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency_query");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    let items = generate_random_u64(LARGE_N);
    let filter = Arc::new(ShardedBloomFilter::<u64>::new(LARGE_N, TARGET_FPR));

    for item in &items[..LARGE_N / 2] {
        filter.insert(item);
    }

    for &thread_count in THREAD_COUNTS {
        group.throughput(Throughput::Elements(LARGE_N as u64));

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &tc| {
                b.iter(|| {
                    let chunk_size = LARGE_N / tc;

                    let handles: Vec<_> = (0..tc)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let start = tid * chunk_size;
                            let end = ((tid + 1) * chunk_size).min(LARGE_N);
                            let chunk = items[start..end].to_vec();

                            thread::spawn(move || {
                                let mut hits = 0;
                                for item in chunk {
                                    if filter.contains(&item) {
                                        hits += 1;
                                    }
                                }
                                black_box(hits);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 5: Mixed Workload
// ============================================================================

fn bench_concurrent_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_mixed_workload");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    let items = generate_random_u64(LARGE_N);

    for &thread_count in &[4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("50_50_readwrite", thread_count),
            &thread_count,
            |b, &tc| {
                b.iter(|| {
                    let filter = Arc::new(ShardedBloomFilter::<u64>::new(LARGE_N, TARGET_FPR));
                    let chunk_size = LARGE_N / tc;

                    let handles: Vec<_> = (0..tc)
                        .map(|tid| {
                            let filter = Arc::clone(&filter);
                            let start = tid * chunk_size;
                            let end = ((tid + 1) * chunk_size).min(LARGE_N);
                            let chunk = items[start..end].to_vec();

                            thread::spawn(move || {
                                let mut rng = StdRng::seed_from_u64(tid as u64);
                                for item in chunk {
                                    if rng.gen_bool(0.5) {
                                        filter.insert(&item);
                                    } else {
                                        black_box(filter.contains(&item));
                                    }
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 6: Shard Count Scaling
// ============================================================================

fn bench_shard_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("shard_count_scaling");

    let items = generate_random_u64(MEDIUM_N);

    for &shard_count in SHARD_COUNTS {
        group.bench_with_input(
            BenchmarkId::new("insert_sequential", shard_count),
            &shard_count,
            |b, &sc| {
                let filter = ShardedBloomFilter::<u64>::with_shard_count(MEDIUM_N, TARGET_FPR, sc);
                b.iter(|| {
                    for item in &items {
                        filter.insert(black_box(item));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("query_sequential", shard_count),
            &shard_count,
            |b, &sc| {
                let filter = ShardedBloomFilter::<u64>::with_shard_count(MEDIUM_N, TARGET_FPR, sc);
                for item in &items {
                    filter.insert(item);
                }

                b.iter(|| {
                    for item in &items {
                        black_box(filter.contains(black_box(item)));
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 7: Load Factor Analysis
// ============================================================================

fn bench_load_factor_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_impact");

    let items = generate_random_u64(MEDIUM_N);

    for &load_factor in LOAD_FACTORS {
        group.bench_with_input(
            BenchmarkId::new("query_performance", format!("{:.0}%", load_factor * 100.0)),
            &load_factor,
            |b, &lf| {
                let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
                populate_to_load_factor(&filter, &items, lf);

                let mut i = 0;
                b.iter(|| {
                    black_box(filter.contains(black_box(&items[i % items.len()])));
                    i += 1;
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("insert_performance", format!("{:.0}%", load_factor * 100.0)),
            &load_factor,
            |b, &lf| {
                b.iter_batched(
                    || {
                        let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
                        populate_to_load_factor(&filter, &items, lf);
                        filter
                    },
                    |filter| {
                        filter.insert(black_box(&999999u64));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 8: Data Types
// ============================================================================

fn bench_data_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_types");

    let u64_items = generate_random_u64(MEDIUM_N);
    group.bench_function("insert_u64", |b| {
        let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0;
        b.iter(|| {
            filter.insert(black_box(&u64_items[i % u64_items.len()]));
            i += 1;
        });
    });

    let string32_items = generate_random_strings(SMALL_N, 32);
    group.bench_function("insert_string32", |b| {
        let filter = ShardedBloomFilter::<String>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0;
        b.iter(|| {
            filter.insert(black_box(&string32_items[i % string32_items.len()]));
            i += 1;
        });
    });

    let string256_items = generate_random_strings(SMALL_N, 256);
    group.bench_function("insert_string256", |b| {
        let filter = ShardedBloomFilter::<String>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0;
        b.iter(|| {
            filter.insert(black_box(&string256_items[i % string256_items.len()]));
            i += 1;
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 9: Access Patterns
// ============================================================================

fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_patterns");

    let sequential = generate_sequential_u64(MEDIUM_N);
    let random = generate_random_u64(MEDIUM_N);
    let zipfian = generate_zipfian_u64(MEDIUM_N, MEDIUM_N / 10);

    group.bench_function("sequential_insert", |b| {
        let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0;
        b.iter(|| {
            filter.insert(black_box(&sequential[i % sequential.len()]));
            i += 1;
        });
    });

    group.bench_function("random_insert", |b| {
        let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0;
        b.iter(|| {
            filter.insert(black_box(&random[i % random.len()]));
            i += 1;
        });
    });

    group.bench_function("zipfian_insert", |b| {
        let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0;
        b.iter(|| {
            filter.insert(black_box(&zipfian[i % zipfian.len()]));
            i += 1;
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 10: Real-World Scenarios
// ============================================================================

fn bench_real_world_url_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_scenarios");

    let urls = generate_urls(MEDIUM_N);

    group.bench_function("url_deduplication_crawler", |b| {
        b.iter(|| {
            let filter = ShardedBloomFilter::<String>::new(MEDIUM_N, TARGET_FPR);
            let mut unique_count = 0;

            for url in &urls {
                if !filter.contains(url) {
                    filter.insert(url);
                    unique_count += 1;
                }
            }

            black_box(unique_count);
        });
    });

    let uuids = generate_uuids(MEDIUM_N);

    group.bench_function("session_tracking_api_gateway", |b| {
        b.iter(|| {
            let filter = ShardedBloomFilter::<String>::new(MEDIUM_N, TARGET_FPR);
            let mut active_sessions = 0;

            for uuid in &uuids {
                if !filter.contains(uuid) {
                    filter.insert(uuid);
                    active_sessions += 1;
                }
            }

            black_box(active_sessions);
        });
    });

    group.finish();
}

fn bench_real_world_log_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_log_dedup");

    let mut rng = StdRng::seed_from_u64(42);
    let unique_logs = generate_random_strings(MEDIUM_N / 10, 64);
    let log_stream: Vec<String> = (0..MEDIUM_N)
        .map(|_| {
            if rng.gen_bool(0.7) {
                unique_logs[rng.gen_range(0..unique_logs.len())].clone()
            } else {
                generate_random_strings(1, 64).pop().unwrap()
            }
        })
        .collect();

    group.bench_function("log_event_deduplication", |b| {
        b.iter(|| {
            let filter = ShardedBloomFilter::<String>::new(MEDIUM_N, TARGET_FPR);
            let mut unique_count = 0;

            for log_entry in &log_stream {
                if !filter.contains(log_entry) {
                    filter.insert(log_entry);
                    unique_count += 1;
                }
            }

            black_box(unique_count);
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 11: Comparison with Standard Filter
// ============================================================================

fn bench_vs_standard_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison_vs_standard");

    let items = generate_random_u64(MEDIUM_N);

    group.bench_function("standard_insert", |b| {
        let filter = StandardBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0;
        b.iter(|| {
            filter.insert(black_box(&items[i % items.len()]));
            i += 1;
        });
    });

    group.bench_function("sharded_insert", |b| {
        let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
        let mut i = 0;
        b.iter(|| {
            filter.insert(black_box(&items[i % items.len()]));
            i += 1;
        });
    });

    let items_arc = Arc::new(items.clone());

    group.bench_function("standard_concurrent_4threads", |b| {
        b.iter(|| {
            let filter = Arc::new(std::sync::Mutex::new(
                StandardBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR)
            ));
            let chunk_size = MEDIUM_N / 4;

            let handles: Vec<_> = (0..4)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items_arc);
                    let start = tid * chunk_size;
                    let end = start + chunk_size;

                    thread::spawn(move || {
                        for i in start..end {
                            let filter = filter.lock().unwrap();
                            filter.insert(&items[i]);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.bench_function("sharded_concurrent_4threads", |b| {
        b.iter(|| {
            let filter = Arc::new(ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR));
            let chunk_size = MEDIUM_N / 4;

            let handles: Vec<_> = (0..4)
                .map(|tid| {
                    let filter = Arc::clone(&filter);
                    let items = Arc::clone(&items_arc);
                    let start = tid * chunk_size;
                    let end = start + chunk_size;

                    thread::spawn(move || {
                        for i in start..end {
                            filter.insert(&items[i]);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 12: Memory Efficiency
// ============================================================================

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    for &size in &[1_000, 10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::new("filter_creation", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    let filter = ShardedBloomFilter::<u64>::new(s, TARGET_FPR);
                    black_box(filter.memory_usage());
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 13: Shard Statistics
// ============================================================================

fn bench_shard_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("shard_statistics");

    let items = generate_random_u64(MEDIUM_N);
    let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);

    for item in &items {
        filter.insert(item);
    }

    group.bench_function("get_shard_stats", |b| {
        b.iter(|| {
            black_box(filter.shard_stats());
        });
    });

    group.bench_function("check_imbalance", |b| {
        b.iter(|| {
            black_box(filter.has_imbalanced_shards());
        });
    });

    group.bench_function("calculate_load_factor", |b| {
        b.iter(|| {
            black_box(filter.load_factor());
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 14: Serialization
// ============================================================================

#[cfg(feature = "serde")]
fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    let items = generate_random_u64(MEDIUM_N);
    let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);

    for item in &items {
        filter.insert(item);
    }

    group.bench_function("extract_raw_bits", |b| {
        b.iter(|| {
            let shard_count = filter.shard_count();
            for i in 0..shard_count {
                black_box(filter.shard_raw_bits(i).unwrap());
            }
        });
    });

    group.bench_function("reconstruct_from_bits", |b| {
        let shard_bits: Vec<Vec<u64>> = (0..filter.shard_count())
            .map(|i| filter.shard_raw_bits(i).unwrap())
            .collect();
        let k = filter.hash_count();

        b.iter(|| {
            black_box(
                ShardedBloomFilter::<u64>::from_shard_bits(
                    shard_bits.clone(),
                    k,
                    MEDIUM_N,
                    TARGET_FPR,
                    StdHasher::default(),
                )
                .unwrap()
            );
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 15: FPR Accuracy
// ============================================================================

fn bench_fpr_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("fpr_accuracy");
    group.sample_size(10);

    let inserted = generate_random_u64(MEDIUM_N);
    let test_set = generate_random_u64(MEDIUM_N);

    group.bench_function("measure_actual_fpr", |b| {
        b.iter(|| {
            let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);

            for item in &inserted {
                filter.insert(item);
            }

            let mut false_positives = 0;
            for item in &test_set {
                if filter.contains(item) {
                    false_positives += 1;
                }
            }

            let actual_fpr = false_positives as f64 / test_set.len() as f64;
            black_box(actual_fpr);
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 16: Adaptive Sharding
// ============================================================================

fn bench_adaptive_sharding(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_sharding");

    let items = generate_random_u64(MEDIUM_N);

    group.bench_function("default_sharding", |b| {
        b.iter(|| {
            let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
            for item in &items[..1000] {
                filter.insert(item);
            }
            black_box(filter);
        });
    });

    group.bench_function("adaptive_sharding", |b| {
        b.iter(|| {
            let filter = ShardedBloomFilter::<u64>::new_adaptive(MEDIUM_N, TARGET_FPR);
            for item in &items[..1000] {
                filter.insert(item);
            }
            black_box(filter);
        });
    });

    group.bench_function("manual_optimal_sharding", |b| {
        let optimal_shards = num_cpus::get() * 2;
        b.iter(|| {
            let filter = ShardedBloomFilter::<u64>::with_shard_count(
                MEDIUM_N,
                TARGET_FPR,
                optimal_shards,
            );
            for item in &items[..1000] {
                filter.insert(item);
            }
            black_box(filter);
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 17: Adversarial Workloads
// ============================================================================

fn bench_adversarial_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("adversarial_workloads");

    group.bench_function("same_shard_attack", |b| {
        let filter = ShardedBloomFilter::<u64>::with_shard_count(MEDIUM_N, TARGET_FPR, 16);

        let items: Vec<u64> = (0..1000)
            .map(|i| i * filter.shard_count() as u64)
            .collect();

        let mut i = 0;
        b.iter(|| {
            filter.insert(black_box(&items[i % items.len()]));
            i += 1;
        });
    });

    group.bench_function("high_contention_same_shard", |b| {
        b.iter(|| {
            let filter = Arc::new(
                ShardedBloomFilter::<u64>::with_shard_count(MEDIUM_N, TARGET_FPR, 16)
            );

            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let filter = Arc::clone(&filter);
                    thread::spawn(move || {
                        for _i in 0..1000 {
                            filter.insert(&0u64);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 18: Clear Under Load
// ============================================================================

fn bench_clear_under_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("clear_under_load");
    group.sample_size(10);

    let items = generate_random_u64(LARGE_N);

    for &load_pct in &[25, 50, 75, 100] {
        group.bench_with_input(
            BenchmarkId::new("clear_at_load", format!("{}%", load_pct)),
            &load_pct,
            |b, &pct| {
                b.iter_batched(
                    || {
                        let filter = ShardedBloomFilter::<u64>::new(LARGE_N, TARGET_FPR);
                        let count = (LARGE_N * pct / 100).min(items.len());
                        for item in &items[..count] {
                            filter.insert(item);
                        }
                        filter
                    },
                    |filter| {
                        filter.clear();
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 19: Clone Performance
// ============================================================================

fn bench_clone_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("clone_performance");

    let items = generate_random_u64(MEDIUM_N);

    for &load_pct in &[0, 25, 50, 75, 100] {
        group.bench_with_input(
            BenchmarkId::new("clone_filter", format!("{}% full", load_pct)),
            &load_pct,
            |b, &pct| {
                let filter = ShardedBloomFilter::<u64>::new(MEDIUM_N, TARGET_FPR);
                let count = (MEDIUM_N * pct / 100).min(items.len());
                for item in &items[..count] {
                    filter.insert(item);
                }

                b.iter(|| {
                    black_box(filter.clone());
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = core_benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(2));
    targets = 
        bench_core_operations,
        bench_batch_operations,
        bench_data_types,
        bench_access_patterns,
);

criterion_group!(
    name = concurrency_benches;
    config = Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3));
    targets =
        bench_concurrency_insert,
        bench_concurrency_query,
        bench_concurrent_mixed_workload,
);

criterion_group!(
    name = scaling_benches;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(7));
    targets =
        bench_shard_count_scaling,
        bench_load_factor_impact,
);

criterion_group!(
    name = real_world_benches;
    config = Criterion::default()
        .sample_size(50);
    targets =
        bench_real_world_url_dedup,
        bench_real_world_log_dedup,
);

criterion_group!(
    name = comparison_benches;
    config = Criterion::default()
        .sample_size(50);
    targets =
        bench_vs_standard_filter,
        bench_memory_efficiency,
);

criterion_group!(
    name = advanced_benches;
    config = Criterion::default()
        .sample_size(30);
    targets =
        bench_shard_statistics,
        bench_fpr_accuracy,
        bench_adaptive_sharding,
        bench_adversarial_workloads,
        bench_clear_under_load,
        bench_clone_performance,
);

#[cfg(feature = "serde")]
criterion_group!(
    name = serde_benches;
    config = Criterion::default()
        .sample_size(30);
    targets = bench_serialization,
);

#[cfg(not(feature = "serde"))]
criterion_main!(
    core_benches,
    concurrency_benches,
    scaling_benches,
    real_world_benches,
    comparison_benches,
    advanced_benches,
);

#[cfg(feature = "serde")]
criterion_main!(
    core_benches,
    concurrency_benches,
    scaling_benches,
    real_world_benches,
    comparison_benches,
    advanced_benches,
    serde_benches,
);