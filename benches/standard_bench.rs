//! Comprehensive benchmark suite for StandardBloomFilter
//!
//! This benchmark suite measures performance across real-world scenarios including:
//! - Single-threaded insert/query performance at various scales
//! - Batch operation efficiency
//! - Concurrent multi-threaded workloads
//! - Memory access patterns and cache behavior
//! - False positive rate validation under load
//! - Different data types and distributions
//! - Filter saturation and degradation characteristics
//!
//! Run with: cargo bench --bench standard_filter
//!
//! For flamegraphs: cargo bench --bench standard_filter -- --profile-time=5

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::hash::StdHasher;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ============================================================================
// SCENARIO 1: Single Insert Performance at Multiple Scales
// ============================================================================
// Real-world: User signup, session tracking, duplicate detection

fn bench_insert_single_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert/single_thread");

    // Test at different scales (small cache-friendly to large memory-bound)
    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}k_items", size / 1000)),
            &size,
            |b, &size| {
                let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();
                let mut counter = 0u64;

                b.iter(|| {
                    filter.insert(black_box(&counter));
                    counter = counter.wrapping_add(1);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 2: Query Performance (Cache Hit/Miss Patterns)
// ============================================================================
// Real-world: URL deduplication, spam detection, cache lookups

fn bench_contains_single_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains/single_thread");

    for size in [10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(1));

        // Benchmark 1: All hits (best case - data in filter)
        group.bench_with_input(
            BenchmarkId::new("all_hits", format!("{}k", size / 1000)),
            &size,
            |b, &size| {
                let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();

                // Pre-populate filter
                for i in 0..size {
                    filter.insert(&(i as u64));
                }

                let mut counter = 0u64;
                b.iter(|| {
                    let result = filter.contains(black_box(&counter));
                    counter = (counter + 1) % (size as u64);
                    black_box(result)
                });
            },
        );

        // Benchmark 2: All misses (worst case - random queries)
        group.bench_with_input(
            BenchmarkId::new("all_misses", format!("{}k", size / 1000)),
            &size,
            |b, &size| {
                let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();

                // Populate with even numbers only
                for i in 0..size {
                    filter.insert(&((i * 2) as u64));
                }

                let mut counter = 1u64; // Query odd numbers
                b.iter(|| {
                    let result = filter.contains(black_box(&counter));
                    counter = counter.wrapping_add(2);
                    black_box(result)
                });
            },
        );

        // Benchmark 3: Mixed (realistic - 50/50 hits/misses)
        group.bench_with_input(
            BenchmarkId::new("mixed_50_50", format!("{}k", size / 1000)),
            &size,
            |b, &size| {
                let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();

                // Populate 50%
                for i in 0..(size / 2) {
                    filter.insert(&(i as u64));
                }

                let mut counter = 0u64;
                b.iter(|| {
                    let result = filter.contains(black_box(&counter));
                    counter = counter.wrapping_add(1) % (size as u64);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 3: Batch Operations (Throughput-Optimized Workloads)
// ============================================================================
// Real-world: Log processing, ETL pipelines, bulk imports

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    for batch_size in [10, 100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(batch_size as u64));

        // Insert batch
        group.bench_with_input(
            BenchmarkId::new("insert", batch_size),
            &batch_size,
            |b, &batch_size| {
                let filter = StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap();
                let items: Vec<u64> = (0..batch_size as u64).collect();

                b.iter(|| {
                    filter.insert_batch(black_box(&items));
                });
            },
        );

        // Contains batch
        group.bench_with_input(
            BenchmarkId::new("contains", batch_size),
            &batch_size,
            |b, &batch_size| {
                let filter = StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap();
                let items: Vec<u64> = (0..batch_size as u64).collect();

                // Pre-populate
                for i in 0..batch_size {
                    filter.insert(&(i as u64));
                }

                b.iter(|| {
                    let results = filter.contains_batch(black_box(&items));
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 4: Concurrent Multi-threaded Performance
// ============================================================================
// Real-world: Web server request deduplication, distributed caching

fn bench_concurrent_inserts(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent/inserts");
    group.sample_size(20); // Fewer samples for expensive benchmarks

    for num_threads in [2, 4, 8, 16] {
        group.throughput(Throughput::Elements(10_000));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}threads", num_threads)),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let filter = Arc::new(StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap());

                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let f = Arc::clone(&filter);
                            thread::spawn(move || {
                                let start = tid * 10_000;
                                for i in start..(start + 10_000) {
                                    f.insert(&(i as u64));
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&filter);
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent/mixed");
    group.sample_size(20);

    for num_threads in [2, 4, 8] {
        group.throughput(Throughput::Elements(10_000));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}threads", num_threads)),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let filter = Arc::new(StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap());

                    // Pre-populate
                    for i in 0..50_000 {
                        filter.insert(&i);
                    }

                    let handles: Vec<_> = (0..num_threads)
                        .map(|tid| {
                            let f = Arc::clone(&filter);
                            thread::spawn(move || {
                                if tid % 2 == 0 {
                                    // Writer threads
                                    for i in 0..5_000 {
                                        f.insert(&(i + 100_000));
                                    }
                                } else {
                                    // Reader threads
                                    for i in 0..5_000 {
                                        black_box(f.contains(&i));
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&filter);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 5: Different Data Types (Real-world Heterogeneous Data)
// ============================================================================
// Real-world: Email addresses, UUIDs, IP addresses, user IDs

fn bench_different_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("types");
    group.throughput(Throughput::Elements(1));

    // Integers (best case - simple hash)
    group.bench_function("u64_insert", |b| {
        let filter = StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap();
        let mut counter = 0u64;

        b.iter(|| {
            filter.insert(black_box(&counter));
            counter = counter.wrapping_add(1);
        });
    });

    group.bench_function("u64_contains", |b| {
        let filter = StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap();
        for i in 0..10_000 {
            filter.insert(&i);
        }
        let mut counter = 0u64;

        b.iter(|| {
            let result = filter.contains(black_box(&counter));
            counter = (counter + 1) % 10_000;
            black_box(result)
        });
    });

    // Strings (common case - variable length)
    group.bench_function("string_insert", |b| {
        let filter = StandardBloomFilter::<String>::new(100_000, 0.01).unwrap();
        let mut counter = 0;

        b.iter(|| {
            let item = format!("user_email_{}@example.com", counter);
            filter.insert(black_box(&item));
            counter += 1;
        });
    });

    group.bench_function("string_contains", |b| {
        let filter = StandardBloomFilter::<String>::new(100_000, 0.01).unwrap();
        for i in 0..10_000 {
            filter.insert(&format!("user_{}", i));
        }
        let mut counter = 0;

        b.iter(|| {
            let item = format!("user_{}", counter);
            let result = filter.contains(black_box(&item));
            counter = (counter + 1) % 10_000;
            black_box(result)
        });
    });

    // UUID-like (128-bit values)
    group.bench_function("u128_insert", |b| {
        let filter = StandardBloomFilter::<u128>::new(100_000, 0.01).unwrap();
        let mut counter = 0u128;

        b.iter(|| {
            filter.insert(black_box(&counter));
            counter = counter.wrapping_add(1);
        });
    });

    // Tuple (composite keys)
    group.bench_function("tuple_insert", |b| {
        let filter = StandardBloomFilter::<(u32, u32, u16)>::new(100_000, 0.01).unwrap();
        let mut counter = 0u32;

        b.iter(|| {
            filter.insert(black_box(&(counter, counter * 2, (counter % 1000) as u16)));
            counter = counter.wrapping_add(1);
        });
    });

    group.finish();
}

// ============================================================================
// SCENARIO 6: Filter Saturation Behavior
// ============================================================================
// Real-world: Understanding degradation over time

fn bench_saturation_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("saturation");

    let capacity = 10_000;
    let filter_size = capacity;

    for fill_percent in [25, 50, 75, 90, 95] {
        let num_items = (capacity * fill_percent) / 100;

        group.bench_with_input(
            BenchmarkId::new("insert_at", format!("{}pct_full", fill_percent)),
            &num_items,
            |b, &num_items| {
                let filter = StandardBloomFilter::<u64>::new(filter_size, 0.01).unwrap();

                // Pre-fill to target saturation
                for i in 0..num_items {
                    filter.insert(&(i as u64));
                }

                let mut counter = num_items as u64;

                b.iter(|| {
                    filter.insert(black_box(&counter));
                    counter = counter.wrapping_add(1);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("query_at", format!("{}pct_full", fill_percent)),
            &num_items,
            |b, &num_items| {
                let filter = StandardBloomFilter::<u64>::new(filter_size, 0.01).unwrap();

                for i in 0..num_items {
                    filter.insert(&(i as u64));
                }

                let mut counter = 0u64;

                b.iter(|| {
                    let result = filter.contains(black_box(&counter));
                    counter = (counter + 1) % (num_items as u64);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 7: Filter Construction Overhead
// ============================================================================
// Real-world: Dynamic filter creation in request handlers

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    for size in [1_000, 10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}k_items", size / 1000)),
            &size,
            |b, &size| {
                b.iter(|| {
                    let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();
                    black_box(filter)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 8: Memory Access Patterns (Cache Behavior)
// ============================================================================
// Real-world: Sequential vs random access patterns

fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_patterns");
    group.throughput(Throughput::Elements(1000));

    let filter_size = 100_000;

    // Sequential access (cache-friendly)
    group.bench_function("sequential_insert", |b| {
        let filter = StandardBloomFilter::<u64>::new(filter_size, 0.01).unwrap();

        b.iter(|| {
            for i in 0..1000 {
                filter.insert(black_box(&(i as u64)));
            }
        });
    });

    group.bench_function("sequential_query", |b| {
        let filter = StandardBloomFilter::<u64>::new(filter_size, 0.01).unwrap();
        for i in 0..10_000 {
            filter.insert(&i);
        }

        b.iter(|| {
            for i in 0..1000 {
                black_box(filter.contains(black_box(&(i as u64))));
            }
        });
    });

    // Random access (cache-hostile)
    group.bench_function("random_insert", |b| {
        let filter = StandardBloomFilter::<u64>::new(filter_size, 0.01).unwrap();

        // Pre-compute random sequence
        let random_seq: Vec<u64> = (0..1000)
            .map(|i: u64| {
                // Simple LCG for deterministic "random" values
                i.wrapping_mul(1103515245).wrapping_add(12345) % 1_000_000
            })
            .collect();

        b.iter(|| {
            for &val in &random_seq {
                filter.insert(black_box(&val));
            }
        });
    });

    group.bench_function("random_query", |b| {
        let filter = StandardBloomFilter::<u64>::new(filter_size, 0.01).unwrap();
        for i in 0..10_000 {
            filter.insert(&i);
        }

        let random_seq: Vec<u64> = (0..1000)
            .map(|i: u64| i.wrapping_mul(1103515245).wrapping_add(12345) % 10_000)
            .collect();

        b.iter(|| {
            for &val in &random_seq {
                black_box(filter.contains(black_box(&val)));
            }
        });
    });

    group.finish();
}

// ============================================================================
// SCENARIO 9: Union/Intersection Operations
// ============================================================================
// Real-world: Merging filters from distributed nodes

fn bench_set_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_operations");

    for size in [10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("union", format!("{}k", size / 1000)),
            &size,
            |b, &size| {
                let filter1 = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();
                let filter2 = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();

                for i in 0..(size / 2) {
                    filter1.insert(&(i as u64));
                    filter2.insert(&((i + size / 2) as u64));
                }

                b.iter(|| {
                    let result = filter1.union(black_box(&filter2)).unwrap();
                    black_box(result)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("intersect", format!("{}k", size / 1000)),
            &size,
            |b, &size| {
                let filter1 = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();
                let filter2 = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();

                for i in 0..(size / 2) {
                    filter1.insert(&(i as u64));
                    if i % 2 == 0 {
                        filter2.insert(&(i as u64));
                    }
                }

                b.iter(|| {
                    let result = filter1.intersect(black_box(&filter2)).unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 10: False Positive Rate Measurement Under Load
// ============================================================================
// Real-world: Validating filter behavior matches theoretical FPR

fn bench_false_positive_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("false_positive_rate");
    group.sample_size(10);

    for target_fpr in [0.001, 0.01, 0.1] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("fpr_{}", target_fpr)),
            &target_fpr,
            |b, &target_fpr| {
                let capacity = 10_000;
                let filter = StandardBloomFilter::<u64>::new(capacity, target_fpr).unwrap();

                // Fill to 50% capacity
                for i in 0..(capacity / 2) {
                    filter.insert(&(i as u64));
                }

                b.iter(|| {
                    // Query items NOT in filter
                    let mut false_positives = 0;
                    for i in capacity..(capacity + 1000) {
                        if filter.contains(&(i as u64)) {
                            false_positives += 1;
                        }
                    }
                    black_box(false_positives)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 11: Clear and Rebuild Performance
// ============================================================================
// Real-world: Periodic filter refresh in long-running services

fn bench_clear_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("clear");

    for size in [10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}k", size / 1000)),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();

                    // Populate
                    for i in 0..(size / 2) {
                        filter.insert(&(i as u64));
                    }

                    // Clear
                    filter.clear();
                    black_box(filter.is_empty())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 12: Health Check Overhead
// ============================================================================
// Real-world: Production monitoring without impacting performance

fn bench_health_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("health_check");

    for size in [10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}k", size / 1000)),
            &size,
            |b, &size| {
                let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();

                for i in 0..(size / 2) {
                    filter.insert(&(i as u64));
                }

                b.iter(|| {
                    let health = filter.health_check();
                    black_box(health)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 13: Statistics Calculation Performance
// ============================================================================
// Real-world: Dashboards and monitoring systems

fn bench_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics");

    let size = 100_000;
    let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();

    for i in 0..(size / 2) {
        filter.insert(&(i as u64));
    }

    group.bench_function("fill_rate", |b| {
        b.iter(|| {
            let rate = filter.fill_rate();
            black_box(rate)
        });
    });

    group.bench_function("estimate_fpr", |b| {
        b.iter(|| {
            let fpr = filter.estimate_fpr();
            black_box(fpr)
        });
    });

    group.bench_function("estimate_cardinality", |b| {
        b.iter(|| {
            let card = filter.estimate_cardinality();
            black_box(card)
        });
    });

    group.bench_function("count_set_bits", |b| {
        b.iter(|| {
            let count = filter.count_set_bits();
            black_box(count)
        });
    });

    group.finish();
}

// ============================================================================
// SCENARIO 14: Clone Performance
// ============================================================================
// Real-world: Snapshotting filters for analysis

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("clone");

    for size in [10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}k", size / 1000)),
            &size,
            |b, &size| {
                let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();

                for i in 0..(size / 2) {
                    filter.insert(&(i as u64));
                }

                b.iter(|| {
                    let cloned = filter.clone();
                    black_box(cloned)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SCENARIO 15: Different Hash Functions Performance
// ============================================================================
// Real-world: Evaluating hash function trade-offs

fn bench_hash_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_functions");
    group.throughput(Throughput::Elements(1));

    let size = 100_000;

    // StdHasher (default)
    group.bench_function("std_hasher_insert", |b| {
        let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();
        let mut counter = 0u64;

        b.iter(|| {
            filter.insert(black_box(&counter));
            counter = counter.wrapping_add(1);
        });
    });

    group.bench_function("std_hasher_contains", |b| {
        let filter = StandardBloomFilter::<u64>::new(size, 0.01).unwrap();
        for i in 0..10_000 {
            filter.insert(&i);
        }
        let mut counter = 0u64;

        b.iter(|| {
            let result = filter.contains(black_box(&counter));
            counter = (counter + 1) % 10_000;
            black_box(result)
        });
    });

    // Custom hasher with seed
    group.bench_function("seeded_hasher_insert", |b| {
        let hasher = StdHasher::with_seed(42);
        let filter = StandardBloomFilter::<u64>::with_hasher(size, 0.01, hasher).unwrap();
        let mut counter = 0u64;

        b.iter(|| {
            filter.insert(black_box(&counter));
            counter = counter.wrapping_add(1);
        });
    });

    group.finish();
}

// ============================================================================
// SCENARIO 16: Real-World Use Case Simulations
// ============================================================================

fn bench_url_deduplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("use_cases/url_dedup");
    group.throughput(Throughput::Elements(1));

    // Simulate web crawler deduplication
    group.bench_function("crawler_check_and_insert", |b| {
        let filter = StandardBloomFilter::<String>::new(1_000_000, 0.01).unwrap();
        let mut counter = 0;

        b.iter(|| {
            let url = format!("https://example.com/page/{}", counter);

            if !filter.contains(&url) {
                filter.insert(&url);
                counter += 1;
            }

            black_box(counter)
        });
    });

    group.finish();
}

fn bench_session_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("use_cases/session_tracking");

    // Simulate API rate limiting
    group.bench_function("rate_limit_check", |b| {
        let filter = StandardBloomFilter::<(u32, u64)>::new(100_000, 0.01).unwrap();
        let mut request_id = 0u64;

        b.iter(|| {
            let user_id = (request_id % 1000) as u32;
            let timestamp = request_id / 100; // Simulate time windows

            let key = (user_id, timestamp);
            let is_rate_limited = filter.contains(&key);

            if !is_rate_limited {
                filter.insert(&key);
            }

            request_id += 1;
            black_box(is_rate_limited)
        });
    });

    group.finish();
}

fn bench_cache_admission(c: &mut Criterion) {
    let mut group = c.benchmark_group("use_cases/cache_admission");

    // Simulate cache admission filter (second-chance cache)
    group.bench_function("admission_filter", |b| {
        let seen_once = StandardBloomFilter::<u64>::new(100_000, 0.01).unwrap();
        let mut key = 0u64;

        b.iter(|| {
            let should_cache = seen_once.contains(&key);

            if !should_cache {
                seen_once.insert(&key);
            }

            key = (key + 1) % 50_000; // Simulate key reuse
            black_box(should_cache)
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets =
        // Core operations
        bench_insert_single_thread,
        bench_contains_single_thread,
        bench_batch_operations,

        // Concurrency
        bench_concurrent_inserts,
        bench_concurrent_mixed_workload,

        // Data types and patterns
        bench_different_types,
        bench_access_patterns,

        // Filter lifecycle
        bench_construction,
        bench_saturation_levels,
        bench_clear_operations,
        bench_clone,

        // Set operations
        bench_set_operations,

        // Monitoring and analysis
        bench_health_check,
        bench_statistics,
        bench_false_positive_validation,

        // Hash functions
        bench_hash_functions,

        // Real-world use cases
        bench_url_deduplication,
        bench_session_tracking,
        bench_cache_admission,
}

criterion_main!(benches);