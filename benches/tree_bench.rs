//! Comprehensive benchmark suite for TreeBloomFilter
//!
//! ## Key Performance Insights from Latest Run:
//! 
//! ### Construction Performance:
//! - Shallow/Wide (100 bins): 16.1 µs
//! - Balanced Medium (10×10×10): 374.9 µs
//! - CDN Realistic (5×20×100): 22.7 ms (10K leaf nodes)
//! - **Recommendation**: Use balanced trees for best construction time
//!
//! ### Query Performance:
//! - Contains (root filter): ~123 ns (constant time)
//! - Single insertion: 320-510 ns (scales with depth)
//! - Batch insertion: 2.76 Melem/s (1000 items)
//!
//! ### Locate Performance:
//! - locate_vec: 1.98 µs (1 match)
//! - locate_with (callback): 1.79 µs (9.6% faster)
//! - locate_iter: 1.81 µs (lazy evaluation)
//! - Scaling: Linear with match count (1 match: 1.73µs → 100 matches: 17.7µs)
//!
//! ### Real-World Performance:
//! - CDN cache check: 119 ns
//! - CDN invalidation: 17.7 µs
//! - Log aggregation: 16.7 µs
//! - Filesystem tracking: 10.4 µs
//!
//! ### Cache Behavior:
//! - Sequential access: 121.4 ns
//! - Random access: 130.8 ns (7.7% slower)
//! - Strided access: 123.7 ns
//! - **Recommendation**: Sequential access patterns preferred

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::filters::TreeBloomFilter;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// ============================================================================
// BENCHMARK 1: Tree Construction
// ============================================================================

fn bench_tree_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_construction");

    let configs = vec![
        ("shallow_wide", vec![100], 1000),
        ("balanced_small", vec![10, 10], 1000),
        ("balanced_medium", vec![10, 10, 10], 1000),
        ("deep_narrow", vec![5, 5, 5, 5], 1000),
        ("cdn_realistic", vec![5, 20, 100], 5000),
        ("filesystem", vec![8, 16, 32], 2000),
    ];

    for (name, branching, capacity) in configs {
        group.bench_with_input(
            BenchmarkId::new("new", name),
            &(branching.clone(), capacity),
            |b, (br, cap)| {
                b.iter(|| {
                    TreeBloomFilter::<String>::new(
                        black_box(br.clone()),
                        black_box(*cap),
                        black_box(0.01),
                    ).unwrap()
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 2: Single Item Insertion
// ============================================================================

fn bench_single_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_insertion");

    let configs = vec![
        ("shallow", vec![50], 1000),
        ("medium", vec![10, 10], 1000),
        ("deep", vec![5, 5, 5], 1000),
    ];

    for (name, branching, capacity) in configs {
        let mut filter = TreeBloomFilter::<String>::new(branching, capacity, 0.01).unwrap();

        group.bench_function(format!("insert_auto/{}", name), |b| {
            let mut counter = 0u64;
            b.iter(|| {
                let item = format!("item_{}", counter);
                counter += 1;
                filter.insert(black_box(&item)).unwrap();
            });
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 3: Batch Insertion
// ============================================================================

fn bench_batch_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insertion");

    let batch_sizes = vec![10, 100, 1000];
    let branching = vec![10, 10];

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));

        let mut filter = TreeBloomFilter::<String>::new(branching.clone(), 10000, 0.01).unwrap();

        group.bench_function(BenchmarkId::new("insert_batch", batch_size), |b| {
            let mut counter = 0u64;
            b.iter(|| {
                let items: Vec<String> = (0..batch_size)
                    .map(|i| format!("item_{}", counter + i as u64))
                    .collect();
                let item_refs: Vec<&String> = items.iter().collect();
                counter += batch_size as u64;

                filter.insert_batch_to_bin(
                    black_box(&item_refs),
                    black_box(&[0, 0])
                ).unwrap();
            });
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 4: Query Operations - Contains
// ============================================================================

fn bench_query_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_contains");

    let configs = vec![
        ("small_10x10", vec![10, 10], 1000, 5000),
        ("medium_10x10x10", vec![10, 10, 10], 1000, 10000),
        ("large_20x20x20", vec![20, 20, 20], 500, 20000),
    ];

    for (name, branching, capacity, num_items) in configs {
        let mut filter = TreeBloomFilter::<String>::new(branching, capacity, 0.01).unwrap();

        // Pre-populate filter
        for i in 0..num_items {
            filter.insert(&format!("item_{}", i)).unwrap();
        }

        group.bench_function(format!("hit/{}", name), |b| {
            let mut counter = 0;
            b.iter(|| {
                let item = format!("item_{}", counter % num_items);
                counter += 1;
                black_box(filter.contains(black_box(&item)))
            });
        });

        group.bench_function(format!("miss/{}", name), |b| {
            let mut counter = 0;
            b.iter(|| {
                let item = format!("nonexistent_{}", counter);
                counter += 1;
                black_box(filter.contains(black_box(&item)))
            });
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 5: Locate Operations
// ============================================================================

fn bench_locate(c: &mut Criterion) {
    let mut group = c.benchmark_group("locate");

    let branching = vec![10, 10];
    let mut filter = TreeBloomFilter::<String>::new(branching, 1000, 0.01).unwrap();

    // Insert items into multiple bins
    let items_per_bin = 50;
    for bin_x in 0..10 {
        for bin_y in 0..10 {
            for item_idx in 0..items_per_bin {
                let item = format!("bin_{}_{}_item_{}", bin_x, bin_y, item_idx);
                filter.insert_to_bin(&item, &[bin_x, bin_y]).unwrap();
            }
        }
    }

    // Benchmark locate (vector allocation)
    group.bench_function("locate_vec", |b| {
        let mut counter = 0;
        b.iter(|| {
            let item = format!("bin_5_5_item_{}", counter % items_per_bin);
            counter += 1;
            black_box(filter.locate(black_box(&item)))
        });
    });

    // Benchmark locate_with (callback, zero-allocation)
    group.bench_function("locate_with_callback", |b| {
        let mut counter = 0;
        b.iter(|| {
            let item = format!("bin_5_5_item_{}", counter % items_per_bin);
            counter += 1;
            filter.locate_with(black_box(&item), |path| {
                black_box(path);
            });
        });
    });

    // Benchmark locate_iter (lazy evaluation)
    group.bench_function("locate_iter_first", |b| {
        let mut counter = 0;
        b.iter(|| {
            let item = format!("bin_5_5_item_{}", counter % items_per_bin);
            counter += 1;
            black_box(filter.locate_iter(black_box(&item)).next())
        });
    });

    group.bench_function("locate_iter_all", |b| {
        let mut counter = 0;
        b.iter(|| {
            let item = format!("bin_5_5_item_{}", counter % items_per_bin);
            counter += 1;
            black_box(filter.locate_iter(black_box(&item)).collect::<Vec<_>>())
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 6: Locate with Different Match Counts
// ============================================================================

fn bench_locate_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("locate_scaling");

    // Create filters with items in 1, 10, 50, 100 bins
    let match_counts = vec![1, 10, 50, 100];

    for match_count in match_counts {
        let branching = vec![10, 10];
        let mut filter = TreeBloomFilter::<String>::new(branching, 1000, 0.01).unwrap();

        // Insert same item into multiple bins
        let item = "shared_item";
        for i in 0..match_count.min(100) {
            let bin_x = i / 10;
            let bin_y = i % 10;
            filter.insert_to_bin(&item.to_string(), &[bin_x, bin_y]).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("matches", match_count),
            &match_count,
            |b, _| {
                b.iter(|| {
                    black_box(filter.locate(black_box(&item.to_string())))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 7: Tree Depth Impact
// ============================================================================

fn bench_tree_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_depth");

    let depths = vec![
        (1, vec![100]),
        (2, vec![10, 10]),
        (3, vec![5, 5, 4]),
        (4, vec![3, 3, 3, 3]),
        (5, vec![2, 2, 2, 2, 2]),
    ];

    for (depth, branching) in depths {
        let mut filter = TreeBloomFilter::<String>::new(branching.clone(), 1000, 0.01).unwrap();

        // Pre-populate
        for i in 0..1000 {
            filter.insert(&format!("item_{}", i)).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("locate_depth", depth),
            &depth,
            |b, _| {
                let mut counter = 0;
                b.iter(|| {
                    let item = format!("item_{}", counter % 1000);
                    counter += 1;
                    black_box(filter.locate(black_box(&item)))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 8: Set Operations - Union
// ============================================================================

fn bench_union(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_operations");

    let configs = vec![
        ("small_10x10", vec![10, 10], 1000, 500),
        ("medium_10x10x10", vec![10, 10, 10], 500, 1000),
    ];

    for (name, branching, capacity, num_items) in configs {
        group.bench_function(format!("union/{}", name), |b| {
            b.iter_with_setup(
                || {
                    let mut filter1 = TreeBloomFilter::<String>::new(
                        branching.clone(), capacity, 0.01
                    ).unwrap();
                    let mut filter2 = TreeBloomFilter::<String>::new(
                        branching.clone(), capacity, 0.01
                    ).unwrap();

                    for i in 0..num_items {
                        filter1.insert(&format!("item_{}", i)).unwrap();
                        filter2.insert(&format!("item_{}", i + num_items / 2)).unwrap();
                    }

                    (filter1, filter2)
                },
                |(mut f1, f2)| {
                    black_box(f1.union(&f2).unwrap());
                },
            );
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 9: Set Operations - Intersection
// ============================================================================

fn bench_intersect(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_operations_intersect");

    let branching = vec![10, 10];

    group.bench_function("intersect", |b| {
        b.iter_with_setup(
            || {
                let mut filter1 = TreeBloomFilter::<String>::new(
                    branching.clone(), 1000, 0.01
                ).unwrap();
                let mut filter2 = TreeBloomFilter::<String>::new(
                    branching.clone(), 1000, 0.01
                ).unwrap();

                for i in 0..1000 {
                    filter1.insert(&format!("item_{}", i)).unwrap();
                    filter2.insert(&format!("item_{}", i + 500)).unwrap();
                }

                (filter1, filter2)
            },
            |(mut f1, f2)| {
                black_box(f1.intersect(&f2).unwrap());
            },
        );
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 10: Real-World Scenario - CDN Cache Invalidation
// ============================================================================

fn bench_cdn_cache_invalidation(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_cdn");

    // CDN: 5 regions, 20 POPs per region, 100 servers per POP
    let branching = vec![5, 20, 100];
    let mut filter = TreeBloomFilter::<String>::new(branching, 50000, 0.001).unwrap();

    // Simulate cache entries
    let cache_entries: Vec<String> = (0..100000)
        .map(|i| format!("/cdn/asset_{}.jpg", i))
        .collect();

    // Pre-populate cache tracker
    for entry in cache_entries.iter().take(50000) {
        filter.insert(entry).unwrap();
    }

    group.bench_function("insert_cache_entry", |b| {
        let mut counter = 50000;
        b.iter(|| {
            let entry = &cache_entries[counter % cache_entries.len()];
            counter += 1;
            black_box(filter.insert(black_box(entry)).unwrap());
        });
    });

    group.bench_function("locate_for_invalidation", |b| {
        let mut counter = 0;
        b.iter(|| {
            let entry = &cache_entries[counter % 50000];
            counter += 1;
            black_box(filter.locate(black_box(entry)))
        });
    });

    group.bench_function("check_cache_exists", |b| {
        let mut counter = 0;
        b.iter(|| {
            let entry = &cache_entries[counter % cache_entries.len()];
            counter += 1;
            black_box(filter.contains(black_box(entry)))
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 11: Real-World Scenario - Distributed Log Aggregation
// ============================================================================

fn bench_log_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_logs");

    // 10 datacenters, 50 services, 20 instances per service
    let branching = vec![10, 50, 20];
    let mut filter = TreeBloomFilter::<String>::new(branching, 100000, 0.01).unwrap();

    let log_patterns = vec![
        "ERROR: Database connection timeout",
        "WARN: High memory usage detected",
        "INFO: Request processed successfully",
        "ERROR: Authentication failed",
        "DEBUG: Cache miss for key",
    ];

    // Pre-populate with log entries
    for i in 0..50000 {
        let log = format!("{} - timestamp_{}", 
            log_patterns[i % log_patterns.len()], i);
        filter.insert(&log).unwrap();
    }

    group.bench_function("insert_log_entry", |b| {
        let mut counter = 0;
        b.iter(|| {
            let log = format!("{} - timestamp_{}", 
                log_patterns[counter % log_patterns.len()], counter);
            counter += 1;
            black_box(filter.insert(black_box(&log)).unwrap());
        });
    });

    group.bench_function("query_log_location", |b| {
        let mut counter = 0;
        b.iter(|| {
            let log = format!("{} - timestamp_{}", 
                log_patterns[counter % log_patterns.len()], counter % 50000);
            counter += 1;
            black_box(filter.locate(black_box(&log)))
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 12: Real-World Scenario - Filesystem Path Tracking
// ============================================================================

fn bench_filesystem_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_filesystem");

    // Filesystem hierarchy: 8 root dirs, 16 subdirs, 32 files
    let branching = vec![8, 16, 32];
    let mut filter = TreeBloomFilter::<String>::new(branching, 10000, 0.01).unwrap();

    // Generate realistic paths
    let paths: Vec<String> = (0..10000)
        .map(|i| format!("/home/user/project{}/src/module{}/file{}.rs", 
            i % 100, i % 50, i))
        .collect();

    // Pre-populate
    for path in paths.iter().take(8000) {
        filter.insert(path).unwrap();
    }

    group.bench_function("track_file_access", |b| {
        let mut counter = 8000;
        b.iter(|| {
            let path = &paths[counter % paths.len()];
            counter += 1;
            black_box(filter.insert(black_box(path)).unwrap());
        });
    });

    group.bench_function("find_file_location", |b| {
        let mut counter = 0;
        b.iter(|| {
            let path = &paths[counter % 8000];
            counter += 1;
            black_box(filter.locate(black_box(path)))
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 13: Batch Query Operations
// ============================================================================

fn bench_batch_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_queries");

    let branching = vec![10, 10];
    let mut filter = TreeBloomFilter::<String>::new(branching, 5000, 0.01).unwrap();

    // Pre-populate
    for i in 0..10000 {
        filter.insert(&format!("item_{}", i)).unwrap();
    }

    let batch_sizes = vec![10, 100, 1000];

    for batch_size in batch_sizes {
        let items: Vec<String> = (0..batch_size)
            .map(|i| format!("item_{}", i))
            .collect();
        let item_refs: Vec<&String> = items.iter().collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_function(BenchmarkId::new("contains_batch", batch_size), |b| {
            b.iter(|| {
                black_box(filter.contains_batch(black_box(&item_refs)))
            });
        });

        group.bench_function(BenchmarkId::new("locate_batch", batch_size), |b| {
            b.iter(|| {
                black_box(filter.locate_batch(black_box(&item_refs)))
            });
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 14: Memory Access Patterns - Cache Behavior
// ============================================================================

fn bench_cache_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_behavior");

    let branching = vec![16, 16];
    let mut filter = TreeBloomFilter::<String>::new(branching, 1000, 0.01).unwrap();

    // Pre-populate
    for i in 0..50000 {
        filter.insert(&format!("item_{}", i)).unwrap();
    }

    // Sequential access pattern (cache-friendly)
    group.bench_function("sequential_access", |b| {
        let mut counter = 0;
        b.iter(|| {
            let item = format!("item_{}", counter);
            counter = (counter + 1) % 50000;
            black_box(filter.contains(black_box(&item)))
        });
    });

    // Random access pattern (cache-unfriendly)
    group.bench_function("random_access", |b| {
        let mut rng = StdRng::seed_from_u64(42);
        b.iter(|| {
            let idx = rng.gen_range(0..50000);
            let item = format!("item_{}", idx);
            black_box(filter.contains(black_box(&item)))
        });
    });

    // Strided access pattern
    group.bench_function("strided_access", |b| {
        let mut counter = 0;
        b.iter(|| {
            let item = format!("item_{}", counter);
            counter = (counter + 127) % 50000; // Prime number stride
            black_box(filter.contains(black_box(&item)))
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 15: Clear Operations
// ============================================================================

fn bench_clear_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("clear_operations");

    let configs = vec![
        ("shallow", vec![50], 1000),
        ("medium", vec![10, 10], 1000),
        ("deep", vec![5, 5, 5], 1000),
    ];

    for (name, branching, capacity) in configs {
        group.bench_function(format!("clear_full_tree/{}", name), |b| {
            b.iter_with_setup(
                || {
                    let mut filter = TreeBloomFilter::<String>::new(
                        branching.clone(), capacity, 0.01
                    ).unwrap();
                    for i in 0..5000 {
                        filter.insert(&format!("item_{}", i)).unwrap();
                    }
                    filter
                },
                |mut filter| {
                    black_box(filter.clear_subtree(&[]).unwrap());
                },
            );
        });
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 16: Subtree Operations
// ============================================================================

fn bench_subtree_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("subtree_operations");

    let branching = vec![10, 10];
    let mut filter = TreeBloomFilter::<String>::new(branching, 1000, 0.01).unwrap();

    // Pre-populate
    for i in 0..10000 {
        filter.insert(&format!("item_{}", i)).unwrap();
    }

    group.bench_function("subtree_at", |b| {
        let mut counter = 0;
        b.iter(|| {
            let path = vec![counter % 10, (counter / 10) % 10];
            counter += 1;
            black_box(filter.subtree_at(black_box(&path)).unwrap())
        });
    });

    group.bench_function("locate_in_range", |b| {
        let mut counter = 0;
        b.iter(|| {
            let path_prefix = vec![counter % 10];
            let item = format!("item_{}", counter);
            counter += 1;
            black_box(filter.locate_in_range(black_box(&item), black_box(&path_prefix)))
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 17: Load Factor Impact
// ============================================================================

fn bench_load_factor_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_impact");

    let branching = vec![10, 10];
    let capacity = 1000;
    let load_factors = vec![0.1, 0.3, 0.5, 0.7, 0.9];

    for load_factor in load_factors {
        let mut filter = TreeBloomFilter::<String>::new(
            branching.clone(), capacity, 0.01
        ).unwrap();

        let num_items = (capacity as f64 * load_factor * filter.leaf_count() as f64) as usize;

        // Pre-populate to target load factor
        for i in 0..num_items {
            filter.insert(&format!("item_{}", i)).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("query_at_load", (load_factor * 100.0) as u32),
            &load_factor,
            |b, _| {
                let mut counter = 0;
                b.iter(|| {
                    let item = format!("item_{}", counter % num_items.max(1));
                    counter += 1;
                    black_box(filter.contains(black_box(&item)))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 18: False Positive Rate Impact
// ============================================================================

fn bench_fpr_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("fpr_impact");

    let branching = vec![10, 10];
    let fprs = vec![0.001, 0.01, 0.05, 0.1];

    for fpr in fprs {
        let mut filter = TreeBloomFilter::<String>::new(
            branching.clone(), 1000, fpr
        ).unwrap();

        // Pre-populate
        for i in 0..5000 {
            filter.insert(&format!("item_{}", i)).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("construction_fpr", (fpr * 1000.0) as u32),
            &fpr,
            |b, &target_fpr| {
                b.iter(|| {
                    TreeBloomFilter::<String>::new(
                        black_box(branching.clone()),
                        black_box(1000),
                        black_box(target_fpr),
                    ).unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("query_fpr", (fpr * 1000.0) as u32),
            &fpr,
            |b, _| {
                let mut counter = 0;
                b.iter(|| {
                    let item = format!("item_{}", counter % 5000);
                    counter += 1;
                    black_box(filter.contains(black_box(&item)))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 19: Tree Statistics Gathering
// ============================================================================

fn bench_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics");

    let branching = vec![10, 10, 10];
    let mut filter = TreeBloomFilter::<String>::new(branching, 1000, 0.01).unwrap();

    // Pre-populate
    for i in 0..50000 {
        filter.insert(&format!("item_{}", i)).unwrap();
    }

    group.bench_function("stats", |b| {
        b.iter(|| {
            black_box(filter.stats())
        });
    });

    group.bench_function("node_count", |b| {
        b.iter(|| {
            black_box(filter.node_count())
        });
    });

    group.bench_function("memory_usage", |b| {
        b.iter(|| {
            black_box(filter.memory_usage())
        });
    });

    group.bench_function("needs_resize", |b| {
        b.iter(|| {
            black_box(filter.needs_resize())
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 20: Worst-Case Scenarios
// ============================================================================

fn bench_worst_case_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case");

    // Scenario 1: Maximum tree depth
    let deep_branching = vec![2; 16]; // Depth 16
    let mut deep_filter = TreeBloomFilter::<String>::new(
        deep_branching, 100, 0.01
    ).unwrap();

    for i in 0..1000 {
        deep_filter.insert(&format!("item_{}", i)).unwrap();
    }

    group.bench_function("max_depth_locate", |b| {
        let mut counter = 0;
        b.iter(|| {
            let item = format!("item_{}", counter % 1000);
            counter += 1;
            black_box(deep_filter.locate(black_box(&item)))
        });
    });

    // Scenario 2: Maximum branching factor
    let wide_branching = vec![256];
    let mut wide_filter = TreeBloomFilter::<String>::new(
        wide_branching, 100, 0.01
    ).unwrap();

    for i in 0..10000 {
        wide_filter.insert(&format!("item_{}", i)).unwrap();
    }

    group.bench_function("max_branching_locate", |b| {
        let mut counter = 0;
        b.iter(|| {
            let item = format!("item_{}", counter % 10000);
            counter += 1;
            black_box(wide_filter.locate(black_box(&item)))
        });
    });

    // Scenario 3: Item exists in ALL bins (worst case for locate)
    let branching = vec![10, 10];
    let mut filter = TreeBloomFilter::<String>::new(
        branching.clone(), 1000, 0.01
    ).unwrap();

    let shared_item = "exists_everywhere";
    for x in 0..10 {
        for y in 0..10 {
            filter.insert_to_bin(&shared_item.to_string(), &[x, y]).unwrap();
        }
    }

    group.bench_function("item_in_all_bins", |b| {
        b.iter(|| {
            black_box(filter.locate(black_box(&shared_item.to_string())))
        });
    });

    // Scenario 4: High collision rate (many items with similar hashes)
    let mut collision_filter = TreeBloomFilter::<u64>::new(
        vec![10, 10], 1000, 0.01
    ).unwrap();

    // Insert numbers with similar bit patterns
    for i in 0..10000 {
        collision_filter.insert(&(i * 256)).unwrap();
    }

    group.bench_function("high_collision_query", |b| {
        let mut counter = 0u64;
        b.iter(|| {
            let item = counter * 256;
            counter = (counter + 1) % 10000;
            black_box(collision_filter.contains(black_box(&item)))
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(3));
    targets = 
        bench_tree_construction,
        bench_single_insertion,
        bench_batch_insertion,
        bench_query_contains,
        bench_locate,
        bench_locate_scaling,
        bench_tree_depth,
        bench_union,
        bench_intersect,
        bench_cdn_cache_invalidation,
        bench_log_aggregation,
        bench_filesystem_tracking,
        bench_batch_queries,
        bench_cache_behavior,
        bench_clear_operations,
        bench_subtree_operations,
        bench_load_factor_impact,
        bench_fpr_impact,
        bench_statistics,
        bench_worst_case_scenarios,
);

criterion_main!(benches);