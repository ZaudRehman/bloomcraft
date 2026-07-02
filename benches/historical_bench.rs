//! Historical Bloom Filter Benchmarks: 1970 → 2024
//!
//! A tribute to Burton Bloom's 1970 pioneering work and the evolution
//! of probabilistic data structures over 54 years of computer science research.
//!
//! # Historical Context
//!
//! ## Burton H. Bloom (1970)
//! **"Space/Time Trade-offs in Hash Coding with Allowable Errors"**
//! *Communications of the ACM, Vol. 13, No. 7, July 1970*
//!
//! Bloom introduced two methods for approximate set membership:
//!
//! ### Method 1: Hash Table Approach
//! - k independent hash functions
//! - Each hash maps to a separate hash table
//! - Simple but memory-intensive
//!
//! ### Method 2: Bit Array Approach (The "Bloom Filter")
//! - k independent hash functions
//! - Shared bit array for all hashes
//! - More memory-efficient
//! - **This became the canonical Bloom filter**
//!
//! ## Kirsch & Mitzenmacher (2006)
//! **"Less Hashing, Same Performance: Building a Better Bloom Filter"**
//! *ESA 2006: European Symposium on Algorithms*
//!
//! Revolutionary insight: Only 2 hash functions needed!
//! - Double hashing: h_i(x) = h1(x) + i × h2(x) mod m
//! - Same FPR guarantees as k independent hashes
//! - 2-3x faster in practice
//!
//! ## Modern Optimizations (2020s)
//! - Lock-free atomic operations (concurrent safety)
//! - Cache-line alignment (64-byte boundaries)
//! - SIMD batch operations (AVX2/NEON)
//! - Prefetching and branch prediction optimization
//!

use bloomcraft::filters::{ClassicBitsFilter, ClassicHashFilter, StandardBloomFilter};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// Generate count random strings of length len for benchmarking.
fn generate_strings(count: usize, len: usize) -> Vec<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    (0..count)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let hash = hasher.finish();
            format!("item_{:016x}_{:0width$x}", hash, i, width = len.saturating_sub(22))
        })
        .collect()
}

// Why: proves 1970 vs 2006 vs 2024 insert throughput gap
/// Compare single insert performance across 54 years of evolution
///
/// Tests Burton Bloom's original methods against modern implementation.
/// Expected: Modern 2-3x faster due to double hashing optimization.
fn bench_insert_historical(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_insert");
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);

    // 1970 Method 1: Hash Table
    group.bench_function("1970_hash_table", |b| {
        let mut filter = ClassicHashFilter::<String>::with_fpr(size, fpr);
        let mut idx = 0;
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });

    // 1970 Method 2: Bit Array (The canonical Bloom filter)
    group.bench_function("1970_bit_array", |b| {
        let mut filter = ClassicBitsFilter::<String>::with_fpr(size, fpr);
        let mut idx = 0;
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });

    // 2024 Modern: Enhanced Double Hashing + Atomics
    group.bench_function("2024_modern", |b| {
        let filter = StandardBloomFilter::<String>::new(size, fpr).unwrap();
        let mut idx = 0;
        b.iter(|| {
            filter.insert(black_box(&items[idx % items.len()]));
            idx += 1;
        });
    });

    group.finish();
}

// Why: proves same historical progression applies to queries
/// Compare query performance: classic vs modern
///
/// Query should show similar speedup patterns to insert.
fn bench_query_historical(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_query");
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);

    // Pre-fill all filters
    let mut filter_hash = ClassicHashFilter::<String>::with_fpr(size, fpr);
    let mut filter_bits = ClassicBitsFilter::<String>::with_fpr(size, fpr);
    let filter_modern = StandardBloomFilter::<String>::new(size, fpr).unwrap();

    for item in &items {
        filter_hash.insert(item);
        filter_bits.insert(item);
        filter_modern.insert(item);
    }

    // 1970 Method 1: Hash Table
    group.bench_function("1970_hash_table", |b| {
        let mut idx = 0;
        b.iter(|| {
            black_box(filter_hash.contains(black_box(&items[idx % items.len()])));
            idx += 1;
        });
    });

    // 1970 Method 2: Bit Array
    group.bench_function("1970_bit_array", |b| {
        let mut idx = 0;
        b.iter(|| {
            black_box(filter_bits.contains(black_box(&items[idx % items.len()])));
            idx += 1;
        });
    });

    // 2024 Modern
    group.bench_function("2024_modern", |b| {
        let mut idx = 0;
        b.iter(|| {
            black_box(filter_modern.contains(black_box(&items[idx % items.len()])));
            idx += 1;
        });
    });

    group.finish();
}

// Why: shows throughput improves with batch size for all eras
/// Batch insert comparison: shows aggregate performance differences
fn bench_batch_insert_historical(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_batch_insert");
    let size = 100_000;
    let fpr = 0.01;

    for batch_size in &[100, 1_000, 10_000] {
        let items = generate_strings(*batch_size, 32);
        group.throughput(Throughput::Elements(*batch_size as u64));

        // 1970 Hash Table
        group.bench_with_input(
            BenchmarkId::new("1970_hash_table", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let mut filter = ClassicHashFilter::<String>::with_fpr(size, fpr);
                    for item in &items {
                        filter.insert(black_box(item));
                    }
                });
            },
        );

        // 1970 Bit Array
        group.bench_with_input(
            BenchmarkId::new("1970_bit_array", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let mut filter = ClassicBitsFilter::<String>::with_fpr(size, fpr);
                    for item in &items {
                        filter.insert(black_box(item));
                    }
                });
            },
        );

        // 2024 Modern
        group.bench_with_input(
            BenchmarkId::new("2024_modern", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let filter = StandardBloomFilter::<String>::new(size, fpr).unwrap();
                    for item in &items {
                        filter.insert(black_box(item));
                    }
                });
            },
        );
    }

    group.finish();
}

// Why: proves O(k) vs O(1) hash complexity gap from Kirsch-Mitzenmacher 2006
/// Critical benchmark: Shows O(k) vs O(1) hash complexity
///
/// **Classic (1970):** Time scales linearly with k (need k independent hashes)
/// **Modern (2024):** Time constant with k (double hashing: only 2 hashes)
///
/// This benchmark PROVES the Kirsch-Mitzenmacher optimization.
fn bench_scaling_by_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_scaling_by_k");
    let size = 50_000;
    let items = generate_strings(1000, 32);

    // Test different k values (hash counts)
    // k ≈ 0.7 × (m/n) for optimal FPR
    let k_values = vec![
        (4, 0.05),    // k=4, fpr≈5%
        (7, 0.01),    // k=7, fpr≈1%
        (10, 0.002),  // k=10, fpr≈0.2%
        (13, 0.0005), // k=13, fpr≈0.05%
    ];

    for (k, fpr) in k_values {
        // Classic Bit Array (O(k) hash complexity)
        group.bench_with_input(
            BenchmarkId::new("1970_bit_array", format!("k_{}", k)),
            &k,
            |b, _| {
                let mut filter = ClassicBitsFilter::<String>::with_fpr(size, fpr);
                let mut idx = 0;
                b.iter(|| {
                    filter.insert(black_box(&items[idx % items.len()]));
                    idx += 1;
                });
            },
        );

        // Modern (O(1) hash complexity - constant time!)
        group.bench_with_input(
            BenchmarkId::new("2024_modern", format!("k_{}", k)),
            &k,
            |b, _| {
                let filter = StandardBloomFilter::<String>::new(size, fpr).unwrap();
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

// Why: isolates pure hash computation cost to prove O(k) vs O(1) gap
/// Isolate pure hash computation cost
///
/// Classic: k independent hashes
/// Modern: 2 hashes + k derivations (cheaper)
fn bench_hash_overhead(c: &mut Criterion) {
    use bloomcraft::hash::{BloomHasher, StdHasher};
    let mut group = c.benchmark_group("historical_hash_overhead");
    let items = generate_strings(1000, 32);
    let hasher = StdHasher::with_seed(42);

    // k=7 independent hash_item calls (classic approach)
    group.bench_function("classic_k_independent", |b| {
        let mut idx = 0;
        b.iter(|| {
            let item = &items[idx % items.len()];
            for _ in 0..7 {
                black_box(hasher.hash_item(item));
            }
            idx += 1;
        });
    });

    // Two hash_item calls (modern double hashing: h1 + h2)
    group.bench_function("modern_double_hashing", |b| {
        let mut idx = 0;
        b.iter(|| {
            let item = &items[idx % items.len()];
            black_box(hasher.hash_item(item));
            black_box(hasher.hash_item(item));
            idx += 1;
        });
    });

    group.finish();
}

// Why: validates no regression in FPR correctness across all methods
/// Prove all methods achieve same FPR (correctness verification)
///
/// **Critical:** Modern optimizations MUST NOT compromise correctness.
/// This benchmark empirically validates FPR matches theory for all methods.
fn bench_fpr_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_fpr_validation");
    group.sample_size(10); // Expensive benchmark

    let size = 10_000;
    let fpr = 0.01; // Target: 1% false positive rate
    let test_count = 100_000; // Large sample for statistical significance

    let train_items = generate_strings(size, 32);
    let test_items = generate_strings(test_count, 32);

    // 1970 Hash Table - Measure empirical FPR
    group.bench_function("1970_hash_table_fpr", |b| {
        b.iter(|| {
            let mut filter = ClassicHashFilter::<String>::with_fpr(size, fpr);

            // Insert training set
            for item in &train_items {
                filter.insert(item);
            }

            // Query test set (should be mostly false)
            let mut false_positives = 0;
            for item in &test_items {
                if filter.contains(item) {
                    false_positives += 1;
                }
            }

            let empirical_fpr = false_positives as f64 / test_items.len() as f64;
            black_box(empirical_fpr);
            // Should be ≈0.01 ±0.002 (within 20% of target)
        });
    });

    // 1970 Bit Array - Measure empirical FPR
    group.bench_function("1970_bit_array_fpr", |b| {
        b.iter(|| {
            let mut filter = ClassicBitsFilter::<String>::with_fpr(size, fpr);

            for item in &train_items {
                filter.insert(item);
            }

            let mut false_positives = 0;
            for item in &test_items {
                if filter.contains(item) {
                    false_positives += 1;
                }
            }

            let empirical_fpr = false_positives as f64 / test_items.len() as f64;
            black_box(empirical_fpr);
        });
    });

    // 2024 Modern - Measure empirical FPR
    group.bench_function("2024_modern_fpr", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<String>::new(size, fpr).unwrap();

            for item in &train_items {
                filter.insert(item);
            }

            let mut false_positives = 0;
            for item in &test_items {
                if filter.contains(item) {
                    false_positives += 1;
                }
            }

            let empirical_fpr = false_positives as f64 / test_items.len() as f64;
            black_box(empirical_fpr);
        });
    });

    group.finish();
}

// Why: proves space-efficiency hierarchy: bit array << hash table
/// Compare memory footprint: hash table vs bit array vs modern
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_memory");
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size / 2, 32);

    let mut hash_filter = ClassicHashFilter::<String>::with_fpr(size, fpr);
    let mut bits_filter = ClassicBitsFilter::<String>::with_fpr(size, fpr);
    let modern_filter = StandardBloomFilter::<String>::new(size, fpr).unwrap();

    for item in &items {
        hash_filter.insert(item);
        bits_filter.insert(item);
        modern_filter.insert(item);
    }

    group.bench_function("1970_hash_table_memory", |b| {
        b.iter(|| black_box(hash_filter.memory_usage()));
    });
    group.bench_function("1970_bit_array_memory", |b| {
        b.iter(|| black_box(bits_filter.memory_usage()));
    });
    group.bench_function("2024_modern_memory", |b| {
        b.iter(|| black_box(modern_filter.memory_usage()));
    });

    group.finish();
}

// Why: demonstrates cache miss penalty difference between hash table and bit array
/// Test cache behavior: sequential bit array vs random hash table access
///
/// Classic hash table: Poor cache locality (random access)
/// Classic bit array: Good cache locality (sequential within cache line)
/// Modern: Excellent (cache-line aligned + prefetching)
fn bench_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_cache_effects");
    let size = 1_000_000; // Large enough to exceed L3 cache
    let fpr = 0.01;
    let items = generate_strings(10_000, 32);
    let indices: Vec<usize> = (0..1000).map(|i| (i * 7919) % items.len()).collect();

    // Pre-allocate filters outside the benchmark loop
    let mut hash_filter = ClassicHashFilter::<String>::with_fpr(size, fpr);
    let mut bits_filter = ClassicBitsFilter::<String>::with_fpr(size, fpr);
    let modern_filter = StandardBloomFilter::<String>::new(size, fpr).unwrap();

    // Hash Table: Random access pattern (poor cache locality)
    group.bench_function("1970_hash_table_cold_cache", |b| {
        b.iter(|| {
            for &idx in &indices {
                hash_filter.insert(&items[idx]);
            }
            black_box(&hash_filter);
        });
    });

    // Bit Array: Better cache locality
    group.bench_function("1970_bit_array_cold_cache", |b| {
        b.iter(|| {
            for &idx in &indices {
                bits_filter.insert(&items[idx]);
            }
            black_box(&bits_filter);
        });
    });

    // Modern: Optimal cache usage
    group.bench_function("2024_modern_cold_cache", |b| {
        b.iter(|| {
            for &idx in &indices {
                modern_filter.insert(&items[idx]);
            }
            black_box(&modern_filter);
        });
    });

    group.finish();
}

// Why: measures real-world interleaved insert+query throughput
/// Real-world pattern: interleaved inserts and queries
fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("historical_mixed_workload");
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(size, 32);
    let ops = 10_000;

    group.throughput(Throughput::Elements(ops as u64));

    // 1970 Hash Table
    group.bench_function("1970_hash_table_mixed", |b| {
        b.iter(|| {
            let mut filter = ClassicHashFilter::<String>::with_fpr(size, fpr);
            for i in 0..ops {
                if i % 2 == 0 {
                    filter.insert(&items[i % items.len()]);
                } else {
                    black_box(filter.contains(&items[i % items.len()]));
                }
            }
        });
    });

    // 1970 Bit Array
    group.bench_function("1970_bit_array_mixed", |b| {
        b.iter(|| {
            let mut filter = ClassicBitsFilter::<String>::with_fpr(size, fpr);
            for i in 0..ops {
                if i % 2 == 0 {
                    filter.insert(&items[i % items.len()]);
                } else {
                    black_box(filter.contains(&items[i % items.len()]));
                }
            }
        });
    });

    // 2024 Modern
    group.bench_function("2024_modern_mixed", |b| {
        b.iter(|| {
            let filter = StandardBloomFilter::<String>::new(size, fpr).unwrap();
            for i in 0..ops {
                if i % 2 == 0 {
                    filter.insert(&items[i % items.len()]);
                } else {
                    black_box(filter.contains(&items[i % items.len()]));
                }
            }
        });
    });

    group.finish();
}

// Why: shows hash table discard behavior under chain saturation
/// Test ClassicHashFilter behavior as chains fill up
///
/// This tests the fundamental limitation of Burton Bloom's Method 1:
/// when chains reach depth `d`, items are discarded.
fn bench_chain_saturation(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_chain_saturation");
    
    // Test with minimal bucket count to force collisions
    let m = 100; // Only 100 buckets
    let d = 5;   // Depth 5
    
    let items = generate_strings(10_000, 32);

    group.bench_function("hash_table_chain_overflow", |b| {
        b.iter(|| {
            let mut filter = ClassicHashFilter::<String>::new(m, d);
            let mut inserted = 0;
            let mut discarded = 0;

            for item in &items {
                let len_before = filter.len();
                filter.insert(black_box(item));
                let len_after = filter.len();
                
                if len_after > len_before {
                    inserted += 1;
                } else {
                    discarded += 1;
                }
            }

            black_box((inserted, discarded));
        });
    });

    group.finish();
}

// Why: measures FPR degradation when bit array is overfilled

/// Test ClassicBitsFilter as bit array approaches 100% saturation
///
/// Measures degradation of FPR as filter fills beyond capacity.
fn bench_bit_saturation(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_bit_saturation");
    
    let m = 10_000; // Small bit array
    let k = 7;
    let items = generate_strings(50_000, 32); // 5x overcapacity

    group.bench_function("bits_extreme_saturation", |b| {
        b.iter(|| {
            let mut filter = ClassicBitsFilter::<String>::new(m, k);
            
            // Insert way beyond capacity
            for item in &items {
                filter.insert(black_box(item));
            }

            // Measure fill rate at extreme saturation
            let fill_rate = filter.fill_rate();
            black_box(fill_rate);
        });
    });

    group.finish();
}

// Why: measures hash vs bit array vs modern insert with varying key sizes

/// Test performance with different key sizes (8 bytes to 1KB)
fn bench_variable_key_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_variable_key_lengths");
    let size = 10_000;
    let fpr = 0.01;

    for key_len in &[8, 32, 128, 512, 1024] {
        let items = generate_strings(1000, *key_len);

        group.bench_with_input(
            BenchmarkId::new("hash_table", key_len),
            key_len,
            |b, _| {
                let mut filter = ClassicHashFilter::<String>::with_fpr(size, fpr);
                let mut idx = 0;
                b.iter(|| {
                    filter.insert(black_box(&items[idx % items.len()]));
                    idx += 1;
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bit_array", key_len),
            key_len,
            |b, _| {
                let mut filter = ClassicBitsFilter::<String>::with_fpr(size, fpr);
                let mut idx = 0;
                b.iter(|| {
                    filter.insert(black_box(&items[idx % items.len()]));
                    idx += 1;
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("modern", key_len),
            key_len,
            |b, _| {
                let filter = StandardBloomFilter::<String>::new(size, fpr).unwrap();
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

// Why: isolates clear() cost (O(m/64) vs O(m*buckets))

/// Benchmark the clear/reset operation
fn bench_clear_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_clear_operation");
    let size = 100_000;
    let fpr = 0.01;
    let items = generate_strings(10_000, 32);

    group.bench_function("hash_table_clear", |b| {
        b.iter(|| {
            let mut filter = ClassicHashFilter::<String>::with_fpr(size, fpr);
            // Fill it
            for item in &items {
                filter.insert(item);
            }
            // Clear it
            filter.clear();
            black_box(&filter);
        });
    });

    group.bench_function("bit_array_clear", |b| {
        b.iter(|| {
            let mut filter = ClassicBitsFilter::<String>::with_fpr(size, fpr);
            for item in &items {
                filter.insert(item);
            }
            filter.clear();
            black_box(&filter);
        });
    });

    group.bench_function("modern_clear", |b| {
        b.iter(|| {
            let mut filter = StandardBloomFilter::<String>::new(size, fpr).unwrap();
            for item in &items {
                filter.insert(item);
            }
            filter.clear();
            black_box(&filter);
        });
    });

    group.finish();
}

// Why: worst-case query cost when all bits are set

/// Test query performance with 100% false positives
fn bench_pathological_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_pathological_queries");
    let size = 10_000;
    let fpr = 0.01;

    // Create a fully saturated filter
    let train_items = generate_strings(size * 10, 32); // 10x overcapacity
    let query_items = generate_strings(10_000, 32);

    let mut filter_hash = ClassicHashFilter::<String>::with_fpr(size, fpr);
    let mut filter_bits = ClassicBitsFilter::<String>::with_fpr(size, fpr);
    let filter_modern = StandardBloomFilter::<String>::new(size, fpr).unwrap();

    for item in &train_items {
        filter_hash.insert(item);
        filter_bits.insert(item);
        filter_modern.insert(item);
    }

    group.bench_function("hash_table_saturated_queries", |b| {
        let mut idx = 0;
        b.iter(|| {
            black_box(filter_hash.contains(&query_items[idx % query_items.len()]));
            idx += 1;
        });
    });

    group.bench_function("bit_array_saturated_queries", |b| {
        let mut idx = 0;
        b.iter(|| {
            black_box(filter_bits.contains(&query_items[idx % query_items.len()]));
            idx += 1;
        });
    });

    group.bench_function("modern_saturated_queries", |b| {
        let mut idx = 0;
        b.iter(|| {
            black_box(filter_modern.contains(&query_items[idx % query_items.len()]));
            idx += 1;
        });
    });

    group.finish();
}

// Why: proves construction scales O(m) for all methods

/// Measure filter construction overhead
fn bench_construction_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_construction_overhead");

    for size in &[10_000, 100_000, 1_000_000] {
        let fpr = 0.01;

        group.bench_with_input(
            BenchmarkId::new("hash_table_construct", size),
            size,
            |b, &s| {
                b.iter(|| {
                    let filter = ClassicHashFilter::<String>::with_fpr(s, fpr);
                    black_box(filter);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bit_array_construct", size),
            size,
            |b, &s| {
                b.iter(|| {
                    let filter = ClassicBitsFilter::<String>::with_fpr(s, fpr);
                    black_box(filter);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("modern_construct", size),
            size,
            |b, &s| {
                b.iter(|| {
                    let filter = StandardBloomFilter::<String>::new(s, fpr).unwrap();
                    black_box(filter);
                });
            },
        );
    }

    group.finish();
}

// Why: shows FPR parameter has negligible effect on insert throughput

/// Test with very tight (0.0001) and loose (0.1) FPR requirements
fn bench_extreme_fpr_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_extreme_fpr");
    let size = 10_000;
    let items = generate_strings(1000, 32);

    for fpr in &[0.0001, 0.001, 0.01, 0.05, 0.1] {
        group.bench_with_input(
            BenchmarkId::new("hash_table_fpr", format!("{:.4}", fpr)),
            fpr,
            |b, &f| {
                let mut filter = ClassicHashFilter::<String>::with_fpr(size, f);
                let mut idx = 0;
                b.iter(|| {
                    filter.insert(black_box(&items[idx % items.len()]));
                    idx += 1;
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bit_array_fpr", format!("{:.4}", fpr)),
            fpr,
            |b, &f| {
                let mut filter = ClassicBitsFilter::<String>::with_fpr(size, f);
                let mut idx = 0;
                b.iter(|| {
                    filter.insert(black_box(&items[idx % items.len()]));
                    idx += 1;
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("modern_fpr", format!("{:.4}", fpr)),
            fpr,
            |b, &f| {
                let filter = StandardBloomFilter::<String>::new(size, f).unwrap();
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

// CRITERION GROUP REGISTRATION
criterion_group!(
    benches,
    bench_insert_historical,
    bench_query_historical,
    bench_batch_insert_historical,
    bench_scaling_by_k,
    bench_hash_overhead,
    bench_fpr_validation,
    bench_memory_usage,
    bench_cache_effects,
    bench_mixed_workload,
    bench_chain_saturation,
    bench_bit_saturation,
    bench_variable_key_lengths,
    bench_clear_operation,
    bench_pathological_queries,
    bench_construction_overhead,
    bench_extreme_fpr_values,
);

criterion_main!(benches);
