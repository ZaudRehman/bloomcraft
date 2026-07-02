//! Benchmarks for ScalableBloomFilter.
//!
//! Run all:       cargo bench --bench scalable_filter
//! Run one group: cargo bench --bench scalable_filter -- sbf/insert
//! HTML reports:  target/criterion/report/index.html
//!
//! # Coverage map
//!
//! ## Insert
//!  1. sbf/insert/sequential          — throughput at 1k/10k/100k/1M items
//!  2. sbf/insert/batch_vs_sequential — insert_batch vs loop at 100–100k
//!  3. sbf/insert/fast                — insert_fast (no total_items update) vs insert
//!  4. sbf/insert/growth_event        — latency of the single insert that fires growth
//!  5. sbf/insert/post_clear          — first 1k inserts immediately after clear
//!
//! ## Contains
//!  6. sbf/contains/hit               — hit path, depth 1/4/6/8
//!  7. sbf/contains/miss              — miss path, depth 1/4/6/8
//!  8. sbf/contains/mixed             — 50/50 at 1k/10k/100k items
//!  9. sbf/contains/batch             — contains_batch, 10/100/1k/10k items
//! 10. sbf/contains/provenance        — contains_with_provenance, hit/miss
//! 11. sbf/contains/fpr_sensitivity   — hit rate vs FPR target (0.1/0.01/0.001/0.0001)
//!
//! ## Query strategy
//! 12. sbf/strategy/query             — Forward vs Reverse, recent/old/miss
//! 13. sbf/strategy/recency_bias      — 90% recent vs 90% old item queries
//!
//! ## Growth
//! 14. sbf/growth/strategies          — Constant/Geometric 2x,1.5x/Bounded/Adaptive insert throughput
//! 15. sbf/growth/error_ratio         — insert throughput at r=0.3/0.5/0.7/0.9
//! 16. sbf/growth/fill_threshold      — insert throughput at threshold 0.45/0.5/0.6/0.7/0.9
//!
//! ## Capacity
//! 17. sbf/capacity_exhausted         — Silent/Error at saturation
//!
//! ## Analytics
//! 18. sbf/analytics/health_metrics   — health_metrics() at 4/6/8 filters
//! 19. sbf/analytics/predict_fpr      — predict_fpr() at 100k–100M
//! 20. sbf/analytics/fpr_breakdown    — filter_fpr_breakdown() at 4/6/8 filters
//! 21. sbf/analytics/cardinality      — estimate_unique_count(), 1x/3x dups
//! 22. sbf/analytics/filter_stats     — filter_stats() at 4/6/8 filters
//! 23. sbf/analytics/memory_usage     — memory_usage() at 4/6/8 filters
//! 24. sbf/analytics/aggregate_fill   — aggregate_fill_rate() at 4/6/8 filters
//! 25. sbf/analytics/fpr_estimators   — estimate_fpr_exact vs max_fpr vs estimate_fpr
//!
//! ## Maintenance
//! 26. sbf/clear                      — clear() at 1/4/6/8 filters
//! 27. sbf/clone                      — clone() at 4/6/8 filters
//!
//! ## Key types
//! 28. sbf/key_types                  — u64/fixed-string/URL hit+miss
//!
//! ## Real-world
//! 29. real_world/url_dedup           — 90%-seen + cold-miss + batch-100
//! 30. real_world/log_dedup           — ingest + window rotation
//! 31. real_world/session_revocation  — valid + revoked session check
//! 32. real_world/recommendation      — filter 100/1k candidates
//! 33. real_world/write_pipeline      — cold-start/warm/deep 1k inserts

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::filters::{
    CapacityExhaustedBehavior, GrowthStrategy, QueryStrategy, ScalableBloomFilter,
};

// ============================================================================
// DATASET GENERATORS
// ============================================================================

fn gen_u64(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn gen_strings(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("key:{:016x}", i)).collect()
}

fn gen_urls(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            format!(
                "https://example.com/path/{}/res?id={}&ts={}",
                i % 1_000,
                i,
                i.wrapping_mul(6_364_136_223_846_793_005)
            )
        })
        .collect()
}

fn gen_session_ids(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            format!(
                "{:032x}",
                (i as u128).wrapping_mul(0x9e3779b97f4a7c15u128)
                    ^ 0xd1b54a32d192ed03u128
            )
        })
        .collect()
}

fn gen_log_lines(n: usize) -> Vec<String> {
    let tpl = [
        "ERROR conn refused host:{}",
        "WARN  slow query {}ms on users",
        "INFO  status 200 in {}ms",
        "DEBUG cache miss key:{}",
        "ERROR timeout svc:{} after {}ms",
    ];
    (0..n)
        .map(|i| tpl[i % tpl.len()].replace("{}", &i.to_string()))
        .collect()
}

// ============================================================================
// HELPERS
// ============================================================================

fn populated_sbf(initial: usize, fpr: f64, n: usize) -> ScalableBloomFilter<u64> {
    let mut f = ScalableBloomFilter::<u64>::new(initial, fpr).unwrap();
    for i in 0..n as u64 {
        f.insert(&i);
    }
    f
}

// ============================================================================
// 1. SINGLE INSERT — SEQUENTIAL THROUGHPUT
// ============================================================================

fn bench_sbf_insert_sequential(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/insert/sequential");
    g.sample_size(10);

    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let keys = gen_u64(n);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_batched(
                || ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
                |mut f| {
                    for k in &keys {
                        f.insert(black_box(k));
                    }
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ============================================================================
// 2. BATCH INSERT vs SEQUENTIAL
// ============================================================================

fn bench_sbf_insert_batch_vs_sequential(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/insert/batch_vs_sequential");

    for &n in &[100usize, 1_000, 10_000, 100_000] {
        let keys = gen_u64(n);
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("batch", n), &n, |b, _| {
            b.iter_batched(
                || (ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(), keys.clone()),
                |(mut f, ks)| {
                    f.insert_batch(black_box(&ks)).unwrap();
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, _| {
            b.iter_batched(
                || (ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(), keys.clone()),
                |(mut f, ks)| {
                    for k in black_box(ks) {
                        f.insert(&k);
                    }
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ============================================================================
// 3. INSERT FAST vs INSERT
//
// insert_fast skips the total_items increment — the doc comment claims ~5-10%
// throughput improvement. This bench verifies that claim and quantifies the
// exact bookkeeping overhead per insert.
// ============================================================================

fn bench_sbf_insert_fast(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/insert/fast");
    g.sample_size(10);

    for &n in &[10_000usize, 100_000, 1_000_000] {
        let keys = gen_u64(n);
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("insert", n), &n, |b, _| {
            b.iter_batched(
                || ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
                |mut f| {
                    for k in &keys {
                        f.insert(black_box(k));
                    }
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("insert_fast", n), &n, |b, _| {
            b.iter_batched(
                || ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
                |mut f| {
                    for k in &keys {
                        f.insert_fast(black_box(k));
                    }
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ============================================================================
// 4. GROWTH EVENT LATENCY — TAIL SPIKE MEASUREMENT
//
// Throughput benchmarks average over many growth events and mask the tail cost
// of a single growth trigger. This bench measures the exact latency of the
// insert that crosses the fill threshold.
//
// With initial_capacity=1_000 and fill_threshold=0.5, the first growth fires
// at approximately 500 items. We prime to (threshold - 1) then measure the
// single insert that triggers try_add_filter.
//
// After optimisation this should be sub-microsecond. A suspicious spike here
// means something is allocating or blocking under the growth lock.
// ============================================================================

fn bench_sbf_insert_growth_event(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/insert/growth_event");
    // Low sample count — we want the raw cost of one growth, not a steady-state
    // average.
    g.sample_size(10);

    // threshold ≈ ceil(1_000 * 0.5) = 500; prime to 499 items.
    let prime_count: u64 = 499;

    g.bench_function("filter-1-to-2", |b| {
        b.iter_batched(
            || {
                let mut f = ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
                for i in 0..prime_count {
                    f.insert(&i);
                }
                f
            },
            |mut f| {
                // This single insert crosses the threshold.
                f.insert(black_box(&prime_count));
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // Also measure growth at depth-4 → depth-5, where the new filter is 16×
    // larger (Geometric 2.0: initial × 2^4 = 16×).
    // At depth-4 the total items is approximately 7_500; threshold for filter-4
    // fires at approximately 7_000.
    let prime_count_deep: u64 = 7_000;
    g.bench_function("filter-4-to-5", |b| {
        b.iter_batched(
            || {
                let mut f = ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
                for i in 0..prime_count_deep {
                    f.insert(&i);
                }
                f
            },
            |mut f| {
                f.insert(black_box(&prime_count_deep));
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 5. INSERT AFTER CLEAR
//
// clear() deallocates all sub-filters and re-allocates one fresh one.
// The first inserts after a clear exercise the cold-allocation path:
// the sub-filter's bit array may have been evicted from cache and the
// growth counter must be rebuilt from fill_rate(). This bench isolates
// that re-warm cost from steady-state insert throughput.
// ============================================================================

fn bench_sbf_insert_post_clear(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/insert/post_clear");
    g.throughput(Throughput::Elements(1_000));

    // Warm filter: insert 50k, clear, then insert 1k.
    g.bench_function("warm_then_clear/1k_inserts", |b| {
        b.iter_batched(
            || {
                let mut f = ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
                for i in 0..50_000u64 {
                    f.insert(&i);
                }
                f.clear();
                f
            },
            |mut f| {
                for i in 0..1_000u64 {
                    f.insert(black_box(&i));
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // Compare with a cold-start filter at the same insert count.
    g.bench_function("cold_start/1k_inserts", |b| {
        b.iter_batched(
            || ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
            |mut f| {
                for i in 0..1_000u64 {
                    f.insert(black_box(&i));
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 6. CONTAINS — HIT PATH
//
// Item counts are calibrated to actual filter depths.
// With initial_capacity=1_000, fill_threshold=0.5 (bit-based), and
// Geometric(2.0) growth, filters fill at approximately 1× their capacity
// (50% bit fill ≈ full intended item count). Depths:
//   500 items  → 1 filter
//   7_500      → 4 filters
//   31_500     → 6 filters
//   127_000    → 8 filters
// ============================================================================

fn bench_sbf_contains_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/contains/hit");

    let scenarios: &[(usize, usize, &str)] = &[
        (1_000,   500,    "depth-1"),
        (1_000,   7_500,  "depth-4"),
        (1_000,  31_500,  "depth-6"),
        (1_000, 127_000,  "depth-8"),
    ];

    for &(cap, n, label) in scenarios {
        let f = populated_sbf(cap, 0.01, n);
        g.throughput(Throughput::Elements(1));
        g.bench_function(label, |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = idx % n as u64;
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });
    }
    g.finish();
}

// ============================================================================
// 7. CONTAINS — MISS PATH
// ============================================================================

fn bench_sbf_contains_miss(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/contains/miss");

    let scenarios: &[(usize, usize, &str)] = &[
        (1_000,   500,    "depth-1"),
        (1_000,   7_500,  "depth-4"),
        (1_000,  31_500,  "depth-6"),
        (1_000, 127_000,  "depth-8"),
    ];

    for &(cap, n, label) in scenarios {
        let f = populated_sbf(cap, 0.01, n);
        let base_miss = (n as u64).wrapping_mul(1_000_000);
        g.throughput(Throughput::Elements(1));
        g.bench_function(label, |b| {
            let mut idx = 0u64;
            b.iter(|| {
                idx = idx.wrapping_add(1);
                let key = base_miss.wrapping_add(idx);
                black_box(f.contains(black_box(&key)))
            });
        });
    }
    g.finish();
}

// ============================================================================
// 8. CONTAINS — MIXED WORKLOAD (50/50)
// ============================================================================

fn bench_sbf_contains_mixed(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/contains/mixed");

    for &n in &[1_000usize, 10_000, 100_000] {
        let f = populated_sbf(1_000, 0.01, n);
        g.throughput(Throughput::Elements(1));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = if idx % 2 == 0 {
                    idx / 2 % n as u64
                } else {
                    n as u64 * 10 + idx
                };
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });
    }
    g.finish();
}

// ============================================================================
// 9. BATCH CONTAINS — VARYING BATCH SIZE
// ============================================================================

fn bench_sbf_contains_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/contains/batch");
    const TOTAL: usize = 50_000;
    let f = populated_sbf(1_000, 0.01, TOTAL);

    for &n in &[10usize, 100, 1_000, 10_000] {
        let step = TOTAL as u64 / n as u64;
        let hits: Vec<u64> = (0..n as u64).map(|i| i * step).collect();
        let misses: Vec<u64> = (TOTAL as u64 * 10..TOTAL as u64 * 10 + n as u64).collect();

        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("hits", n), &n, |b, _| {
            b.iter_batched(
                || hits.clone(),
                |ks| black_box(f.contains_batch(black_box(&ks))),
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("misses", n), &n, |b, _| {
            b.iter_batched(
                || misses.clone(),
                |ks| black_box(f.contains_batch(black_box(&ks))),
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ============================================================================
// 10. CONTAINS WITH PROVENANCE
// ============================================================================

fn bench_sbf_contains_provenance(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/contains/provenance");
    const N: usize = 50_000;
    let f = populated_sbf(1_000, 0.01, N);

    g.throughput(Throughput::Elements(1));
    g.bench_function("hit/newest", |b| {
        let key = (N - 1) as u64;
        b.iter(|| black_box(f.contains_with_provenance(black_box(&key))))
    });
    g.bench_function("hit/oldest", |b| {
        let key = 0u64;
        b.iter(|| black_box(f.contains_with_provenance(black_box(&key))))
    });
    g.bench_function("miss", |b| {
        let miss = N as u64 * 1_000_000;
        b.iter(|| black_box(f.contains_with_provenance(black_box(&miss))))
    });
    g.finish();
}

// ============================================================================
// 11. CONTAINS — FPR SENSITIVITY
//
// The FPR target affects how many bits per item each sub-filter allocates,
// which in turn determines the hash count k and the cost of each contains()
// call. A filter at 0.0001 has more bits per item and more hash evaluations
// per query than one at 0.1.
//
// This bench isolates that dimension by holding depth constant (same number
// of items relative to initial capacity) while varying target_fpr.
// ============================================================================

fn bench_sbf_contains_fpr_sensitivity(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/contains/fpr_sensitivity");
    // 4-filter depth for all variants to make the comparison fair.
    const N: usize = 7_500;

    for &(fpr, label) in &[
        (0.1f64,    "fpr_0.1"),
        (0.01,      "fpr_0.01"),
        (0.001,     "fpr_0.001"),
        (0.0001,    "fpr_0.0001"),
    ] {
        let f = populated_sbf(1_000, fpr, N);
        let miss_key = (N as u64).wrapping_mul(1_000_000);

        g.throughput(Throughput::Elements(1));
        g.bench_function(format!("{label}/hit"), |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = idx % N as u64;
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });
        g.bench_function(format!("{label}/miss"), |b| {
            let mut idx = 0u64;
            b.iter(|| {
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&(miss_key.wrapping_add(idx)))))
            });
        });
    }
    g.finish();
}

// ============================================================================
// 12. QUERY STRATEGY: FORWARD vs REVERSE
// ============================================================================

fn bench_sbf_query_strategy(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/strategy/query");
    const N: usize = 50_000;

    for &(strategy, label) in &[
        (QueryStrategy::Reverse, "reverse"),
        (QueryStrategy::Forward, "forward"),
    ] {
        let mut f = ScalableBloomFilter::<u64>::new(1_000, 0.01)
            .unwrap()
            .with_query_strategy(strategy);
        for i in 0..N as u64 {
            f.insert(&i);
        }

        let recent = (N - 1) as u64;
        let old    = 0u64;

        g.throughput(Throughput::Elements(1));
        g.bench_function(format!("{label}/recent"), |b| {
            b.iter(|| black_box(f.contains(black_box(&recent))))
        });
        g.bench_function(format!("{label}/old"), |b| {
            b.iter(|| black_box(f.contains(black_box(&old))))
        });
        g.bench_function(format!("{label}/miss"), |b| {
            let miss = N as u64 * 1_000;
            b.iter(|| black_box(f.contains(black_box(&miss))))
        });
    }
    g.finish();
}

// ============================================================================
// 13. RECENCY BIAS — FORWARD vs REVERSE WITH SKEWED ACCESS PATTERN
//
// bench 12 tests a single fixed key. This bench tests a realistic access
// pattern where 90% of queries target recently-inserted items (the last 10%
// of the key space) and 10% target old items. It verifies that QueryStrategy
// choices translate to real-world throughput differences, not just microbench
// arithmetic.
// ============================================================================

fn bench_sbf_strategy_recency_bias(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/strategy/recency_bias");
    const N: usize = 50_000;
    // "Recent" = top 10% of inserted keys (most likely in the newest sub-filter).
    const RECENT_CUTOFF: u64 = (N as u64 * 9) / 10;

    for &(strategy, label) in &[
        (QueryStrategy::Reverse, "reverse"),
        (QueryStrategy::Forward, "forward"),
    ] {
        let mut f = ScalableBloomFilter::<u64>::new(1_000, 0.01)
            .unwrap()
            .with_query_strategy(strategy);
        for i in 0..N as u64 {
            f.insert(&i);
        }

        g.throughput(Throughput::Elements(1));

        // 90% recent: most queries hit the newest sub-filter (Reverse wins).
        g.bench_function(format!("{label}/90pct_recent"), |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = if idx % 10 == 0 {
                    idx / 10 % RECENT_CUTOFF // old item (10% of queries)
                } else {
                    RECENT_CUTOFF + (idx % (N as u64 - RECENT_CUTOFF)) // recent
                };
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });

        // 90% old: most queries hit the oldest sub-filter (Forward wins).
        g.bench_function(format!("{label}/90pct_old"), |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = if idx % 10 == 0 {
                    RECENT_CUTOFF + (idx / 10 % (N as u64 - RECENT_CUTOFF)) // recent
                } else {
                    idx % RECENT_CUTOFF // old
                };
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });
    }
    g.finish();
}

// ============================================================================
// 14. GROWTH STRATEGIES — INSERT THROUGHPUT
// ============================================================================

fn bench_sbf_growth_strategies(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/growth/strategies");
    const N: usize = 50_000;

    let strategies: &[(GrowthStrategy, &str)] = &[
        (GrowthStrategy::Constant, "constant"),
        (GrowthStrategy::Geometric(2.0), "geometric_2x"),
        (GrowthStrategy::Geometric(1.5), "geometric_1.5x"),
        (
            GrowthStrategy::Bounded { scale: 2.0, max_filter_size: 10_000 },
            "bounded_10k",
        ),
        (
            GrowthStrategy::Adaptive { initial_ratio: 0.5, min_ratio: 0.3, max_ratio: 0.9 },
            "adaptive",
        ),
    ];

    for &(strategy, label) in strategies {
        g.throughput(Throughput::Elements(N as u64));
        g.bench_function(label, |b| {
            b.iter_batched(
                || ScalableBloomFilter::<u64>::with_strategy(1_000, 0.01, 0.5, strategy).unwrap(),
                |mut f| {
                    for i in 0..N as u64 {
                        f.insert(black_box(&i));
                    }
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ============================================================================
// 15. ERROR RATIO — INSERT THROUGHPUT
//
// The error_ratio r controls how many bits each successive sub-filter uses:
// lower r → more bits per item → higher k → slower insert and contains.
// This bench quantifies the throughput cost of choosing a tighter r.
//
// All variants use Geometric(2.0) growth and the same item count so the only
// variable is k, which grows as log2(1/r^i) per filter index i.
// ============================================================================

fn bench_sbf_growth_error_ratio(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/growth/error_ratio");
    g.sample_size(10);
    const N: usize = 50_000;

    for &(ratio, label) in &[
        (0.9f64, "r_0.9"),
        (0.7,    "r_0.7"),
        (0.5,    "r_0.5"),
        (0.3,    "r_0.3"),
    ] {
        g.throughput(Throughput::Elements(N as u64));
        g.bench_function(label, |b| {
            b.iter_batched(
                || {
                    ScalableBloomFilter::<u64>::with_strategy(
                        1_000, 0.01, ratio, GrowthStrategy::Geometric(2.0),
                    )
                    .unwrap()
                },
                |mut f| {
                    for i in 0..N as u64 {
                        f.insert(black_box(&i));
                    }
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ============================================================================
// 16. FILL THRESHOLD — INSERT THROUGHPUT
//
// fill_threshold controls when growth fires. A higher threshold means fewer,
// denser sub-filters (worse FPR per filter, fewer growth events, higher k on
// contains). A lower threshold means more filters but each is less dense.
//
// This bench measures total insert throughput across the threshold range.
// The doc warns that values below 0.45 invalidate the FPR bound; we skip 0.4
// and stay within the validated range.
// ============================================================================

fn bench_sbf_growth_fill_threshold(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/growth/fill_threshold");
    g.sample_size(10);
    const N: usize = 50_000;

    for &(threshold, label) in &[
        (0.45f64, "threshold_0.45"),
        (0.5,     "threshold_0.50"),
        (0.6,     "threshold_0.60"),
        (0.7,     "threshold_0.70"),
        (0.9,     "threshold_0.90"),
    ] {
        g.throughput(Throughput::Elements(N as u64));
        g.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let mut f = ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
                    f.set_fill_threshold(threshold).unwrap();
                    f
                },
                |mut f| {
                    for i in 0..N as u64 {
                        f.insert(black_box(&i));
                    }
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ============================================================================
// 17. CAPACITY EXHAUSTED BEHAVIOR
// ============================================================================

fn bench_sbf_capacity_behavior(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/capacity_exhausted");

    let build_saturated = |behavior: CapacityExhaustedBehavior| -> ScalableBloomFilter<u64> {
        let mut f = ScalableBloomFilter::<u64>::with_strategy(
            1,
            0.5,
            0.5,
            GrowthStrategy::Bounded { scale: 1.01, max_filter_size: 5 },
        )
        .unwrap()
        .with_capacity_behavior(behavior);
        for i in 0..10_000u64 {
            f.insert(&i);
            if f.is_at_max_capacity() {
                break;
            }
        }
        f
    };

    let silent = build_saturated(CapacityExhaustedBehavior::Silent);
    let error  = build_saturated(CapacityExhaustedBehavior::Error);

    g.throughput(Throughput::Elements(1));

    g.bench_function("silent/saturated_insert", |b| {
        let mut f = silent.clone();
        let mut i = 100_000u64;
        b.iter(|| {
            i = i.wrapping_add(1);
            f.insert(black_box(&i));
        });
    });

    g.bench_function("error/saturated_insert_checked", |b| {
        let mut f = error.clone();
        let mut i = 100_000u64;
        b.iter(|| {
            i = i.wrapping_add(1);
            let _ = black_box(f.insert_checked(black_box(&i)));
        });
    });

    g.finish();
}

// ============================================================================
// 18. HEALTH METRICS
// ============================================================================

fn bench_sbf_health_metrics(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/analytics/health_metrics");

    for &(cap, n, label) in &[
        (1_000usize,  7_500usize, "4-filters"),
        (1_000,      31_500,      "6-filters"),
        (1_000,     127_000,      "8-filters"),
    ] {
        let f = populated_sbf(cap, 0.01, n);
        g.bench_function(label, |b| b.iter(|| black_box(f.health_metrics())));
    }
    g.finish();
}

// ============================================================================
// 19. FPR PREDICTION
// ============================================================================

fn bench_sbf_predict_fpr(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/analytics/predict_fpr");
    let f = populated_sbf(1_000, 0.01, 50_000);

    for &target in &[100_000usize, 1_000_000, 10_000_000, 100_000_000] {
        g.bench_with_input(BenchmarkId::from_parameter(target), &target, |b, &t| {
            b.iter(|| black_box(f.predict_fpr(black_box(t))))
        });
    }
    g.finish();
}

// ============================================================================
// 20. FPR BREAKDOWN
// ============================================================================

fn bench_sbf_fpr_breakdown(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/analytics/fpr_breakdown");

    for &(cap, n, label) in &[
        (1_000usize,  7_500usize, "4-filters"),
        (1_000,      31_500,      "6-filters"),
        (1_000,     127_000,      "8-filters"),
    ] {
        let f = populated_sbf(cap, 0.01, n);
        g.bench_function(label, |b| b.iter(|| black_box(f.filter_fpr_breakdown())));
    }
    g.finish();
}

// ============================================================================
// 21. CARDINALITY ESTIMATION
// ============================================================================

fn bench_sbf_cardinality(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/analytics/cardinality");

    for &(unique, reps, label) in &[
        (10_000usize, 1usize, "10k_unique_1x"),
        (10_000, 3,           "10k_unique_3x_dups"),
        (100_000, 1,          "100k_unique_1x"),
    ] {
        let mut f = ScalableBloomFilter::<u64>::new(10_000, 0.01)
            .unwrap()
            .with_cardinality_tracking();
        for _ in 0..reps {
            for i in 0..unique as u64 {
                f.insert(&i);
            }
        }
        g.bench_function(label, |b| b.iter(|| black_box(f.estimate_unique_count())));
    }
    g.finish();
}

// ============================================================================
// 22. FILTER STATS
// ============================================================================

fn bench_sbf_filter_stats(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/analytics/filter_stats");

    for &(cap, n, label) in &[
        (1_000usize,  7_500usize, "4-filters"),
        (1_000,      31_500,      "6-filters"),
        (1_000,     127_000,      "8-filters"),
    ] {
        let f = populated_sbf(cap, 0.01, n);
        g.bench_function(label, |b| b.iter(|| black_box(f.filter_stats())));
    }
    g.finish();
}

// ============================================================================
// 23. MEMORY USAGE
// ============================================================================

fn bench_sbf_memory_usage(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/analytics/memory_usage");

    for &(cap, n, label) in &[
        (1_000usize,  7_500usize, "4-filters"),
        (1_000,      31_500,      "6-filters"),
        (1_000,     127_000,      "8-filters"),
    ] {
        let f = populated_sbf(cap, 0.01, n);
        g.bench_function(label, |b| b.iter(|| black_box(f.memory_usage())));
    }
    g.finish();
}

// ============================================================================
// 24. AGGREGATE FILL RATE
//
// aggregate_fill_rate() iterates all sub-filters, sums total bits and set bits,
// and divides. This is O(n_filters × m/64) — more expensive than the O(1)
// current_fill_rate(). This bench quantifies that cost at different depths so
// callers know whether to call it on a hot path.
// ============================================================================

fn bench_sbf_aggregate_fill_rate(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/analytics/aggregate_fill_rate");

    for &(cap, n, label) in &[
        (1_000usize,  7_500usize, "4-filters"),
        (1_000,      31_500,      "6-filters"),
        (1_000,     127_000,      "8-filters"),
    ] {
        let f = populated_sbf(cap, 0.01, n);
        g.bench_function(label, |b| b.iter(|| black_box(f.aggregate_fill_rate())));
    }
    g.finish();
}

// ============================================================================
// 25. FPR ESTIMATORS — estimate_fpr_exact vs max_fpr vs estimate_fpr
//
// The filter provides three FPR estimators with different costs and semantics:
// - estimate_fpr_exact / estimate_fpr: complement rule (tighter, iterates filters)
// - max_fpr: union bound (faster, just a sum)
//
// This bench tells callers which to use on hot monitoring paths. max_fpr
// should be cheaper because it avoids the multiplication chain.
// ============================================================================

fn bench_sbf_fpr_estimators(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/analytics/fpr_estimators");
    // 8-filter depth gives the most interesting difference between the methods.
    let f = populated_sbf(1_000, 0.01, 127_000);

    g.bench_function("estimate_fpr_exact", |b| {
        b.iter(|| black_box(f.estimate_fpr_exact()))
    });
    g.bench_function("estimate_fpr_alias", |b| {
        b.iter(|| black_box(f.estimate_fpr()))
    });
    g.bench_function("max_fpr", |b| {
        b.iter(|| black_box(f.max_fpr()))
    });
    g.finish();
}

// ============================================================================
// 26. CLEAR
// ============================================================================

fn bench_sbf_clear(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/clear");

    for &(cap, n, label) in &[
        (1_000usize, 1_000usize,  "1-filter"),
        (1_000,       7_500,      "4-filters"),
        (1_000,      31_500,      "6-filters"),
        (1_000,     127_000,      "8-filters"),
    ] {
        g.bench_function(label, |b| {
            b.iter_batched(
                || populated_sbf(cap, 0.01, n),
                |mut f| black_box(f.clear()),
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ============================================================================
// 27. CLONE COST
// ============================================================================

fn bench_sbf_clone(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/clone");

    for &(cap, n, label) in &[
        (1_000usize,  7_500usize, "4-filters"),
        (1_000,      31_500,      "6-filters"),
        (1_000,     127_000,      "8-filters"),
    ] {
        let f = populated_sbf(cap, 0.01, n);
        g.bench_function(label, |b| b.iter(|| black_box(f.clone())));
    }
    g.finish();
}

// ============================================================================
// 28. KEY TYPES — HASH COST
// ============================================================================

fn bench_sbf_key_types(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/key_types");
    const N: usize = 50_000;

    let f_int = populated_sbf(1_000, 0.01, N);

    let str_keys = gen_strings(N);
    let mut f_str: ScalableBloomFilter<String> = ScalableBloomFilter::new(1_000, 0.01).unwrap();
    for s in &str_keys {
        f_str.insert(s);
    }

    let url_keys = gen_urls(N);
    let mut f_url: ScalableBloomFilter<String> = ScalableBloomFilter::new(1_000, 0.01).unwrap();
    for s in &url_keys {
        f_url.insert(s);
    }

    let miss_int = N as u64 * 999_999;
    let miss_str = "key:ffffffffffffffff_absent".to_string();
    let miss_url = "https://absent.example.invalid/path".to_string();

    g.throughput(Throughput::Elements(1));

    g.bench_function("u64/hit", |b| {
        let k = (N / 2) as u64;
        b.iter(|| black_box(f_int.contains(black_box(&k))))
    });
    g.bench_function("u64/miss", |b| {
        b.iter(|| black_box(f_int.contains(black_box(&miss_int))))
    });
    g.bench_function("string/hit", |b| {
        let k = str_keys[N / 2].clone();
        b.iter(|| black_box(f_str.contains(black_box(&k))))
    });
    g.bench_function("string/miss", |b| {
        b.iter(|| black_box(f_str.contains(black_box(&miss_str))))
    });
    g.bench_function("url/hit", |b| {
        let k = url_keys[N / 2].clone();
        b.iter(|| black_box(f_url.contains(black_box(&k))))
    });
    g.bench_function("url/miss", |b| {
        b.iter(|| black_box(f_url.contains(black_box(&miss_url))))
    });

    g.finish();
}

// ============================================================================
// 29. REAL-WORLD: WEB CRAWLER URL DEDUPLICATION
// ============================================================================

fn bench_real_world_url_dedup(c: &mut Criterion) {
    let mut g = c.benchmark_group("real_world/url_dedup");
    const VISITED: usize = 500_000;

    let visited  = gen_urls(VISITED);
    let new_urls: Vec<String> = gen_urls(VISITED)
        .into_iter()
        .map(|u| format!("{}/unseen", u))
        .collect();

    let mut f: ScalableBloomFilter<&str> =
        ScalableBloomFilter::new(100_000, 0.00001).unwrap();
    for url in &visited {
        f.insert(&url.as_str());
    }

    g.throughput(Throughput::Elements(1));

    g.bench_function("check/90pct_seen", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let url: &str = if idx % 10 == 0 {
                new_urls[idx / 10 % new_urls.len()].as_str()
            } else {
                visited[idx % VISITED].as_str()
            };
            idx = idx.wrapping_add(1);
            black_box(f.contains(black_box(&url)))
        });
    });

    g.bench_function("check/cold_all_miss", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let url = new_urls[idx % new_urls.len()].as_str();
            idx = idx.wrapping_add(1);
            black_box(f.contains(black_box(&url)))
        });
    });

    g.throughput(Throughput::Elements(100));
    g.bench_function("batch_check/100_urls", |b| {
        let batch: Vec<&str> = visited[..100].iter().map(|s| s.as_str()).collect();
        b.iter_batched(
            || batch.clone(),
            |ks| black_box(f.contains_batch(black_box(&ks))),
            criterion::BatchSize::SmallInput,
        );
    });

    g.finish();
}

// ============================================================================
// 30. REAL-WORLD: LOG DEDUPLICATION PIPELINE
// ============================================================================

fn bench_real_world_log_dedup(c: &mut Criterion) {
    let mut g = c.benchmark_group("real_world/log_dedup");
    const WINDOW: usize = 100_000;

    let logs = gen_log_lines(WINDOW * 2);

    g.throughput(Throughput::Elements(1));
    g.bench_function("ingest_single_entry", |b| {
        b.iter_batched(
            || {
                let mut f: ScalableBloomFilter<&str> =
                    ScalableBloomFilter::new(WINDOW, 0.001).unwrap();
                for log in &logs[..WINDOW / 5] {
                    f.insert(&log.as_str());
                }
                f
            },
            |mut f| {
                let log = black_box(logs[WINDOW / 5 + 1].as_str());
                let is_dup = f.contains(&log);
                if !is_dup {
                    f.insert(&log);
                }
                black_box((f, is_dup))
            },
            criterion::BatchSize::LargeInput,
        )
    });

    g.bench_function("window_rotation/clear_reseed_1k", |b| {
        b.iter_batched(
            || {
                let mut f: ScalableBloomFilter<&str> =
                    ScalableBloomFilter::new(WINDOW, 0.001).unwrap();
                for log in &logs[..WINDOW] {
                    f.insert(&log.as_str());
                }
                f
            },
            |mut f| {
                f.clear();
                for log in &logs[WINDOW..WINDOW + 1_000] {
                    f.insert(&log.as_str());
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    g.finish();
}

// ============================================================================
// 31. REAL-WORLD: SESSION REVOCATION CHECK
// ============================================================================

fn bench_real_world_session_check(c: &mut Criterion) {
    let mut g = c.benchmark_group("real_world/session_revocation");
    const REVOKED: usize = 50_000;

    let revoked = gen_session_ids(REVOKED);
    let valid: Vec<String> = gen_session_ids(REVOKED)
        .into_iter()
        .map(|s| format!("v_{s}"))
        .collect();

    let mut f: ScalableBloomFilter<&str> =
        ScalableBloomFilter::new(REVOKED, 0.000_001).unwrap();
    for id in &revoked {
        f.insert(&id.as_str());
    }

    g.throughput(Throughput::Elements(1));

    g.bench_function("valid_session_check", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let s = valid[idx % REVOKED].as_str();
            idx = idx.wrapping_add(1);
            black_box(f.contains(black_box(&s)))
        });
    });

    g.bench_function("revoked_session_check", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let s = revoked[idx % REVOKED].as_str();
            idx = idx.wrapping_add(1);
            black_box(f.contains(black_box(&s)))
        });
    });

    g.finish();
}

// ============================================================================
// 32. REAL-WORLD: RECOMMENDATION ENGINE
// ============================================================================

fn bench_real_world_recommendation(c: &mut Criterion) {
    let mut g = c.benchmark_group("real_world/recommendation");
    const SEEN: usize = 50_000;

    let mut f: ScalableBloomFilter<u64> = ScalableBloomFilter::new(SEEN, 0.001).unwrap();
    for i in 0..SEEN as u64 {
        f.insert(&i);
    }

    let candidates_100: Vec<u64> = (0..100u64)
        .map(|i| if i % 2 == 0 { i } else { SEEN as u64 + i })
        .collect();
    let candidates_1k: Vec<u64> = (0..1_000u64)
        .map(|i| if i % 2 == 0 { i } else { SEEN as u64 + i })
        .collect();

    g.throughput(Throughput::Elements(100));
    g.bench_function("filter_100_candidates", |b| {
        b.iter_batched(
            || candidates_100.clone(),
            |ks| black_box(f.contains_batch(black_box(&ks))),
            criterion::BatchSize::SmallInput,
        );
    });

    g.throughput(Throughput::Elements(1_000));
    g.bench_function("filter_1k_candidates", |b| {
        b.iter_batched(
            || candidates_1k.clone(),
            |ks| black_box(f.contains_batch(black_box(&ks))),
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 33. REAL-WORLD: HIGH-THROUGHPUT WRITE PIPELINE
// ============================================================================

fn bench_real_world_write_pipeline(c: &mut Criterion) {
    let mut g = c.benchmark_group("real_world/write_pipeline");

    g.throughput(Throughput::Elements(1_000));
    g.bench_function("cold_start/1k_inserts", |b| {
        b.iter_batched(
            || ScalableBloomFilter::<u64>::new(10_000, 0.01).unwrap(),
            |mut f| {
                for i in 0..1_000u64 {
                    f.insert(black_box(&i));
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.throughput(Throughput::Elements(1_000));
    g.bench_function("warm/1k_inserts_at_50k", |b| {
        b.iter_batched(
            || populated_sbf(1_000, 0.01, 50_000),
            |mut f| {
                let base = f.total_items() as u64;
                for i in 0..1_000u64 {
                    f.insert(black_box(&(base + i)));
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.throughput(Throughput::Elements(1_000));
    g.bench_function("deep/1k_inserts_at_100k", |b| {
        b.iter_batched(
            || populated_sbf(1_000, 0.01, 100_000),
            |mut f| {
                let base = f.len() as u64;
                for i in 0..1_000u64 {
                    f.insert(black_box(&(base + i)));
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// CRITERION GROUPS
// ============================================================================

criterion_group!(
    sbf_benches,
    // Insert
    bench_sbf_insert_sequential,
    bench_sbf_insert_batch_vs_sequential,
    bench_sbf_insert_fast,
    bench_sbf_insert_growth_event,
    bench_sbf_insert_post_clear,
    // Contains
    bench_sbf_contains_hit,
    bench_sbf_contains_miss,
    bench_sbf_contains_mixed,
    bench_sbf_contains_batch,
    bench_sbf_contains_provenance,
    bench_sbf_contains_fpr_sensitivity,
    // Query strategy
    bench_sbf_query_strategy,
    bench_sbf_strategy_recency_bias,
    // Growth
    bench_sbf_growth_strategies,
    bench_sbf_growth_error_ratio,
    bench_sbf_growth_fill_threshold,
    // Capacity
    bench_sbf_capacity_behavior,
    // Analytics
    bench_sbf_health_metrics,
    bench_sbf_predict_fpr,
    bench_sbf_fpr_breakdown,
    bench_sbf_cardinality,
    bench_sbf_filter_stats,
    bench_sbf_memory_usage,
    bench_sbf_aggregate_fill_rate,
    bench_sbf_fpr_estimators,
    // Maintenance
    bench_sbf_clear,
    bench_sbf_clone,
    // Key types
    bench_sbf_key_types,
    // Real-world
    bench_real_world_url_dedup,
    bench_real_world_log_dedup,
    bench_real_world_session_check,
    bench_real_world_recommendation,
    bench_real_world_write_pipeline,
);

criterion_main!(sbf_benches);