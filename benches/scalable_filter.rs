//! Production-grade benchmarks for ScalableBloomFilter and AtomicScalableBloomFilter.
//!
//! Run all:        cargo bench --bench scalable_filter
//! Run concurrent: cargo bench --bench scalable_filter --features concurrent
//! Run one group:  cargo bench --bench scalable_filter -- sbf/contains
//! HTML reports:   target/criterion/report/index.html

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::filters::{
    CapacityExhaustedBehavior, GrowthStrategy, QueryStrategy, ScalableBloomFilter,
};

#[cfg(feature = "concurrent")]
use bloomcraft::filters::AtomicScalableBloomFilter;

#[cfg(feature = "concurrent")]
use std::sync::Arc;

#[cfg(feature = "concurrent")]
use std::thread;

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

#[cfg(feature = "concurrent")]
fn populated_asbf(initial: usize, fpr: f64, n: usize) -> AtomicScalableBloomFilter<u64> {
    let f = AtomicScalableBloomFilter::<u64>::new(initial, fpr).unwrap();
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
                || {
                    (
                        ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
                        keys.clone(),
                    )
                },
                |(mut f, ks)| {
                    f.insert_batch(black_box(&ks)).unwrap();
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, _| {
            b.iter_batched(
                || {
                    (
                        ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
                        keys.clone(),
                    )
                },
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
// 3. CONTAINS — HIT PATH
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
// 4. CONTAINS — MISS PATH
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
// 5. CONTAINS — MIXED WORKLOAD (50/50)
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
// 6. BATCH CONTAINS — VARYING BATCH SIZE
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
// 7. QUERY STRATEGY: FORWARD vs REVERSE
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
        let old = 0u64;

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
// 8. GROWTH STRATEGIES
// ============================================================================

fn bench_sbf_growth_strategies(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/growth/strategies");
    const N: usize = 50_000;

    let strategies: &[(GrowthStrategy, &str)] = &[
        (GrowthStrategy::Constant, "constant"),
        (GrowthStrategy::Geometric(2.0), "geometric_2x"),
        (GrowthStrategy::Geometric(1.5), "geometric_1.5x"),
        (
            GrowthStrategy::Bounded {
                scale: 2.0,
                max_filter_size: 10_000,
            },
            "bounded_10k",
        ),
        (
            GrowthStrategy::Adaptive {
                initial_ratio: 0.5,
                min_ratio: 0.3,
                max_ratio: 0.9,
            },
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
// 9. CAPACITY EXHAUSTED BEHAVIOR
// ============================================================================

fn bench_sbf_capacity_behavior(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/capacity_exhausted");

    let build_saturated = |behavior: CapacityExhaustedBehavior| -> ScalableBloomFilter<u64> {
        let mut f = ScalableBloomFilter::<u64>::with_strategy(
            1,
            0.5,
            0.5,
            GrowthStrategy::Bounded {
                scale: 1.01,
                max_filter_size: 5,
            },
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
    let error = build_saturated(CapacityExhaustedBehavior::Error);

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
// 10. HEALTH METRICS
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
// 11. FPR PREDICTION
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
// 12. FPR BREAKDOWN
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
// 13. CARDINALITY ESTIMATION
// ============================================================================

fn bench_sbf_cardinality(c: &mut Criterion) {
    let mut g = c.benchmark_group("sbf/analytics/cardinality");

    for &(unique, reps, label) in &[
        (10_000usize, 1usize, "10k_unique_1x"),
        (10_000, 3, "10k_unique_3x_dups"),
        (100_000, 1, "100k_unique_1x"),
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
// 14. CLEAR
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
// 15. CONTAINS WITH PROVENANCE
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
// 16. FILTER STATS
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
// 17. REAL-WORLD: WEB CRAWLER URL DEDUPLICATION
// ============================================================================

fn bench_real_world_url_dedup(c: &mut Criterion) {
    let mut g = c.benchmark_group("real_world/url_dedup");
    const VISITED: usize = 500_000;

    let visited = gen_urls(VISITED);
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
// 18. REAL-WORLD: LOG DEDUPLICATION PIPELINE
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
// 19. REAL-WORLD: SESSION REVOCATION CHECK
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
// 20. REAL-WORLD: RECOMMENDATION ENGINE
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
// 21. REAL-WORLD: HIGH-THROUGHPUT WRITE PIPELINE
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
// 22. KEY TYPES — HASH COST
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
// 24. CLONE COST
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
// 25. ASBF — SINGLE-THREADED INSERT BASELINE
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_insert_single_thread(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/insert/single_thread");
    g.sample_size(10);

    for &n in &[1_000usize, 10_000, 100_000] {
        let keys = gen_u64(n);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_batched(
                || AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
                |f| {
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
// 26. ASBF — CONCURRENT INSERT SCALING
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_concurrent_insert_scaling(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/insert/concurrent_scaling");
    const ITEMS_PER_THREAD: usize = 10_000;

    for &threads in &[1usize, 2, 4, 8, 16] {
        let total = threads * ITEMS_PER_THREAD;
        g.throughput(Throughput::Elements(total as u64));
        g.bench_with_input(BenchmarkId::from_parameter(threads), &threads, |b, &t| {
            b.iter_batched(
                || Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap()),
                |f| {
                    let handles: Vec<_> = (0..t)
                        .map(|tid| {
                            let f = Arc::clone(&f);
                            thread::spawn(move || {
                                let base = (tid * ITEMS_PER_THREAD) as u64;
                                for i in 0..ITEMS_PER_THREAD as u64 {
                                    f.insert(black_box(&(base + i)));
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
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
// 27. ASBF — CONCURRENT CONTAINS SCALING
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_concurrent_contains_scaling(c: &mut Criterion) {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, Ordering};  // AtomicBool, not AtomicUsize

    let mut g = c.benchmark_group("asbf/contains/concurrent_scaling");
    const N: usize = 50_000;
    const OPS_PER_THREAD: usize = 10_000;

    let f = Arc::new(populated_asbf(1_000, 0.01, N));

    for &threads in &[1usize, 2, 4, 8, 16] {
        let total = threads * OPS_PER_THREAD;
        g.throughput(Throughput::Elements(total as u64));

        g.bench_with_input(BenchmarkId::new("hit", threads), &threads, |b, &t| {
            let start = Arc::new(Barrier::new(t + 1));
            let done  = Arc::new(Barrier::new(t + 1));
            let stop  = Arc::new(AtomicBool::new(false));

            let handles: Vec<_> = (0..t)
                .map(|tid| {
                    let f     = Arc::clone(&f);
                    let start = Arc::clone(&start);
                    let done  = Arc::clone(&done);
                    let stop  = Arc::clone(&stop);
                    thread::spawn(move || loop {
                        start.wait();
                        if stop.load(Ordering::Acquire) {
                            return;
                        }
                        let base = (tid * OPS_PER_THREAD) as u64 % N as u64;
                        for i in 0..OPS_PER_THREAD as u64 {
                            black_box(f.contains(black_box(&((base + i) % N as u64))));
                        }
                        done.wait();
                    })
                })
                .collect();

            b.iter(|| {
                start.wait();
                done.wait();
            });

            stop.store(true, Ordering::Release);
            start.wait();
            for h in handles {
                h.join().unwrap();
            }
        });

        g.bench_with_input(BenchmarkId::new("miss", threads), &threads, |b, &t| {
            let start = Arc::new(Barrier::new(t + 1));
            let done  = Arc::new(Barrier::new(t + 1));
            let stop  = Arc::new(AtomicBool::new(false));

            let handles: Vec<_> = (0..t)
                .map(|tid| {
                    let f     = Arc::clone(&f);
                    let start = Arc::clone(&start);
                    let done  = Arc::clone(&done);
                    let stop  = Arc::clone(&stop);
                    thread::spawn(move || loop {
                        start.wait();
                        if stop.load(Ordering::Acquire) {
                            return;
                        }
                        let base = N as u64 * 1_000 + (tid * OPS_PER_THREAD) as u64;
                        for i in 0..OPS_PER_THREAD as u64 {
                            black_box(f.contains(black_box(&(base + i))));
                        }
                        done.wait();
                    })
                })
                .collect();

            b.iter(|| {
                start.wait();
                done.wait();
            });

            stop.store(true, Ordering::Release);
            start.wait();
            for h in handles {
                h.join().unwrap();
            }
        });
    }
    g.finish();
}

// ============================================================================
// 28. ASBF — MIXED READ/WRITE UNDER CONTENTION
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_mixed_rw_contention(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/mixed/read_write");
    const READER_THREADS: usize = 4;
    const WRITER_THREADS: usize = 1;
    const OPS_EACH: usize = 5_000;

    g.throughput(Throughput::Elements(
        ((READER_THREADS + WRITER_THREADS) * OPS_EACH) as u64,
    ));

    g.bench_function("4r_1w", |b| {
        b.iter_batched(
            || Arc::new(populated_asbf(1_000, 0.01, 10_000)),
            |f| {
                let mut handles = vec![];

                for w in 0..WRITER_THREADS {
                    let f = Arc::clone(&f);
                    handles.push(thread::spawn(move || {
                        let base = 100_000u64 + (w * OPS_EACH) as u64;
                        for i in 0..OPS_EACH as u64 {
                            f.insert(black_box(&(base + i)));
                        }
                    }));
                }

                for r in 0..READER_THREADS {
                    let f = Arc::clone(&f);
                    handles.push(thread::spawn(move || {
                        let base = (r * OPS_EACH) as u64 % 10_000;
                        for i in 0..OPS_EACH as u64 {
                            let key = (base + i) % 10_000;
                            black_box(f.contains(black_box(&key)));
                        }
                    }));
                }

                for h in handles {
                    h.join().unwrap();
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 29. ASBF — CLEAR UNDER CONCURRENT LOAD
//
// Uses a Barrier to synchronise clear() and reader threads so that clear()
// fires while readers are genuinely querying live data, not an already-empty
// filter. stop is set before readers observe the barrier so they exit promptly.
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_clear_concurrent(c: &mut Criterion) {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut g = c.benchmark_group("asbf/clear/concurrent");

    for &readers in &[0usize, 2, 4, 8] {
        g.bench_with_input(
            BenchmarkId::new("clear_with_readers", readers),
            &readers,
            |b, &r| {
                b.iter_batched(
                    || Arc::new(populated_asbf(1_000, 0.01, 10_000)),
                    |f| {
                        let stop = Arc::new(AtomicBool::new(false));
                        // +1 for the clear thread itself; all threads start together.
                        let barrier = Arc::new(Barrier::new(r + 1));

                        let reader_handles: Vec<_> = (0..r)
                            .map(|_| {
                                let f = Arc::clone(&f);
                                let stop = Arc::clone(&stop);
                                let barrier = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    barrier.wait();
                                    let mut i = 0u64;
                                    while !stop.load(Ordering::Relaxed) {
                                        black_box(f.contains(black_box(&(i % 10_000))));
                                        i = i.wrapping_add(1);
                                    }
                                })
                            })
                            .collect();

                        // Synchronise: this wait completes when all reader
                        // threads are alive and spinning — clear() fires under
                        // genuine concurrent read load.
                        barrier.wait();
                        black_box(f.clear());
                        stop.store(true, Ordering::Relaxed);

                        for h in reader_handles {
                            h.join().unwrap();
                        }
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    g.finish();
}

// ============================================================================
// 30. ASBF — REAL-WORLD CONCURRENT DEDUPLICATION PIPELINE
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_real_world_dedup_pipeline(c: &mut Criterion) {
    let mut g = c.benchmark_group("real_world/asbf/dedup_pipeline");
    const PRODUCERS: usize = 4;
    const CONSUMERS: usize = 4;
    const EVENTS_EACH: usize = 5_000;

    g.throughput(Throughput::Elements((PRODUCERS * EVENTS_EACH) as u64));
    g.bench_function("4p_4c", |b| {
        b.iter_batched(
            || Arc::new(AtomicScalableBloomFilter::<u64>::new(10_000, 0.001).unwrap()),
            |f| {
                let mut handles = vec![];

                for p in 0..PRODUCERS {
                    let f = Arc::clone(&f);
                    handles.push(thread::spawn(move || {
                        let base = (p * EVENTS_EACH) as u64;
                        for i in 0..EVENTS_EACH as u64 {
                            f.insert(black_box(&(base + i)));
                        }
                    }));
                }

                for consumer in 0..CONSUMERS {
                    let f = Arc::clone(&f);
                    handles.push(thread::spawn(move || {
                        let base = (consumer * EVENTS_EACH * 2) as u64;
                        for i in 0..EVENTS_EACH as u64 {
                            let key = base + i;
                            black_box(f.contains(black_box(&key)));
                        }
                    }));
                }

                for h in handles {
                    h.join().unwrap();
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 31. SBF vs ASBF — SINGLE-THREADED OVERHEAD COMPARISON
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_sbf_vs_asbf_single_thread(c: &mut Criterion) {
    let mut g = c.benchmark_group("comparison/sbf_vs_asbf/single_thread");
    const N: usize = 50_000;

    let sbf = populated_sbf(1_000, 0.01, N);
    let asbf = Arc::new(populated_asbf(1_000, 0.01, N));

    let key_hit = (N / 2) as u64;
    let key_miss = N as u64 * 1_000_000;

    g.throughput(Throughput::Elements(1));
    g.bench_function("sbf/contains/hit", |b| {
        b.iter(|| black_box(sbf.contains(black_box(&key_hit))))
    });
    g.bench_function("asbf/contains/hit", |b| {
        b.iter(|| black_box(asbf.contains(black_box(&key_hit))))
    });
    g.bench_function("sbf/contains/miss", |b| {
        b.iter(|| black_box(sbf.contains(black_box(&key_miss))))
    });
    g.bench_function("asbf/contains/miss", |b| {
        b.iter(|| black_box(asbf.contains(black_box(&key_miss))))
    });
    g.bench_function("sbf/insert", |b| {
        let mut f = ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        let mut i = 0u64;
        b.iter(|| {
            i = i.wrapping_add(1);
            f.insert(black_box(&i));
        })
    });
    g.bench_function("asbf/insert", |b| {
        let f = AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap();
        let mut i = 0u64;
        b.iter(|| {
            i = i.wrapping_add(1);
            f.insert(black_box(&i));
        })
    });
    g.finish();
}

// ============================================================================
// 32. ASBF — with_preallocated vs new INSERT THROUGHPUT
//
// Directly measures the value of the with_preallocated fix. with_preallocated
// should show lower and more consistent insert latency at scale because growth
// events are pointer advances (no allocation) rather than ShardedFilter
// construction under a write lock.
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_with_preallocated(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/insert/with_preallocated");
    g.sample_size(10);

    for &(n, label) in &[
        (10_000usize,  "10k"),
        (100_000usize, "100k"),
        (500_000usize, "500k"),
    ] {
        let keys = gen_u64(n);
        g.throughput(Throughput::Elements(n as u64));

        // Baseline: new() allocates a fresh ShardedFilter on every growth event.
        g.bench_with_input(BenchmarkId::new("new", label), &n, |b, _| {
            b.iter_batched(
                || AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
                |f| {
                    for k in &keys {
                        f.insert(black_box(k));
                    }
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });

        // Subject: with_preallocated() — growth events are O(1) pointer advances.
        g.bench_with_input(BenchmarkId::new("with_preallocated", label), &n, |b, _| {
            b.iter_batched(
                || {
                    AtomicScalableBloomFilter::<u64>::with_preallocated(
                        1_000,
                        0.01,
                        n * 2,
                    )
                    .unwrap()
                },
                |f| {
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
// 33. ASBF CONTAINS — HIT PATH BY DEPTH
//
// Mirrors sbf/contains/hit (bench 3). Enables direct SBF vs ASBF latency
// comparison at identical filter depths. Required to verify the "1.5-2x SBF"
// single-thread overhead claim after the Mutex removal.
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_contains_hit_depth(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/contains/hit");

    let scenarios: &[(usize, usize, &str)] = &[
        (1_000,   500,    "depth-1"),
        (1_000,   7_500,  "depth-4"),
        (1_000,  31_500,  "depth-6"),
        (1_000, 127_000,  "depth-8"),
    ];

    for &(cap, n, label) in scenarios {
        let f = populated_asbf(cap, 0.01, n);
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
// 34. ASBF CONTAINS — MISS PATH BY DEPTH
//
// Mirrors sbf/contains/miss (bench 4). Without this, you cannot know whether
// ASBF depth-8 miss cost is dominated by lock overhead or filter traversal.
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_contains_miss_depth(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/contains/miss");

    let scenarios: &[(usize, usize, &str)] = &[
        (1_000,   500,    "depth-1"),
        (1_000,   7_500,  "depth-4"),
        (1_000,  31_500,  "depth-6"),
        (1_000, 127_000,  "depth-8"),
    ];

    for &(cap, n, label) in scenarios {
        let f = populated_asbf(cap, 0.01, n);
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
// 35. ASBF INSERT — SCALE WITH GROWTH EVENTS VISIBLE
//
// The existing bench_asbf_concurrent_insert uses 10_000 items/thread, which
// sits under the growth cliff. This benchmark deliberately crosses it so the
// throughput drop (if allocation is still under the write lock) is visible.
//
// After the fix, throughput must be monotonically increasing across item counts
// and 2-thread must beat 1-thread at every scale point.
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_insert_scale(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/insert/scale");
    g.sample_size(10);

    for &(n_threads, items_per_thread) in &[
        (1usize, 10_000usize),
        (1,      50_000),
        (1,     100_000),
        (2,      10_000),
        (2,      50_000),
        (2,     100_000),
        (4,      10_000),
        (4,      50_000),
        (4,     100_000),
    ] {
        let total = n_threads * items_per_thread;
        g.throughput(Throughput::Elements(total as u64));
        g.bench_with_input(
            BenchmarkId::new(format!("threads-{}", n_threads), items_per_thread),
            &(n_threads, items_per_thread),
            |b, &(nt, ipt)| {
                b.iter_batched(
                    || Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap()),
                    |f| {
                        let handles: Vec<_> = (0..nt)
                            .map(|tid| {
                                let fc = Arc::clone(&f);
                                thread::spawn(move || {
                                    let base = (tid * ipt) as u64;
                                    for i in 0..ipt as u64 {
                                        fc.insert(black_box(&(base + i)));
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(f)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    g.finish();
}

// ============================================================================
// 36. ASBF GROWTH EVENT LATENCY — TAIL SPIKE MEASUREMENT
//
// Throughput benchmarks hide tail spikes. This measures the exact latency of
// the single insert that fires try_grow(), forcing developers to see the P99
// cost of a growth event. After the fix this should be <5 µs. Before the fix
// it is 50–100 ms at filter-6 with Geometric(2.0).
//
// Method: insert exactly (threshold - 1) items to prime the filter, then
// measure a single insert that will trigger growth. iter_custom gives us
// per-iteration wall time including the growth spike.
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_growth_event_latency(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/latency/growth_event");
    // One sample per run — we want the raw latency of the growth event, not
    // a smoothed average. Criterion's default 100 samples would re-trigger
    // growth 100 times which is what we want to measure, but sample_size(10)
    // keeps the bench run time reasonable.
    g.sample_size(10);

    // With initial_capacity=1_000 and fill_threshold=0.5, the first growth
    // fires at approximately 500 items. Subtract 1 to prime to the edge.
    let prime_count: u64 = 499;

    g.bench_function("filter-1-to-2", |b| {
        b.iter_batched(
            || {
                let f = Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap());
                for i in 0..prime_count {
                    f.insert(&i);
                }
                f
            },
            |f| {
                // This single insert crosses the threshold and fires try_grow.
                f.insert(black_box(&prime_count));
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 37. ASBF MIXED R/W RATIOS
//
// The existing bench_asbf_mixed_rw only tests 4r/1w. Production workloads
// span the full spectrum. This benchmark gives developers the data to choose
// the right tool:
//   - 8r/1w:  cache / deduplication (read-heavy)
//   - 2r/2w:  balanced pipeline
//   - 1r/4w:  stream ingest (write-heavy)
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_mixed_rw_ratios(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/mixed/rw_ratio");
    g.sample_size(10);

    // (reader_threads, writer_threads, ops_per_thread, label)
    let scenarios: &[(usize, usize, usize, &str)] = &[
        (8, 1, 5_000, "8r_1w"),
        (2, 2, 5_000, "2r_2w"),
        (1, 4, 5_000, "1r_4w"),
    ];

    for &(readers, writers, ops, label) in scenarios {
        let total_ops = (readers + writers) * ops;
        g.throughput(Throughput::Elements(total_ops as u64));
        g.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let f = Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap());
                    // Pre-populate so readers have something to find.
                    for i in 0..5_000u64 {
                        f.insert(&i);
                    }
                    f
                },
                |f| {
                    let mut handles = Vec::with_capacity(readers + writers);

                    for _ in 0..readers {
                        let fc = Arc::clone(&f);
                        handles.push(thread::spawn(move || {
                            let mut hits = 0u64;
                            for i in 0..ops as u64 {
                                if fc.contains(black_box(&(i % 5_000))) {
                                    hits += 1;
                                }
                            }
                            hits
                        }));
                    }

                    for tid in 0..writers {
                        let fc = Arc::clone(&f);
                        handles.push(thread::spawn(move || {
                            let base = (5_000 + tid * ops) as u64;
                            for i in 0..ops as u64 {
                                fc.insert(black_box(&(base + i)));
                            }
                            0u64
                        }));
                    }

                    let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
                    black_box(total)
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ============================================================================
// 38. ASBF CONCURRENT BATCH INSERT
//
// There is no concurrent batch insert benchmark. insert_batch has a completely
// different concurrency model from insert (shard-grouped access, per-bucket
// lock re-acquisition). This benchmark exposes whether batch insert actually
// outperforms individual inserts under concurrent load.
// ============================================================================

#[cfg(feature = "concurrent")]
fn bench_asbf_concurrent_batch_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/insert/batch_concurrent");
    g.sample_size(10);

    for &(n_threads, batch_size) in &[
        (2usize,  100usize),
        (2,       1_000),
        (4,       100),
        (4,       1_000),
        (8,       100),
        (8,       1_000),
    ] {
        let total = n_threads * batch_size * 10; // 10 batches per thread
        g.throughput(Throughput::Elements(total as u64));
        g.bench_with_input(
            BenchmarkId::new(format!("threads-{}", n_threads), batch_size),
            &(n_threads, batch_size),
            |b, &(nt, bs)| {
                b.iter_batched(
                    || {
                        let batches: Vec<Vec<u64>> = (0..nt)
                            .map(|tid| {
                                (0..(bs * 10) as u64)
                                    .map(|i| (tid as u64 * bs as u64 * 10) + i)
                                    .collect()
                            })
                            .collect();
                        let f = Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap());
                        (f, batches)
                    },
                    |(f, batches)| {
                        let handles: Vec<_> = batches
                            .into_iter()
                            .map(|batch| {
                                let fc = Arc::clone(&f);
                                thread::spawn(move || {
                                    // 10 batches of batch_size each
                                    for chunk in batch.chunks(batch_size) {
                                        fc.insert_batch(black_box(chunk)).ok();
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(f)
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    g.finish();
}

// ============================================================================
// CRITERION GROUPS
// ============================================================================

criterion_group!(
    sbf_benches,
    bench_sbf_insert_sequential,
    bench_sbf_insert_batch_vs_sequential,
    bench_sbf_contains_hit,
    bench_sbf_contains_miss,
    bench_sbf_contains_mixed,
    bench_sbf_contains_batch,
    bench_sbf_query_strategy,
    bench_sbf_growth_strategies,
    bench_sbf_capacity_behavior,
    bench_sbf_health_metrics,
    bench_sbf_predict_fpr,
    bench_sbf_fpr_breakdown,
    bench_sbf_cardinality,
    bench_sbf_clear,
    bench_sbf_contains_provenance,
    bench_sbf_filter_stats,
    bench_sbf_memory_usage,
    bench_sbf_clone,
    bench_sbf_key_types,
    bench_real_world_url_dedup,
    bench_real_world_log_dedup,
    bench_real_world_session_check,
    bench_real_world_recommendation,
    bench_real_world_write_pipeline,
);

#[cfg(feature = "concurrent")]
criterion_group!(
    asbf_benches,
    bench_asbf_insert_single_thread,
    bench_asbf_concurrent_insert_scaling,
    bench_asbf_concurrent_contains_scaling,
    bench_asbf_mixed_rw_contention,
    bench_asbf_clear_concurrent,
    bench_asbf_real_world_dedup_pipeline,
    bench_sbf_vs_asbf_single_thread,
    bench_asbf_with_preallocated,
    bench_asbf_contains_hit_depth,
    bench_asbf_contains_miss_depth,
    bench_asbf_insert_scale,
    bench_asbf_growth_event_latency,
    bench_asbf_mixed_rw_ratios,
    bench_asbf_concurrent_batch_insert,
);

#[cfg(not(feature = "concurrent"))]
criterion_main!(sbf_benches);

#[cfg(feature = "concurrent")]
criterion_main!(sbf_benches, asbf_benches);
