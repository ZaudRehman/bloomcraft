//! Production-grade benchmarks for AtomicScalableBloomFilter.
//!
//! Requires the `concurrent` feature:
//!
//!     cargo bench --bench atomic_scalable_filter --features concurrent
//!
//! Run one group:  cargo bench --bench atomic_scalable_filter -- asbf/insert
//! HTML reports:   target/criterion/report/index.html
//!
//! # Coverage map
//!
//! ## Single-threaded baselines
//!  1. asbf/insert/single_thread         — throughput at 1k/10k/100k (vs SBF)
//!  2. asbf/contains/hit                 — hit latency at depth 1/4/6/8
//!  3. asbf/contains/miss                — miss latency at depth 1/4/6/8
//!  4. asbf/contains/batch               — contains_batch, 10/100/1k/10k items
//!  5. asbf/contains/fpr_sensitivity     — hit+miss latency at FPR 0.1/0.01/0.001/0.0001
//!
//! ## Concurrent insert
//!  6. asbf/insert/concurrent_scaling    — 1/2/4/8/16 threads × 10k items each
//!  7. asbf/insert/scale                 — threads × items crossing growth events
//!  8. asbf/insert/with_preallocated     — new() vs with_preallocated() at 10k/100k/500k
//!  9. asbf/insert/batch_concurrent      — insert_batch concurrent, 2/4/8 threads
//! 10. asbf/insert/batch_vs_loop         — insert_batch vs loop-insert, single thread
//! 11. asbf/insert/growth_event          — tail latency of a single growth trigger
//!
//! ## Concurrent contains
//! 12. asbf/contains/concurrent_scaling  — 1/2/4/8/16 reader threads hit+miss
//!
//! ## Mixed read/write
//! 13. asbf/mixed/read_write             — 4r/1w fixed contention bench
//! 14. asbf/mixed/rw_ratio               — 8r/1w, 2r/2w, 1r/4w spectrum
//! 15. asbf/mixed/rw_write_heavy         — escalating writer count, fixed readers
//!
//! ## Clone semantics (Arc overhead)
//! 16. asbf/clone                        — O(1) Arc clone cost at depth 1/4/8
//!
//! ## Maintenance
//! 17. asbf/clear/concurrent             — clear() at 0/2/4/8 concurrent readers
//! 18. asbf/clear/and_reinsert           — clear() + 1k reinserts (window rotation)
//!
//! ## Analytics
//! 19. asbf/analytics/estimate_fpr       — estimate_fpr() at depth 1/4/8
//! 20. asbf/analytics/memory_usage       — memory_usage() at depth 1/4/8
//! 21. asbf/analytics/bit_statistics     — bit_statistics() at depth 1/4/8
//! 22. asbf/analytics/current_fill_rate  — current_fill_rate() under concurrent load
//! 23. asbf/analytics/accessors          — len/is_empty/filter_count/total_capacity/shard_count
//!
//! ## SBF comparison
//! 24. comparison/sbf_vs_asbf/single_thread — hit/miss/insert overhead of concurrency primitives
//! 25. comparison/sbf_vs_asbf/depth         — overhead at each depth
//!
//! ## Growth strategies
//! 26. asbf/growth/strategies             — Constant/Geometric/Bounded/Adaptive insert throughput
//!
//! ## Real-world
//! 27. real_world/asbf/dedup_pipeline     — 4 producers + 4 consumers
//! 28. real_world/asbf/url_dedup          — concurrent URL deduplication
//! 29. real_world/asbf/session_revocation — concurrent session-revocation check

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::filters::{AtomicScalableBloomFilter, GrowthStrategy, ScalableBloomFilter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Barrier;
use std::thread;

// ============================================================================
// DATASET GENERATORS
// ============================================================================

fn gen_u64(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn gen_urls(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            format!(
                "https://example.com/path/{}/res?id={}&ts={}",
                i % 1_000,
                i,
                i.wrapping_mul(6_364_136_223_846_793_005usize),
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

fn populated_asbf(initial: usize, fpr: f64, n: usize) -> AtomicScalableBloomFilter<u64> {
    let f = AtomicScalableBloomFilter::<u64>::new(initial, fpr).unwrap();
    for i in 0..n as u64 {
        f.insert(&i);
    }
    f
}

// ============================================================================
// 1. SINGLE-THREADED INSERT BASELINE
// ============================================================================

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
// 2. CONTAINS — HIT PATH BY DEPTH
//
// Mirrors sbf/contains/hit exactly so the two can be compared in the same
// Criterion report. The read-lock acquisition (~10 ns per call) is the primary
// overhead over SBF at depth-1; the gap narrows at depth-8 where traversal
// dominates.
// ============================================================================

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
// 3. CONTAINS — MISS PATH BY DEPTH
//
// A miss forces traversal of every sub-filter. The cost grows with depth and
// is the worst case the reader lock is held for. This bench shows whether the
// miss cost is dominated by lock overhead (flat curve) or filter traversal
// (rising curve) — critical for knowing whether lock-free iteration is needed.
// ============================================================================

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
// 4. CONTAINS — BATCH
//
// contains_batch is a plain iterator over contains(), so it acquires and
// releases the read-lock per item. This bench exposes whether the lock
// acquisition dominates at small batch sizes (where per-call overhead matters)
// or disappears at large batches (where traversal amortises it).
// ============================================================================

fn bench_asbf_contains_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/contains/batch");
    const TOTAL: usize = 50_000;
    let f = populated_asbf(1_000, 0.01, TOTAL);

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
// 5. CONTAINS — FPR SENSITIVITY
//
// Higher FPR targets → fewer bits per item → lower k → cheaper contains().
// The read-lock overhead is constant; this bench isolates how much of the
// per-call time is hash evaluation vs lock acquisition. Holding depth constant
// at 4 filters makes the comparison fair.
// ============================================================================

fn bench_asbf_contains_fpr_sensitivity(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/contains/fpr_sensitivity");
    // Depth-4 for all variants so the only variable is k (hash count).
    const N: usize = 7_500;

    for &(fpr, label) in &[
        (0.1f64,  "fpr_0.1"),
        (0.01,    "fpr_0.01"),
        (0.001,   "fpr_0.001"),
        (0.0001,  "fpr_0.0001"),
    ] {
        let f = populated_asbf(1_000, fpr, N);
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
// 6. CONCURRENT INSERT SCALING
// ============================================================================

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
// 7. INSERT SCALE — CROSSING GROWTH EVENTS
//
// bench 6 uses 10k items/thread, which stays under most growth thresholds.
// This bench deliberately crosses growth cliffs to expose the write-lock
// stall if allocation still runs inside the lock. After the three-phase
// growth fix, throughput must be monotonically non-decreasing across item
// counts and multi-thread must outperform single-thread at every point.
// ============================================================================

fn bench_asbf_insert_scale(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/insert/scale");
    g.sample_size(10);

    for &(n_threads, items_per_thread) in &[
        (1usize, 10_000usize),
        (1,       50_000),
        (1,      100_000),
        (2,       10_000),
        (2,       50_000),
        (2,      100_000),
        (4,       10_000),
        (4,       50_000),
        (4,      100_000),
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
// 8. WITH_PREALLOCATED vs NEW
//
// with_preallocated() zeroes all sub-filter bit arrays at construction so
// growth events become O(1) pointer advances rather than ShardedFilter::new()
// calls that zero potentially megabytes of bit arrays. This bench quantifies
// the benefit across item counts where multiple growth events would otherwise
// fire, as well as the upfront allocation penalty paid at construction.
// ============================================================================

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

        g.bench_with_input(BenchmarkId::new("with_preallocated", label), &n, |b, _| {
            b.iter_batched(
                || AtomicScalableBloomFilter::<u64>::with_preallocated(1_000, 0.01, n * 2).unwrap(),
                |f| {
                    for k in &keys {
                        f.insert(black_box(k));
                    }
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });

        // Also measure the construction cost alone so callers can weigh
        // upfront cost against insert-time savings.
        g.bench_with_input(BenchmarkId::new("construction_cost", label), &n, |b, _| {
            b.iter(|| {
                black_box(
                    AtomicScalableBloomFilter::<u64>::with_preallocated(1_000, 0.01, n * 2)
                        .unwrap(),
                )
            });
        });
    }
    g.finish();
}

// ============================================================================
// 9. CONCURRENT BATCH INSERT
//
// insert_batch() differs from a loop of insert() calls in two ways:
// (1) it groups items by shard upfront, buying cache locality;
// (2) it acquires the read-lock O(shard_count) times per batch rather than
//     O(n). Under high concurrency, shard grouping means each thread's writes
//     cluster in one contiguous region of the bit array rather than scattering
//     across all shards, reducing false sharing between threads.
// ============================================================================

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
        let total = n_threads * batch_size * 10;
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
                        let f = Arc::new(
                            AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
                        );
                        (f, batches)
                    },
                    |(f, batches)| {
                        let handles: Vec<_> = batches
                            .into_iter()
                            .map(|batch| {
                                let fc = Arc::clone(&f);
                                thread::spawn(move || {
                                    for chunk in batch.chunks(bs) {
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
// 10. SINGLE-THREADED BATCH INSERT vs LOOP INSERT
//
// The doc comment on insert_batch claims shard grouping gives 3-4× faster
// writes than a loop of insert() calls due to sequential writes hitting the
// same shard's cache lines repeatedly. This bench verifies that claim in the
// single-threaded case where there is no contention — isolating the pure
// cache-locality benefit from any concurrency effect.
// ============================================================================

fn bench_asbf_insert_batch_vs_loop(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/insert/batch_vs_loop");

    for &n in &[100usize, 1_000, 10_000, 100_000] {
        let keys = gen_u64(n);
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("batch", n), &n, |b, _| {
            b.iter_batched(
                || AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(),
                |f| {
                    f.insert_batch(black_box(&keys)).unwrap();
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("loop", n), &n, |b, _| {
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
// 11. GROWTH EVENT LATENCY — TAIL SPIKE MEASUREMENT
//
// The three-phase growth protocol allocates outside the write lock so that the
// lock hold time is ~10 ns rather than the full ShardedFilter::new() duration.
// This bench measures the tail latency of the insert that triggers growth,
// making any regression in the write-lock hold time immediately visible.
//
// filter-1→2: the first growth event, zeroing the smallest new sub-filter.
// filter-4→5: the fifth growth event; at Geometric(2.0) the new filter is
//             16× larger than the initial one, so zeroing takes proportionally
//             longer — this is where a regression hurts most.
// ============================================================================

fn bench_asbf_growth_event_latency(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/latency/growth_event");
    g.sample_size(10);

    // initial_capacity=1_000, fill_threshold=0.5 → growth fires at ~500 items.
    let prime_1_to_2: u64 = 499;

    g.bench_function("filter-1-to-2", |b| {
        b.iter_batched(
            || {
                let f = Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap());
                for i in 0..prime_1_to_2 {
                    f.insert(&i);
                }
                f
            },
            |f| {
                f.insert(black_box(&prime_1_to_2));
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // filter-4→5: total items at depth-4 ≈ 7_000; the threshold for filter-4
    // fires at approximately 7_000. We prime just below it.
    let prime_4_to_5: u64 = 6_999;

    g.bench_function("filter-4-to-5", |b| {
        b.iter_batched(
            || {
                let f = Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap());
                for i in 0..prime_4_to_5 {
                    f.insert(&i);
                }
                f
            },
            |f| {
                f.insert(black_box(&prime_4_to_5));
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // preallocated path: the same trigger point but the filter was pre-built,
    // so the growth event is an O(1) pointer advance. This is the lower bound.
    g.bench_function("filter-4-to-5/preallocated", |b| {
        b.iter_batched(
            || {
                let f = Arc::new(
                    AtomicScalableBloomFilter::<u64>::with_preallocated(1_000, 0.01, 50_000)
                        .unwrap(),
                );
                for i in 0..prime_4_to_5 {
                    f.insert(&i);
                }
                f
            },
            |f| {
                f.insert(black_box(&prime_4_to_5));
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 12. CONCURRENT CONTAINS SCALING
// ============================================================================

fn bench_asbf_concurrent_contains_scaling(c: &mut Criterion) {
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
                        if stop.load(Ordering::Acquire) { return; }
                        let base = (tid * OPS_PER_THREAD) as u64 % N as u64;
                        for i in 0..OPS_PER_THREAD as u64 {
                            black_box(f.contains(black_box(&((base + i) % N as u64))));
                        }
                        done.wait();
                    })
                })
                .collect();

            b.iter(|| { start.wait(); done.wait(); });

            stop.store(true, Ordering::Release);
            start.wait();
            for h in handles { h.join().unwrap(); }
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
                        if stop.load(Ordering::Acquire) { return; }
                        let base = N as u64 * 1_000 + (tid * OPS_PER_THREAD) as u64;
                        for i in 0..OPS_PER_THREAD as u64 {
                            black_box(f.contains(black_box(&(base + i))));
                        }
                        done.wait();
                    })
                })
                .collect();

            b.iter(|| { start.wait(); done.wait(); });

            stop.store(true, Ordering::Release);
            start.wait();
            for h in handles { h.join().unwrap(); }
        });
    }
    g.finish();
}

// ============================================================================
// 13. MIXED READ/WRITE — FIXED CONTENTION (4r/1w)
// ============================================================================

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

                for h in handles { h.join().unwrap(); }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 14. MIXED R/W RATIOS
//
// Production workloads span the full read/write spectrum. This bench gives
// the data needed to decide whether the ASBF's RwLock model provides
// meaningful parallelism or whether a different data structure is needed:
//   - 8r/1w: cache/deduplication workloads (read-heavy; RwLock should scale)
//   - 2r/2w: balanced pipelines
//   - 1r/4w: stream ingest (write-heavy; write lock becomes the bottleneck)
// ============================================================================

fn bench_asbf_mixed_rw_ratios(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/mixed/rw_ratio");
    g.sample_size(10);

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
                    for i in 0..5_000u64 { f.insert(&i); }
                    f
                },
                |f| {
                    let mut handles = Vec::with_capacity(readers + writers);

                    for _ in 0..readers {
                        let fc = Arc::clone(&f);
                        handles.push(thread::spawn(move || {
                            let mut hits = 0u64;
                            for i in 0..ops as u64 {
                                if fc.contains(black_box(&(i % 5_000))) { hits += 1; }
                            }
                            hits
                        }));
                    }

                    for tid in 0..writers {
                        let fc = Arc::clone(&f);
                        handles.push(thread::spawn(move || {
                            let base = (5_000 + tid * ops) as u64;
                            for i in 0..ops as u64 { fc.insert(black_box(&(base + i))); }
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
// 15. WRITE-HEAVY SCALING
//
// The RwLock becomes a contention point under write-heavy loads because
// insert() re-acquires the read-lock on every call (to snapshot the current
// filter pointer). This bench escalates from 1 to 8 writers at a fixed
// reader count to find the writer count at which throughput stops scaling.
// ============================================================================

fn bench_asbf_write_heavy_scaling(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/mixed/rw_write_heavy");
    g.sample_size(10);
    const READERS: usize = 2;
    const OPS: usize = 5_000;

    for &writers in &[1usize, 2, 4, 8] {
        let total_ops = (READERS + writers) * OPS;
        g.throughput(Throughput::Elements(total_ops as u64));
        g.bench_with_input(
            BenchmarkId::new(format!("r{READERS}"), writers),
            &writers,
            |b, &w| {
                b.iter_batched(
                    || {
                        let f =
                            Arc::new(AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap());
                        for i in 0..5_000u64 { f.insert(&i); }
                        f
                    },
                    |f| {
                        let mut handles = Vec::with_capacity(READERS + w);

                        for _ in 0..READERS {
                            let fc = Arc::clone(&f);
                            handles.push(thread::spawn(move || {
                                for i in 0..OPS as u64 {
                                    black_box(fc.contains(black_box(&(i % 5_000))));
                                }
                            }));
                        }
                        for tid in 0..w {
                            let fc = Arc::clone(&f);
                            handles.push(thread::spawn(move || {
                                let base = (5_000 + tid * OPS) as u64;
                                for i in 0..OPS as u64 {
                                    fc.insert(black_box(&(base + i)));
                                }
                            }));
                        }

                        for h in handles { h.join().unwrap(); }
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
// 16. CLONE — ARC REFERENCE COUNT OVERHEAD
//
// Cloning an ASBF is O(1): it increments an Arc reference count, not the
// bit arrays. This bench verifies that promise and gives callers the concrete
// cost of sharing the filter across threads by cloning before spawning.
// The cost must be independent of filter depth.
// ============================================================================

fn bench_asbf_clone(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/clone");

    for &(cap, n, label) in &[
        (1_000usize, 500usize,    "depth-1"),
        (1_000,      7_500,       "depth-4"),
        (1_000,      127_000,     "depth-8"),
    ] {
        let f = Arc::new(populated_asbf(cap, 0.01, n));
        // clone() on the inner filter (the O(1) Arc clone).
        g.bench_function(label, |b| b.iter(|| black_box((*f).clone())));
    }
    g.finish();
}

// ============================================================================
// 17. CLEAR UNDER CONCURRENT READERS
//
// The doc states ~162× overhead vs uncontested clear at 8 concurrent readers.
// This bench verifies that claim and quantifies the curve so callers know
// the actual cost of using clear() for window rotation under read load.
// ============================================================================

fn bench_asbf_clear_concurrent(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/clear/concurrent");

    for &readers in &[0usize, 2, 4, 8] {
        g.bench_with_input(
            BenchmarkId::new("clear_with_readers", readers),
            &readers,
            |b, &r| {
                b.iter_batched(
                    || Arc::new(populated_asbf(1_000, 0.01, 10_000)),
                    |f| {
                        let stop    = Arc::new(AtomicBool::new(false));
                        let barrier = Arc::new(Barrier::new(r + 1));

                        let reader_handles: Vec<_> = (0..r)
                            .map(|_| {
                                let f       = Arc::clone(&f);
                                let stop    = Arc::clone(&stop);
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

                        barrier.wait();
                        black_box(f.clear());
                        stop.store(true, Ordering::Relaxed);

                        for h in reader_handles { h.join().unwrap(); }
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    g.finish();
}

// ============================================================================
// 18. CLEAR AND REINSERT (WINDOW ROTATION)
//
// A common pattern is clear() followed immediately by reinsertion of a new
// batch — log deduplication, sliding-window rate limiting, cache invalidation.
// This bench models that pattern end-to-end, measuring the combined cost of
// clear() + 1k reinserts, both uncontested and under 4 concurrent readers.
// ============================================================================

fn bench_asbf_clear_and_reinsert(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/clear/and_reinsert");
    g.throughput(Throughput::Elements(1_000));

    // Uncontested: clear and reinsert with no concurrent readers.
    g.bench_function("uncontested", |b| {
        b.iter_batched(
            || {
                let f = Arc::new(populated_asbf(1_000, 0.01, 10_000));
                f
            },
            |f| {
                f.clear();
                for i in 0..1_000u64 {
                    f.insert(black_box(&i));
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // Contended: 4 readers running while clear() fires.
    g.bench_function("with_4_readers", |b| {
        b.iter_batched(
            || Arc::new(populated_asbf(1_000, 0.01, 10_000)),
            |f| {
                let stop    = Arc::new(AtomicBool::new(false));
                let barrier = Arc::new(Barrier::new(5)); // 4 readers + 1 writer

                let reader_handles: Vec<_> = (0..4)
                    .map(|_| {
                        let f       = Arc::clone(&f);
                        let stop    = Arc::clone(&stop);
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

                barrier.wait();
                f.clear();
                for i in 0..1_000u64 {
                    f.insert(black_box(&i));
                }
                stop.store(true, Ordering::Relaxed);
                for h in reader_handles { h.join().unwrap(); }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 19. ANALYTICS — estimate_fpr
//
// estimate_fpr() acquires the read-lock and reads atomic counters across all
// shards of all sub-filters — O(n_filters × shard_count × m/64). This bench
// quantifies the cost at each filter depth so callers know whether it is safe
// to call on a monitoring hot path.
// ============================================================================

fn bench_asbf_estimate_fpr(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/analytics/estimate_fpr");

    for &(cap, n, label) in &[
        (1_000usize, 500usize,    "depth-1"),
        (1_000,      7_500,       "depth-4"),
        (1_000,      127_000,     "depth-8"),
    ] {
        let f = populated_asbf(cap, 0.01, n);
        g.bench_function(label, |b| b.iter(|| black_box(f.estimate_fpr())));
    }
    g.finish();
}

// ============================================================================
// 20. ANALYTICS — memory_usage
//
// memory_usage() holds the read-lock and sums bit-array sizes across every
// shard. This bench gives the O(n_filters × shard_count) constant.
// ============================================================================

fn bench_asbf_memory_usage(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/analytics/memory_usage");

    for &(cap, n, label) in &[
        (1_000usize, 500usize,    "depth-1"),
        (1_000,      7_500,       "depth-4"),
        (1_000,      127_000,     "depth-8"),
    ] {
        let f = populated_asbf(cap, 0.01, n);
        g.bench_function(label, |b| b.iter(|| black_box(f.memory_usage())));
    }
    g.finish();
}

// ============================================================================
// 21. ANALYTICS — bit_statistics
//
// bit_statistics() locks, iterates every shard of every sub-filter, and
// counts set bits via popcount. The docstring warns against calling it during
// active inserts. This bench confirms its cost relative to cheaper accessors
// like current_fill_rate() and len().
// ============================================================================

fn bench_asbf_bit_statistics(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/analytics/bit_statistics");

    for &(cap, n, label) in &[
        (1_000usize, 500usize,    "depth-1"),
        (1_000,      7_500,       "depth-4"),
        (1_000,      127_000,     "depth-8"),
    ] {
        let f = populated_asbf(cap, 0.01, n);
        g.bench_function(label, |b| b.iter(|| black_box(f.bit_statistics())));
    }
    g.finish();
}

// ============================================================================
// 22. ANALYTICS — current_fill_rate UNDER CONCURRENT LOAD
//
// current_fill_rate() acquires the read-lock and reads one ShardedFilter's
// bit counters. This bench compares its cost in the quiet case (no concurrent
// inserts) versus when 4 writer threads are hammering the filter, measuring
// how much the writer-driven cache invalidation raises the read cost.
// ============================================================================

fn bench_asbf_current_fill_rate_under_load(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/analytics/current_fill_rate");

    let f = Arc::new(populated_asbf(1_000, 0.01, 7_500));

    // Quiet: no concurrent writers.
    g.bench_function("quiet", |b| b.iter(|| black_box(f.current_fill_rate())));

    // Hot: 4 writers inserting while fill_rate is sampled.
    g.bench_function("under_4_writers", |b| {
        let stop = Arc::new(AtomicBool::new(false));
        let handles: Vec<_> = (0..4)
            .map(|w| {
                let f    = Arc::clone(&f);
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    let mut i = (w * 100_000) as u64;
                    while !stop.load(Ordering::Relaxed) {
                        f.insert(&i);
                        i = i.wrapping_add(1);
                    }
                })
            })
            .collect();

        b.iter(|| black_box(f.current_fill_rate()));

        stop.store(true, Ordering::Relaxed);
        for h in handles { h.join().unwrap(); }
    });

    g.finish();
}

// ============================================================================
// 23. ANALYTICS — CHEAP ACCESSORS
//
// len(), is_empty(), filter_count(), total_capacity(), and shard_count() are
// the cheapest analytics calls. len() and is_empty() are single Relaxed
// atomic loads; filter_count() and total_capacity() acquire the read-lock.
// shard_count() reads an immutable config field — no lock, no atomic.
//
// This bench gives their absolute cost and confirms the lock-free promise for
// len()/is_empty().
// ============================================================================

fn bench_asbf_cheap_accessors(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/analytics/accessors");
    let f = Arc::new(populated_asbf(1_000, 0.01, 7_500));

    // O(1) accessors: single Relaxed load or immutable config read.
    g.bench_function("len",            |b| b.iter(|| black_box(f.len())));
    g.bench_function("is_empty",       |b| b.iter(|| black_box(f.is_empty())));
    g.bench_function("shard_count",    |b| b.iter(|| black_box(f.shard_count())));

    // Read-lock accessors: acquire the filter-list lock.
    g.bench_function("filter_count",   |b| b.iter(|| black_box(f.filter_count())));
    g.bench_function("total_capacity", |b| b.iter(|| black_box(f.total_capacity())));
    g.bench_function("hash_count",     |b| b.iter(|| black_box(f.hash_count())));

    // O(n iterator) accessors: iterate all shards of all filters.
    g.bench_function("bit_count",      |b| b.iter(|| black_box(f.bit_count())));
    g.bench_function("count_set_bits", |b| b.iter(|| black_box(f.count_set_bits())));
    g.bench_function("estimate_count", |b| b.iter(|| black_box(f.estimate_count())));

    g.finish();
}

// ============================================================================
// 24. SBF vs ASBF — SINGLE-THREADED OVERHEAD
//
// This is the baseline comparison that justifies (or refutes) choosing ASBF
// over SBF in single-threaded code. The overhead comes from:
// - insert:   one read-lock acquire per insert (vs zero for SBF)
// - contains: one read-lock acquire per call (vs zero for SBF)
// If overhead exceeds 1.5–2×, callers in ST code should prefer SBF and
// wrap it in Arc<Mutex<_>> rather than using ASBF.
// ============================================================================

fn bench_sbf_vs_asbf_single_thread(c: &mut Criterion) {
    let mut g = c.benchmark_group("comparison/sbf_vs_asbf/single_thread");
    const N: usize = 50_000;

    let sbf  = populated_sbf(1_000, 0.01, N);
    let asbf = Arc::new(populated_asbf(1_000, 0.01, N));

    let key_hit  = (N / 2) as u64;
    let key_miss = N as u64 * 1_000_000;

    g.throughput(Throughput::Elements(1));
    g.bench_function("sbf/contains/hit",  |b| b.iter(|| black_box(sbf.contains(black_box(&key_hit)))));
    g.bench_function("asbf/contains/hit", |b| b.iter(|| black_box(asbf.contains(black_box(&key_hit)))));
    g.bench_function("sbf/contains/miss", |b| b.iter(|| black_box(sbf.contains(black_box(&key_miss)))));
    g.bench_function("asbf/contains/miss",|b| b.iter(|| black_box(asbf.contains(black_box(&key_miss)))));
    g.bench_function("sbf/insert", |b| {
        b.iter_batched(
            || (ScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(), 0u64),
            |(mut f, mut i)| {
                i = i.wrapping_add(1);
                f.insert(black_box(&i));
                black_box((f, i))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("asbf/insert", |b| {
        b.iter_batched(
            || (AtomicScalableBloomFilter::<u64>::new(1_000, 0.01).unwrap(), 0u64),
            |(f, mut i)| {
                i = i.wrapping_add(1);
                f.insert(black_box(&i));
                black_box((f, i))
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

// ============================================================================
// 25. SBF vs ASBF — OVERHEAD AT EACH DEPTH
//
// The read-lock overhead is a fixed addend per call; traversal cost scales
// with depth. This bench shows how quickly the traversal dominates the lock
// overhead so callers know at what depth the two implementations converge.
// ============================================================================

fn bench_sbf_vs_asbf_depth(c: &mut Criterion) {
    let mut g = c.benchmark_group("comparison/sbf_vs_asbf/depth");

    let scenarios: &[(usize, usize, &str)] = &[
        (1_000,  500,    "depth-1"),
        (1_000,  7_500,  "depth-4"),
        (1_000,  31_500, "depth-6"),
        (1_000,  127_000,"depth-8"),
    ];

    for &(cap, n, label) in scenarios {
        let sbf  = populated_sbf(cap, 0.01, n);
        let asbf = populated_asbf(cap, 0.01, n);
        let miss = (n as u64).wrapping_mul(1_000_000);

        g.throughput(Throughput::Elements(1));
        g.bench_function(format!("sbf/hit/{label}"),   |b| {
            let mut idx = 0u64;
            b.iter(|| { let k = idx % n as u64; idx = idx.wrapping_add(1); black_box(sbf.contains(black_box(&k))) })
        });
        g.bench_function(format!("asbf/hit/{label}"),  |b| {
            let mut idx = 0u64;
            b.iter(|| { let k = idx % n as u64; idx = idx.wrapping_add(1); black_box(asbf.contains(black_box(&k))) })
        });
        g.bench_function(format!("sbf/miss/{label}"),  |b| {
            let mut idx = 0u64;
            b.iter(|| { idx = idx.wrapping_add(1); black_box(sbf.contains(black_box(&(miss.wrapping_add(idx))))) })
        });
        g.bench_function(format!("asbf/miss/{label}"), |b| {
            let mut idx = 0u64;
            b.iter(|| { idx = idx.wrapping_add(1); black_box(asbf.contains(black_box(&(miss.wrapping_add(idx))))) })
        });
    }
    g.finish();
}

// ============================================================================
// 26. GROWTH STRATEGIES — INSERT THROUGHPUT
//
// All four GrowthStrategy variants are supported. Constant uses a fixed
// shard_count per filter; Adaptive adjusts error_ratio under the write lock.
// This bench measures total insert throughput across 50k items so that
// growth-event frequency differences between strategies are visible.
// ============================================================================

fn bench_asbf_growth_strategies(c: &mut Criterion) {
    let mut g = c.benchmark_group("asbf/growth/strategies");
    g.sample_size(10);
    const N: usize = 50_000;

    let strategies: &[(GrowthStrategy, f64, &str)] = &[
        (GrowthStrategy::Constant,              0.5, "constant"),
        (GrowthStrategy::Geometric(2.0),        0.5, "geometric_2x"),
        (GrowthStrategy::Geometric(1.5),        0.5, "geometric_1.5x"),
        (GrowthStrategy::Bounded { scale: 2.0, max_filter_size: 10_000 }, 0.5, "bounded_10k"),
        (GrowthStrategy::Adaptive { initial_ratio: 0.5, min_ratio: 0.3, max_ratio: 0.9 }, 0.5, "adaptive"),
    ];

    for &(strategy, error_ratio, label) in strategies {
        g.throughput(Throughput::Elements(N as u64));
        g.bench_function(label, |b| {
            b.iter_batched(
                || {
                    AtomicScalableBloomFilter::<u64>::with_strategy(
                        1_000, 0.01, error_ratio, strategy,
                    )
                    .unwrap()
                },
                |f| {
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
// 27. REAL-WORLD: CONCURRENT DEDUPLICATION PIPELINE
// ============================================================================

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
                        for i in 0..EVENTS_EACH as u64 { f.insert(black_box(&(base + i))); }
                    }));
                }

                for consumer in 0..CONSUMERS {
                    let f = Arc::clone(&f);
                    handles.push(thread::spawn(move || {
                        let base = (consumer * EVENTS_EACH * 2) as u64;
                        for i in 0..EVENTS_EACH as u64 { black_box(f.contains(black_box(&(base + i)))); }
                    }));
                }

                for h in handles { h.join().unwrap(); }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 28. REAL-WORLD: CONCURRENT URL DEDUPLICATION
//
// Web crawlers share a single bloom filter across N fetcher threads. Each
// thread checks URL membership before enqueuing, then inserts the URL. This
// bench models the check-then-insert pattern under concurrent load, with a
// 90% hit rate to reflect a well-populated filter late in a crawl.
// ============================================================================

fn bench_asbf_real_world_url_dedup(c: &mut Criterion) {
    let mut g = c.benchmark_group("real_world/asbf/url_dedup");
    const VISITED: usize = 100_000;
    const FETCHER_THREADS: usize = 8;
    const OPS_PER_THREAD: usize = 5_000;

    let visited  = gen_urls(VISITED);
    let new_urls = gen_urls(VISITED)
        .into_iter()
        .map(|u| format!("{}/new", u))
        .collect::<Vec<_>>();

    let f = Arc::new(AtomicScalableBloomFilter::<String>::new(VISITED, 0.00001).unwrap());
    for url in &visited { f.insert(url); }

    g.throughput(Throughput::Elements((FETCHER_THREADS * OPS_PER_THREAD) as u64));

    // 90% hit (URL already visited): typical late-crawl workload.
    g.bench_function("90pct_seen/8_threads", |b| {
        b.iter_batched(
            || Arc::clone(&f),
            |f| {
                let handles: Vec<_> = (0..FETCHER_THREADS)
                    .map(|tid| {
                        let f         = Arc::clone(&f);
                        let visited   = visited.clone();
                        let new_urls  = new_urls.clone();
                        thread::spawn(move || {
                            let mut idx = tid * OPS_PER_THREAD;
                            for _ in 0..OPS_PER_THREAD {
                                let url = if idx % 10 == 0 {
                                    &new_urls[idx % new_urls.len()]
                                } else {
                                    &visited[idx % VISITED]
                                };
                                if !f.contains(url) { f.insert(url); }
                                idx = idx.wrapping_add(1);
                            }
                        })
                    })
                    .collect();
                for h in handles { h.join().unwrap(); }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // 100% miss (all-new URLs): models a fresh-domain crawl burst.
    g.bench_function("all_miss/8_threads", |b| {
        b.iter_batched(
            || Arc::new(AtomicScalableBloomFilter::<String>::new(VISITED, 0.00001).unwrap()),
            |f| {
                let handles: Vec<_> = (0..FETCHER_THREADS)
                    .map(|tid| {
                        let f        = Arc::clone(&f);
                        let new_urls = new_urls.clone();
                        thread::spawn(move || {
                            let base = tid * OPS_PER_THREAD;
                            for i in 0..OPS_PER_THREAD {
                                let url = &new_urls[(base + i) % new_urls.len()];
                                if !f.contains(url) { f.insert(url); }
                            }
                        })
                    })
                    .collect();
                for h in handles { h.join().unwrap(); }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ============================================================================
// 29. REAL-WORLD: CONCURRENT SESSION REVOCATION
//
// Auth services check whether a session token has been revoked on every
// API request. Many reader threads check concurrently; one background writer
// adds newly-revoked tokens. This bench models that pattern and measures
// whether the RwLock gives true read scalability for this workload.
// ============================================================================

fn bench_asbf_real_world_session_revocation(c: &mut Criterion) {
    let mut g = c.benchmark_group("real_world/asbf/session_revocation");
    const REVOKED: usize = 50_000;
    const CHECK_THREADS: usize = 8;
    const CHECKS_PER_THREAD: usize = 5_000;

    let revoked = gen_session_ids(REVOKED);
    let valid: Vec<String> = gen_session_ids(REVOKED)
        .into_iter()
        .map(|s| format!("v_{s}"))
        .collect();

    let f = Arc::new(AtomicScalableBloomFilter::<String>::new(REVOKED, 0.000_001).unwrap());
    for id in &revoked { f.insert(id); }

    // 8 threads checking valid (miss) sessions with 1 background writer adding
    // revoked tokens. Models a high-traffic API gateway.
    g.throughput(Throughput::Elements((CHECK_THREADS * CHECKS_PER_THREAD) as u64));
    g.bench_function("8_checkers_1_writer", |b| {
        b.iter_batched(
            || Arc::clone(&f),
            |f| {
                let done = Arc::new(AtomicUsize::new(0));
                let mut handles = Vec::with_capacity(CHECK_THREADS + 1);

                // 1 writer adding new revocations until all readers finish.
                {
                    let f    = Arc::clone(&f);
                    let done = Arc::clone(&done);
                    handles.push(thread::spawn(move || {
                        let mut i = REVOKED as u64;
                        while done.load(Ordering::Relaxed) < CHECK_THREADS {
                            let id = format!("new_revoked_{}", i);
                            f.insert(&id);
                            i += 1;
                        }
                    }));
                }

                // 8 readers checking sessions.
                for tid in 0..CHECK_THREADS {
                    let f     = Arc::clone(&f);
                    let valid = valid.clone();
                    let done  = Arc::clone(&done);
                    handles.push(thread::spawn(move || {
                        let base = tid * CHECKS_PER_THREAD;
                        for i in 0..CHECKS_PER_THREAD {
                            let id = &valid[(base + i) % REVOKED];
                            black_box(f.contains(id));
                        }
                        done.fetch_add(1, Ordering::Relaxed);
                    }));
                }

                for h in handles { h.join().unwrap(); }
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
    asbf_benches,
    // Single-threaded baselines
    bench_asbf_insert_single_thread,
    bench_asbf_contains_hit_depth,
    bench_asbf_contains_miss_depth,
    bench_asbf_contains_batch,
    bench_asbf_contains_fpr_sensitivity,
    // Concurrent insert
    bench_asbf_concurrent_insert_scaling,
    bench_asbf_insert_scale,
    bench_asbf_with_preallocated,
    bench_asbf_concurrent_batch_insert,
    bench_asbf_insert_batch_vs_loop,
    bench_asbf_growth_event_latency,
    // Concurrent contains
    bench_asbf_concurrent_contains_scaling,
    // Mixed read/write
    bench_asbf_mixed_rw_contention,
    bench_asbf_mixed_rw_ratios,
    bench_asbf_write_heavy_scaling,
    // Clone
    bench_asbf_clone,
    // Maintenance
    bench_asbf_clear_concurrent,
    bench_asbf_clear_and_reinsert,
    // Analytics
    bench_asbf_estimate_fpr,
    bench_asbf_memory_usage,
    bench_asbf_bit_statistics,
    bench_asbf_current_fill_rate_under_load,
    bench_asbf_cheap_accessors,
    // SBF comparison
    bench_sbf_vs_asbf_single_thread,
    bench_sbf_vs_asbf_depth,
    // Growth strategies
    bench_asbf_growth_strategies,
    // Real-world
    bench_asbf_real_world_dedup_pipeline,
    bench_asbf_real_world_url_dedup,
    bench_asbf_real_world_session_revocation,
);

criterion_main!(asbf_benches);