//! Benchmarks for [`PartitionedBloomFilter`].
//!
//! Run all:        cargo bench --bench partitioned_filter
//! Run one group:  cargo bench --bench partitioned_filter -- partitioned/contains
//! HTML reports:   target/criterion/report/index.html
//!
//! # Coverage map
//!
//!  1. partitioned/construction            — new() allocation cost at 1k/100k/10M items
//!  2. partitioned/construction/cache_tuned — new() vs new_cache_tuned() vs with_alignment()
//!  3. partitioned/insert/sequential       — throughput at 1k/10k/100k/1M items
//!  4. partitioned/insert/batch_vs_loop    — insert_batch vs loop-insert at varying sizes
//!  5. partitioned/insert/fill_curve       — insert throughput at 25/50/75/100/150% fill
//!  6. partitioned/contains/hit            — hit latency at 10/50/100% fill
//!  7. partitioned/contains/miss           — miss latency at 10/50/100% fill
//!  8. partitioned/contains/batch          — contains_batch at varying batch sizes
//!  9. partitioned/contains/mixed          — 50/50, 90/10, 10/90 hit/miss ratios
//! 10. partitioned/k_sensitivity           — contains latency across FPR 0.5→0.0001 (varies k)
//! 11. partitioned/alignment_sweep         — contains latency at 16/32/64/128/256-byte alignment
//! 12. partitioned/scale                   — contains latency at 1k/100k/10M capacity
//! 13. partitioned/clear                   — clear() cost (memset) at 1k/100k/10M items
//! 14. partitioned/clone                   — clone() cost (full realloc+memcpy) at varying sizes
//! 15. partitioned/set_ops/union           — union() cost at varying sizes
//! 16. partitioned/set_ops/union_new       — union_new() vs union() (alloc overhead)
//! 17. partitioned/set_ops/intersect       — intersect() cost at varying sizes
//! 18. partitioned/analytics/saturation    — saturation() cost (full bit scan) at varying sizes
//! 19. partitioned/analytics/estimate_count — estimate_count() cost at varying fill
//! 20. partitioned/analytics/partition_stats — partition_stats() cost (per-partition popcount)
//! 21. partitioned/key_types               — u64 vs String insert+contains cost
//! 22. partitioned/vs_standard             — partitioned vs StandardBloomFilter insert+contains
//! 23. partitioned/real_world/cache_guard  — hot-path negative-cache simulation
//! 24. partitioned/real_world/write_then_read — bulk ingest then query burst

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::filters::{PartitionedBloomFilter, StandardBloomFilter};
use bloomcraft::core::filter::BloomFilter;

// ── Helpers ───────────────────────────────────────────────────────────────────

type Pbf = PartitionedBloomFilter<u64>;

fn gen_u64(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn gen_strings(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("key:{:016x}", i)).collect()
}

/// Build a filter pre-populated with `n` items.
fn populated(capacity: usize, fpr: f64, n: usize) -> Pbf {
    let mut f = Pbf::new(capacity, fpr).unwrap();
    for i in 0..n as u64 {
        f.insert(&i);
    }
    f
}

// ── 1. CONSTRUCTION ───────────────────────────────────────────────────────────
//
// new() allocates a single flat 64-byte-aligned buffer via std::alloc and
// zeroes it with write_bytes. This bench measures that cost at different
// capacities — the cost should scale linearly with bit_count() / 8.

fn bench_construction(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/construction");

    for &(n, label) in &[
        (1_000usize,      "1k"),
        (100_000usize,    "100k"),
        (10_000_000usize, "10M"),
    ] {
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(label, |b| {
            b.iter(|| black_box(Pbf::new(black_box(n), 0.01).unwrap()))
        });
    }
    g.finish();
}

// ── 2. CONSTRUCTION VARIANTS ──────────────────────────────────────────────────
//
// new_cache_tuned() adds a cache-detection syscall/CPUID step before
// allocation; with_alignment() skips detection but still validates a custom
// alignment. This bench isolates the cost of cache detection from the
// allocation itself.

fn bench_construction_variants(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/construction/variants");
    const N: usize = 100_000;

    g.bench_function("new", |b| {
        b.iter(|| black_box(Pbf::new(black_box(N), 0.01).unwrap()))
    });
    g.bench_function("new_cache_tuned", |b| {
        b.iter(|| black_box(Pbf::new_cache_tuned(black_box(N), 0.01).unwrap()))
    });
    g.bench_function("with_alignment_64", |b| {
        b.iter(|| black_box(Pbf::with_alignment(black_box(N), 0.01, 64).unwrap()))
    });
    g.bench_function("with_alignment_128", |b| {
        b.iter(|| black_box(Pbf::with_alignment(black_box(N), 0.01, 128).unwrap()))
    });

    g.finish();
}

// ── 3. INSERT — SEQUENTIAL THROUGHPUT ─────────────────────────────────────────

fn bench_insert_sequential(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/insert/sequential");
    g.sample_size(10);

    for &(n, label) in &[
        (1_000usize,       "1k"),
        (10_000usize,      "10k"),
        (100_000usize,     "100k"),
        (1_000_000usize,   "1M"),
    ] {
        let keys = gen_u64(n);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(label), &n, |b, &n| {
            b.iter_batched(
                || Pbf::new(n, 0.01).unwrap(),
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

// ── 4. INSERT_BATCH vs LOOP INSERT ────────────────────────────────────────────
//
// insert_batch currently degrades to a per-item insert() loop regardless of
// the simd feature flag (the SIMD branch is unimplemented and falls through
// to the same loop). This bench documents that reality empirically — the two
// should show statistically indistinguishable throughput. If a real SIMD
// kernel is added later, this bench becomes the regression detector for it.

fn bench_insert_batch_vs_loop(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/insert/batch_vs_loop");

    for &n in &[8usize, 64, 1_000, 10_000, 100_000] {
        let keys = gen_u64(n);
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("insert_batch", n), &n, |b, _| {
            b.iter_batched(
                || Pbf::new(n.max(1_000), 0.01).unwrap(),
                |mut f| {
                    f.insert_batch(black_box(&keys));
                    black_box(f)
                },
                criterion::BatchSize::LargeInput,
            );
        });

        g.bench_with_input(BenchmarkId::new("loop", n), &n, |b, _| {
            b.iter_batched(
                || Pbf::new(n.max(1_000), 0.01).unwrap(),
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

// ── 5. INSERT — FILL CURVE ────────────────────────────────────────────────────
//
// Inserts a fixed batch of 1,000 items at different fill levels of a
// 100k-capacity filter. Each insert touches one bit per partition regardless
// of fill, so this curve should stay flat — any rise points to unexpected
// cost from saturation-dependent branching (there is none in the current
// implementation, so this is a regression guard).

fn bench_insert_fill_curve(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/insert/fill_curve");
    const CAP: usize = 100_000;
    const BATCH: usize = 1_000;
    let batch_keys: Vec<u64> = (CAP as u64..CAP as u64 + BATCH as u64).collect();

    for &(prefill, label) in &[
        (CAP / 4,       "25pct_fill"),
        (CAP / 2,       "50pct_fill"),
        (CAP * 3 / 4,   "75pct_fill"),
        (CAP,           "100pct_fill"),
        (CAP + CAP / 2, "150pct_fill"),
    ] {
        g.throughput(Throughput::Elements(BATCH as u64));
        g.bench_function(label, |b| {
            b.iter_batched(
                || populated(CAP, 0.01, prefill),
                |mut f| {
                    for k in &batch_keys {
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

// ── 6. CONTAINS — HIT PATH ────────────────────────────────────────────────────
//
// All k partitions are checked sequentially with one cache miss per
// partition in the worst case (k cache misses total, vs. k random misses for
// a standard filter spread over the full bit array). This bench shows
// whether hit latency stays flat as fill rises (it should — same k checks
// regardless of fill).

fn bench_contains_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/contains/hit");
    const CAP: usize = 100_000;

    for &(fill, label) in &[
        (CAP / 10,  "10pct_fill"),
        (CAP / 2,   "50pct_fill"),
        (CAP,       "100pct_fill"),
    ] {
        let f = populated(CAP, 0.01, fill);
        g.throughput(Throughput::Elements(1));
        g.bench_function(label, |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = idx % fill as u64;
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });
    }
    g.finish();
}

// ── 7. CONTAINS — MISS PATH ───────────────────────────────────────────────────
//
// contains() returns false on the first zero bit (early exit), so miss
// latency should be lower than hit latency on average, especially at low
// fill where most partitions have many zero bits.

fn bench_contains_miss(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/contains/miss");
    const CAP: usize = 100_000;

    for &(fill, label) in &[
        (CAP / 10,  "10pct_fill"),
        (CAP / 2,   "50pct_fill"),
        (CAP,       "100pct_fill"),
    ] {
        let f = populated(CAP, 0.01, fill);
        let miss_base = CAP as u64 * 1_000_000;
        g.throughput(Throughput::Elements(1));
        g.bench_function(label, |b| {
            let mut idx = 0u64;
            b.iter(|| {
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&(miss_base.wrapping_add(idx)))))
            });
        });
    }
    g.finish();
}

// ── 8. CONTAINS_BATCH ─────────────────────────────────────────────────────────
//
// contains_batch is a plain iterator map over contains() with no batching
// optimisation in the current implementation. This bench gives the absolute
// per-call cost at varying batch sizes, useful as a baseline if a prefetching
// or SIMD batch path is added later.

fn bench_contains_batch(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/contains/batch");
    const TOTAL: usize = 50_000;
    let f = populated(TOTAL, 0.01, TOTAL);

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

// ── 9. CONTAINS — MIXED HIT/MISS RATIOS ──────────────────────────────────────

fn bench_contains_mixed(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/contains/mixed");
    const CAP: usize = 100_000;
    const FILL: usize = CAP / 2;
    let f = populated(CAP, 0.01, FILL);
    let miss_base = CAP as u64 * 1_000_000;

    for &(hit_pct, label) in &[
        (50u64, "50pct_hit"),
        (90u64, "90pct_hit"),
        (10u64, "10pct_hit"),
    ] {
        g.throughput(Throughput::Elements(1));
        g.bench_function(label, |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = if (idx % 100) < hit_pct {
                    idx % FILL as u64
                } else {
                    miss_base.wrapping_add(idx)
                };
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });
    }
    g.finish();
}

// ── 10. k SENSITIVITY ─────────────────────────────────────────────────────────
//
// The FPR target controls k (number of partitions = number of hash probes).
// Lower FPR → more partitions → more cache misses per query (one per
// partition, worst case). This bench isolates the per-partition cost of
// contains() by holding item count constant and only varying the FPR target.

fn bench_k_sensitivity(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/k_sensitivity");
    const N: usize = 50_000;

    for &(fpr, label) in &[
        (0.5f64,    "fpr_0.5"),
        (0.1,       "fpr_0.1"),
        (0.01,      "fpr_0.01"),
        (0.001,     "fpr_0.001"),
        (0.0001,    "fpr_0.0001"),
    ] {
        let f = populated(N, fpr, N);
        let k = f.partition_count();
        let miss_base = N as u64 * 1_000_000;

        g.throughput(Throughput::Elements(1));
        g.bench_function(format!("{label}_k{k}/hit"), |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = idx % N as u64;
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });
        g.bench_function(format!("{label}_k{k}/miss"), |b| {
            let mut idx = 0u64;
            b.iter(|| {
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&(miss_base.wrapping_add(idx)))))
            });
        });
    }
    g.finish();
}

// ── 11. ALIGNMENT SWEEP ───────────────────────────────────────────────────────
//
// The partition_stride is rounded up to `alignment` bytes. Larger alignment
// means more padding between partitions (worse memory density) but
// potentially better false-sharing avoidance in concurrent scenarios and
// SIMD compatibility. This bench shows the single-threaded query cost across
// alignment values — should be roughly flat since the access pattern is
// always one partition at a time.

fn bench_alignment_sweep(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/alignment_sweep");
    const N: usize = 50_000;

    for &alignment in &[16usize, 32, 64, 128, 256] {
        let mut f = Pbf::with_alignment(N, 0.01, alignment).unwrap();
        for i in 0..N as u64 { f.insert(&i); }
        let miss = N as u64 * 1_000_000;

        g.throughput(Throughput::Elements(1));
        g.bench_function(format!("align_{alignment}B/hit"), |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = idx % N as u64;
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });
        g.bench_function(format!("align_{alignment}B/miss"), |b| {
            b.iter(|| black_box(f.contains(black_box(&miss))))
        });
    }
    g.finish();
}

// ── 12. SCALE — L1 vs LLC vs DRAM ─────────────────────────────────────────────
//
// Partition size grows with capacity. At 1k items each partition fits
// entirely in L1; at 10M items partitions may exceed L2/L3, turning the
// "1-2 cache misses" promise into more misses per partition scan. This bench
// spans that range to show where the promise starts to degrade.

fn bench_scale(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/scale");

    for &(cap, label) in &[
        (1_000usize,        "1k"),
        (100_000usize,      "100k"),
        (10_000_000usize,   "10M"),
    ] {
        let n = cap / 2;
        let f = populated(cap, 0.01, n);
        let miss_base = cap as u64 * 1_000_000;

        g.throughput(Throughput::Elements(1));
        g.bench_function(format!("{label}/hit"), |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = idx % n as u64;
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&key)))
            });
        });
        g.bench_function(format!("{label}/miss"), |b| {
            b.iter(|| black_box(f.contains(black_box(&miss_base))))
        });
    }
    g.finish();
}

// ── 13. CLEAR ──────────────────────────────────────────────────────────────────
//
// clear() is a single write_bytes memset over allocated_bytes, independent
// of fill level. Cost scales purely with filter size.

fn bench_clear(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/clear");

    for &(cap, label) in &[
        (1_000usize,        "1k"),
        (100_000usize,      "100k"),
        (10_000_000usize,   "10M"),
    ] {
        g.bench_function(label, |b| {
            b.iter_batched(
                || populated(cap, 0.01, cap / 2),
                |mut f| black_box(f.clear()),
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ── 14. CLONE ──────────────────────────────────────────────────────────────────
//
// Clone allocates a fresh buffer via std::alloc and copies allocated_bytes
// with copy_nonoverlapping. This is strictly more expensive than a Vec clone
// because it goes through the raw allocator API directly. This bench gives
// the real cost so callers using clone for snapshot patterns know the price.

fn bench_clone(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/clone");

    for &(cap, label) in &[
        (1_000usize,        "1k"),
        (100_000usize,      "100k"),
        (10_000_000usize,   "10M"),
    ] {
        let f = populated(cap, 0.01, cap / 2);
        g.bench_function(label, |b| b.iter(|| black_box(f.clone())));
    }
    g.finish();
}

// ── 15. UNION (MUTATING) ──────────────────────────────────────────────────────
//
// union() ORs every word of every partition between two compatible filters.
// Cost is O(bit_count() / 64), independent of item_count.

fn bench_union(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/set_ops/union");

    for &(cap, label) in &[
        (1_000usize,    "1k"),
        (100_000usize,  "100k"),
        (1_000_000usize,"1M"),
    ] {
        g.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let a = populated(cap, 0.01, cap / 2);
                    let mut b2 = Pbf::new(cap, 0.01).unwrap();
                    for i in (cap as u64)..(cap as u64 + (cap / 2) as u64) { b2.insert(&i); }
                    (a, b2)
                },
                |(mut a, b2)| {
                    a.union(black_box(&b2)).unwrap();
                    black_box(a)
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ── 16. UNION_NEW vs UNION ────────────────────────────────────────────────────
//
// union_new() additionally allocates and copies self's buffer before ORing,
// so it pays an extra alloc+memcpy over union(). This bench quantifies that
// overhead directly so callers can decide whether the non-mutating API is
// worth the extra cost for their workload.

fn bench_union_new_vs_union(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/set_ops/union_new_vs_union");
    const CAP: usize = 100_000;

    g.bench_function("union_mutating", |b| {
        b.iter_batched(
            || {
                let a = populated(CAP, 0.01, CAP / 2);
                let mut b2 = Pbf::new(CAP, 0.01).unwrap();
                for i in (CAP as u64)..(CAP as u64 + (CAP / 2) as u64) { b2.insert(&i); }
                (a, b2)
            },
            |(mut a, b2)| { a.union(black_box(&b2)).unwrap(); black_box(a) },
            criterion::BatchSize::LargeInput,
        );
    });

    g.bench_function("union_new_non_mutating", |b| {
        b.iter_batched(
            || {
                let a = populated(CAP, 0.01, CAP / 2);
                let mut b2 = Pbf::new(CAP, 0.01).unwrap();
                for i in (CAP as u64)..(CAP as u64 + (CAP / 2) as u64) { b2.insert(&i); }
                (a, b2)
            },
            |(a, b2)| black_box(a.union_new(black_box(&b2)).unwrap()),
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ── 17. INTERSECT ──────────────────────────────────────────────────────────────
//
// intersect() is structurally identical to union() (full word-by-word scan)
// but ANDs instead of ORs. Cost should match union() exactly.

fn bench_intersect(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/set_ops/intersect");

    for &(cap, label) in &[
        (1_000usize,    "1k"),
        (100_000usize,  "100k"),
        (1_000_000usize,"1M"),
    ] {
        g.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let a = populated(cap, 0.01, cap / 2);
                    let b2 = populated(cap, 0.01, cap / 2);
                    (a, b2)
                },
                |(mut a, b2)| {
                    a.intersect(black_box(&b2)).unwrap();
                    black_box(a)
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ── 18. ANALYTICS — saturation() ──────────────────────────────────────────────
//
// saturation() scans every word of every partition via raw pointer reads and
// popcounts. Cost is O(bit_count() / 64), the same order as union/intersect.
// This bench gives the absolute cost so callers know not to poll it in a hot
// loop at large filter sizes.

fn bench_saturation(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/analytics/saturation");

    for &(cap, label) in &[
        (1_000usize,        "1k"),
        (100_000usize,      "100k"),
        (10_000_000usize,   "10M"),
    ] {
        let f = populated(cap, 0.01, cap / 2);
        g.bench_function(label, |b| b.iter(|| black_box(f.saturation())));
    }
    g.finish();
}

// ── 19. ANALYTICS — estimate_count() ──────────────────────────────────────────
//
// estimate_count() has a fast path (returns item_count directly) when
// saturation < 1%, and a full bit-scan path (same cost as saturation())
// otherwise. This bench exercises both paths explicitly so the fast-path
// speedup is visible in the report.

fn bench_estimate_count(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/analytics/estimate_count");
    const CAP: usize = 1_000_000;

    // All paths now share a single scan via total_set_bits(). The "fast path"
    // avoids the natural-log estimate calculation but still pays for the scan.
    // b.iter_batched is used to prevent compiler hoisting.
    g.bench_function("fast_path_sub_1pct_saturation", |b| {
        b.iter_batched(
            || populated(CAP, 0.01, CAP / 200),
            |f| black_box(f.estimate_count()),
            criterion::BatchSize::SmallInput,
        );
    });

    for &(fill_pct, label) in &[(50usize, "scan_path_50pct"), (100, "scan_path_100pct")] {
        g.bench_function(label, |b| {
            b.iter_batched(
                || populated(CAP, 0.01, CAP * fill_pct / 100),
                |f| black_box(f.estimate_count()),
                criterion::BatchSize::SmallInput,
            );
        });
    }

    g.finish();
}

// ── 20. ANALYTICS — partition_stats() ─────────────────────────────────────────
//
// partition_stats() does a full per-partition popcount and allocates a
// Vec<(usize, usize, f64)> of length k. Cost is the same bit-scan as
// saturation() plus the allocation. This bench gives the cost at varying k
// (driven by FPR target) since the Vec allocation size scales with k.

fn bench_partition_stats(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/analytics/partition_stats");
    const N: usize = 100_000;

    for &(fpr, label) in &[(0.1f64, "low_k"), (0.01, "mid_k"), (0.0001, "high_k")] {
        let f = populated(N, fpr, N / 2);
        let k = f.partition_count();
        g.bench_function(format!("{label}_k{k}"), |b| {
            b.iter(|| black_box(f.partition_stats()))
        });
    }
    g.finish();
}

// ── 21. KEY TYPES ──────────────────────────────────────────────────────────────

fn bench_key_types(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/key_types");
    const N: usize = 50_000;

    let u64_keys = gen_u64(N);
    let mut f_u64 = Pbf::new(N, 0.01).unwrap();
    for k in &u64_keys { f_u64.insert(k); }
    let miss_u64 = N as u64 * 999_999;

    g.throughput(Throughput::Elements(1));
    g.bench_function("u64/insert", |b| {
        b.iter_batched(
            || Pbf::new(N, 0.01).unwrap(),
            |mut f| { for k in &u64_keys { f.insert(black_box(k)); } black_box(f) },
            criterion::BatchSize::LargeInput,
        );
    });
    g.bench_function("u64/contains_hit", |b| {
        let k = u64_keys[N / 2];
        b.iter(|| black_box(f_u64.contains(black_box(&k))))
    });
    g.bench_function("u64/contains_miss", |b| {
        b.iter(|| black_box(f_u64.contains(black_box(&miss_u64))))
    });

    let str_keys = gen_strings(N);
    let mut f_str: PartitionedBloomFilter<String> = PartitionedBloomFilter::new(N, 0.01).unwrap();
    for k in &str_keys { f_str.insert(k); }
    let miss_str = "key:ffffffffffffffff_absent".to_string();

    g.bench_function("string/contains_hit", |b| {
        let k = str_keys[N / 2].clone();
        b.iter(|| black_box(f_str.contains(black_box(&k))))
    });
    g.bench_function("string/contains_miss", |b| {
        b.iter(|| black_box(f_str.contains(black_box(&miss_str))))
    });

    g.finish();
}

// ── 22. vs STANDARD BLOOM FILTER ─────────────────────────────────────────────
//
// The module doc claims 2-4x faster queries via cache locality, at the cost
// of 2-5% higher FPR. This bench gives the raw insert/contains numbers to
// verify that trade-off directly against StandardBloomFilter at matched
// (n, fpr) parameters.

fn bench_vs_standard(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/vs_standard");
    const CAP: usize = 100_000;
    const FPR: f64 = 0.01;
    const FILL: usize = CAP / 2;

    let pbf = populated(CAP, FPR, FILL);

    let mut sbf: StandardBloomFilter<u64> = StandardBloomFilter::new(CAP, FPR).unwrap();
    for i in 0..FILL as u64 { sbf.insert(&i); }

    let key_hit = (FILL / 2) as u64;
    let key_miss = CAP as u64 * 1_000_000;

    g.throughput(Throughput::Elements(1));

    g.bench_function("partitioned/insert", |b| {
        b.iter_batched(
            || Pbf::new(CAP, FPR).unwrap(),
            |mut f| { f.insert(black_box(&42u64)); black_box(f) },
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("standard/insert", |b| {
        b.iter_batched(
            || StandardBloomFilter::<u64>::new(CAP, FPR).unwrap(),
            |mut f| { f.insert(black_box(&42u64)); black_box(f) },
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("partitioned/contains_hit",  |b| b.iter(|| black_box(pbf.contains(black_box(&key_hit)))));
    g.bench_function("standard/contains_hit",     |b| b.iter(|| black_box(sbf.contains(black_box(&key_hit)))));
    g.bench_function("partitioned/contains_miss", |b| b.iter(|| black_box(pbf.contains(black_box(&key_miss)))));
    g.bench_function("standard/contains_miss",    |b| b.iter(|| black_box(sbf.contains(black_box(&key_miss)))));

    g.finish();
}

// ── 23. REAL-WORLD: CACHE GUARD (NEGATIVE CACHE) ─────────────────────────────
//
// Hot-path negative cache: before an expensive backend lookup, check whether
// the key is known-absent. 80% of queries hit cached (known) keys; 20% are
// genuinely new. Pure read workload — filter populated once at setup.

fn bench_real_world_cache_guard(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/real_world/cache_guard");
    const CACHED_KEYS: usize = 500_000;
    const QUERIES: usize = 10_000;

    let guard = populated(CACHED_KEYS, 0.001, CACHED_KEYS);

    g.throughput(Throughput::Elements(QUERIES as u64));
    g.bench_function("80pct_cached_20pct_uncached", |b| {
        let mut state = 0u64;
        b.iter(|| {
            let mut fetches = 0u32;
            for _ in 0..QUERIES {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let key = if state % 100 < 80 {
                    state % CACHED_KEYS as u64
                } else {
                    CACHED_KEYS as u64 + state
                };
                if !guard.contains(&key) {
                    fetches += 1;
                }
            }
            black_box(fetches)
        });
    });

    g.finish();
}

// ── 24. REAL-WORLD: BULK WRITE THEN READ BURST ────────────────────────────────
//
// ETL-style pattern: ingest a dataset, then issue a burst of mixed hit/miss
// queries. Tests whether the flat cache-aligned layout helps both phases.

fn bench_real_world_write_then_read(c: &mut Criterion) {
    let mut g = c.benchmark_group("partitioned/real_world/write_then_read");
    g.sample_size(10);
    const READ_QUERIES: usize = 50_000;

    for &(n, label) in &[
        (10_000usize,      "10k"),
        (100_000usize,     "100k"),
        (1_000_000usize,   "1M"),
    ] {
        let write_keys = gen_u64(n);
        let read_keys: Vec<u64> = (0..READ_QUERIES as u64)
            .map(|i| if i % 2 == 0 { i % n as u64 } else { n as u64 * 10 + i })
            .collect();

        g.throughput(Throughput::Elements((n + READ_QUERIES) as u64));
        g.bench_with_input(BenchmarkId::from_parameter(label), &n, |b, _| {
            b.iter_batched(
                || Pbf::new(n, 0.01).unwrap(),
                |mut f| {
                    for k in &write_keys {
                        f.insert(black_box(k));
                    }
                    let mut hits = 0u32;
                    for k in &read_keys {
                        if f.contains(black_box(k)) {
                            hits += 1;
                        }
                    }
                    black_box((f, hits))
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ── CRITERION GROUPS ─────────────────────────────────────────────────────────

criterion_group!(
    partitioned_benches,
    bench_construction,
    bench_construction_variants,
    bench_insert_sequential,
    bench_insert_batch_vs_loop,
    bench_insert_fill_curve,
    bench_contains_hit,
    bench_contains_miss,
    bench_contains_batch,
    bench_contains_mixed,
    bench_k_sensitivity,
    bench_alignment_sweep,
    bench_scale,
    bench_clear,
    bench_clone,
    bench_union,
    bench_union_new_vs_union,
    bench_intersect,
    bench_saturation,
    bench_estimate_count,
    bench_partition_stats,
    bench_key_types,
    bench_vs_standard,
    bench_real_world_cache_guard,
    bench_real_world_write_then_read,
);

criterion_main!(partitioned_benches);