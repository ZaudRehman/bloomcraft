//! Benchmarks for [`RegisterBlockedBloomFilter`].
//!
//! Run all:        cargo bench --bench register_blocked_filter
//! Run one group:  cargo bench --bench register_blocked_filter -- rbbf/contains
//! HTML reports:   target/criterion/report/index.html
//!
//! # Coverage map
//!
//!  1. rbbf/construction          — allocation cost at 1k/100k/10M items
//!  2. rbbf/insert/sequential     — throughput at 1k/10k/100k/1M items
//!  3. rbbf/insert/fill_curve     — insert throughput at 25/50/75/100/150% fill
//!  4. rbbf/contains/hit          — hit latency at 10%/50%/100% fill
//!  5. rbbf/contains/miss         — miss latency at 10%/50%/100% fill
//!  6. rbbf/contains/mixed        — 50/50, 90/10, 10/90 hit/miss ratios
//!  7. rbbf/contains/sequential   — sequential access (cache warm) vs random
//!  8. rbbf/fpr_targets           — contains latency across FPR 0.5→0.0001 (varies k)
//!  9. rbbf/scale                 — contains latency at 1k/100k/10M capacity (L1→DRAM)
//! 10. rbbf/clear                 — clear() cost at 1k/100k/10M items
//! 11. rbbf/analytics/false_positive_rate  — false_positive_rate() at 10/50/100% fill
//! 12. rbbf/analytics/estimate_count       — estimate_count() at 10/50/100% fill
//! 13. rbbf/analytics/count_set_bits       — count_set_bits() at varying sizes
//! 14. rbbf/key_types             — u64 vs String vs &[u8] insert+contains cost
//! 15. rbbf/clone                 — clone() cost at 1k/100k/10M items
//! 16. rbbf/vs_standard           — rbbf vs StandardBloomFilter insert+contains
//! 17. rbbf/real_world/packet_filter       — network packet dedup simulation
//! 18. rbbf/real_world/cache_guard         — hot-path cache guard simulation
//! 19. rbbf/real_world/write_then_read     — bulk insert followed by query burst

use bloomcraft::core::filter::BloomFilter;
use bloomcraft::filters::{RegisterBlockedBloomFilter, StandardBloomFilter};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ── Helpers ───────────────────────────────────────────────────────────────────

type Rbbf = RegisterBlockedBloomFilter<u64>;

fn gen_u64(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn gen_strings(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("key:{:016x}", i)).collect()
}

/// Build a filter pre-populated with `n` items.
fn populated(capacity: usize, fpr: f64, n: usize) -> Rbbf {
    let mut f = Rbbf::new(capacity, fpr).unwrap();
    for i in 0..n as u64 {
        f.insert(&i);
    }
    f
}

// ── 1. CONSTRUCTION ───────────────────────────────────────────────────────────
//
// Measures the allocation cost of new() — zeroing the bit array is the
// dominant cost at large capacities. This is the baseline for every subsequent
// bench that uses iter_batched with a fresh filter in the setup closure.

fn bench_construction(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/construction");

    for &(n, label) in &[
        (1_000usize, "1k"),
        (100_000usize, "100k"),
        (10_000_000usize, "10M"),
    ] {
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(label, |b| {
            b.iter(|| black_box(Rbbf::new(black_box(n), 0.01).unwrap()))
        });
    }
    g.finish();
}

// ── 2. INSERT — SEQUENTIAL THROUGHPUT ─────────────────────────────────────────
//
// Measures raw insert throughput from a cold filter to full capacity.
// Shows how throughput changes as the number of items grows and blocks
// saturate. At large n, memory bandwidth becomes the bottleneck.

fn bench_insert_sequential(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/insert/sequential");
    g.sample_size(10);

    for &(n, label) in &[
        (1_000usize, "1k"),
        (10_000usize, "10k"),
        (100_000usize, "100k"),
        (1_000_000usize, "1M"),
    ] {
        let keys = gen_u64(n);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::from_parameter(label), &n, |b, &n| {
            b.iter_batched(
                || Rbbf::new(n, 0.01).unwrap(),
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

// ── 3. INSERT — FILL CURVE ────────────────────────────────────────────────────
//
// Inserts a fixed batch of 1,000 items at different fill levels of a
// 100k-capacity filter. Reveals whether insert throughput degrades as the
// block fill rate rises (more bit collisions, no new cache misses — should
// stay flat; any degradation points to unexpected branching).

fn bench_insert_fill_curve(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/insert/fill_curve");
    const CAP: usize = 100_000;
    const BATCH: usize = 1_000;
    let batch_keys: Vec<u64> = (CAP as u64..CAP as u64 + BATCH as u64).collect();

    for &(prefill, label) in &[
        (CAP / 4, "25pct_fill"),
        (CAP / 2, "50pct_fill"),
        (CAP * 3 / 4, "75pct_fill"),
        (CAP, "100pct_fill"),
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

// ── 4. CONTAINS — HIT PATH ────────────────────────────────────────────────────
//
// Hit latency at different fill levels. The filter guarantees one cache miss
// per query regardless of fill; this bench confirms that cache miss cost
// dominates and hit latency stays flat as fill increases.

fn bench_contains_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/contains/hit");
    const CAP: usize = 100_000;

    for &(fill, label) in &[
        (CAP / 10, "10pct_fill"),
        (CAP / 2, "50pct_fill"),
        (CAP, "100pct_fill"),
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

// ── 5. CONTAINS — MISS PATH ───────────────────────────────────────────────────
//
// Miss latency at different fill levels. A miss always checks all k bits
// (no early exit until all pass), so miss cost > hit cost at low fill.
// At high fill almost every bit is set so the first check usually passes
// and misses take as long as hits.

fn bench_contains_miss(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/contains/miss");
    const CAP: usize = 100_000;

    for &(fill, label) in &[
        (CAP / 10, "10pct_fill"),
        (CAP / 2, "50pct_fill"),
        (CAP, "100pct_fill"),
    ] {
        let f = populated(CAP, 0.01, fill);
        // Miss keys are well outside the inserted range.
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

// ── 6. CONTAINS — MIXED HIT/MISS RATIOS ──────────────────────────────────────
//
// Real workloads are rarely pure hit or pure miss. This bench covers three
// realistic ratios at 50% fill to model cache guards (90% miss) and
// deduplication (90% hit).

fn bench_contains_mixed(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/contains/mixed");
    const CAP: usize = 100_000;
    const FILL: usize = CAP / 2;
    let f = populated(CAP, 0.01, FILL);
    let miss_base = CAP as u64 * 1_000_000;

    // (hit_every_n, label): hit when idx % n == 0, else miss.
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

// ── 7. CONTAINS — SEQUENTIAL vs RANDOM ACCESS PATTERN ────────────────────────
//
// The single-cache-miss guarantee holds regardless of access pattern, but
// prefetcher behaviour differs between sequential and random access. This
// bench quantifies whether the prefetcher gives any additional benefit on
// sequential key streams.

fn bench_contains_access_pattern(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/contains/sequential_vs_random");
    const CAP: usize = 100_000;
    const N: usize = 50_000;
    let f = populated(CAP, 0.01, N);

    // Sequential: keys 0, 1, 2, ... wrap at N.
    g.throughput(Throughput::Elements(1));
    g.bench_function("sequential", |b| {
        let mut idx = 0u64;
        b.iter(|| {
            let key = idx % N as u64;
            idx = idx.wrapping_add(1);
            black_box(f.contains(black_box(&key)))
        });
    });

    // Pseudo-random: multiplicative hash scatter to defeat the prefetcher.
    g.bench_function("random", |b| {
        let mut state = 0u64;
        b.iter(|| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let key = state % N as u64;
            black_box(f.contains(black_box(&key)))
        });
    });

    g.finish();
}

// ── 8. FPR TARGET SWEEP — k SENSITIVITY ──────────────────────────────────────
//
// The FPR target drives k (hash count). Lower FPR → higher k → more bit
// checks per query. This bench isolates the per-k cost of contains() by
// holding fill at 50% of capacity and varying only the FPR target.
// Run cargo bench -- rbbf/fpr_targets and look at absolute ns to see
// the incremental cost of each additional hash probe.

fn bench_fpr_targets(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/fpr_targets");
    const CAP: usize = 100_000;

    for &(fpr, label) in &[
        (0.5f64, "fpr_0.5"),
        (0.1, "fpr_0.1"),
        (0.01, "fpr_0.01"),
        (0.001, "fpr_0.001"),
        (0.0001, "fpr_0.0001"),
    ] {
        let fill = CAP / 2;
        let f = populated(CAP, fpr, fill);
        let k = f.hash_count();
        let miss_base = CAP as u64 * 1_000_000;

        g.throughput(Throughput::Elements(1));
        // Label includes k so the report shows the relationship directly.
        g.bench_function(format!("{label}_k{k}/hit"), |b| {
            let mut idx = 0u64;
            b.iter(|| {
                let key = idx % fill as u64;
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

// ── 9. SCALE — L1 CACHE vs LLC vs DRAM ───────────────────────────────────────
//
// The single-cache-miss guarantee is the filter's central claim. This bench
// tests it across filter sizes that span cache hierarchy levels:
//   - 1k items   → filter fits in L1 (32 KB on most x86) → no real miss
//   - 100k items → filter in LLC (8–32 MB on modern chips) → LLC miss
//   - 10M items  → filter in DRAM → true memory latency
// The latency increase from L1→LLC→DRAM should follow hardware cache miss
// costs (~4 ns, ~12 ns, ~70 ns), demonstrating the single-miss model is real.

fn bench_scale(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/scale");

    for &(cap, label) in &[
        (1_000usize, "1k_l1"),
        (100_000usize, "100k_llc"),
        (10_000_000usize, "10M_dram"),
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
            let mut idx = 0u64;
            b.iter(|| {
                idx = idx.wrapping_add(1);
                black_box(f.contains(black_box(&(miss_base.wrapping_add(idx)))))
            });
        });
    }
    g.finish();
}

// ── 10. CLEAR ─────────────────────────────────────────────────────────────────
//
// clear() calls blocks.fill(0), which is a memset over the whole backing Vec.
// Cost is purely proportional to allocated size, not to fill level.
// This bench shows the memset cost at each filter size — relevant for
// window-rotation patterns where a filter is cleared and reused periodically.

fn bench_clear(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/clear");

    for &(cap, label) in &[
        (1_000usize, "1k"),
        (100_000usize, "100k"),
        (10_000_000usize, "10M"),
    ] {
        g.bench_function(label, |b| {
            b.iter_batched(
                || populated(cap, 0.01, cap / 2),
                |mut f| {
                    f.clear();
                    black_box(())
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    g.finish();
}

// ── 11. ANALYTICS — false_positive_rate() ─────────────────────────────────────
//
// false_positive_rate() is a pure arithmetic function with no memory access
// beyond reading three fields (item_count, num_blocks, k). Its cost should be
// O(1) and independent of filter size. This bench confirms that and gives the
// absolute cost so callers know whether it is safe on a tight hot path.

fn bench_false_positive_rate(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/analytics/false_positive_rate");
    const CAP: usize = 100_000;

    for &(fill, label) in &[
        (CAP / 10, "10pct_fill"),
        (CAP / 2, "50pct_fill"),
        (CAP, "100pct_fill"),
    ] {
        let f = populated(CAP, 0.01, fill);
        g.bench_function(label, |b| b.iter(|| black_box(f.false_positive_rate())));
    }
    g.finish();
}

// ── 12. ANALYTICS — estimate_count() ──────────────────────────────────────────
//
// estimate_count() iterates the whole blocks Vec via count_ones() (popcount).
// Cost is O(num_blocks) — proportional to filter size, not fill level.
// This bench gives the absolute cost so callers know not to call it in tight
// per-packet loops at large filter sizes.

fn bench_estimate_count(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/analytics/estimate_count");

    for &(cap, fill_pct, label) in &[
        (100_000usize, 50usize, "100k_50pct"),
        (100_000usize, 100, "100k_100pct"),
        (10_000_000usize, 50, "10M_50pct"),
    ] {
        let fill = cap * fill_pct / 100;
        let f = populated(cap, 0.01, fill);
        g.bench_function(label, |b| b.iter(|| black_box(f.estimate_count())));
    }
    g.finish();
}

// ── 13. ANALYTICS — count_set_bits() ──────────────────────────────────────────
//
// count_set_bits() is the popcount inner loop: the same traversal as
// estimate_count() but returning a raw count rather than an estimate.
// Benchmarked separately to give the cost of the pure popcount kernel
// versus the arithmetic on top of it in estimate_count().

fn bench_count_set_bits(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/analytics/count_set_bits");

    for &(cap, label) in &[
        (1_000usize, "1k"),
        (100_000usize, "100k"),
        (10_000_000usize, "10M"),
    ] {
        let f = populated(cap, 0.01, cap / 2);
        g.bench_function(label, |b| b.iter(|| black_box(f.count_set_bits())));
    }
    g.finish();
}

// ── 14. KEY TYPES ─────────────────────────────────────────────────────────────
//
// The filter's hot path is: hash(item) → block select → k bit checks.
// Hashing cost varies by key type. u64 is cheapest; heap-allocated String
// hashes the full UTF-8 bytes; &[u8] is similar. This bench isolates the
// hash cost from the bit-array cost by comparing insert and contains across
// all three types at the same filter size.

fn bench_key_types(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/key_types");
    const N: usize = 50_000;

    // u64
    let u64_keys = gen_u64(N);
    let mut f_u64: Rbbf = Rbbf::new(N, 0.01).unwrap();
    for k in &u64_keys {
        f_u64.insert(k);
    }
    let miss_u64 = N as u64 * 999_999;

    g.throughput(Throughput::Elements(1));
    g.bench_function("u64/insert", |b| {
        b.iter_batched(
            || Rbbf::new(N, 0.01).unwrap(),
            |mut f| {
                for k in &u64_keys {
                    f.insert(black_box(k));
                }
                black_box(f)
            },
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

    // String
    let str_keys = gen_strings(N);
    let mut f_str: RegisterBlockedBloomFilter<String> =
        RegisterBlockedBloomFilter::new(N, 0.01).unwrap();
    for k in &str_keys {
        f_str.insert(k);
    }
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

// ── 15. CLONE ─────────────────────────────────────────────────────────────────
//
// clone() deep-copies the full backing Vec<u64>. Cost is O(num_blocks) and
// dominated by memcpy bandwidth. This bench gives the raw cost so callers
// using clone for snapshot-and-query patterns know the upfront price.

fn bench_clone(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/clone");

    for &(cap, label) in &[
        (1_000usize, "1k"),
        (100_000usize, "100k"),
        (10_000_000usize, "10M"),
    ] {
        let f = populated(cap, 0.01, cap / 2);
        g.bench_function(label, |b| b.iter(|| black_box(f.clone())));
    }
    g.finish();
}

// ── 16. vs STANDARD BLOOM FILTER ─────────────────────────────────────────────
//
// The module doc claims 20–30% faster queries than a standard Bloom filter.
// This bench gives the raw numbers to verify that claim. Both filters are
// configured for the same target FPR and item count. The RBBF uses more
// memory (1.3–1.5×) in exchange for the single-cache-miss guarantee.
//
// StandardBloomFilter spreads k probes across the full bit array, incurring
// up to k cache misses per query; RBBF concentrates all k probes in one
// 64-byte block, incurring exactly 1. The gap widens with k (lower FPR).

fn bench_vs_standard(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/vs_standard");
    const CAP: usize = 100_000;
    const FPR: f64 = 0.01;
    const FILL: usize = CAP / 2;

    // RBBF
    let rbbf = populated(CAP, FPR, FILL);

    // StandardBloomFilter
    let sbf: StandardBloomFilter<u64> = StandardBloomFilter::new(CAP, FPR).unwrap();
    for i in 0..FILL as u64 {
        sbf.insert(&i);
    }

    let key_hit = (FILL / 2) as u64;
    let key_miss = CAP as u64 * 1_000_000;

    g.throughput(Throughput::Elements(1));

    g.bench_function("rbbf/insert", |b| {
        b.iter_batched(
            || Rbbf::new(CAP, FPR).unwrap(),
            |mut f| {
                f.insert(black_box(&42u64));
                black_box(f)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("standard/insert", |b| {
        b.iter_batched(
            || StandardBloomFilter::<u64>::new(CAP, FPR).unwrap(),
            |f| {
                f.insert(black_box(&42u64));
                black_box(f)
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("rbbf/contains_hit", |b| {
        b.iter(|| black_box(rbbf.contains(black_box(&key_hit))))
    });
    g.bench_function("standard/contains_hit", |b| {
        b.iter(|| black_box(sbf.contains(black_box(&key_hit))))
    });
    g.bench_function("rbbf/contains_miss", |b| {
        b.iter(|| black_box(rbbf.contains(black_box(&key_miss))))
    });
    g.bench_function("standard/contains_miss", |b| {
        b.iter(|| black_box(sbf.contains(black_box(&key_miss))))
    });

    g.finish();
}

// ── 17. REAL-WORLD: NETWORK PACKET FILTER ─────────────────────────────────────
//
// Models a high-frequency network packet deduplication path. Each arriving
// packet hash is checked against the filter; if not seen, it is inserted.
// The access pattern is pseudo-random (packets arrive from arbitrary sources).
// 95% of packets are new (miss-then-insert); 5% are duplicates (hit-only).
// Throughput in elements/sec is the metric: this is the sustainable packet
// dedup rate the filter can support on a single core.

fn bench_real_world_packet_filter(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/real_world/packet_filter");

    // Network packet identifier: 5-tuple hash represented as u64.
    const WINDOW: usize = 1_000_000; // distinct flows in sliding window
    const BATCH: usize = 10_000; // packets per measurement iteration

    g.throughput(Throughput::Elements(BATCH as u64));

    g.bench_function("95pct_new_5pct_dup", |b| {
        b.iter_batched(
            || {
                // Pre-seed with 5% of the window as "already seen" duplicates.
                let mut f = Rbbf::new(WINDOW, 0.001).unwrap();
                for i in 0..(WINDOW / 20) as u64 {
                    f.insert(&i);
                }
                f
            },
            |mut f| {
                let mut state = 0xdeadbeef_u64;
                for _ in 0..BATCH {
                    // Pseudo-random packet hash.
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let pkt = state % WINDOW as u64;
                    if !f.contains(&pkt) {
                        f.insert(&pkt);
                    }
                }
                black_box(f)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    g.finish();
}

// ── 18. REAL-WORLD: CACHE GUARD (HOT PATH) ────────────────────────────────────
//
// Models a cache guard: before issuing an expensive backend lookup, the
// caller checks whether the key is known-missing (negative cache). The filter
// holds known-present keys; a miss in the filter means the key definitely
// needs a backend fetch. A hit in the filter means likely-cached (though
// false positives cause unnecessary cache lookups).
//
// This workload is purely read (no inserts in the hot loop); the filter is
// populated once at setup. Throughput is queries/sec.

fn bench_real_world_cache_guard(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/real_world/cache_guard");

    const CACHED_KEYS: usize = 500_000;
    const QUERIES: usize = 10_000;

    // Build the filter once.
    let guard = populated(CACHED_KEYS, 0.001, CACHED_KEYS);

    g.throughput(Throughput::Elements(QUERIES as u64));

    // Hot path: 80% of queries hit cached keys (filter returns true → no fetch).
    g.bench_function("80pct_cached_20pct_uncached", |b| {
        let mut state = 0u64;
        b.iter(|| {
            let mut fetches = 0u32;
            for _ in 0..QUERIES {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let key = if state % 100 < 80 {
                    state % CACHED_KEYS as u64 // likely cached
                } else {
                    CACHED_KEYS as u64 + state // definitely not cached
                };
                if !guard.contains(&key) {
                    // Would trigger a backend fetch in production.
                    fetches += 1;
                }
            }
            black_box(fetches)
        });
    });

    g.finish();
}

// ── 19. REAL-WORLD: BULK WRITE THEN READ BURST ────────────────────────────────
//
// Models an ETL or batch-processing pattern: ingest a large dataset into the
// filter (write phase), then issue a burst of membership queries (read phase).
// This two-phase pattern tests whether the filter's memory layout is friendly
// to both sequential writes and subsequent random reads.
// Each bench variant represents a different dataset size.

fn bench_real_world_write_then_read(c: &mut Criterion) {
    let mut g = c.benchmark_group("rbbf/real_world/write_then_read");
    g.sample_size(10);

    const READ_QUERIES: usize = 50_000;

    for &(n, label) in &[
        (10_000usize, "10k"),
        (100_000usize, "100k"),
        (1_000_000usize, "1M"),
    ] {
        let write_keys = gen_u64(n);
        // Mix of hits (inserted keys) and misses (unseen keys).
        let read_keys: Vec<u64> = (0..READ_QUERIES as u64)
            .map(|i| {
                if i % 2 == 0 {
                    i % n as u64
                } else {
                    n as u64 * 10 + i
                }
            })
            .collect();

        g.throughput(Throughput::Elements((n + READ_QUERIES) as u64));
        g.bench_with_input(BenchmarkId::from_parameter(label), &n, |b, _| {
            b.iter_batched(
                || Rbbf::new(n, 0.01).unwrap(),
                |mut f| {
                    // Write phase.
                    for k in &write_keys {
                        f.insert(black_box(k));
                    }
                    // Read phase.
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
    rbbf_benches,
    bench_construction,
    bench_insert_sequential,
    bench_insert_fill_curve,
    bench_contains_hit,
    bench_contains_miss,
    bench_contains_mixed,
    bench_contains_access_pattern,
    bench_fpr_targets,
    bench_scale,
    bench_clear,
    bench_false_positive_rate,
    bench_estimate_count,
    bench_count_set_bits,
    bench_key_types,
    bench_clone,
    bench_vs_standard,
    bench_real_world_packet_filter,
    bench_real_world_cache_guard,
    bench_real_world_write_then_read,
);

criterion_main!(rbbf_benches);
