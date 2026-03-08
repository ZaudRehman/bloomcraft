//! Comprehensive benchmark suite for `StandardBloomFilter`.
//!
//! Covers the complete public API surface: construction, single insert/query,
//! batch operations, concurrent scaling, set algebra, key-type sensitivity,
//! saturation curves, FPR-target sweep, access-pattern cache behaviour,
//! statistics/observability, health classification, and five production
//! simulation scenarios.
//!
//! Bench entry: `standard_bench`
//!
//! ```
//! cargo bench --bench standard_bench --features "concurrent,metrics"
//! ```
//!
//! For profile-guided flamegraphs:
//! ```
//! cargo bench --bench standard_bench -- --profile-time=5
//! ```

use bloomcraft::core::filter::{ConcurrentBloomFilter, MergeableBloomFilter};
use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::hash::StdHasher;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Zipf};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ─── Deterministic data generators ───────────────────────────────────────────
// Benchmarks must be reproducible: always seed the RNG.

fn rng() -> StdRng {
    StdRng::seed_from_u64(0xDEAD_BEEF_CAFE_1337)
}

fn rand_u64s(n: usize) -> Vec<u64> {
    let mut r = rng();
    (0..n).map(|_| r.gen()).collect()
}

/// `len`-char ASCII strings — short (≤32 B) stay in HashBytes inline buffer,
/// long (>128 B) force a heap spill. Benchmarking both isolates alloc cost.
fn rand_strings(n: usize, len: usize) -> Vec<String> {
    let mut r = rng();
    let charset: Vec<char> = "abcdefghijklmnopqrstuvwxyz0123456789".chars().collect();
    (0..n)
        .map(|_| (0..len).map(|_| charset[r.gen_range(0..charset.len())]).collect())
        .collect()
}

/// Zipf-distributed indices (s=1.1) over `vocab` unique keys.
/// Models real production traffic: a small hot-set sees most requests.
fn zipf_u64s(n: usize, vocab: u64) -> Vec<u64> {
    let mut r = rng();
    let dist = Zipf::new(vocab, 1.1).expect("valid Zipf parameters");
    (0..n).map(|_| dist.sample(&mut r) as u64).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Construction
//    Measures: new(), with_hasher(), with_params() at multiple scales.
//    Real-world: dynamic filter creation per tenant / shard in SaaS systems.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/construction");

    for &n in &[1_000usize, 10_000, 100_000, 1_000_000, 10_000_000] {
        group.bench_with_input(
            BenchmarkId::new("new", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    black_box(StandardBloomFilter::<u64>::new(black_box(n), 0.01).unwrap())
                });
            },
        );
    }

    group.bench_function("with_hasher_seeded", |b| {
        b.iter(|| {
            black_box(
                StandardBloomFilter::<u64, _>::with_hasher(
                    black_box(100_000),
                    0.01,
                    StdHasher::with_seed(42),
                )
                .unwrap(),
            )
        });
    });

    // with_params bypasses automatic m/k derivation — measures pure BitVec allocation.
    group.bench_function("with_params_explicit_mk", |b| {
        b.iter(|| {
            black_box(
                StandardBloomFilter::<u64, _>::with_params(
                    black_box(958_506), // m for (100k items, 1% FPR)
                    7,
                    StdHasher::new(),
                )
                .unwrap(),
            )
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Single Insert — Capacity Sweep
//    Measures: lock-free set path latency as m grows (L1 → L2 → L3 → DRAM).
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_insert_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/insert/capacity_sweep");
    group.throughput(Throughput::Elements(1));

    for &capacity in &[1_000usize, 10_000, 100_000, 1_000_000, 10_000_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(capacity),
            &capacity,
            |b, &capacity| {
                let filter = StandardBloomFilter::<u64>::new(capacity, 0.01).unwrap();
                let mut i = 0u64;
                b.iter(|| {
                    filter.insert(black_box(&i));
                    i = i.wrapping_add(1);
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Single Contains — Hit / Miss / Mixed
//    Measures: early-exit on first missing bit (miss) vs all-k-bits check (hit).
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_contains_hit_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/contains/hit_miss");
    group.throughput(Throughput::Elements(1));

    let capacity = 100_000usize;
    let filter = StandardBloomFilter::<u64>::new(capacity, 0.01).unwrap();
    for i in 0..capacity as u64 {
        filter.insert(&i);
    }

    // All hits — all k bit positions must be checked.
    {
        let mut i = 0u64;
        group.bench_function("all_hits", |b| {
            b.iter(|| {
                let r = filter.contains(black_box(&(i % capacity as u64)));
                i = i.wrapping_add(1);
                black_box(r)
            });
        });
    }

    // All misses — early exit after first unset bit; latency is dominated by
    // the first hash position probe.
    {
        let mut i = capacity as u64; // never inserted
        group.bench_function("all_misses", |b| {
            b.iter(|| {
                let r = filter.contains(black_box(&i));
                i = i.wrapping_add(1);
                black_box(r)
            });
        });
    }

    // 50/50 — representative of most production queries.
    {
        let mixed: Vec<u64> = (0..capacity as u64 * 2).collect();
        let mut idx = 0usize;
        group.bench_function("mixed_50_50", |b| {
            b.iter(|| {
                let r = filter.contains(black_box(&mixed[idx % mixed.len()]));
                idx = idx.wrapping_add(1);
                black_box(r)
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Batch Insert
//    Measures: insert_batch vs insert_batch_ref across batch sizes.
//    Expected: batch shows 1.3–1.5× speedup over per-item loop at ≥100 items
//    (cache prefetch amortisation).
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/batch/insert");

    for &batch in &[10usize, 100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(batch as u64));
        let items: Vec<u64> = (0..batch as u64).collect();

        group.bench_with_input(
            BenchmarkId::new("insert_batch", batch),
            &batch,
            |b, _| {
                let filter = StandardBloomFilter::<u64>::new(batch * 4, 0.01).unwrap();
                b.iter(|| filter.insert_batch(black_box(&items)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("insert_batch_ref", batch),
            &batch,
            |b, _| {
                let filter = StandardBloomFilter::<u64>::new(batch * 4, 0.01).unwrap();
                let refs: Vec<&u64> = items.iter().collect();
                b.iter(|| filter.insert_batch_ref(black_box(&refs)));
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Batch Contains
//    Measures: contains_batch vs contains_batch_ref; pre-allocated Vec<bool>
//    path vs per-item contains.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_batch_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/batch/contains");

    for &batch in &[10usize, 100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(batch as u64));

        let items: Vec<u64> = (0..batch as u64).collect();
        let refs: Vec<&u64> = items.iter().collect(); // built once per batch, outside iter()
        let filter = StandardBloomFilter::<u64>::new(batch * 4, 0.01).unwrap();
        filter.insert_batch(&items);

        group.bench_with_input(
            BenchmarkId::new("contains_batch", batch),
            &batch,
            |b, _| b.iter(|| black_box(filter.contains_batch(black_box(items.as_slice())))),
        );

        group.bench_with_input(
            BenchmarkId::new("contains_batch_ref", batch),
            &batch,
            |b, _| b.iter(|| black_box(filter.contains_batch_ref(black_box(refs.as_slice())))),
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Key-Type Sensitivity
//    Measures: HashBytes pipeline cost for each T. Exposes the stack-alloc
//    (≤128 B) vs heap-spill (>128 B) boundary in HashBytes::write().
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_key_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/key_types");
    group.throughput(Throughput::Elements(1));

    macro_rules! scalar_insert {
        ($name:expr, $T:ty, $init:expr, $next:expr) => {{
            let f = StandardBloomFilter::<$T>::new(100_000, 0.01).unwrap();
            let mut v: $T = $init;
            group.bench_function($name, |b| {
                b.iter(|| {
                    f.insert(black_box(&v));
                    v = $next(v);
                });
            });
        }};
    }

    scalar_insert!("u32",  u32,  0u32,  |x: u32|  x.wrapping_add(1));
    scalar_insert!("u64",  u64,  0u64,  |x: u64|  x.wrapping_add(1));
    scalar_insert!("u128", u128, 0u128, |x: u128| x.wrapping_add(1));
    scalar_insert!("i64",  i64,  0i64,  |x: i64|  x.wrapping_add(1));

    // [u8; 16] — raw UUID bytes: 16 B, inline, no alloc.
    {
        let f = StandardBloomFilter::<[u8; 16]>::new(100_000, 0.01).unwrap();
        let mut v = 0u128;
        group.bench_function("uuid_bytes_16", |b| {
            b.iter(|| {
                let key = v.to_le_bytes();
                f.insert(black_box(&key));
                v = v.wrapping_add(1);
            });
        });
    }

    // Short string: 24 chars — inline HashBytes, zero heap allocation.
    {
        let strings = rand_strings(50_000, 24);
        let f = StandardBloomFilter::<String>::new(100_000, 0.01).unwrap();
        let mut idx = 0usize;
        group.bench_function("string_24_inline", |b| {
            b.iter(|| {
                f.insert(black_box(&strings[idx % strings.len()]));
                idx = idx.wrapping_add(1);
            });
        });
    }

    // Long string: 256 chars — forces HashBytes heap spill.
    {
        let strings = rand_strings(10_000, 256);
        let f = StandardBloomFilter::<String>::new(100_000, 0.01).unwrap();
        let mut idx = 0usize;
        group.bench_function("string_256_spill", |b| {
            b.iter(|| {
                f.insert(black_box(&strings[idx % strings.len()]));
                idx = idx.wrapping_add(1);
            });
        });
    }

    // Composite tuple key: (user_id: u32, shard: u16, window: u64) — struct-like.
    {
        let f = StandardBloomFilter::<(u32, u16, u64)>::new(100_000, 0.01).unwrap();
        let mut i = 0u64;
        group.bench_function("tuple_u32_u16_u64", |b| {
            b.iter(|| {
                let key = ((i % 100_000) as u32, (i % 512) as u16, i / 100_000);
                f.insert(black_box(&key));
                i = i.wrapping_add(1);
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. FPR-Target Sweep
//    Different FPR targets imply different m (filter size) and k (hash rounds).
//    Measures: how per-op latency scales with k and m as FPR tightens.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_fpr_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/fpr_targets");
    group.throughput(Throughput::Elements(1));

    for &fpr in &[0.1f64, 0.01, 0.001, 0.0001] {
        let capacity = 100_000usize;
        let filter = StandardBloomFilter::<u64>::new(capacity, fpr).unwrap();
        for i in 0..50_000u64 {
            filter.insert(&i);
        }

        let label = format!("{:.4}", fpr);
        let mut i = 0u64;

        group.bench_with_input(
            BenchmarkId::new("insert", label.clone()),
            &fpr,
            |b, _| {
                b.iter(|| {
                    filter.insert(black_box(&i));
                    i = i.wrapping_add(1);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("contains_hit", label),
            &fpr,
            |b, _| {
                b.iter(|| {
                    let r = filter.contains(black_box(&(i % 50_000)));
                    i = i.wrapping_add(1);
                    black_box(r)
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. Saturation Degradation
//    Measures insert/contains latency as fill rate climbs 0% → 100%.
//    Exposes: cache miss rate increase as more bits are set (atomic loads touch
//    more distinct cache lines).
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_saturation(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/saturation");
    group.throughput(Throughput::Elements(1));

    let capacity = 100_000usize;

    for &fill_pct in &[0u64, 10, 25, 50, 75, 90, 100] {
        let pre_fill = (capacity as u64 * fill_pct / 100) as usize;

        group.bench_with_input(
            BenchmarkId::new("insert_at_pct", fill_pct),
            &pre_fill,
            |b, &pre_fill| {
                let f = StandardBloomFilter::<u64>::new(capacity, 0.01).unwrap();
                for i in 0..pre_fill as u64 {
                    f.insert(&i);
                }
                let mut i = pre_fill as u64;
                b.iter(|| {
                    f.insert(black_box(&i));
                    i = i.wrapping_add(1);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("contains_at_pct", fill_pct),
            &pre_fill,
            |b, &pre_fill| {
                let f = StandardBloomFilter::<u64>::new(capacity, 0.01).unwrap();
                for i in 0..pre_fill as u64 {
                    f.insert(&i);
                }
                let len = pre_fill.max(1) as u64;
                let mut i = 0u64;
                b.iter(|| {
                    let r = f.contains(black_box(&(i % len)));
                    i = i.wrapping_add(1);
                    black_box(r)
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. Access Patterns — Sequential vs Random vs Zipf
//    Same filter, same items — only query ordering differs.
//    Isolates CPU prefetcher effectiveness and cache-line reuse.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/access_patterns");
    group.throughput(Throughput::Elements(1_000));

    // All three patterns query only keys in [0, VOCAB) — all inserted.
    // Only the access ORDER differs.
    const VOCAB: u64   = 100_000;
    const POOL:  usize = 2_000_000; // 16 MB — exceeds typical 8–32 MB L3

    let filter = StandardBloomFilter::<u64>::new(1_000_000, 0.01).unwrap();
    for i in 0..VOCAB {
        filter.insert(&i);
    }

    // Sequential: monotone key stream, predictable prefetcher behaviour.
    let mut seq_idx = 0u64;
    group.bench_function("sequential/1k", |b| {
        b.iter(|| {
            for _ in 0..1_000 {
                black_box(filter.contains(black_box(&(seq_idx % VOCAB))));
                seq_idx = seq_idx.wrapping_add(1);
            }
        })
    });

    // Uniform random: maximally cache-hostile, no spatial locality.
    // Keys are drawn from [0, VOCAB) so membership is identical to sequential.
    let random_pool: Vec<u64> = {
        let mut r = rng();
        (0..POOL).map(|_| r.gen_range(0..VOCAB)).collect()
    };
    let mut rand_idx = 0usize;
    group.bench_function("uniform_random/1k", |b| {
        b.iter(|| {
            for _ in 0..1_000 {
                black_box(filter.contains(black_box(&random_pool[rand_idx % POOL])));
                rand_idx = rand_idx.wrapping_add(1);
            }
        })
    });

    // Zipf s=1.1: hot-set reuse, models real production traffic.
    // Mapped into [0, VOCAB) so all keys are guaranteed members.
    let zipf_pool: Vec<u64> = zipf_u64s(POOL, VOCAB)
        .into_iter()
        .map(|v| v % VOCAB)
        .collect();
    let mut zipf_idx = 0usize;
    group.bench_function("zipf_s1.1/1k", |b| {
        b.iter(|| {
            for _ in 0..1_000 {
                black_box(filter.contains(black_box(&zipf_pool[zipf_idx % POOL])));
                zipf_idx = zipf_idx.wrapping_add(1);
            }
        })
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. Concurrent Insert Scaling
//     Measures: throughput vs thread count. Ideal: near-linear (lock-free
//     AtomicU64 fetch_or). Deviation reveals CAS contention or false sharing.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_concurrent_insert_scaling(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("standard/concurrent/insert_scaling");
    group.sample_size(20);

    for n_threads in [1usize, 2, 4, 8, 16] {
        const ITEMS_PER_THREAD: usize = 10_000;
        group.throughput(Throughput::Elements((n_threads * ITEMS_PER_THREAD) as u64));

        group.bench_with_input(
            BenchmarkId::new("threads", n_threads),
            &n_threads,
            |b, &n_threads| {
                b.iter_batched(
                    || {
                        // Setup (excluded): allocate fresh filter.
                        Arc::new(
                            StandardBloomFilter::<u64>::new(
                                n_threads * ITEMS_PER_THREAD * 2,
                                0.01,
                            )
                            .unwrap(),
                        )
                    },
                    |filter| {
                        let handles: Vec<_> = (0..n_threads)
                            .map(|tid| {
                                let f = Arc::clone(&filter);
                                thread::spawn(move || {
                                    let base = (tid * ITEMS_PER_THREAD) as u64;
                                    for i in 0..ITEMS_PER_THREAD as u64 {
                                        f.insert_concurrent(&(base + i));
                                    }
                                })
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.count_set_bits())
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 11. Concurrent Mixed Read/Write
//     Measures: latency under various reader:writer ratios.
//     Real-world: most systems are read-heavy; r8w1 is most representative.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_concurrent_mixed(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("standard/concurrent/mixed_rw");
    group.sample_size(20);

    for (r, w) in [(1usize, 1usize), (3, 1), (7, 1), (15, 1), (4, 4)] {
        let label = format!("{}r{}w", r, w);
        group.throughput(Throughput::Elements(((r + w) * 5_000) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &(r, w),
            |b, &(r, w)| {
                b.iter_batched(
                    || {
                        // Setup (excluded): build and pre-fill filter.
                        let f = Arc::new(
                            StandardBloomFilter::<u64>::new(500_000, 0.01).unwrap(),
                        );
                        for i in 0..100_000u64 {
                            f.insert(&i);
                        }
                        f
                    },
                    |filter| {
                        let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();

                        for wid in 0..w {
                            let f = Arc::clone(&filter);
                            handles.push(thread::spawn(move || {
                                let base = 100_000u64 + wid as u64 * 5_000;
                                for i in 0..5_000u64 {
                                    f.insert_concurrent(&(base + i));
                                }
                            }));
                        }
                        for _ in 0..r {
                            let f = Arc::clone(&filter);
                            handles.push(thread::spawn(move || {
                                for i in 0..5_000u64 {
                                    black_box(f.contains_concurrent(&(i % 100_000)));
                                }
                            }));
                        }

                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(filter.fill_rate())
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 12. Concurrent Batch Operations
//     Measures: insert_batch_concurrent and contains_batch_concurrent on Arc.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_concurrent_batch(c: &mut Criterion) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("standard/concurrent/batch");
    group.sample_size(20);

    for batch in [100usize, 1_000, 10_000] {
        let items: Vec<u64> = (0..batch as u64).collect();
        group.throughput(Throughput::Elements((4 * batch) as u64));

        // Variant A: 4 threads each insert the batch concurrently.
        group.bench_with_input(
            BenchmarkId::new("insert_batch_4t", batch),
            &batch,
            |b, _| {
                b.iter_batched(
                    || Arc::new(StandardBloomFilter::<u64>::new(batch * 10, 0.01).unwrap()),
                    |f| {
                        let handles: Vec<_> = (0..4)
                            .map(|_| {
                                let ff = Arc::clone(&f);
                                let it = items.clone();
                                thread::spawn(move || ff.insert_batch_concurrent(&it))
                            })
                            .collect();
                        for h in handles {
                            h.join().unwrap();
                        }
                        black_box(f.count_set_bits())
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        // Variant B: read-only — build the filter once, outside b.iter.
        let populated = {
            let f = Arc::new(StandardBloomFilter::<u64>::new(batch * 10, 0.01).unwrap());
            f.insert_batch(&items);
            f
        };

        group.bench_with_input(
            BenchmarkId::new("contains_batch_concurrent_4t", batch),
            &batch,
            |b, _| {
                b.iter(|| {
                    let handles: Vec<_> = (0..4)
                        .map(|_| {
                            let ff = Arc::clone(&populated);
                            let it = items.clone();
                            thread::spawn(move || black_box(ff.contains_batch_concurrent(&it)))
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

// ═══════════════════════════════════════════════════════════════════════════════
// 13. Set Algebra — Union & Intersection
//     Constructive (inherent, returns new Self) and in-place (UFCS trait).
//     Throughput expressed in bytes of bit-array processed.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_set_algebra(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/set_algebra");

    for &n in &[10_000usize, 100_000, 1_000_000] {
        // Pre-compute filter pair outside the iter loop.
        let f1 = StandardBloomFilter::<u64>::new(n, 0.01).unwrap();
        let f2 = StandardBloomFilter::<u64>::new(n, 0.01).unwrap();
        for i in 0..(n / 2) as u64 {
            f1.insert(&i);
            f2.insert(&(i + n as u64 / 2));
        }
        group.throughput(Throughput::Bytes(f1.memory_usage() as u64));

        group.bench_with_input(
            BenchmarkId::new("union_constructive", n),
            &n,
            |b, _| b.iter(|| black_box(f1.union(black_box(&f2)).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("intersect_constructive", n),
            &n,
            |b, _| b.iter(|| black_box(f1.intersect(black_box(&f2)).unwrap())),
        );

        // In-place union: clone each iteration to avoid accumulating state.
        group.bench_with_input(
            BenchmarkId::new("union_inplace_ufcs", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut target = f1.clone();
                    MergeableBloomFilter::union(&mut target, &f2).unwrap();
                    black_box(target.size())
                });
            },
        );

        // In-place intersect
        group.bench_with_input(
            BenchmarkId::new("intersect_inplace_ufcs", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut target = f1.clone();
                    MergeableBloomFilter::intersect(&mut target, &f2).unwrap();
                    black_box(target.size())
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 14. Clone
//     Measures: deep-copy cost of the underlying BitVec at multiple sizes.
//     Real-world: filter snapshotting for distributed broadcast, analytics forks.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/clone");

    for &n in &[10_000usize, 100_000, 1_000_000] {
        let filter = StandardBloomFilter::<u64>::new(n, 0.01).unwrap();
        for i in 0..(n / 2) as u64 {
            filter.insert(&i);
        }
        group.throughput(Throughput::Bytes(filter.memory_usage() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            &n,
            |b, _| b.iter(|| black_box(filter.clone())),
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 15. Clear
//     Measures: memset-equivalent cost of zeroing the BitVec.
//     Real-world: periodic filter rotation in long-running services.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_clear(c: &mut Criterion) {
    use criterion::BatchSize;
    let mut group = c.benchmark_group("standard/clear");
    for n in [10_000usize, 100_000, 1_000_000] {
        group.throughput(Throughput::Bytes(n as u64 / 8));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut f = StandardBloomFilter::<u64>::new(n, 0.01).unwrap();
                    for i in 0..n as u64 / 2 { f.insert(&i); }
                    f
                },
                |mut f| { f.clear(); black_box(f.is_empty()) },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 16. Statistics & Accessors
//     Measures individual diagnostic method overhead.
//     Real-world: Prometheus scrape handler, alerting sidecar, capacity planner.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/statistics");

    let n = 100_000usize;
    let filter = StandardBloomFilter::<u64>::new(n, 0.01).unwrap();
    for i in 0..(n / 2) as u64 {
        filter.insert(&i);
    }

    group.bench_function("fill_rate",            |b| b.iter(|| black_box(filter.fill_rate())));
    group.bench_function("count_set_bits",        |b| b.iter(|| black_box(filter.count_set_bits())));
    group.bench_function("estimate_fpr",          |b| b.iter(|| black_box(filter.estimate_fpr())));
    group.bench_function("estimate_cardinality",  |b| b.iter(|| black_box(filter.estimate_cardinality())));
    group.bench_function("memory_usage",          |b| b.iter(|| black_box(filter.memory_usage())));
    group.bench_function("is_empty",              |b| b.iter(|| black_box(filter.is_empty())));
    group.bench_function("is_full",               |b| b.iter(|| black_box(filter.is_full())));
    group.bench_function("hash_strategy",         |b| b.iter(|| black_box(filter.hash_strategy())));
    group.bench_function("hasher_name",           |b| b.iter(|| black_box(filter.hasher_name())));
    group.bench_function("size",                  |b| b.iter(|| black_box(filter.size())));
    group.bench_function("hash_count",            |b| b.iter(|| black_box(filter.hash_count())));
    group.bench_function("raw_bits",              |b| b.iter(|| black_box(filter.raw_bits())));

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 17. Health Check Classification
//     Measures health_check() at each state (Healthy / Degraded / Critical).
//     Exercises fill_rate + estimate_fpr + estimate_cardinality in one call.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_health_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/health_check");

    let capacity = 100_000usize;

    for &(fill_pct, label) in &[
        (20u64, "fill_20pct"),
        (60u64, "fill_60pct"),
        (85u64, "fill_85pct"),
    ] {
        let filter = StandardBloomFilter::<u64>::new(capacity, 0.01).unwrap();
        let n = (capacity as u64 * fill_pct / 100) as usize;
        for i in 0..n as u64 {
            filter.insert(&i);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &fill_pct,
            |b, _| b.iter(|| black_box(filter.health_check())),
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 18. Hasher Variants
//     Default StdHasher vs seeded instances. Measures seed-mixing overhead and
//     verifies that seeded construction is cost-equivalent to default.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_hasher_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/hasher_variants");
    group.throughput(Throughput::Elements(1));

    let n = 100_000usize;

    // Default seed (compile-time constant — fastest path)
    {
        let f = StandardBloomFilter::<u64>::new(n, 0.01).unwrap();
        let mut i = 0u64;
        group.bench_function("default_insert", |b| {
            b.iter(|| {
                f.insert(black_box(&i));
                i = i.wrapping_add(1);
            });
        });
        let mut j = 0u64;
        group.bench_function("default_contains", |b| {
            b.iter(|| {
                let r = f.contains(black_box(&(j % 50_000)));
                j = j.wrapping_add(1);
                black_box(r)
            });
        });
    }

    // Seed = 42
    {
        let f = StandardBloomFilter::<u64, _>::with_hasher(n, 0.01, StdHasher::with_seed(42))
            .unwrap();
        let mut i = 0u64;
        group.bench_function("seed_42_insert", |b| {
            b.iter(|| {
                f.insert(black_box(&i));
                i = i.wrapping_add(1);
            });
        });
    }

    // Seed = 0xDEAD_BEEF
    {
        let f = StandardBloomFilter::<u64, _>::with_hasher(
            n,
            0.01,
            StdHasher::with_seed(0xDEAD_BEEF),
        )
        .unwrap();
        let mut i = 0u64;
        group.bench_function("seed_deadbeef_insert", |b| {
            b.iter(|| {
                f.insert(black_box(&i));
                i = i.wrapping_add(1);
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 19. False Positive Rate Measurement Under Load
//     Quantifies actual FPR at 50% and 100% fill vs theoretical target.
//     Not a latency bench — measures correctness degradation characteristic.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_fpr_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/fpr_measurement");
    group.sample_size(10); // expensive: 10k probe loop per sample

    for &target_fpr in &[0.1f64, 0.01, 0.001] {
        let capacity = 50_000usize;
        let probe_start = capacity * 2; // never inserted — guaranteed negatives

        for &fill in &[50u64, 100u64] {
            let filter = StandardBloomFilter::<u64>::new(capacity, target_fpr).unwrap();
            let n = (capacity as u64 * fill / 100) as usize;
            for i in 0..n as u64 {
                filter.insert(&i);
            }

            let label = format!("fpr{:.3}_fill{}pct", target_fpr, fill);
            group.bench_with_input(
                BenchmarkId::from_parameter(label),
                &probe_start,
                |b, &probe_start| {
                    b.iter(|| {
                        let mut fp = 0u32;
                        for i in probe_start..(probe_start + 10_000) {
                            if filter.contains(&(i as u64)) {
                                fp += 1;
                            }
                        }
                        black_box(fp)
                    });
                },
            );
        }
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// 20. Real-World Simulations
// ═══════════════════════════════════════════════════════════════════════════════

/// Scenario A — Web crawler URL deduplication.
/// The hot path of a distributed crawler: check-then-insert on a 1M-URL
/// frontier. String hashing through HashBytes inline path (48-char URLs).
fn bench_sim_url_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/sim/url_dedup");
    group.throughput(Throughput::Elements(1));

    let urls = rand_strings(500_000, 48);

    group.bench_function("check_then_insert", |b| {
        let filter = StandardBloomFilter::<String>::new(1_000_000, 0.005).unwrap();
        let mut idx = 0usize;
        b.iter(|| {
            let url = &urls[idx % urls.len()];
            if !filter.contains(black_box(url)) {
                filter.insert(url);
            }
            idx = idx.wrapping_add(1);
            black_box(idx)
        });
    });

    group.finish();
}

/// Scenario B — API rate-limit guard.
/// Key = (user_id: u32, time_window: u32). Pre-check before Redis.
/// ~10 k active users × 100 time windows = 1 M unique keys.
fn bench_sim_rate_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/sim/rate_limit");
    group.throughput(Throughput::Elements(1));

    let filter = StandardBloomFilter::<(u32, u32)>::new(500_000, 0.01).unwrap();
    let mut req = 0u64;

    group.bench_function("guard_check_insert", |b| {
        b.iter(|| {
            let user    = (req % 10_000) as u32;
            let window  = (req / 10_000 % 100) as u32;
            let key = (user, window);
            let blocked = filter.contains(black_box(&key));
            if !blocked {
                filter.insert(&key);
            }
            req = req.wrapping_add(1);
            black_box(blocked)
        });
    });

    group.finish();
}

/// Scenario C — Cache admission filter (second-chance / TinyLFU style).
/// First access: miss → insert (seen-once). Second access: hit → admit to L1.
/// Zipf traffic over 50k keys models hot-set reuse accurately.
fn bench_sim_cache_admission(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/sim/cache_admission");
    group.throughput(Throughput::Elements(1));

    let filter = StandardBloomFilter::<u64>::new(200_000, 0.01).unwrap();
    let traffic = zipf_u64s(1_000_000, 50_000);
    let mut idx = 0usize;

    group.bench_function("second_chance", |b| {
        b.iter(|| {
            let key = traffic[idx % traffic.len()];
            let admit = filter.contains(black_box(&key));
            if !admit {
                filter.insert(&key);
            }
            idx = idx.wrapping_add(1);
            black_box(admit)
        });
    });

    group.finish();
}

/// Scenario D — Distributed log deduplication.
/// 4 producer threads ingest unique log-line hashes concurrently.
/// 1 consumer thread reads already-committed keys (no false negatives expected).
fn bench_sim_log_dedup_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/sim/log_dedup_4p1c");
    group.sample_size(20);
    group.throughput(Throughput::Elements(5 * 10_000));

    group.bench_function("4_producers_1_consumer", |b| {
        b.iter_batched(
            // Setup (excluded): allocate a fresh filter per sample.
            || Arc::new(StandardBloomFilter::<u64>::new(500_000, 0.01).unwrap()),
            |filter| {
                // Producers: keys 0..40_000, partitioned by thread id.
                let producers: Vec<_> = (0..4u64)
                    .map(|tid| {
                        let f = Arc::clone(&filter);
                        thread::spawn(move || {
                            let base = tid * 10_000;
                            for i in 0..10_000u64 {
                                f.insert_concurrent(&(base + i));
                            }
                        })
                    })
                    .collect();

                // Consumer: reads keys 0..10_000 — overlaps with producer 0.
                // Concurrent with producers; hit count varies by scheduling.
                let f = Arc::clone(&filter);
                let consumer = thread::spawn(move || {
                    let mut hits = 0u64;
                    for i in 0..10_000u64 {
                        if f.contains_concurrent(&i) {
                            hits += 1;
                        }
                    }
                    hits
                });

                for h in producers {
                    h.join().unwrap();
                }
                black_box(consumer.join().unwrap());
                black_box(filter.count_set_bits())
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Scenario E — Nightly ETL pipeline.
/// Batch-insert yesterday's keys; batch-probe today's keys for overlap.
/// Pure throughput benchmark: measures insert_batch + contains_batch pipeline.
fn bench_sim_etl_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard/sim/etl_pipeline");

    for &batch in &[10_000usize, 100_000] {
        let yesterday: Vec<u64> = (0..batch as u64).collect();
        // 50% overlap with yesterday — realistic dedup scenario.
        let today: Vec<u64> = (batch as u64 / 2..(batch as u64 / 2 + batch as u64)).collect();

        group.throughput(Throughput::Elements(batch as u64 * 2));
        group.bench_with_input(
            BenchmarkId::new("insert_then_probe", batch),
            &batch,
            |b, &batch| {
                b.iter_batched(
                    // Setup (excluded): allocate filter sized for yesterday + headroom.
                    || StandardBloomFilter::<u64>::new(batch * 3, 0.01).unwrap(),
                    |f| {
                        f.insert_batch(black_box(&yesterday));
                        let results = f.contains_batch(black_box(&today));
                        black_box(results)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

// ─── Criterion groups ─────────────────────────────────────────────────────────

criterion_group! {
    name = g_construction;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(8))
        .sample_size(50);
    targets = bench_construction
}

criterion_group! {
    name = g_single_ops;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(8))
        .sample_size(200);
    targets =
        bench_insert_scale,
        bench_contains_hit_miss,
        bench_key_types,
        bench_fpr_targets,
        bench_saturation,
        bench_access_patterns,
        bench_hasher_variants
}

criterion_group! {
    name = g_batch_ops;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(8))
        .sample_size(100);
    targets =
        bench_batch_insert,
        bench_batch_contains
}

criterion_group! {
    name = g_concurrent;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .sample_size(20);
    targets =
        bench_concurrent_insert_scaling,
        bench_concurrent_mixed,
        bench_concurrent_batch
}

criterion_group! {
    name = g_set_algebra;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(8))
        .sample_size(50);
    targets =
        bench_set_algebra,
        bench_clone,
        bench_clear
}

criterion_group! {
    name = g_observability;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(200);
    targets =
        bench_statistics,
        bench_health_check
}

criterion_group! {
    name = g_correctness;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(8))
        .sample_size(10);
    targets =
        bench_fpr_measurement
}

criterion_group! {
    name = g_simulations;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .sample_size(30);
    targets =
        bench_sim_url_dedup,
        bench_sim_rate_limit,
        bench_sim_cache_admission,
        bench_sim_log_dedup_concurrent,
        bench_sim_etl_pipeline
}

criterion_main!(
    g_construction,
    g_single_ops,
    g_batch_ops,
    g_concurrent,
    g_set_algebra,
    g_observability,
    g_correctness,
    g_simulations
);
