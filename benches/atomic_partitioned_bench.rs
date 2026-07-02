//! Full-scale Criterion benchmark suite for `AtomicPartitionedBloomFilter`.
//!
//! Covers construction cost, single-threaded insert/query throughput,
//! lock-free concurrent scaling (1..16 threads) and oversubscription
//! (32/64 threads), cache-line contention, a realistic mixed read/write
//! workload, whole-filter introspection cost (`saturation`,
//! `estimate_count`, `count_set_bits`, `clone`), extreme scale (10M
//! items), behavior at 5x design capacity, cache-line alignment choice,
//! and the cost of a tighter target FPR (larger `k`).
//!
//! 14 benchmark groups, ~39 individual measured cases in total.
//!
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench atomic_partitioned_bench --features concurrent
//! ```
//!
//! For the concurrency benchmarks to actually demonstrate scaling, run
//! on a machine with >= 16 physical cores, or interpret the higher
//! thread-count results as oversubscription behavior instead.

use bloomcraft::core::{BloomFilter, ConcurrentBloomFilter};
use bloomcraft::filters::AtomicPartitionedBloomFilter;
use bloomcraft::hash::StdHasher;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

const TARGET_FPR: f64 = 0.01;

// Thread counts swept across the two headline concurrency benchmarks.
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];

// Secondary mixed-workload sweep.
const MIXED_THREAD_COUNTS: &[usize] = &[2, 8, 16];

// Oversubscription sweep — deliberately exceeds typical physical core
// counts to see how the lock-free design degrades (or doesn't) under
// scheduler contention rather than memory contention.
const OVERSUBSCRIPTION_THREAD_COUNTS: &[usize] = &[32, 64];

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn build_filter(n: usize) -> AtomicPartitionedBloomFilter<u64> {
    AtomicPartitionedBloomFilter::<u64>::new(n, TARGET_FPR).unwrap()
}

fn build_prefilled(n: usize) -> AtomicPartitionedBloomFilter<u64> {
    let filter = build_filter(n);
    for i in 0..n as u64 {
        filter.insert_concurrent(&i);
    }
    filter
}

/// Spawns `threads` workers, releases them simultaneously via a barrier,
/// and times only the work that happens after release. Used for every
/// concurrent benchmark so setup/teardown never pollutes the measurement.
fn timed_concurrent<F>(threads: usize, work: F) -> Duration
where
    F: Fn(usize) + Send + Sync + 'static,
{
    let work = Arc::new(work);
    let barrier = Arc::new(Barrier::new(threads + 1));

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let work = Arc::clone(&work);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                work(tid);
            })
        })
        .collect();

    barrier.wait();
    let start = Instant::now();
    for h in handles {
        h.join().unwrap();
    }
    start.elapsed()
}

// ---------------------------------------------------------------------
// 1. Construction cost — allocation + zeroing scales with bit count
// ---------------------------------------------------------------------

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    for &n in &[1_000usize, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("new", n), &n, |b, &n| {
            b.iter(|| black_box(AtomicPartitionedBloomFilter::<u64>::new(n, TARGET_FPR).unwrap()));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------
// 2. Single-threaded insert throughput
// ---------------------------------------------------------------------

fn bench_sequential_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_insert");
    for &n in &[1_000usize, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("insert_concurrent", n), &n, |b, &n| {
            b.iter_batched(
                || build_filter(n),
                |filter| {
                    for i in 0..n as u64 {
                        filter.insert_concurrent(black_box(&i));
                    }
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------
// 3. Single-threaded query throughput — pure hit vs pure miss
// ---------------------------------------------------------------------

fn bench_sequential_contains(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let filter = build_prefilled(N);

    let mut group = c.benchmark_group("sequential_contains");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("hit", |b| {
        b.iter(|| {
            for i in 0..N as u64 {
                black_box(filter.contains(&i));
            }
        });
    });

    group.bench_function("miss", |b| {
        b.iter(|| {
            for i in (N as u64)..(2 * N as u64) {
                black_box(filter.contains(&i));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------
// 4. Concurrent insert scaling — disjoint key shards per thread,
//    measures wait-free insert throughput as core count grows.
// ---------------------------------------------------------------------

fn bench_concurrent_insert_scaling(c: &mut Criterion) {
    const ITEMS_PER_THREAD: usize = 100_000;

    let mut group = c.benchmark_group("concurrent_insert_scaling");
    for &threads in THREAD_COUNTS {
        let total_items = (ITEMS_PER_THREAD * threads) as u64;
        group.throughput(Throughput::Elements(total_items));
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let filter = Arc::new(build_filter(ITEMS_PER_THREAD * threads));
                        let elapsed = timed_concurrent(threads, move |tid| {
                            let base = (tid * ITEMS_PER_THREAD) as u64;
                            for i in 0..ITEMS_PER_THREAD as u64 {
                                filter.insert_concurrent(black_box(&(base + i)));
                            }
                        });
                        total += elapsed;
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// 5. Concurrent query scaling — every thread reads the full key space
//    against a pre-filled filter (pure Relaxed atomic loads).
// ---------------------------------------------------------------------

fn bench_concurrent_contains_scaling(c: &mut Criterion) {
    const N: usize = 200_000;
    let filter = Arc::new(build_prefilled(N));

    let mut group = c.benchmark_group("concurrent_contains_scaling");
    for &threads in THREAD_COUNTS {
        group.throughput(Throughput::Elements(N as u64));
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let filter = Arc::clone(&filter);
                        let elapsed = timed_concurrent(threads, move |tid| {
                            let chunk = N / threads;
                            let start = tid * chunk;
                            let end = if tid == threads - 1 { N } else { start + chunk };
                            for i in start..end {
                                black_box(filter.contains(&(i as u64)));
                            }
                        });
                        total += elapsed;
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// 6. Cache-line contention: every thread hammering the SAME keys
//    (worst case — all threads CAS-loop on the same words) vs disjoint
//    keys (best case — no shared cache lines). Same thread count (8),
//    isolates contention cost from raw thread-count scaling.
// ---------------------------------------------------------------------

fn bench_contention_comparison(c: &mut Criterion) {
    const THREADS: usize = 8;
    const ITEMS: usize = 50_000;

    let mut group = c.benchmark_group("contention_comparison");
    group.throughput(Throughput::Elements((ITEMS * THREADS) as u64));

    group.bench_function("disjoint_keys_low_contention", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let filter = Arc::new(build_filter(ITEMS * THREADS));
                let elapsed = timed_concurrent(THREADS, move |tid| {
                    let base = (tid * ITEMS) as u64;
                    for i in 0..ITEMS as u64 {
                        filter.insert_concurrent(black_box(&(base + i)));
                    }
                });
                total += elapsed;
            }
            total
        });
    });

    group.bench_function("shared_keys_high_contention", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let filter = Arc::new(build_filter(ITEMS));
                let elapsed = timed_concurrent(THREADS, move |_tid| {
                    // All threads race to set the *same* bits in the
                    // *same* partitions — maximal fetch_or contention
                    // on a small number of cache lines.
                    for i in 0..ITEMS as u64 {
                        filter.insert_concurrent(black_box(&i));
                    }
                });
                total += elapsed;
            }
            total
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------
// 7. Realistic mixed workload — 90% reads / 10% writes, scaled across
//    thread counts. Models a cache/dedup-style usage pattern.
// ---------------------------------------------------------------------

fn bench_mixed_workload(c: &mut Criterion) {
    const OPS_PER_THREAD: usize = 50_000;
    const PREFILL: usize = 100_000;

    let mut group = c.benchmark_group("mixed_workload_90read_10write");
    for &threads in MIXED_THREAD_COUNTS {
        group.throughput(Throughput::Elements((OPS_PER_THREAD * threads) as u64));
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let filter = Arc::new(build_prefilled(PREFILL));
                        let elapsed = timed_concurrent(threads, move |tid| {
                            let mut next_write = (tid * 1_000_000) as u64;
                            for op in 0..OPS_PER_THREAD {
                                if op % 10 == 0 {
                                    // 10% writes: insert a fresh, thread-unique key.
                                    filter.insert_concurrent(black_box(&next_write));
                                    next_write += 1;
                                } else {
                                    // 90% reads against the pre-existing key space.
                                    let key = (op % PREFILL) as u64;
                                    black_box(filter.contains(&key));
                                }
                            }
                        });
                        total += elapsed;
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// 8. Whole-filter scan cost — saturation / estimate_count /
//    count_set_bits / clone all walk every AtomicU64 word,
//    O(k * partition_size) regardless of item_count.
// ---------------------------------------------------------------------

fn bench_whole_filter_scans(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let filter = build_prefilled(N);

    let mut group = c.benchmark_group("whole_filter_scans");

    group.bench_function("saturation", |b| {
        b.iter(|| black_box(filter.saturation()));
    });

    group.bench_function("estimate_count", |b| {
        b.iter(|| black_box(filter.estimate_count()));
    });

    group.bench_function("count_set_bits", |b| {
        b.iter(|| black_box(filter.count_set_bits()));
    });

    group.bench_function("clone_1m_items", |b| {
        b.iter(|| black_box(filter.clone()));
    });

    group.finish();
}

// ---------------------------------------------------------------------
// 9. Hashing overhead by key type — String allocation/hashing vs
//    fixed-width u64, both through the same insert/contains path.
// ---------------------------------------------------------------------

fn bench_key_type_overhead(c: &mut Criterion) {
    const N: usize = 200_000;
    let strings: Vec<String> = (0..N).map(|i| format!("benchmark-key-{i:08}")).collect();

    let mut group = c.benchmark_group("key_type_overhead");
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("insert_u64", |b| {
        b.iter_batched(
            || build_filter(N),
            |filter| {
                for i in 0..N as u64 {
                    filter.insert_concurrent(black_box(&i));
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.bench_function("insert_string", |b| {
        b.iter_batched(
            || AtomicPartitionedBloomFilter::<String>::new(N, TARGET_FPR).unwrap(),
            |filter| {
                for s in &strings {
                    filter.insert_concurrent(black_box(s));
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------
// 10. Extreme scale — 10M items. Pushes allocation size, cache footprint,
//     and per-op cost well past anything the other benchmarks exercise.
// ---------------------------------------------------------------------

fn bench_extreme_scale(c: &mut Criterion) {
    const N: usize = 10_000_000;

    let mut group = c.benchmark_group("extreme_scale_10m");
    group.sample_size(10); // each iteration is expensive; keep the run time sane
    group.throughput(Throughput::Elements(N as u64));

    group.bench_function("insert_concurrent", |b| {
        b.iter_batched(
            || build_filter(N),
            |filter| {
                for i in 0..N as u64 {
                    filter.insert_concurrent(black_box(&i));
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });

    let filter = build_prefilled(N);
    group.bench_function("contains_hit", |b| {
        b.iter(|| {
            for i in (0..N as u64).step_by(7) {
                black_box(filter.contains(&i));
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------
// 11. Oversubscription — 32 and 64 threads against typically <= 16-32
//     physical cores, to see how throughput behaves once the OS
//     scheduler, not memory contention, is the bottleneck.
// ---------------------------------------------------------------------

fn bench_oversubscription(c: &mut Criterion) {
    const ITEMS_PER_THREAD: usize = 25_000;

    let mut group = c.benchmark_group("oversubscription_insert");
    for &threads in OVERSUBSCRIPTION_THREAD_COUNTS {
        let total_items = (ITEMS_PER_THREAD * threads) as u64;
        group.throughput(Throughput::Elements(total_items));
        group.bench_with_input(
            BenchmarkId::new("threads", threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let filter = Arc::new(build_filter(ITEMS_PER_THREAD * threads));
                        let elapsed = timed_concurrent(threads, move |tid| {
                            let base = (tid * ITEMS_PER_THREAD) as u64;
                            for i in 0..ITEMS_PER_THREAD as u64 {
                                filter.insert_concurrent(black_box(&(base + i)));
                            }
                        });
                        total += elapsed;
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// 12. Near-full saturation — `contains` cost (and false-positive rate)
//     when the filter is loaded far past its design capacity, vs a
//     lightly loaded filter of the same size. Bit density affects
//     branch prediction on the early-exit in `contains`.
// ---------------------------------------------------------------------

fn bench_saturation_degradation(c: &mut Criterion) {
    const CAPACITY: usize = 100_000;
    const QUERY_COUNT: usize = 200_000;

    // Lightly loaded: ~10% of design capacity.
    let light = build_filter(CAPACITY);
    for i in 0..(CAPACITY / 10) as u64 {
        light.insert_concurrent(&i);
    }

    // Overloaded: 5x design capacity, pushed well past the target FPR.
    let heavy = build_filter(CAPACITY);
    for i in 0..(CAPACITY * 5) as u64 {
        heavy.insert_concurrent(&i);
    }

    let mut group = c.benchmark_group("saturation_degradation");
    group.throughput(Throughput::Elements(QUERY_COUNT as u64));

    group.bench_function("contains_at_10pct_load", |b| {
        b.iter(|| {
            for i in 0..QUERY_COUNT as u64 {
                black_box(light.contains(&i));
            }
        });
    });

    group.bench_function("contains_at_500pct_load", |b| {
        b.iter(|| {
            for i in 0..QUERY_COUNT as u64 {
                black_box(heavy.contains(&i));
            }
        });
    });

    group.finish();

    eprintln!(
        "[saturation_degradation] light saturation={:.3} fpr~{:.4} | heavy saturation={:.3} fpr~{:.4}",
        light.saturation(),
        light.estimated_fpr(),
        heavy.saturation(),
        heavy.estimated_fpr(),
    );
}

// ---------------------------------------------------------------------
// 13. Cache-line alignment impact — this implementation pads every
//     partition to a configurable byte alignment specifically to avoid
//     false sharing. Compare the default 64B (one cache line) against
//     256B and 4096B (page-sized) alignment.
// ---------------------------------------------------------------------

fn bench_alignment_impact(c: &mut Criterion) {
    const N: usize = 500_000;
    const ALIGNMENTS: &[usize] = &[64, 256, 4096];

    let mut group = c.benchmark_group("alignment_impact_insert");
    group.throughput(Throughput::Elements(N as u64));

    for &alignment in ALIGNMENTS {
        group.bench_with_input(
            BenchmarkId::new("alignment_bytes", alignment),
            &alignment,
            |b, &alignment| {
                b.iter_batched(
                    || {
                        AtomicPartitionedBloomFilter::<u64>::with_hasher_and_alignment(
                            N,
                            TARGET_FPR,
                            StdHasher::new(),
                            alignment,
                        )
                        .unwrap()
                    },
                    |filter| {
                        for i in 0..N as u64 {
                            filter.insert_concurrent(black_box(&i));
                        }
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------
// 14. Partition count (k) impact — a tighter target FPR raises the
//     optimal hash count `k`, meaning more partitions touched per
//     insert/query. Compares loose (k is small) vs tight (k is large)
//     FPR targets at the same item count.
// ---------------------------------------------------------------------

fn bench_partition_count_impact(c: &mut Criterion) {
    const N: usize = 500_000;
    const FPRS: &[f64] = &[0.1, 0.01, 0.0001];

    let mut group = c.benchmark_group("partition_count_impact_insert");
    group.throughput(Throughput::Elements(N as u64));

    for &fpr in FPRS {
        group.bench_with_input(BenchmarkId::new("target_fpr", fpr), &fpr, |b, &fpr| {
            b.iter_batched(
                || AtomicPartitionedBloomFilter::<u64>::new(N, fpr).unwrap(),
                |filter| {
                    for i in 0..N as u64 {
                        filter.insert_concurrent(black_box(&i));
                    }
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_sequential_insert,
    bench_sequential_contains,
    bench_concurrent_insert_scaling,
    bench_concurrent_contains_scaling,
    bench_contention_comparison,
    bench_mixed_workload,
    bench_whole_filter_scans,
    bench_key_type_overhead,
    bench_extreme_scale,
    bench_oversubscription,
    bench_saturation_degradation,
    bench_alignment_impact,
    bench_partition_count_impact,
);
criterion_main!(benches);