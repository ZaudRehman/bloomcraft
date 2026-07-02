//! Criterion benchmarks for [`CountingBloomFilter`].
//!
//! # What is covered
//!
//! | Group | What is measured |
//! |-------|-----------------|
//! | `insert` | Throughput of single inserts across all three counter sizes |
//! | `insert_fast` | Lock-free fast-path insert (no item-count tracking) |
//! | `insert_batch` | Batch insert throughput at realistic cardinalities |
//! | `contains_hit` | Query latency when item is present (true positive) |
//! | `contains_miss` | Query latency when item is absent (true negative) |
//! | `contains_batch` | Bulk membership queries |
//! | `contains_all` | Short-circuit all-present predicate |
//! | `contains_any` | Short-circuit any-present predicate |
//! | `delete` | Single delete with two-phase safety check |
//! | `delete_batch` | Bulk delete throughput |
//! | `delete_all_or_none` | Transactional batch delete |
//! | `insert_delete_cycle` | Sustained churn: insert N, delete N, repeat |
//! | `high_load_factor` | Insert at 2× design capacity (FPR degradation zone) |
//! | `overflow_pressure_4bit` | Saturate 4-bit counters; measure insert behaviour post-saturation |
//! | `health_metrics` | Cost of computing the full `HealthMetrics` snapshot |
//! | `hotspots` | Top-N hot-counter scan |
//! | `counter_distribution` | Full counter distribution statistics |
//! | `counter_histogram` | Full histogram allocation |
//! | `clear` | Zero-out a pre-filled filter |
//! | `memory_overhead_comparison` | Side-by-side memory usage across counter sizes |
//!
//! # Running
//!
//! cargo bench --bench counting
//! # HTML reports
//! cargo bench --bench counting -- --output-format html
//! # Single group
//! cargo bench --bench counting -- insert
//!

use bloomcraft::filters::{CounterSize, CountingBloomFilter};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Design capacity used for most benchmarks.
const CAPACITY: usize = 100_000;

/// Target FPR that drives optimal sizing.
const FPR: f64 = 0.01;

/// Number of items in the hot working set (fits comfortably in L2/L3).
const HOT_SET: usize = 10_000;

/// Items that are never inserted, used for miss-path benchmarks.
const MISS_OFFSET: u64 = 1_000_000_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a pre-filled 8-bit filter containing `n` distinct items.
fn prefilled(n: usize) -> CountingBloomFilter<u64> {
    let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
    for i in 0..n as u64 {
        f.insert(&i);
    }
    f
}

/// Build a `Vec<u64>` working set.
fn workset(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

// ---------------------------------------------------------------------------
// 1. insert – single insert, all counter sizes
// ---------------------------------------------------------------------------

fn bench_insert(c: &mut Criterion) {
    let sizes = [
        ("4bit", CounterSize::FourBit),
        ("8bit", CounterSize::EightBit),
        ("16bit", CounterSize::SixteenBit),
    ];

    let mut group = c.benchmark_group("insert");
    group.throughput(Throughput::Elements(1));

    for (label, cs) in sizes {
        group.bench_with_input(BenchmarkId::from_parameter(label), &cs, |b, &cs| {
            let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, cs);
            let mut i: u64 = 0;
            b.iter(|| {
                f.insert(black_box(&i));
                i = i.wrapping_add(1);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 2. insert_fast – lock-free fast path (shared reference, no item count)
// ---------------------------------------------------------------------------

fn bench_insert_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_fast");
    group.throughput(Throughput::Elements(1));

    group.bench_function("8bit", |b| {
        // insert_fast takes &self, so no &mut needed.
        let f = CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
        let mut i: u64 = 0;
        b.iter(|| {
            f.insert_fast(black_box(&i));
            i = i.wrapping_add(1);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 3. insert_batch – bulk insert at several cardinalities
// ---------------------------------------------------------------------------

fn bench_insert_batch(c: &mut Criterion) {
    let cardinalities: &[usize] = &[100, 1_000, 10_000, 50_000];

    let mut group = c.benchmark_group("insert_batch");
    for &n in cardinalities {
        let items = workset(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &items, |b, items| {
            b.iter(|| {
                let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
                f.insert_batch(black_box(items));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 4. contains_hit – true-positive query (all items present)
// ---------------------------------------------------------------------------

fn bench_contains_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_hit");
    group.throughput(Throughput::Elements(1));

    let f = prefilled(HOT_SET);
    let keys: Vec<u64> = (0..HOT_SET as u64).collect();
    let mut idx = 0usize;

    group.bench_function("8bit", |b| {
        b.iter(|| {
            let k = keys[idx % HOT_SET];
            idx = idx.wrapping_add(1);
            black_box(f.contains(black_box(&k)));
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 5. contains_miss – true-negative query (no item present)
// ---------------------------------------------------------------------------

fn bench_contains_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_miss");
    group.throughput(Throughput::Elements(1));

    let f = prefilled(HOT_SET);
    let mut i: u64 = MISS_OFFSET;

    group.bench_function("8bit", |b| {
        b.iter(|| {
            // Items starting at MISS_OFFSET are never inserted.
            let k = i;
            i = i.wrapping_add(1);
            black_box(f.contains(black_box(&k)));
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 6. contains_batch – bulk query
// ---------------------------------------------------------------------------

fn bench_contains_batch(c: &mut Criterion) {
    let sizes: &[usize] = &[100, 1_000, 10_000];

    let mut group = c.benchmark_group("contains_batch");
    for &n in sizes {
        let f = prefilled(n);
        let queries: Vec<u64> = (0..n as u64).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &queries, |b, queries| {
            b.iter(|| black_box(f.contains_batch(black_box(queries))));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 7. contains_all – short-circuit all-present predicate
// ---------------------------------------------------------------------------

fn bench_contains_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_all");

    // Best case: all present → no early exit until the last element.
    let f = prefilled(HOT_SET);
    let all_present: Vec<u64> = (0..HOT_SET as u64).collect();
    group.throughput(Throughput::Elements(HOT_SET as u64));
    group.bench_function("all_present", |b| {
        b.iter(|| black_box(f.contains_all(black_box(&all_present))));
    });

    // Worst case: miss on the last item → no short-circuit.
    let mut mostly_present = all_present.clone();
    *mostly_present.last_mut().unwrap() = MISS_OFFSET;
    group.bench_function("last_miss", |b| {
        b.iter(|| black_box(f.contains_all(black_box(&mostly_present))));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 8. contains_any – short-circuit any-present predicate
// ---------------------------------------------------------------------------

fn bench_contains_any(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_any");

    let f = prefilled(HOT_SET);

    // Best case: first item present → immediate return.
    let first_hit: Vec<u64> = std::iter::once(0u64)
        .chain((0..HOT_SET as u64).map(|i| MISS_OFFSET + i))
        .collect();
    group.throughput(Throughput::Elements(first_hit.len() as u64));
    group.bench_function("first_hit", |b| {
        b.iter(|| black_box(f.contains_any(black_box(&first_hit))));
    });

    // Worst case: no items present → full scan.
    let all_miss: Vec<u64> = (0..HOT_SET as u64).map(|i| MISS_OFFSET + i).collect();
    group.bench_function("all_miss", |b| {
        b.iter(|| black_box(f.contains_any(black_box(&all_miss))));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 9. delete – single delete with two-phase safety check
// ---------------------------------------------------------------------------

fn bench_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("delete");
    group.throughput(Throughput::Elements(1));

    // Keep re-inserting and deleting the same item to isolate delete cost.
    group.bench_function("8bit_cycling", |b| {
        let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
        // Warm up: insert enough times that counters stay non-zero across iterations.
        for _ in 0..32 {
            f.insert(&42u64);
        }
        b.iter(|| {
            // Insert one to keep the counters from reaching zero permanently.
            f.insert(&black_box(42u64));
            black_box(f.delete(&black_box(42u64)));
        });
    });

    // Miss path: deleting items that are definitely absent.
    group.bench_function("8bit_miss", |b| {
        let f = CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
        let mut f = f;
        let mut i: u64 = MISS_OFFSET;
        b.iter(|| {
            i = i.wrapping_add(1);
            black_box(f.delete(black_box(&i)));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 10. delete_batch – bulk delete throughput
// ---------------------------------------------------------------------------

fn bench_delete_batch(c: &mut Criterion) {
    let sizes: &[usize] = &[100, 1_000, 10_000];
    let mut group = c.benchmark_group("delete_batch");

    for &n in sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    // Setup: fresh filter pre-filled with exactly the items we delete.
                    let mut f =
                        CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
                    let items: Vec<u64> = (0..n as u64).collect();
                    f.insert_batch(&items);
                    (f, items)
                },
                |(mut f, items)| {
                    black_box(f.delete_batch(black_box(&items)));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 11. delete_all_or_none – transactional batch delete
// ---------------------------------------------------------------------------

fn bench_delete_all_or_none(c: &mut Criterion) {
    let mut group = c.benchmark_group("delete_all_or_none");
    let n = 1_000usize;
    group.throughput(Throughput::Elements(n as u64));

    // Success path: all items present.
    group.bench_function("all_present", |b| {
        b.iter_batched(
            || {
                let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
                let items: Vec<u64> = (0..n as u64).collect();
                f.insert_batch(&items);
                (f, items)
            },
            |(mut f, items)| {
                let _ = black_box(f.delete_all_or_none(black_box(&items)));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Failure path: last item absent → aborts at index n-1.
    group.bench_function("last_absent", |b| {
        b.iter_batched(
            || {
                let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
                let inserted: Vec<u64> = (0..n as u64 - 1).collect();
                f.insert_batch(&inserted);
                let mut items = inserted.clone();
                items.push(MISS_OFFSET); // The item that is absent.
                (f, items)
            },
            |(mut f, items)| {
                let _ = f.delete_all_or_none(black_box(&items));
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 12. insert_delete_cycle – sustained churn workload
// ---------------------------------------------------------------------------

fn bench_insert_delete_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_delete_cycle");
    let cycle_size = 1_000usize;
    // Throughput = items processed per iteration = insert + delete per item.
    group.throughput(Throughput::Elements(cycle_size as u64 * 2));

    group.bench_function("8bit", |b| {
        let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
        let items: Vec<u64> = (0..cycle_size as u64).collect();
        b.iter(|| {
            f.insert_batch(black_box(&items));
            black_box(f.delete_batch(black_box(&items)));
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 13. high_load_factor – insert at 2× design capacity
//
// Measures behaviour in the FPR degradation zone where the filter is
// significantly over-capacity. This quantifies the cost of a saturated
// filter: insert time should remain O(k) but FPR will be elevated.
// ---------------------------------------------------------------------------

fn bench_high_load_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_load_factor");
    let overload = CAPACITY * 2;
    group.throughput(Throughput::Elements(overload as u64));

    group.bench_function("2x_capacity_8bit", |b| {
        b.iter(|| {
            let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, CounterSize::EightBit);
            for i in 0..overload as u64 {
                f.insert(black_box(&i));
            }
            black_box(f.estimate_fpr());
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 14. overflow_pressure_4bit – counter saturation on 4-bit filters
//
// Inserts the same key repeatedly to drive 4-bit counters to saturation,
// then measures steady-state insert cost (CAS loop contention at max value).
// ---------------------------------------------------------------------------

fn bench_overflow_pressure_4bit(c: &mut Criterion) {
    let mut group = c.benchmark_group("overflow_pressure_4bit");
    // Saturate a tiny filter deliberately.
    let tiny_capacity = 64usize;
    group.throughput(Throughput::Elements(1));

    group.bench_function("post_saturation_insert", |b| {
        // Pre-saturate: insert key 0 many times until all k counters hit 15.
        let mut f = CountingBloomFilter::with_size(tiny_capacity, 0.5, CounterSize::FourBit);
        for _ in 0..256 {
            f.insert(&0u64);
        }
        let mut i: u64 = 0;
        b.iter(|| {
            // All relevant counters are saturated; measures the fast-exit path.
            f.insert(black_box(&i));
            i = i.wrapping_add(1);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 15. health_metrics – full introspection snapshot cost
// ---------------------------------------------------------------------------

fn bench_health_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("health_metrics");

    for &fill in &[0.1f64, 0.5, 0.9] {
        let n = (CAPACITY as f64 * fill) as usize;
        let f = prefilled(n);
        let label = format!("fill_{:.0}pct", fill * 100.0);
        group.bench_function(&label, |b| {
            b.iter(|| black_box(f.health_metrics()));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 16. hotspots – top-N counter scan
// ---------------------------------------------------------------------------

fn bench_hotspots(c: &mut Criterion) {
    let mut group = c.benchmark_group("hotspots");

    let f = prefilled(HOT_SET);
    for &n in &[1usize, 10, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| black_box(f.hotspots(black_box(n))));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 17. counter_distribution – statistics across the counter array
// ---------------------------------------------------------------------------

fn bench_counter_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("counter_distribution");

    let f = prefilled(HOT_SET);
    group.bench_function("8bit_10k_items", |b| {
        b.iter(|| black_box(f.counter_distribution()));
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 18. counter_histogram – full histogram allocation
// ---------------------------------------------------------------------------

fn bench_counter_histogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("counter_histogram");

    for &cs in &[CounterSize::FourBit, CounterSize::EightBit] {
        let label = if cs == CounterSize::FourBit {
            "4bit"
        } else {
            "8bit"
        };
        let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, cs);
        for i in 0..HOT_SET as u64 {
            f.insert(&i);
        }
        group.bench_function(label, |b| {
            b.iter(|| black_box(f.counter_histogram()));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 19. clear – zeroing out a pre-filled filter
// ---------------------------------------------------------------------------

fn bench_clear(c: &mut Criterion) {
    let mut group = c.benchmark_group("clear");

    // Measure cost proportional to filter size: re-fill then clear each iteration.
    for &cs in &[
        CounterSize::FourBit,
        CounterSize::EightBit,
        CounterSize::SixteenBit,
    ] {
        let label = match cs {
            CounterSize::FourBit => "4bit",
            CounterSize::EightBit => "8bit",
            CounterSize::SixteenBit => "16bit",
        };
        group.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let mut f = CountingBloomFilter::with_size(CAPACITY, FPR, cs);
                    for i in 0..HOT_SET as u64 {
                        f.insert(&i);
                    }
                    f
                },
                |mut f| {
                    f.clear();
                    black_box(())
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 20. memory_overhead_comparison – memory_usage() across counter sizes
//
// Not a latency benchmark; validates and records reported memory overhead
// so regressions in memory accounting are caught by the Criterion baseline.
// ---------------------------------------------------------------------------

fn bench_memory_overhead_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead_comparison");

    for &cs in &[
        CounterSize::FourBit,
        CounterSize::EightBit,
        CounterSize::SixteenBit,
    ] {
        let label = match cs {
            CounterSize::FourBit => "4bit",
            CounterSize::EightBit => "8bit",
            CounterSize::SixteenBit => "16bit",
        };
        let f = CountingBloomFilter::<u64>::with_size(CAPACITY, FPR, cs);
        group.bench_function(label, |b| {
            b.iter(|| black_box(f.memory_usage()));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion registration
// ---------------------------------------------------------------------------

criterion_group!(
    counting_benches,
    bench_insert,
    bench_insert_fast,
    bench_insert_batch,
    bench_contains_hit,
    bench_contains_miss,
    bench_contains_batch,
    bench_contains_all,
    bench_contains_any,
    bench_delete,
    bench_delete_batch,
    bench_delete_all_or_none,
    bench_insert_delete_cycle,
    bench_high_load_factor,
    bench_overflow_pressure_4bit,
    bench_health_metrics,
    bench_hotspots,
    bench_counter_distribution,
    bench_counter_histogram,
    bench_clear,
    bench_memory_overhead_comparison,
);
criterion_main!(counting_benches);
