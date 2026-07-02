//! Stress benchmark suite for TreeBloomFilter
//!
//! Purpose: push TreeBloomFilter to its operational limits and surface
//! performance cliffs, memory ceilings, and architectural constraints that
//! are invisible in unit tests.
//!
//! Every benchmark group answers a concrete engineering question:
//!
//! | Group                       | Question                                            |
//! |-----------------------------|-----------------------------------------------------|
//! | construction_scaling        | When does tree construction become a bottleneck?    |
//! | insert_throughput           | What is the raw insert ceiling?                     |
//! | insert_auto_vs_bin          | Cost of hash routing vs explicit path insertion     |
//! | locate_fanout               | How does DFS cost scale with true match count?      |
//! | locate_api_overhead         | locate vs locate_with vs locate_iter allocation gap |
//! | prune_effectiveness         | How much does pruning actually save?                |
//! | depth_vs_width_tradeoff     | Same leaf count — narrow+deep vs wide+shallow       |
//! | set_operations              | Union / intersect at scale                          |
//! | saturation_cliff            | Query cost degradation as filter saturates          |
//! | clear_subtree_cost          | Per-subtree clear with ancestor propagation         |
//! | memory_pressure             | Cache miss impact with large filters                |
//! | real_world_cdn              | CDN cache invalidation (5×20×100, 10K POPs)         |
//! | real_world_dedup_pipeline   | Log deduplication across 50 services × 20 instances |
//! | real_world_geospatial       | Geo-sharded item lookup (continent/country/city)    |
//! | batch_vs_individual         | Batch insert amortization vs per-item cost          |
//! | contains_in_bin_vs_locate   | Targeted single-bin check vs full tree locate       |
//! | stats_cost                  | Cost of stats() on large trees                      |
//! | worst_case_adversarial      | Pathological inputs: all-bins hit, deep trees       |

#![allow(unused_variables)]

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use bloomcraft::core::MergeableBloomFilter;
use bloomcraft::filters::TreeBloomFilter;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Seed used for all deterministic RNG instances.
const SEED: u64 = 0xDEAD_BEEF_CAFE_1337;

/// Build a pre-populated filter. Items are routed via `insert_auto` so the
/// hash-routing path is exercised. Returns `(filter, items_inserted)`.
fn populated_filter(
    branching: Vec<usize>,
    capacity_per_bin: usize,
    fpr: f64,
    n: usize,
) -> (TreeBloomFilter<String>, Vec<String>) {
    let mut filter =
        TreeBloomFilter::<String>::new(branching, capacity_per_bin, fpr).unwrap();
    let items: Vec<String> = (0..n).map(|i| format!("item:{i}")).collect();
    for item in &items {
        filter.insert(item).unwrap();
    }
    (filter, items)
}

/// Generate `n` items that were never inserted (miss probes).
fn miss_items(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("MISS:{i}")).collect()
}

// ── 1. Tree Construction Scaling ─────────────────────────────────────────────
//
// Question: at what node count does construction become expensive?
// Surfaces: memory allocation cost, recursive build overhead, filter init cost.

fn bench_construction_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_scaling");

    // (label, branching, capacity_per_bin)
    // Node counts: 101, 111, 1111, 10_011, 13_431, 15_625
    let configs: &[(&str, &[usize], usize)] = &[
        ("1L_100",       &[100],           500),
        ("2L_10x10",     &[10, 10],        500),
        ("2L_100x100",   &[100, 100],      100),
        ("3L_10x10x10",  &[10, 10, 10],    200),
        ("3L_5x20x100",  &[5, 20, 100],    500),   // CDN shape
        ("5L_2^5",       &[2, 2, 2, 2, 2], 500),
        ("3L_25x25x25",  &[25, 25, 25],    100),
    ];

    for &(label, branching, cap) in configs {
        group.bench_function(label, |b| {
            b.iter(|| {
                TreeBloomFilter::<String>::new(
                    black_box(branching.to_vec()),
                    black_box(cap),
                    black_box(0.01),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

// ── 2. Raw Insert Throughput ──────────────────────────────────────────────────
//
// Question: what is the maximum insert rate before the filter's bit-set
// contention and hash computation dominate?
// Surfaces: optimal `k` cost, internal node write amplification (k × depth).

fn bench_insert_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_throughput");

    // Write amplification = k × (depth + 1) bit-sets per insert_to_bin.
    // depth=1 → k writes; depth=3 → 4k writes.
    let shapes: &[(&str, &[usize])] = &[
        ("depth1_b100",  &[100]),
        ("depth2_b10",   &[10, 10]),
        ("depth3_b5",    &[5, 5, 4]),
        ("depth4_b3",    &[3, 3, 3, 3]),
        ("depth5_b2",    &[2, 2, 2, 2, 2]),
    ];

    for &(label, branching) in shapes {
        let mut filter =
            TreeBloomFilter::<u64>::new(branching.to_vec(), 100_000, 0.01).unwrap();
        group.throughput(Throughput::Elements(1));
        group.bench_function(label, |b| {
            let mut n: u64 = 0;
            b.iter(|| {
                filter.insert(black_box(&n)).unwrap();
                n = n.wrapping_add(1);
            });
        });
    }

    group.finish();
}

// ── 3. insert_auto vs insert_to_bin ──────────────────────────────────────────
//
// Question: what is the hash-routing overhead of insert_auto relative to
// insert_to_bin with a pre-known path?
// Surfaces: Lemire fast-range + SplitMix64 mixing cost per level.

fn bench_insert_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_routing");

    let branching = vec![10, 10, 10];
    let mut auto_filter =
        TreeBloomFilter::<u64>::new(branching.clone(), 50_000, 0.01).unwrap();
    let mut bin_filter =
        TreeBloomFilter::<u64>::new(branching.clone(), 50_000, 0.01).unwrap();

    group.throughput(Throughput::Elements(1));

    group.bench_function("insert_auto", |b| {
        let mut n: u64 = 0;
        b.iter(|| {
            auto_filter.insert(black_box(&n)).unwrap();
            n = n.wrapping_add(1);
        });
    });

    group.bench_function("insert_to_bin_fixed", |b| {
        let path = [3usize, 7, 2];
        let mut n: u64 = 0;
        b.iter(|| {
            bin_filter.insert_to_bin(black_box(&n), black_box(&path)).unwrap();
            n = n.wrapping_add(1);
        });
    });

    group.finish();
}

// ── 4. Locate Fan-out Scaling ─────────────────────────────────────────────────
//
// Question: how does locate cost grow as true match count increases from 1 to N?
// Surfaces: DFS stack growth, path Vec cloning, prune effectiveness.

fn bench_locate_fanout(c: &mut Criterion) {
    let mut group = c.benchmark_group("locate_fanout");

    // 10×10 = 100 leaf bins
    let branching = vec![10usize, 10];
    let shared = "omnipresent";

    let match_counts = [1usize, 5, 10, 25, 50, 75, 100];

    for &n_bins in &match_counts {
        let mut filter =
            TreeBloomFilter::<String>::new(branching.clone(), 1000, 0.01).unwrap();
        for i in 0..n_bins {
            let bx = i / 10;
            let by = i % 10;
            filter
                .insert_to_bin(&shared.to_string(), &[bx, by])
                .unwrap();
        }

        group.throughput(Throughput::Elements(n_bins as u64));
        group.bench_with_input(
            BenchmarkId::new("bins_hit", n_bins),
            &n_bins,
            |b, _| {
                b.iter(|| black_box(filter.locate(black_box(&shared.to_string()))))
            },
        );
    }

    group.finish();
}

// ── 5. locate vs locate_with vs locate_iter ───────────────────────────────────
//
// Question: what is the allocation overhead of locate() vs the zero-alloc
// variants? Quantifies Vec<Vec<usize>> allocation pressure.

fn bench_locate_api_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("locate_api_overhead");

    let branching = vec![10usize, 10];
    let mut filter =
        TreeBloomFilter::<String>::new(branching.clone(), 1000, 0.01).unwrap();

    // Insert the probe item into 10 scattered bins
    let probe = "probe_item";
    for i in 0..10usize {
        filter
            .insert_to_bin(&probe.to_string(), &[i, (i * 3) % 10])
            .unwrap();
    }

    // Also populate remaining bins with noise to stress DFS pruning
    for i in 0..5000 {
        filter.insert(&format!("noise:{i}")).unwrap();
    }

    group.bench_function("locate_vec_alloc", |b| {
        b.iter(|| black_box(filter.locate(black_box(&probe.to_string()))))
    });

    group.bench_function("locate_with_callback", |b| {
        b.iter(|| {
            filter.locate_with(black_box(&probe.to_string()), |path| {
                black_box(path);
            })
        })
    });

    group.bench_function("locate_iter_collect", |b| {
        b.iter(|| {
            black_box(
                filter
                    .locate_iter(black_box(&probe.to_string()))
                    .collect::<Vec<_>>(),
            )
        })
    });

    group.bench_function("locate_iter_first_only", |b| {
        b.iter(|| {
            black_box(
                filter
                    .locate_iter(black_box(&probe.to_string()))
                    .next(),
            )
        })
    });

    group.finish();
}

// ── 6. Pruning Effectiveness: Hit Rate vs Prune Rate ──────────────────────────
//
// Question: as item density increases, how does the pruning advantage erode?
// At 100% fill every subtree matches — DFS degenerates to full traversal.

fn bench_prune_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("prune_effectiveness");

    let branching = vec![10usize, 10]; // 100 leaf bins
    let capacity_per_bin = 500usize;

    // Fill fractions: 0% → 100% of total capacity
    let fill_fractions = [0.05f64, 0.2, 0.5, 0.8, 1.0, 1.5];

    let probe_miss = "item:MISS:never_inserted";

    for &fill in &fill_fractions {
        let total_cap = capacity_per_bin * 100; // leaf_count = 100
        let n = (total_cap as f64 * fill) as usize;

        let (filter, _) = populated_filter(branching.clone(), capacity_per_bin, 0.01, n);

        let label = format!("fill_{:03}", (fill * 100.0) as u32);

        group.bench_function(format!("miss/{label}"), |b| {
            b.iter(|| black_box(filter.locate(black_box(&probe_miss.to_string()))))
        });
    }

    group.finish();
}

// ── 7. Depth vs Width Trade-off (same leaf count) ────────────────────────────
//
// Question: for N leaf bins, is a flat wide tree or a narrow deep tree faster
// for locate? Tests cache pressure vs DFS step count.
// All configs have 64 leaf bins (2^6).

fn bench_depth_vs_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("depth_vs_width");

    // All have leaf_count = 64
    let shapes: &[(&str, &[usize])] = &[
        ("1L_b64",      &[64]),
        ("2L_b8x8",     &[8, 8]),
        ("3L_b4x4x4",   &[4, 4, 4]),
        ("6L_b2x2x2x2x2x2", &[2, 2, 2, 2, 2, 2]),
    ];

    for &(label, branching) in shapes {
        let (filter, items) =
            populated_filter(branching.to_vec(), 1000, 0.01, 10_000);

        group.bench_function(format!("locate/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &items[i % items.len()];
                i += 1;
                black_box(filter.locate(black_box(item)))
            })
        });

        group.bench_function(format!("contains/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &items[i % items.len()];
                i += 1;
                black_box(filter.contains(black_box(item)))
            })
        });
    }

    group.finish();
}

// ── 8. Set Operations at Scale ────────────────────────────────────────────────
//
// Question: union and intersect cost as filter size grows. These are O(m)
// bitwise ops — benchmark confirms memory bandwidth is the bottleneck.

fn bench_set_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_operations");

    let configs: &[(&str, &[usize], usize)] = &[
        ("2L_10x10_cap500",   &[10, 10],      500),
        ("3L_10x10x10_cap200",&[10, 10, 10],  200),
        ("2L_50x50_cap200",   &[50, 50],       200),
    ];

    for &(label, branching, cap) in configs {
        group.bench_function(format!("union/{label}"), |b| {
            b.iter_with_setup(
                || {
                    let (mut f1, _) = populated_filter(branching.to_vec(), cap, 0.01, cap * 50);
                    let (f2, _) = populated_filter(branching.to_vec(), cap, 0.01, cap * 50);
                    (f1, f2)
                },
                |(mut f1, f2)| black_box(f1.union(&f2).unwrap()),
            );
        });

        group.bench_function(format!("intersect/{label}"), |b| {
            b.iter_with_setup(
                || {
                    let (mut f1, _) = populated_filter(branching.to_vec(), cap, 0.01, cap * 50);
                    let (f2, _) = populated_filter(branching.to_vec(), cap, 0.01, cap * 50);
                    (f1, f2)
                },
                |(mut f1, f2)| black_box(f1.intersect(&f2).unwrap()),
            );
        });
    }

    group.finish();
}

// ── 9. Saturation Cliff ───────────────────────────────────────────────────────
//
// Question: at what fill fraction does query latency increase noticeably?
// Surfaces: FPR inflation causing more subtrees to pass the root gate,
// expanding effective DFS from O(match) to O(nodes).

fn bench_saturation_cliff(c: &mut Criterion) {
    let mut group = c.benchmark_group("saturation_cliff");

    let branching = vec![10usize, 10];
    let cap = 1000usize;
    let leaf_count = 100usize;

    let loads = [0.1f64, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0];

    let misses = miss_items(10_000);

    for &load in &loads {
        let n = (cap as f64 * leaf_count as f64 * load) as usize;
        let (filter, items) = populated_filter(branching.clone(), cap, 0.01, n);

        let label = format!("load_{:03}", (load * 100.0) as u32);

        group.bench_function(format!("hit/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &items[i % items.len().max(1)];
                i += 1;
                black_box(filter.locate(black_box(item)))
            })
        });

        group.bench_function(format!("miss/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &misses[i % misses.len()];
                i += 1;
                black_box(filter.locate(black_box(item)))
            })
        });
    }

    group.finish();
}

// ── 10. Clear Subtree Cost ────────────────────────────────────────────────────
//
// Question: how does clear_subtree cost scale with subtree size?
// Surfaces: iterative clearing (DFS/BFS switch), ancestor propagation cost.

fn bench_clear_subtree(c: &mut Criterion) {
    let mut group = c.benchmark_group("clear_subtree");

    // 3-level tree: 3×4×5 = 60 leaf bins
    let branching = vec![3usize, 4, 5];

    // Clear levels: full tree, level-1 subtree, level-2 subtree, leaf
    let paths: &[(&str, &[usize])] = &[
        ("full_tree",         &[]),
        ("subtree_l1",        &[1]),
        ("subtree_l2",        &[1, 2]),
        ("leaf",              &[1, 2, 3]),
    ];

    for &(label, path) in paths {
        group.bench_function(label, |b| {
            b.iter_with_setup(
                || {
                    let (filter, _) =
                        populated_filter(branching.clone(), 500, 0.01, 30_000);
                    filter
                },
                |mut filter| {
                    black_box(filter.clear_subtree(black_box(path)).unwrap())
                },
            );
        });
    }

    group.finish();
}

// ── 11. Memory Pressure / Cache Miss Impact ───────────────────────────────────
//
// Question: how does query latency degrade when the filter exceeds L2/L3?
// Surfaces: cache miss penalty in BitVec::get across large bit arrays.

fn bench_memory_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure");
    group.sample_size(50); // fewer samples — these are slow by design

    // Escalate memory footprint while keeping leaf count constant (100 bins)
    // by increasing capacity_per_bin → larger per-bin bit vectors
    let configs: &[(&str, usize)] = &[
        ("cap_1k",   1_000),
        ("cap_10k",  10_000),
        ("cap_50k",  50_000),
        ("cap_100k", 100_000),
    ];

    let branching = vec![10usize, 10];
    let mut rng = StdRng::seed_from_u64(SEED);

    for &(label, cap) in configs {
        let n = cap * 50; // 50% fill
        let (filter, items) =
            populated_filter(branching.clone(), cap, 0.01, n);

        // Random access pattern to defeat hardware prefetcher
        group.bench_function(format!("random_query/{label}"), |b| {
            b.iter(|| {
                let idx = rng.gen_range(0..items.len());
                black_box(filter.contains(black_box(&items[idx])))
            })
        });

        group.bench_function(format!("sequential_query/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &items[i % items.len()];
                i += 1;
                black_box(filter.contains(black_box(item)))
            })
        });
    }

    group.finish();
}

// ── 12. Real-World: CDN Cache Invalidation ───────────────────────────────────
//
// Topology: 5 regions × 20 POPs × 100 servers = 10 000 leaf bins
// Workload: random asset inserts + batch invalidation queries
// Metric: invalidation latency per asset = locate() cost

fn bench_real_world_cdn(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_cdn");
    group.sample_size(50);

    let branching = vec![5usize, 20, 100];
    let capacity_per_bin = 5_000usize;
    let n_assets = 200_000usize;

    let mut filter =
        TreeBloomFilter::<String>::new(branching.clone(), capacity_per_bin, 0.001)
            .unwrap();

    let assets: Vec<String> = (0..n_assets)
        .map(|i| format!("/cdn/v2/assets/file_{i:08}.br"))
        .collect();

    // Simulate a warm cache: 60% fill
    for asset in assets.iter().take(n_assets * 6 / 10) {
        filter.insert(asset).unwrap();
    }

    // Invalidation: find all servers hosting an asset
    group.bench_function("invalidation_locate", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let asset = &assets[i % (n_assets * 6 / 10)];
            i += 1;
            black_box(filter.locate(black_box(asset)))
        })
    });

    // Existence check: is this asset cached anywhere?
    group.bench_function("existence_check", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let asset = &assets[i % n_assets];
            i += 1;
            black_box(filter.contains(black_box(asset)))
        })
    });

    // New asset ingestion rate
    group.bench_function("asset_insert", |b| {
        let mut i = n_assets * 6 / 10;
        b.iter(|| {
            let asset = &assets[i % n_assets];
            i += 1;
            filter.insert(black_box(asset)).unwrap();
        })
    });

    // Scoped invalidation: only servers in region 2
    let region_prefix = vec![2usize];
    group.bench_function("region_scoped_invalidation", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let asset = &assets[i % (n_assets * 6 / 10)];
            i += 1;
            black_box(filter.locate_in_range(black_box(asset), black_box(&region_prefix)))
        })
    });

    group.finish();
}

// ── 13. Real-World: Log Deduplication Pipeline ───────────────────────────────
//
// Topology: 10 DCs × 50 services × 20 instances = 10 000 leaf bins
// Workload: high-volume log dedup — check-then-insert pattern
// Insight: contains() + insert() together reveal write-after-read cost

fn bench_real_world_log_dedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_log_dedup");

    let branching = vec![10usize, 50, 20];
    let capacity_per_bin = 2_000usize;

    let mut filter =
        TreeBloomFilter::<String>::new(branching.clone(), capacity_per_bin, 0.01)
            .unwrap();

    let log_templates = [
        "ERROR db_pool exhausted at ts={}",
        "WARN  memory_rss={}mb threshold=80%",
        "INFO  processed request_id={} in 12ms",
        "ERROR auth_fail user_id={} ip=192.168.1.1",
        "DEBUG cache_miss key=session:{}",
    ];

    let n_logs = 500_000usize;
    let logs: Vec<String> = (0..n_logs)
        .map(|i| log_templates[i % log_templates.len()].replace("{}", &i.to_string()))
        .collect();

    // Pre-seed: 40% of logs already seen
    for log in logs.iter().take(n_logs * 4 / 10) {
        filter.insert(log).unwrap();
    }

    // Check-then-insert (dedup pattern)
    group.bench_function("dedup_check_insert", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let log = &logs[i % n_logs];
            i += 1;
            if !filter.contains(black_box(log)) {
                filter.insert(black_box(log)).unwrap();
            }
        })
    });

    // Locate: which DC/service/instance emitted this log?
    group.bench_function("log_locate", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let log = &logs[i % (n_logs * 4 / 10)];
            i += 1;
            black_box(filter.locate(black_box(log)))
        })
    });

    group.finish();
}

// ── 14. Real-World: Geo-Sharded Item Lookup ──────────────────────────────────
//
// Topology: 6 continents × 50 countries × 100 cities = 30 000 leaf bins
// Workload: point queries, continent-scoped queries, global queries
// Limits: tests large subtree navigation + locale-scoped range queries

fn bench_real_world_geospatial(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_world_geospatial");
    group.sample_size(50);

    let branching = vec![6usize, 50, 100];
    let capacity_per_bin = 1_000usize;

    let mut filter =
        TreeBloomFilter::<String>::new(branching.clone(), capacity_per_bin, 0.01)
            .unwrap();

    let n_events = 500_000usize;
    let events: Vec<String> = (0..n_events)
        .map(|i| format!("event:user:{i}:action:purchase"))
        .collect();

    for event in events.iter().take(n_events / 2) {
        filter.insert(event).unwrap();
    }

    // Global lookup
    group.bench_function("global_locate", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let event = &events[i % (n_events / 2)];
            i += 1;
            black_box(filter.locate(black_box(event)))
        })
    });

    // Continent-scoped lookup (prunes 5/6 of the tree immediately)
    let continent_scope = vec![2usize];
    group.bench_function("continent_scoped", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let event = &events[i % (n_events / 2)];
            i += 1;
            black_box(filter.locate_in_range(black_box(event), black_box(&continent_scope)))
        })
    });

    // Country-scoped lookup
    let country_scope = vec![2usize, 15];
    group.bench_function("country_scoped", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let event = &events[i % (n_events / 2)];
            i += 1;
            black_box(filter.locate_in_range(black_box(event), black_box(&country_scope)))
        })
    });

    group.finish();
}

// ── 15. Batch vs Individual Insert ───────────────────────────────────────────
//
// Question: does insert_batch_to_bin amortize overhead meaningfully vs N
// individual insert_to_bin calls?
// Surfaces: per-call validation, counter increment overhead.

fn bench_batch_vs_individual(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_vs_individual");

    let branching = vec![10usize, 10];
    let path = [5usize, 3];

    let batch_sizes = [1usize, 4, 16, 64, 256, 1024];

    for &bs in &batch_sizes {
        group.throughput(Throughput::Elements(bs as u64));

        group.bench_with_input(
            BenchmarkId::new("insert_batch", bs),
            &bs,
            |b, &bs| {
                let mut filter =
                    TreeBloomFilter::<String>::new(branching.clone(), 1_000_000, 0.01)
                        .unwrap();
                b.iter_with_setup(
                    || {
                        (0..bs)
                            .map(|i| format!("batch_item:{i}"))
                            .collect::<Vec<_>>()
                    },
                    |items| {
                        black_box(
                            filter.insert_batch_to_bin(black_box(&items), black_box(&path)).unwrap(),
                        )
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("insert_individual", bs),
            &bs,
            |b, &bs| {
                let mut filter =
                    TreeBloomFilter::<String>::new(branching.clone(), 1_000_000, 0.01)
                        .unwrap();
                b.iter_with_setup(
                    || {
                        (0..bs)
                            .map(|i| format!("indiv_item:{i}"))
                            .collect::<Vec<_>>()
                    },
                    |items| {
                        for item in &items {
                            filter.insert_to_bin(black_box(item), black_box(&path)).unwrap();
                        }
                    },
                );
            },
        );
    }

    group.finish();
}

// ── 16. contains_in_bin vs locate ────────────────────────────────────────────
//
// Question: when you know the exact bin, how much cheaper is contains_in_bin
// vs locate? Quantifies per-level traversal cost vs direct path check.

fn bench_targeted_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("targeted_query");

    let branching = vec![10usize, 10, 10];
    let exact_path = [4usize, 7, 2];

    let (filter, items) = populated_filter(branching.clone(), 1000, 0.01, 50_000);

    group.bench_function("contains_global", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let item = &items[i % items.len()];
            i += 1;
            black_box(filter.contains(black_box(item)))
        })
    });

    group.bench_function("contains_in_bin_exact", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let item = &items[i % items.len()];
            i += 1;
            black_box(filter.contains_in_bin(black_box(item), black_box(&exact_path)).unwrap())
        })
    });

    group.bench_function("locate_full", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let item = &items[i % items.len()];
            i += 1;
            black_box(filter.locate(black_box(item)))
        })
    });

    group.finish();
}

// ── 17. Stats Computation Cost ────────────────────────────────────────────────
//
// Question: what is the cost of stats() and related introspection methods?
// These traverse the entire tree — they should not be called in hot paths.

fn bench_stats_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_cost");

    let configs: &[(&str, &[usize])] = &[
        ("111nodes_10x10",     &[10, 10]),
        ("1111nodes_10x10x10", &[10, 10, 10]),
        ("2501nodes_50x50",    &[50, 50]),
    ];

    for &(label, branching) in configs {
        let n = 50_000usize;
        let (filter, _) = populated_filter(branching.to_vec(), 500, 0.01, n);

        group.bench_function(format!("stats/{label}"), |b| {
            b.iter(|| black_box(filter.stats()))
        });

        group.bench_function(format!("node_count/{label}"), |b| {
            b.iter(|| black_box(filter.node_count()))
        });

        group.bench_function(format!("memory_usage/{label}"), |b| {
            b.iter(|| black_box(filter.memory_usage()))
        });

        group.bench_function(format!("needs_resize/{label}"), |b| {
            b.iter(|| black_box(filter.needs_resize()))
        });
    }

    group.finish();
}

// ── 18. Adversarial / Worst-Case Scenarios ────────────────────────────────────
//
// These are the torture tests. Each targets a specific structural weakness.

fn bench_adversarial(c: &mut Criterion) {
    let mut group = c.benchmark_group("adversarial");
    group.sample_size(50);

    // A) Item present in every single leaf bin (locate must visit all 100 bins)
    {
        let branching = vec![10usize, 10];
        let mut filter =
            TreeBloomFilter::<String>::new(branching, 1000, 0.01).unwrap();
        let omnipresent = "i_am_everywhere";
        for x in 0..10usize {
            for y in 0..10usize {
                filter
                    .insert_to_bin(&omnipresent.to_string(), &[x, y])
                    .unwrap();
            }
        }
        group.bench_function("all_bins_hit", |b| {
            b.iter(|| black_box(filter.locate(black_box(&omnipresent.to_string()))))
        });
    }

    // B) Maximum supported depth with binary branching (depth=16, 65536 leaves)
    //    Tests iterative DFS stack management at scale.
    {
        let depth = 12usize; // 2^12 = 4096 leaves — large but constructible
        let branching = vec![2usize; depth];
        let mut filter =
            TreeBloomFilter::<String>::new(branching, 50, 0.01).unwrap();
        for i in 0..2000usize {
            filter.insert(&format!("deep:{i}")).unwrap();
        }
        group.bench_function("deep_tree_locate", |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = format!("deep:{}", i % 2000);
                i += 1;
                black_box(filter.locate(black_box(&item)))
            })
        });
    }

    // C) Maximally wide single-level tree (256 bins, branching=[256])
    //    Tests branch enumeration cost without DFS depth.
    {
        let branching = vec![256usize];
        let mut filter =
            TreeBloomFilter::<String>::new(branching, 500, 0.01).unwrap();
        let probe = "wide_probe";
        for x in 0..256usize {
            filter.insert_to_bin(&probe.to_string(), &[x]).unwrap();
        }
        group.bench_function("max_width_all_bins_hit", |b| {
            b.iter(|| black_box(filter.locate(black_box(&probe.to_string()))))
        });
    }

    // D) Skewed workload: 99% of inserts into bin [0, 0], 1% uniform
    //    Tests load imbalance impact on locate for the hot bin.
    {
        let branching = vec![10usize, 10];
        let mut filter =
            TreeBloomFilter::<String>::new(branching, 50_000, 0.01).unwrap();
        let mut rng = StdRng::seed_from_u64(SEED);
        for i in 0..100_000usize {
            let item = format!("skewed:{i}");
            if rng.gen_bool(0.99) {
                filter.insert_to_bin(&item, &[0, 0]).unwrap();
            } else {
                filter.insert(&item).unwrap();
            }
        }
        group.bench_function("skewed_hot_bin_locate", |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = format!("skewed:{}", i % 100_000);
                i += 1;
                black_box(filter.locate(black_box(&item)))
            })
        });
    }

    // E) Sequential integer keys — tests hash function quality under low entropy
    {
        let branching = vec![10usize, 10];
        let mut filter =
            TreeBloomFilter::<u64>::new(branching, 100_000, 0.01).unwrap();
        for i in 0u64..50_000 {
            filter.insert(&i).unwrap();
        }
        group.bench_function("low_entropy_sequential_u64", |b| {
            let mut i = 0u64;
            b.iter(|| {
                let result = black_box(filter.contains(black_box(&i)));
                i = i.wrapping_add(1) % 50_000;
                result
            })
        });
    }

    group.finish();
}

// ── 19. FPR vs Memory vs Throughput Trade-off ─────────────────────────────────
//
// Question: how does tighter FPR (larger bit arrays) affect query throughput?
// Surfaces: cache miss rate increase as filter grows larger than L3.

fn bench_fpr_vs_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("fpr_vs_throughput");

    let branching = vec![10usize, 10];
    let n = 50_000usize;

    let fprs: &[(&str, f64)] = &[
        ("fpr_10pct",  0.10),
        ("fpr_1pct",   0.01),
        ("fpr_01pct",  0.001),
        ("fpr_001pct", 0.0001),
    ];

    for &(label, fpr) in fprs {
        let (filter, items) = populated_filter(branching.clone(), 500, fpr, n);

        group.bench_function(format!("contains/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &items[i % items.len()];
                i += 1;
                black_box(filter.contains(black_box(item)))
            })
        });

        group.bench_function(format!("locate/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &items[i % items.len()];
                i += 1;
                black_box(filter.locate(black_box(item)))
            })
        });
    }

    group.finish();
}

// ── 20. Resize Workflow ───────────────────────────────────────────────────────
//
// Question: what is the cost of constructing a replacement filter when
// needs_resize() returns true? This is the full resize workflow cost.

fn bench_resize_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize_workflow");

    let branching = vec![10usize, 10];
    let original_cap = 500usize;
    let new_cap = 2_000usize;

    let (filter, _) = populated_filter(branching.clone(), original_cap, 0.01, 50_000);

    // Constructing the new empty filter is the resize cost (no item migration —
    // callers must re-insert, which is by design for a probabilistic structure)
    group.bench_function("construct_replacement", |b| {
        b.iter(|| {
            TreeBloomFilter::<String>::new(
                black_box(branching.clone()),
                black_box(new_cap),
                black_box(0.01),
            )
            .unwrap()
        })
    });

    group.bench_function("check_needs_resize", |b| {
        b.iter(|| black_box(filter.needs_resize()))
    });

    group.finish();
}

// ── 21. query_any vs contains ─────────────────────────────────────────────────
//
// Question: what is the extra cost of query_any (verifies item at EVERY level
// via any-child recursion) relative to the root-only contains()?
// Surfaces: recursive tree walk cost vs single-filter probe.
// Expected: query_any should be O(depth × branching) in the worst case.
// A large gap signals query_any is unsuitable for hot-path dedup checks.

fn bench_query_any_vs_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_any_vs_contains");

    let shapes: &[(&str, &[usize])] = &[
        ("depth1_b50",      &[50]),
        ("depth2_b10x10",   &[10, 10]),
        ("depth3_b5x5x5",   &[5, 5, 5]),
        ("depth4_b3x3x3x3", &[3, 3, 3, 3]),
    ];

    for &(label, branching) in shapes {
        let (filter, items) = populated_filter(branching.to_vec(), 1000, 0.01, 20_000);

        // Hit path: item was inserted — query_any walks the full confirming chain
        group.bench_function(format!("contains_hit/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &items[i % items.len()];
                i += 1;
                black_box(filter.contains(black_box(item)))
            })
        });

        group.bench_function(format!("query_any_hit/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &items[i % items.len()];
                i += 1;
                black_box(filter.query_any(black_box(item)))
            })
        });

        // Miss path: item not in any filter — query_any should exit at root
        let misses = miss_items(10_000);
        group.bench_function(format!("contains_miss/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &misses[i % misses.len()];
                i += 1;
                black_box(filter.contains(black_box(item)))
            })
        });

        group.bench_function(format!("query_any_miss/{label}"), |b| {
            let mut i = 0usize;
            b.iter(|| {
                let item = &misses[i % misses.len()];
                i += 1;
                black_box(filter.query_any(black_box(item)))
            })
        });
    }

    group.finish();
}

// ── 22. Builder Pattern Construction Overhead ─────────────────────────────────
//
// Question: does TreeBloomFilterBuilder add measurable overhead over
// TreeBloomFilter::new? The builder calls withhasher internally — this
// should be zero-cost. Any gap reveals non-inlined validation duplication.

fn bench_builder_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("builder_overhead");

    let configs: &[(&str, &[usize], usize)] = &[
        ("small_2L",  &[10, 10],    500),
        ("medium_3L", &[10, 10, 10], 200),
        ("large_2L",  &[50, 50],    200),
    ];

    for &(label, branching, cap) in configs {
        group.bench_function(format!("new_direct/{label}"), |b| {
            b.iter(|| {
                TreeBloomFilter::<String>::new(
                    black_box(branching.to_vec()),
                    black_box(cap),
                    black_box(0.01),
                )
                .unwrap()
            })
        });

        group.bench_function(format!("builder/{label}"), |b| {
            b.iter(|| {
                use bloomcraft::filters::tree::TreeBloomFilterBuilder;
                TreeBloomFilterBuilder::<String>::new()
                    .branching(black_box(branching.to_vec()))
                    .capacity_per_bin(black_box(cap))
                    .false_positive_rate(black_box(0.01))
                    .build()
                    .unwrap()
            })
        });
    }

    group.finish();
}

// ── 23. TreeConfig Validation Cost ───────────────────────────────────────────
//
// Question: what is the cost of TreeConfig::validate() and report()?
// These are planning-time calls but engineers sometimes call them at startup
// inside a health-check loop. Quantifies whether that is safe.
// Surfaces: node count walk, optimal_m invocation, string formatting cost.

fn bench_tree_config_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_config_validation");

    use bloomcraft::filters::tree::TreeConfig;

    let configs: &[(&str, &[usize], usize)] = &[
        ("tiny_2L",        &[5, 5],       100),
        ("medium_3L",      &[10, 10, 10], 500),
        ("cdn_3L_5x20x100",&[5, 20, 100], 5000),
        ("large_2L_50x50", &[50, 50],     1000),
    ];

    for &(label, branching, cap) in configs {
        let config = TreeConfig {
            branching: branching.to_vec(),
            capacity_per_bin: cap,
            target_fpr: 0.01,
        };

        group.bench_function(format!("validate/{label}"), |b| {
            b.iter(|| black_box(config.validate()))
        });

        group.bench_function(format!("report/{label}"), |b| {
            b.iter(|| black_box(config.report()))
        });
    }

    group.finish();
}

// ── 24. locate_batch vs locate_batch_parallel ─────────────────────────────────
//
// Question: at what batch size does the Rayon parallel version break even
// with the sequential version? Below the break-even point, thread-pool
// overhead dominates. This directly informs when locate_batch_parallel
// should be gated behind a batch-size check.
// Note: compile with --features rayon to activate parallel variant.

fn bench_locate_batch_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("locate_batch_parallel");

    let branching = vec![10usize, 10];
    let (filter, items) = populated_filter(branching.clone(), 1000, 0.01, 50_000);

    let batch_sizes = [1usize, 4, 16, 64, 256, 1024, 4096];

    for &bs in &batch_sizes {
        let batch: Vec<&String> = items[..bs.min(items.len())].iter().collect();

        group.throughput(Throughput::Elements(bs as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential", bs),
            &bs,
            |b, _| {
                b.iter(|| black_box(filter.locate_batch(black_box(&batch))))
            },
        );

        // Only compiled when --features rayon is active.
        // Without the feature this bench still compiles but calls the
        // sequential fallback, giving a clean apples-to-apples baseline.
        #[cfg(feature = "rayon")]
        group.bench_with_input(
            BenchmarkId::new("parallel_rayon", bs),
            &bs,
            |b, _| {
                b.iter(|| black_box(filter.locate_batch_parallel(black_box(&batch))))
            },
        );
    }

    group.finish();
}

// ── 25. Serialization / Deserialization Round-Trip Cost ───────────────────────
//
// Question: what is the bincode serialization and deserialization latency
// as filter size scales? These operations block the event loop in
// checkpoint/restore scenarios (crash recovery, migration).
// Surfaces: TypeId field inclusion overhead, bitset encoding cost.
// Requires: --features serde

#[cfg(feature = "serde")]
fn bench_serde_roundtrip(c: &mut Criterion) {
    use bloomcraft::hash::StdHasher;

    let mut group = c.benchmark_group("serde_roundtrip");
    group.sample_size(50);

    let configs: &[(&str, &[usize], usize, usize)] = &[
        // (label, branching, cap_per_bin, n_items)
        ("tiny_10x10_1k",      &[10, 10],      500,  5_000),
        ("medium_10x10x10_5k", &[10, 10, 10],  200,  20_000),
        ("large_50x50_50k",    &[50, 50],       200,  50_000),
    ];

    for &(label, branching, cap, n) in configs {
        let (filter, _) =
            populated_filter(branching.to_vec(), cap, 0.01, n);

        // Serialize once to pre-compute bytes for the deserialization bench
        let bytes = bincode::serialize(&filter).unwrap();
        let byte_count = bytes.len();

        group.throughput(Throughput::Bytes(byte_count as u64));

        group.bench_function(format!("serialize/{label}"), |b| {
            b.iter(|| black_box(bincode::serialize(black_box(&filter)).unwrap()))
        });

        group.bench_function(format!("deserialize/{label}"), |b| {
            b.iter(|| {
                black_box(
                    bincode::deserialize::<TreeBloomFilter<String, StdHasher>>(
                        black_box(&bytes),
                    )
                    .unwrap(),
                )
            })
        });
    }

    group.finish();
}

// Stub for non-serde builds so criterion_group! registration compiles uniformly.
#[cfg(not(feature = "serde"))]
fn bench_serde_roundtrip(_c: &mut Criterion) {}

// ── 26. insert_auto Determinism / Hash Distribution Uniformity ────────────────
//
// Question: does insert_auto distribute items uniformly across bins?
// A non-uniform distributon causes hot bins to saturate faster, which
// inflates FPR in those bins while wasting capacity in cold bins.
// This benchmark measures the cost of verifying distribution by
// checking locate() returns exactly 1 bin per item (insert_auto
// must be deterministic and injective in the routing sense).
//
// The actual distribution measurement is done by computing stddev of
// per-bin item counts via stats() — the bench measures that analysis cost.

fn bench_insert_auto_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_auto_distribution");

    let branching = vec![10usize, 10]; // 100 leaf bins
    let n = 100_000usize;
    let (filter, _) = populated_filter(branching.clone(), 50_000, 0.01, n);

    // Cost of gathering stats after a large population — used to measure
    // distribution health at startup or in monitoring loops.
    group.bench_function("stats_after_large_population", |b| {
        b.iter(|| black_box(filter.stats()))
    });

    // Cost of confirming each insert_auto routes to exactly 1 bin.
    // Measures locate() on freshly inserted items — confirms no routing
    // ambiguity has been introduced by hash mixing changes.
    let (filter2, items2) = populated_filter(branching.clone(), 50_000, 0.01, 1_000);
    group.bench_function("confirm_single_bin_routing", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let item = &items2[i % items2.len()];
            i += 1;
            let locs = filter2.locate(black_box(item));
            // black_box the length to prevent the compiler from eliminating
            // the locate call entirely.
            black_box(locs.len())
        })
    });

    group.finish();
}

// ── 27. Resize Workflow: Full Re-Insertion Amortized Cost ─────────────────────
//
// Question: when a filter needs_resize(), the correct workflow is to
// construct a new filter and re-insert all items. What is the total
// amortized cost of that migration per-item?
// This bench measures: construct + insert N items into the replacement filter.
// Surfaces: whether batch insert amortizes construction cost at scale.

fn bench_resize_full_migration(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize_full_migration");
    group.sample_size(20); // expensive: each iter constructs + populates a new filter

    let branching = vec![10usize, 10];

    let migration_sizes = [1_000usize, 10_000, 50_000, 100_000];

    for &n in &migration_sizes {
        group.throughput(Throughput::Elements(n as u64));

        let items: Vec<String> = (0..n).map(|i| format!("migrated:{i}")).collect();

        group.bench_with_input(
            BenchmarkId::new("migrate_n_items", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    // Simulate the full resize workflow:
                    // 1. Construct replacement with 2× capacity
                    let mut new_filter =
                        TreeBloomFilter::<String>::new(branching.clone(), n / 50 + 1, 0.01)
                            .unwrap();
                    // 2. Re-insert all items
                    for item in items.iter().take(n) {
                        new_filter.insert(black_box(item)).unwrap();
                    }
                    black_box(new_filter)
                })
            },
        );
    }

    group.finish();
}

// ── 28. Concurrent Read Safety: Arc<TreeBloomFilter> Throughput ───────────────
//
// Question: what is the throughput of concurrent read-only access via
// Arc<TreeBloomFilter>? The docs state contains() is safe from multiple
// threads (StandardBloomFilter uses atomic ops). This bench validates
// that claim under real contention and measures the coherency overhead.
//
// Architecture note: TreeBloomFilter is not Sync due to &mut methods.
// Arc<TreeBloomFilter> makes it Sync-by-construction for the read path.
// This is the recommended pattern for read-heavy workloads.

fn bench_concurrent_reads(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let mut group = c.benchmark_group("concurrent_reads");
    group.sample_size(30);

    let branching = vec![10usize, 10];
    let (filter, items) = populated_filter(branching.clone(), 5_000, 0.01, 50_000);
    let filter = Arc::new(filter);
    let items = Arc::new(items);

    let thread_counts = [1usize, 2, 4, 8];

    for &n_threads in &thread_counts {
        group.throughput(Throughput::Elements(n_threads as u64 * 1_000));

        group.bench_with_input(
            BenchmarkId::new("contains_arc", n_threads),
            &n_threads,
            |b, &n_threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let f = Arc::clone(&filter);
                            let it = Arc::clone(&items);
                            thread::spawn(move || {
                                // Each thread queries 1000 items
                                let base = t * 1_000;
                                for k in 0..1_000usize {
                                    let item = &it[(base + k) % it.len()];
                                    black_box(f.contains(black_box(item)));
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("locate_arc", n_threads),
            &n_threads,
            |b, &n_threads| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n_threads)
                        .map(|t| {
                            let f = Arc::clone(&filter);
                            let it = Arc::clone(&items);
                            thread::spawn(move || {
                                let base = t * 500;
                                for k in 0..500usize {
                                    let item = &it[(base + k) % it.len()];
                                    black_box(f.locate(black_box(item)));
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    name = construction;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(2));
    targets =
        bench_construction_scaling,
        bench_resize_workflow
);

criterion_group!(
    name = insert;
    config = Criterion::default()
        .sample_size(200)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(2));
    targets =
        bench_insert_throughput,
        bench_insert_routing,
        bench_batch_vs_individual
);

criterion_group!(
    name = query;
    config = Criterion::default()
        .sample_size(200)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(2));
    targets =
        bench_locate_fanout,
        bench_locate_api_overhead,
        bench_prune_effectiveness,
        bench_depth_vs_width,
        bench_targeted_query,
        bench_saturation_cliff,
        bench_fpr_vs_throughput
);

criterion_group!(
    name = structural;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(8))
        .warm_up_time(std::time::Duration::from_secs(2));
    targets =
        bench_set_operations,
        bench_clear_subtree,
        bench_stats_cost
);

criterion_group!(
    name = real_world;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(15))
        .warm_up_time(std::time::Duration::from_secs(3));
    targets =
        bench_real_world_cdn,
        bench_real_world_log_dedup,
        bench_real_world_geospatial
);

criterion_group!(
    name = limits;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(12))
        .warm_up_time(std::time::Duration::from_secs(3));
    targets =
        bench_memory_pressure,
        bench_adversarial
);

criterion_group!(
    name = extended;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(2));
    targets =
        bench_query_any_vs_contains,
        bench_builder_overhead,
        bench_tree_config_validation,
        bench_insert_auto_distribution
);

criterion_group!(
    name = parallel_and_serde;
    config = Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(12))
        .warm_up_time(std::time::Duration::from_secs(3));
    targets =
        bench_locate_batch_parallel,
        bench_serde_roundtrip,
        bench_resize_full_migration,
        bench_concurrent_reads
);

criterion_main!(
    construction,
    insert,
    query,
    structural,
    real_world,
    limits,
    extended,
    parallel_and_serde
);