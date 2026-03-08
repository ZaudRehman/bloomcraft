# BloomCraft

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

A production-grade Bloom filter library for Rust. BloomCraft provides twelve filter variants — from the classical space-optimal filter to scalable, partitioned, register-blocked, and concurrent implementations — unified under a coherent trait hierarchy with type-state builders, pluggable hash strategies, and optional serde, metrics, and SIMD support.

## Why BloomCraft?

Most Rust Bloom filter crates ship one or two variants behind a single trait.
BloomCraft ships twelve — covering every practical trade-off between space, speed, deletion, scalability, and concurrency — all under a single, coherent API with three distinct concurrency models (`&mut self` with external lock, `&self` via `AtomicU64` CAS, and `&self` via interior mutability), type-state builders that make misconfiguration a compile error, and pluggable hash strategies from SipHash to SIMD-accelerated WyHash.

If you need a filter you can delete from, one that grows without bounds, one that saturates a single cache line per query, or one that accepts concurrent writes from 64 threads without a `Mutex` in sight — this crate has a specific, documented type for each of those requirements rather than asking you to bolt synchronization onto a single generic struct.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Filter Selection](#filter-selection)
- [Filter Variants](#filter-variants)
- [Concurrency Models](#concurrency-models)
- [Type-State Builders](#type-state-builders)
- [Hash Strategies](#hash-strategies)
- [Feature Flags](#feature-flags)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [References](#references)
- [Contributing](#contributing)
- [Changelog](#changelog)
- [License](#license)

---

## Installation

```toml
[dependencies]
bloomcraft = "0.1"
```

With optional features:

```toml
[dependencies]
bloomcraft = { version = "0.1", features = ["serde", "wyhash", "metrics", "concurrent"] }
```

**Minimum supported Rust version:** 1.70

---

## Quick Start

```rust
use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::core::BloomFilter;

let mut filter = StandardBloomFilter::<String>::new(10_000, 0.01)?;

filter.insert(&"hello".to_string());
filter.insert(&"world".to_string());

assert!(filter.contains(&"hello".to_string()));   // definitely present
assert!(!filter.contains(&"rust".to_string()));   // definitely absent
// contains() may return true for items never inserted (bounded by 1% FPR)
```

---

## Filter Selection

| Use Case | Filter | Feature Gate | Notes |
|---|---|---|---|
| General purpose, known capacity | `StandardBloomFilter` | always | Optimal space, supports union/intersect |
| Need deletion | `CountingBloomFilter` | always | 4–16× memory, per-element counters |
| Unknown or growing dataset | `ScalableBloomFilter` | always | Auto-grows, FPR stays bounded |
| Concurrent, growing dataset | `AtomicScalableBloomFilter` | `concurrent` | Lock-free inserts, sharded internals |
| Query-heavy, cache-sensitive | `PartitionedBloomFilter` | always | 2–4× faster queries, cache-line aligned |
| Ultra-fast queries, FPR-tolerant | `RegisterBlockedBloomFilter` | always | 512-bit AVX blocks, 20–30% faster, 2–3× FPR |
| Concurrent, cache-optimized | `AtomicPartitionedBloomFilter` | `concurrent` | Wait-free inserts + cache-line partitions |
| Multi-level / location-aware | `TreeBloomFilter` | always | Returns which subtree contains item |
| High-concurrency writes (lock-free) | `ShardedBloomFilter` | always | `&self` insert, no `Mutex` |
| High-concurrency, memory-constrained | `StripedBloomFilter` | always | Fine-grained `RwLock` striping, `&self` |
| Historical / research reference | `ClassicHashFilter` | always | Bloom 1970 Method 1 |
| Historical / research reference | `ClassicBitsFilter` | always | Bloom 1970 Method 2 |

**Concurrency quick-reference:**

| Filter | Ownership for Insert | Mechanism |
|---|---|---|
| `StandardBloomFilter` | `&mut self` or `&self` via `ConcurrentBloomFilter` | Atomic CAS on `AtomicU64` words |
| `CountingBloomFilter` | `&mut self` → `Arc<Mutex<T>>` | External lock |
| `ScalableBloomFilter` | `&mut self` → `Arc<Mutex<T>>` | External lock |
| `AtomicScalableBloomFilter` | `&self` | Lock-free shards + `RwLock` for growth |
| `PartitionedBloomFilter` | `&mut self` → `Arc<RwLock<T>>` | External lock |
| `RegisterBlockedBloomFilter` | `&mut self` → `Arc<Mutex<T>>` | External lock |
| `AtomicPartitionedBloomFilter` | `&self` via `ConcurrentBloomFilter` | `AtomicU64` fetchor, wait-free |
| `TreeBloomFilter` | `&mut self` → `Arc<RwLock<T>>` | External lock |
| `ShardedBloomFilter` | `&self` via `SharedBloomFilter` | Lock-free atomic shards |
| `StripedBloomFilter` | `&self` via `SharedBloomFilter` | Striped `RwLock` |

---

## Filter Variants

### StandardBloomFilter

The classic space-optimal Bloom filter. Its bit array is backed by `AtomicU64` words, making it also suitable for wait-free concurrent writes via the `ConcurrentBloomFilter` extension trait — no external lock required. Supports union and intersection merges with compatible filters.

```rust
use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::core::{BloomFilter, MergeableBloomFilter};

let mut filter = StandardBloomFilter::<&str>::new(50_000, 0.001)?;

filter.insert(&"key");
assert!(filter.contains(&"key"));

println!("bits: {}",         filter.bit_count());
println!("hash fns: {}",     filter.hash_count());
println!("estimated FPR: {:.6}", filter.estimate_fpr());

// Merge two filters of identical configuration
let union        = filter_a.union(&filter_b)?;
let intersection = filter_a.intersect(&filter_b)?;
```

---

### CountingBloomFilter

Extends the standard filter with per-slot counters, enabling safe deletion. Each slot uses 4, 8, or 16 bits (configurable via `CounterSize`); overflow is tracked so callers can detect correctness degradation before it becomes silent.

```rust
use bloomcraft::filters::CountingBloomFilter;
use bloomcraft::core::{BloomFilter, DeletableBloomFilter};

let mut filter = CountingBloomFilter::<String>::new(10_000, 0.01)?;

filter.insert(&"item".to_string());
assert!(filter.contains(&"item".to_string()));

// delete() is a no-op and returns false if the item is not present.
let removed = filter.delete(&"item".to_string());
assert!(removed);
assert!(!filter.contains(&"item".to_string()));

println!("overflowed slots: {}", filter.overflow_count());
println!("has overflowed:   {}", filter.has_overflowed());
```

---

### ScalableBloomFilter

Maintains a chain of fixed-size filter slices. When the active slice's fill ratio exceeds `fill_threshold` (default 0.5 — the mathematically proven optimal), a new slice is appended with capacity multiplied by the `growth_factor` and FPR multiplied by the `tightening_ratio`. The overall compound FPR is bounded:

```
FPR_overall ≤ target_fpr / (1 − tightening_ratio)
```

Four growth strategies are available:

| Strategy | Description |
|---|---|
| `Geometric(f64)` | Each slice is `scale ×` the previous. Default: `Geometric(2.0)` |
| `Constant` | All slices have the same capacity as the initial one |
| `Adaptive { initial_ratio, min_ratio, max_ratio }` | Tightening ratio adapts to the observed fill rate of each completed slice |
| `Bounded { scale, max_filter_size }` | Geometric until a per-filter capacity ceiling is hit |

```rust
use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy, QueryStrategy};
use bloomcraft::core::BloomFilter;

// Adaptive growth: tightens FPR more aggressively when fill spikes
let mut filter = ScalableBloomFilter::<u64>::with_strategy(
    1_000, 0.01, 0.5,
    GrowthStrategy::Adaptive {
        initial_ratio: 0.5,
        min_ratio:     0.3,
        max_ratio:     0.9,
    },
)?;

for i in 0..50_000_u64 {
    filter.insert(&i);
}

// Query strategy: Reverse (default) checks the newest slice first,
// yielding O(1) latency for recently-inserted items.
// Forward is appropriate when queries are uniformly distributed.

let metrics = filter.health_metrics();
println!("slices:          {}", metrics.filter_count);
println!("total capacity:  {}", metrics.total_capacity);
println!("estimated FPR:   {:.6}", metrics.estimated_fpr);
println!("FPR upper bound: {:.6}", metrics.max_fpr);

// Optional HyperLogLog-based unique-item estimation (±0.81%)
let mut filter = ScalableBloomFilter::<u64>::new(1_000, 0.01)?
    .with_cardinality_tracking();

for _ in 0..5 {
    for i in 0..10_000_u64 { filter.insert(&i); }
}

println!("total inserts: {}", filter.len());           // 50,000
println!("unique items:  {}", filter.estimate_unique_count()); // ~10,000
```

---

### AtomicScalableBloomFilter

*(Requires `concurrent` feature)*

A concurrent, automatically-growing Bloom filter backed by a chain of sharded sub-filters. Each sub-filter is independently divided into `shard_count` `StandardBloomFilter` instances, so concurrent inserts to different shards are completely wait-free with zero cross-shard contention. Growth uses a double-checked locking pattern: only one thread wins the CAS and allocates the new slice; all other threads continue inserting into the current slice without blocking.

```rust
#[cfg(feature = "concurrent")]
{
    use bloomcraft::filters::AtomicScalableBloomFilter;
    use std::sync::Arc;

    // Shard count is chosen automatically from available_parallelism(), capped at 16.
    let filter = Arc::new(AtomicScalableBloomFilter::<u64>::new(10_000, 0.01)?);

    let handles: Vec<_> = (0..8_u64).map(|thread_id| {
        let f = Arc::clone(&filter);
        std::thread::spawn(move || {
            for i in 0..1_000 {
                f.insert(thread_id * 1_000 + i);  // &self, wait-free
            }
        })
    }).collect();

    for h in handles { h.join().unwrap(); }
    assert_eq!(filter.len(), 8_000);

    // Pre-allocate all slices upfront to eliminate growth-lock contention
    // during a known-size bulk-insert phase.
    let pre = AtomicScalableBloomFilter::<u64>::with_preallocated(
        100_000, 0.01, 10_000_000
    )?;
}
```

---

### PartitionedBloomFilter

Divides the bit array into `k` equal partitions — one per hash function — so each hash function probes within a single cache line. This eliminates cross-partition false sharing during queries and delivers 2–4× higher lookup throughput on typical hardware.

```rust
use bloomcraft::filters::PartitionedBloomFilter;
use bloomcraft::core::BloomFilter;

// Auto-tune partitions to the L1/L2 boundary
let mut filter = PartitionedBloomFilter::<String>::new_cache_tuned(10_000, 0.01)?;

// Or specify a fixed cache-line alignment explicitly
let mut filter = PartitionedBloomFilter::<String>::with_alignment(10_000, 0.01, 64)?;

filter.insert(&"item".to_string());
assert!(filter.contains(&"item".to_string()));

println!("partitions:      {}", filter.partition_count());
println!("bits/partition:  {}", filter.partition_size());
```

---

### RegisterBlockedBloomFilter

Divides the bit array into 512-bit (64-byte) blocks. The first hash selects the block; the remaining `k − 1` hashes select bits within it. Because all probed bits lie within a single AVX-512 register-sized block, **every query incurs at most one cache miss** regardless of `k`. This makes it 20–30% faster than `PartitionedBloomFilter` for query-dominated workloads. The trade-off is a 2–3× higher false positive rate at equivalent memory compared to a standard filter, since block-constrained hashing reduces the effective independence of hash positions.

```rust
use bloomcraft::filters::RegisterBlockedBloomFilter;
use bloomcraft::core::BloomFilter;

let mut filter = RegisterBlockedBloomFilter::<u64>::new(100_000, 0.01)?;

filter.insert(&42);
assert!(filter.contains(&42));

println!("blocks:       {}", filter.num_blocks());        // 512-bit each
println!("bits/block:   {}", filter.bits_per_block());    // always 512
println!("target FPR:   {:.2}%", filter.target_fpr() * 100.0);
// Actual empirical FPR is ~2.5–3× the target due to blocking overhead.
```

**Performance vs. other filters:**

| Metric | `StandardBloomFilter` | `PartitionedBloomFilter` | `RegisterBlockedBloomFilter` |
|---|---|---|---|
| Query latency | 15–20 ns | 10–15 ns | 8–12 ns |
| Cache misses / query | 1–k | 1 (per partition) | 1 (guaranteed) |
| FPR overhead | 1× target | 1× target | 2.5–3× target |
| Memory efficiency | Optimal | Near-optimal | Good |

---

### AtomicPartitionedBloomFilter

*(Requires `concurrent` feature)*

Combines the cache-line partition layout of `PartitionedBloomFilter` with `AtomicU64`-backed storage for wait-free concurrent inserts and lock-free concurrent queries. All hot-path atomics use `Ordering::Relaxed` — correct because Bloom filter bit-set operations are idempotent and no inter-thread causality is required.

```rust
#[cfg(feature = "concurrent")]
{
    use bloomcraft::filters::AtomicPartitionedBloomFilter;
    use bloomcraft::core::ConcurrentBloomFilter;
    use std::sync::Arc;

    let filter = Arc::new(
        AtomicPartitionedBloomFilter::<u64>::new(1_000_000, 0.01)?
    );

    // Expected throughput scaling (insert_concurrent):
    // 2 threads → ~1.9× | 4 threads → ~3.7× | 8 threads → ~7.0× | 16 threads → ~13×
    let handles: Vec<_> = (0..8).map(|t| {
        let f = Arc::clone(&filter);
        std::thread::spawn(move || {
            for i in 0..10_000_usize {
                f.insert_concurrent(t * 10_000 + i);  // wait-free
            }
        })
    }).collect();

    for h in handles { h.join().unwrap(); }
    println!("saturation: {:.1}%", filter.saturation() * 100.0);
    println!("estimated FPR: {:.4}%", filter.estimated_fpr() * 100.0);
}
```

---

### TreeBloomFilter

A hierarchical filter that maps items to leaf nodes in a configurable tree. A `locate()` call returns every path whose subtree might contain the item, enabling location-aware lookups across tiered storage systems (e.g., continent → datacenter → rack → node).

```rust
use bloomcraft::filters::TreeBloomFilter;
use bloomcraft::core::BloomFilter;

// Two-level tree: 4 regions × 8 datacenters = 32 leaf bins
let mut router = TreeBloomFilter::<String>::new(vec![4, 8], 1_000, 0.01)?;

router.insert_to_bin(&"session:alice".to_string(), &[1, 3])?;

// locate() returns all leaf paths that might contain the item
let locations = router.locate(&"session:alice".to_string());
for path in &locations {
    println!("may be at: {:?}", path);  // → [][4][3]
}

// Bin-scoped query: check a specific path without scanning the whole tree
assert!( router.contains_in_bin(&"session:alice".to_string(), &)?);[3][4]
assert!(!router.contains_in_bin(&"session:alice".to_string(), &)?);[5]

let stats = router.stats();
println!("depth:       {}", stats.depth);       // 2
println!("leaf bins:   {}", stats.leaf_bins);   // 32
println!("memory:      {} bytes", stats.memory_usage);
```

---

### ShardedBloomFilter / StripedBloomFilter

Both filters live in the `sync` module and implement `SharedBloomFilter`, whose `insert`, `contains`, and `clear` methods take `&self`. No external lock is needed — concurrency is managed entirely within the type.

`ShardedBloomFilter` uses fully independent atomic shards routed by the upper bits of the item hash. `StripedBloomFilter` uses a striped `RwLock` array, where the stripe is selected via Lemire's fast range reduction (`(hash as u128 * num_stripes as u128) >> 64`) — approximately 7× faster than `hash % num_stripes`. Each stripe is padded to a full 64-byte cache line to prevent false sharing.

```rust
use bloomcraft::sync::{ShardedBloomFilter, StripedBloomFilter};
use bloomcraft::core::SharedBloomFilter;
use std::sync::Arc;

// ShardedBloomFilter: ideal for read-heavy, high-parallelism workloads
let sharded = Arc::new(ShardedBloomFilter::<String>::new(100_000, 0.01));

// StripedBloomFilter: ideal for write-heavy, moderate-concurrency workloads.
// Stripe count = clamp(next_pow2(threads × 4), 16, 4096).
// with_concurrency(n) derives this automatically.
let striped = Arc::new(
    StripedBloomFilter::<String>::with_concurrency(100_000, 0.01, 64)?
    // → 256 stripes (64 × 4, rounded to next power of two)
);

let handles: Vec<_> = (0..8).map(|t| {
    let f = Arc::clone(&sharded);  // swap with &striped — identical call site
    std::thread::spawn(move || {
        for i in 0..1_000_usize {
            f.insert(&format!("t{}-{}", t, i));  // &self, no Mutex
        }
    })
}).collect();

for h in handles { h.join().unwrap(); }
```

**`StripedBloomFilter` throughput (ops/sec under contention):**

| Threads | 16 stripes | 256 stripes | 1,024 stripes |
|---|---|---|---|
| 1  |  6.3 M/s |  6.2 M/s |  6.0 M/s |
| 8  | 12   M/s | 18   M/s | 20   M/s |
| 16 | 14   M/s | 22   M/s | 28   M/s |
| 64 | 16   M/s | 35   M/s | 55   M/s |

---

### ClassicHashFilter / ClassicBitsFilter

Reference implementations of Burton Bloom's original 1970 paper — Method 1 (hash table with chaining) and Method 2 (bit array). Useful for education, benchmarking against modern variants, and reproducing original results.

```rust
use bloomcraft::filters::{ClassicBitsFilter, ClassicHashFilter};
use bloomcraft::core::BloomFilter;

// Method 2: the direct ancestor of every modern Bloom filter
let mut bits = ClassicBitsFilter::<&str>::new(10_000, 7)?;
bits.insert(&"hello");
assert!(bits.contains(&"hello"));

// Method 1: hash table with chaining
let mut hash = ClassicHashFilter::<&str>::new(1_000, 3)?;
hash.insert(&"hello");
println!("avg chain length: {:.2}", hash.avg_chain_length());
```

---

## Concurrency Models

BloomCraft exposes three distinct concurrency patterns. They are not unified because each makes a different guarantee at the type level.

### 1 — Single-threaded (`BloomFilter` trait, `&mut self`)

Standard filters — `CountingBloomFilter`, `ScalableBloomFilter`, `PartitionedBloomFilter`, `RegisterBlockedBloomFilter`, `TreeBloomFilter`, and the Classic filters — all require `&mut self` for writes. Wrap in `Arc<Mutex<T>>` or `Arc<RwLock<T>>` when sharing across threads. Zero synchronization overhead when used single-threaded.

```rust
use bloomcraft::filters::CountingBloomFilter;
use bloomcraft::core::BloomFilter;
use std::sync::{Arc, Mutex};

let filter = Arc::new(Mutex::new(
    CountingBloomFilter::<String>::new(10_000, 0.01)?
));

let f = Arc::clone(&filter);
std::thread::spawn(move || {
    f.lock().unwrap().insert(&"shared".to_string());
}).join().unwrap();

assert!(filter.lock().unwrap().contains(&"shared".to_string()));
```

### 2 — Lock-free atomic (`ConcurrentBloomFilter` trait, `&self`)

`StandardBloomFilter` and `AtomicPartitionedBloomFilter` implement `ConcurrentBloomFilter`, whose `insert_concurrent` (and `contains`) methods take `&self`. Both use `AtomicU64::fetch_or` with `Ordering::Relaxed` — correct because bit-set operations are idempotent and Bloom filter semantics permit false positives. No `Mutex` is needed.

```rust
use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::core::ConcurrentBloomFilter;
use std::sync::Arc;

let filter = Arc::new(StandardBloomFilter::<String>::new(100_000, 0.01)?);

let handles: Vec<_> = (0..8).map(|t| {
    let f = Arc::clone(&filter);
    std::thread::spawn(move || {
        for i in 0..1_000_usize {
            f.insert_concurrent(&format!("thread-{}-item-{}", t, i));
        }
    })
}).collect();

for h in handles { h.join().unwrap(); }
```

### 3 — Interior mutability (`SharedBloomFilter` trait, `&self`)

`ShardedBloomFilter` and `StripedBloomFilter` implement `SharedBloomFilter`, whose `insert`, `contains`, and `clear` methods take `&self`. Concurrency is managed entirely inside the type. No external wrapper is required.

For concurrent growing workloads, `AtomicScalableBloomFilter` also takes `&self` for inserts, but it implements `BloomFilter` directly (not `SharedBloomFilter`) because it carries its own growth-coordination state.

---

## Type-State Builders

Builders enforce correct parameter ordering at compile time. Calling `.build()` before providing required parameters is a type error, not a runtime panic.

### `StandardBloomFilterBuilder`

```rust
use bloomcraft::builder::StandardBloomFilterBuilder;
use bloomcraft::filters::StandardBloomFilter;

// Type-state: Initial → WithCapacity → Complete
let filter: StandardBloomFilter<String> = StandardBloomFilterBuilder::new()
    .expected_items(100_000)      // required
    .false_positive_rate(0.001)   // required
    .build()?;

// build_with_metadata() returns (filter, FilterMetadata) for capacity planning
let (filter, meta) = StandardBloomFilterBuilder::new()
    .expected_items(100_000)
    .false_positive_rate(0.001)
    .build_with_metadata::<String>()?;

println!("bit count:  {}", meta.filter_size);
println!("hash fns:   {}", meta.num_hashes);
println!("memory:     {} bytes", meta.filter_size / 8);
```

### `ScalableBloomFilterBuilder`

```rust
use bloomcraft::builder::ScalableBloomFilterBuilder;
use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy};

let mut filter: ScalableBloomFilter<String> = ScalableBloomFilterBuilder::new()
    .initial_capacity(1_000)     // required — capacity of first slice
    .false_positive_rate(0.01)   // required — FPR of first slice
    .growth_factor(2.0)          // optional, default 2.0
    .tightening_ratio(0.85)      // optional, default 0.85
    .build()?;

// Capacity planning before first insert:
// FPR upper bound = 0.01 / (1 − 0.85) = 0.067
let (_, meta) = ScalableBloomFilterBuilder::new()
    .initial_capacity(1_000)
    .false_positive_rate(0.01)
    .build_with_metadata::<String>()?;

for n in 0..5 {
    println!(
        "slice {}: capacity={}, FPR={:.4}",
        n, meta.slice_capacity(n), meta.slice_fp_rate(n)
    );
}
println!("FPR upper bound: {:.4}", meta.max_fp_rate_bound);
```

---

## Hash Strategies

All filters accept a `HashStrategy` that controls how `k` bit indices are derived from two 64-bit seeds. The strategy is part of the serialized form; deserializing with a mismatched strategy is rejected.

| Strategy | Formula | Default |
|---|---|---|
| `Double` | `h(i) = h₁ + i·h₂` | — |
| `EnhancedDouble` | `h(i) = h₁ + i·h₂ + (i²+i)/2` | ✓ |
| `Triple` | `h(i) = h₁ + i·h₂ + i²·h₃` | — |

`EnhancedDouble` is the default. It breaks the linear dependency between consecutive indices at the cost of only a triangular-number addition per probe — no additional hash computation. The underlying hash function is swappable via the `BloomHasher` trait:

| Hasher | Feature | Notes |
|---|---|---|
| `StdHasher` | built-in | SipHash-1-3, DoS-resistant |
| `WyHasher` | `wyhash` | ~0.5 cycles/byte, strong avalanche |
| `XxHasher` | `xxhash` | XXH3, widely benchmarked |
| `SimdHasher` | `simd` | AVX2/SSE4.1/NEON vectorized batch hashing |

```rust
use bloomcraft::builder::StandardBloomFilterBuilder;
use bloomcraft::hash::HashStrategy;

let filter = StandardBloomFilterBuilder::new()
    .expected_items(100_000)
    .false_positive_rate(0.01)
    .hash_strategy(HashStrategy::EnhancedDouble)
    .build::<String>()?;
```

---

## Feature Flags

| Flag | Enables |
|---|---|
| `serde` | `Serialize`/`Deserialize` for all filter types; zero-copy binary format |
| `xxhash` | `XxHasher` (XXH3) |
| `wyhash` | `WyHasher` |
| `rayon` | Parallel batch insert / query |
| `simd` | AVX2/SSE4.1/NEON vectorized batch hashing |
| `metrics` | `MetricsCollector`, `FalsePositiveTracker`, `LatencyHistogram` |
| `trace` | `tracing`-compatible span instrumentation; exports `QueryTrace`, `QueryTraceBuilder` |
| `concurrent` | Exports `AtomicScalableBloomFilter` and `AtomicPartitionedBloomFilter` |
| `all-features` | All of the above; used by docs.rs |

```toml
# docs.rs and CI
bloomcraft = { version = "0.1", features = ["all-features"] }
```

### Serialization (`serde` feature)

```rust
#[cfg(feature = "serde")]
{
    use bloomcraft::filters::StandardBloomFilter;

    let mut filter = StandardBloomFilter::<String>::new(1_000, 0.01)?;
    filter.insert(&"hello".to_string());

    // JSON
    let json = serde_json::to_string(&filter)?;
    let restored: StandardBloomFilter<String> = serde_json::from_str(&json)?;

    // Bincode
    let bytes = bincode::serialize(&filter)?;
    let restored: StandardBloomFilter<String> = bincode::deserialize(&bytes)?;

    // Zero-copy: bypasses field-by-field decoding; <1 ms for 1 M-item filters
    use bloomcraft::serde_support::zerocopy::ZeroCopyBloomFilter;
    let zc_bytes = ZeroCopyBloomFilter::serialize(&filter);
    let zc_restored = ZeroCopyBloomFilter::deserialize(&zc_bytes)?;
}
```

Zero-copy format layout (32-byte header + raw `u64` words):

```
[0..4]   Magic       "BLOM"
[4..6]   Version     u16, little-endian
[6..8]   FilterType  u16  (0=Standard, 1=Counting, 2=Scalable)
[8..16]  Size        u64  — total bit count
[16..20] NumHashes   u32
     HashStrat   u8   (0=Double, 1=EnhancedDouble, 2=Triple)
[21..32] Reserved    zeroed
[32..]   Data        raw u64 words, little-endian
```

### Metrics (`metrics` feature)

```rust
#[cfg(feature = "metrics")]
{
    use bloomcraft::metrics::MetricsCollector;
    use std::time::Duration;

    let metrics = MetricsCollector::with_histogram();

    metrics.record_insert();
    metrics.record_query_latency(true, Duration::from_micros(45));
    // Call when you can verify actual membership externally:
    metrics.record_confirmed_query(/*filter_said=*/true, /*actually_present=*/false);

    let snap = metrics.snapshot();
    println!("inserts:     {}",    snap.total_inserts);
    println!("queries/sec: {:.0}", snap.queries_per_second());
    println!("FP rate:     {:.4}", snap.fp_tracker.current_fp_rate);

    // Prometheus text format for scraping
    let prom = snap.to_prometheus_format("bloomcraft");

    // Per-stripe diagnostics for StripedBloomFilter
    use bloomcraft::sync::StripedBloomFilter;
    let striped = StripedBloomFilter::<u64>::with_stripe_count(10_000, 0.01, 64)?;
    let stats = striped.stripe_stats();
    let hot   = striped.most_contended_stripes(5);
}
```

---

## Architecture

```
bloomcraft/
├── src/
│   ├── lib.rs              — public API surface, prelude, feature gates
│   │
│   ├── core/
│   │   ├── filter.rs       — BloomFilter, ConcurrentBloomFilter, SharedBloomFilter traits
│   │   │                     DeletableBloomFilter, MergeableBloomFilter
│   │   ├── bitvec.rs       — AtomicU64-backed bit vector; set/get/union/intersect
│   │   └── params.rs       — optimal_bit_count(), optimal_hash_count(), validate_params()
│   │
│   ├── filters/
│   │   ├── standard.rs         — StandardBloomFilter  (AtomicU64; ConcurrentBloomFilter)
│   │   ├── counting.rs         — CountingBloomFilter  (4/8/16-bit counters; DeletableBloomFilter)
│   │   ├── scalable.rs         — ScalableBloomFilter, GrowthStrategy, CapacityExhaustedBehavior,
│   │   │                         QueryStrategy, ScalableHealthMetrics, HyperLogLog,
│   │   │                         AtomicScalableBloomFilter (sub-module, feature: concurrent)
│   │   ├── partitioned.rs      — PartitionedBloomFilter (cache-line partitioned, &mut self)
│   │   ├── atomic_partitioned/ — AtomicPartitionedBloomFilter (feature: concurrent; ConcurrentBloomFilter)
│   │   ├── register_blocked.rs — RegisterBlockedBloomFilter (512-bit blocks; 1 cache miss/query)
│   │   ├── tree.rs             — TreeBloomFilter, TreeConfig, TreeStats, LocateIter
│   │   ├── classic_bits.rs     — ClassicBitsFilter    (Bloom 1970 Method 2)
│   │   └── classic_hash.rs     — ClassicHashFilter    (Bloom 1970 Method 1)
│   │
│   ├── sync/
│   │   ├── sharded.rs      — ShardedBloomFilter   (SharedBloomFilter; lock-free atomic shards)
│   │   └── striped.rs      — StripedBloomFilter   (SharedBloomFilter; RwLock striping,
│   │                                               Lemire range reduction, 64-byte padding)
│   │
│   ├── builder/
│   │   ├── standard.rs     — StandardBloomFilterBuilder  (type-state)
│   │   ├── counting.rs     — CountingBloomFilterBuilder  (type-state)
│   │   └── scalable.rs     — ScalableBloomFilterBuilder  (type-state + growth metadata)
│   │
│   ├── hash/
│   │   ├── mod.rs          — BloomHasher trait, HashStrategy enum
│   │   ├── strategies.rs   — DoubleHashing, EnhancedDoubleHashing, TripleHashing
│   │   ├── std_hasher.rs   — StdHasher  (SipHash-1-3)
│   │   ├── wyhash.rs       — WyHasher   (feature: wyhash)
│   │   ├── xxhash.rs       — XxHasher   (feature: xxhash)
│   │   └── simd.rs         — SimdHasher (feature: simd; AVX2/SSE4.1/NEON)
│   │
│   ├── metrics/            — (feature: metrics)
│   │   ├── collector.rs    — MetricsCollector, FilterMetrics snapshot
│   │   ├── tracker.rs      — FalsePositiveTracker, sliding-window FPR
│   │   └── histogram.rs    — LatencyHistogram, percentile queries
│   │
│   ├── serde_support/      — (feature: serde)
│   │   ├── standard.rs     — Serialize/Deserialize for StandardBloomFilter
│   │   ├── counting.rs     — Serialize/Deserialize for CountingBloomFilter
│   │   └── zerocopy.rs     — ZeroCopyBloomFilter; fixed header + raw u64 words
│   │
│   └── error.rs            — BloomCraftError enum; typed constructors
│
├── benches/
├── examples/
│   └── capacity_planning.rs
└── Cargo.toml
```

### Trait Hierarchy

```
BloomFilter<T>                         ← all twelve filter types
├── DeletableBloomFilter<T>            ← CountingBloomFilter
└── MergeableBloomFilter<T>            ← StandardBloomFilter

ConcurrentBloomFilter<T>               ← StandardBloomFilter, AtomicPartitionedBloomFilter
                                         extends BloomFilter; adds insert_concurrent(&self)

SharedBloomFilter<T>                   ← ShardedBloomFilter, StripedBloomFilter
                                         distinct trait; all methods take &self
```

`ConcurrentBloomFilter` extends `BloomFilter` — the filter supports both `&mut self` (single-threaded) and `&self` (atomic) paths. `SharedBloomFilter` is a wholly separate trait for filters whose entire public surface is `&self`; they have no single-threaded `&mut self` mode. The two cannot be merged without losing these type-level guarantees.

`AtomicScalableBloomFilter` implements `BloomFilter` directly (not `ConcurrentBloomFilter` or `SharedBloomFilter`) because its growth-coordination machinery — a `RwLock`-guarded filter list plus `AtomicBool` CAS for growth election — does not fit either of the simpler concurrent patterns.

---

## Benchmarks

```bash
# All suites
cargo bench

# Specific targets
cargo bench insert
cargo bench query
cargo bench hash_functions
cargo bench comparison            # all filter variants side-by-side
cargo bench concurrent            # multi-threaded throughput scaling
cargo bench memory                # allocation and footprint
cargo bench striped               # StripedBloomFilter stripe-count scaling
cargo bench sharded_bloom_filter
cargo bench register_blocked      # RegisterBlockedBloomFilter vs partitioned
cargo bench atomic_partitioned    # AtomicPartitionedBloomFilter scaling
```

HTML reports are written to `target/criterion/`.

**Space efficiency reference** — `m = −n·ln(p) / (ln 2)²`, `k = (m/n)·ln 2`:

| Target FPR | Bits / element | Memory for 1 M items |
|---|---|---|
| 10%   | ~4.8  | ~600 KB |
| 1%    | ~9.6  | ~1.2 MB |
| 0.1%  | ~14.4 | ~1.8 MB |
| 0.01% | ~19.2 | ~2.4 MB |

---

## References

- Bloom, B. H. (1970). *Space/time trade-offs in hash coding with allowable errors*. CACM.
- Kirsch, A. & Mitzenmacher, M. (2006). *Less Hashing, Same Performance: Building a Better Bloom Filter*. ESA.
- Almeida, P. et al. (2007). *Scalable Bloom Filters*. Information Processing Letters.
- Putze, F. et al. (2009). *Cache-, hash- and space-efficient Bloom filters*. JEA.
- Flajolet, P. et al. (2007). *HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm*. DMTCS.
- Lemire, D. (2016). *A fast alternative to the modulo reduction*. arXiv:1805.10941.

---

## Contributing

Bug reports, API feedback, and pull requests are welcome.
Before opening a PR, please read the contribution guide:

- **Issues** — [github.com/ZaudRehman/BloomCraft/issues](https://github.com/ZaudRehman/BloomCraft/issues)
  Open a ticket for bugs, missing filter variants, API concerns, or documentation gaps.
  Label your issue `bug`, `enhancement`, or `question` accordingly.

- **Pull requests** — Target the `main` branch.
  All public API additions require:
  - Doc comments (`///`) with at least one runnable example
  - Unit tests covering the happy path and at least one edge case
  - An entry in `CHANGELOG.md` under `[Unreleased]`

- **Unsafe code** — Any new `unsafe` block must include an explicit `// SAFETY:` comment
  justifying why the operation is sound. PRs that add `unsafe` without justification
  will not be merged.

- **MSRV** — BloomCraft targets Rust 1.70. Do not use features stabilized after that version
  without updating the badge, `Cargo.toml`, and CI matrix.

---

## Changelog

All notable changes are documented in [CHANGELOG.md](CHANGELOG.md) following
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions and
[Semantic Versioning](https://semver.org/).

---

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
