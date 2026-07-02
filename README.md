# BloomCraft

[![Rust Version](https://img.shields.io/badge/rust-1.73%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/bloomcraft.svg)](https://crates.io/crates/bloomcraft)
[![docs.rs](https://docs.rs/bloomcraft/badge.svg)](https://docs.rs/bloomcraft)

A production-grade Bloom filter library for Rust. BloomCraft provides twelve filter variants, from the classical space-optimal filter to scalable, partitioned, register-blocked, and concurrent implementations, unified under a coherent trait hierarchy with type-state builders, pluggable hash strategies, and optional Serde, metrics, and SIMD support.

## Why BloomCraft?

BloomCraft ships twelve, covering every practical trade-off between space, speed, deletion, scalability, and concurrency, all under a single, coherent API featuring:

* **Three distinct concurrency models:** `&mut self` with external locking, `&self` lock-free operations via `AtomicU64` CAS, and `&self` wait-free operations via interior mutability.
* **Type-state builders:** Misconfiguration is a compile-time error, not a runtime panic.
* **Pluggable hash strategies:** From standard SipHash to SIMD-accelerated WyHash and XXH3.
* **Audited unsafe internals:** The public API stays safe; any internal `unsafe` is tightly scoped, documented, and reviewed.

If you need a filter you can delete from, one that grows without bounds, one that saturates a single cache line per query, or one that accepts concurrent writes from 64 threads without a Mutex in sight, this crate provides a specific, mathematically-verified type for your requirement rather than bolting synchronization onto a generic struct.

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
- [Contact](#contact)
- [Contributing](#contributing)
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

**Minimum Supported Rust Version (MSRV):** 1.73

---

## Quick Start

```rust
use bloomcraft::core::BloomFilter;
use bloomcraft::filters::StandardBloomFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut filter = StandardBloomFilter::<String>::new(10_000, 0.01)?;

    filter.insert(&"hello".to_string());
    filter.insert(&"world".to_string());

    assert!(filter.contains(&"hello".to_string()));
    assert!(!filter.contains(&"rust".to_string()));

    Ok(())
}
```

---

## Filter Selection

| Use case | Filter | Feature gate | Notes |
|---|---|---|---|
| General purpose, known capacity | `StandardBloomFilter` | always | Space-optimal, supports union/intersect |
| Need deletion | `CountingBloomFilter` | always | 4-16x memory overhead, per-element counters |
| Unknown or growing dataset | `ScalableBloomFilter` | always | Auto-grows, bounded compound FPR |
| Concurrent, growing dataset | `AtomicScalableBloomFilter` | `concurrent` | Sharded internals, CAS-based growth |
| Query-heavy, cache-sensitive | `PartitionedBloomFilter` | always | Partitioned bit array, cache-aligned |
| High throughput, FPR-tolerant | `RegisterBlockedBloomFilter` | always | 512-bit register blocks, one cache-line touch per query |
| Concurrent, cache-optimized | `AtomicPartitionedBloomFilter` | `concurrent` | Atomic partitioned filter |
| Location-aware queries | `TreeBloomFilter` | always | Hierarchical bins, returns matching subtree |
| High-concurrency writes | `ShardedBloomFilter` | always | `&self` insert via atomic shards |
| High-concurrency, low memory | `StripedBloomFilter` | always | Striped `RwLock`, `&self` |
| Educational baseline | `ClassicHashFilter` | always | Bloom (1970) Method 1 |
| Educational baseline | `ClassicBitsFilter` | always | Bloom (1970) Method 2 |

### Concurrency quick-reference

| Filter | Insert requires | Mechanism |
|---|---|---|
| `StandardBloomFilter` | `&mut self` or `&self` | Atomic CAS on `AtomicU64` |
| `CountingBloomFilter` | `&mut self` | External `Mutex` |
| `ScalableBloomFilter` | `&mut self` | External `Mutex` |
| `AtomicScalableBloomFilter` | `&self` | Shards + `RwLock` for growth |
| `PartitionedBloomFilter` | `&mut self` | External `RwLock` |
| `RegisterBlockedBloomFilter` | `&mut self` | External `Mutex` |
| `AtomicPartitionedBloomFilter` | `&self` | `AtomicU64` `fetch_or` |
| `TreeBloomFilter` | `&mut self` | External `RwLock` |
| `ShardedBloomFilter` | `&self` | Atomic shards |
| `StripedBloomFilter` | `&self` | Striped `RwLock` array |

---

## Filter Variants

### StandardBloomFilter

Classic space-optimal Bloom filter backed by `AtomicU64` words. Supports
`&self` concurrent writes via the `ConcurrentBloomFilter` extension trait.

```rust
use bloomcraft::core::{BloomFilter, MergeableBloomFilter};
use bloomcraft::filters::StandardBloomFilter;

let mut filter_a = StandardBloomFilter::<&str>::new(50_000, 0.001)?;
let mut filter_b = StandardBloomFilter::<&str>::new(50_000, 0.001)?;

filter_a.insert(&"key");

println!("bits: {}", filter_a.bit_count());
println!("hash fns: {}", filter_a.hash_count());
println!("estimated FPR: {:.6}", filter_a.estimate_fpr());

let union = filter_a.union(&filter_b)?;
```

### CountingBloomFilter

Extends the standard filter with per-slot counters for safe deletion.
Counter width is configurable to 4, 8, or 16 bits per slot.

```rust
use bloomcraft::core::{BloomFilter, DeletableBloomFilter};
use bloomcraft::filters::CountingBloomFilter;

let mut filter = CountingBloomFilter::<String>::new(10_000, 0.01);
let item = "item".to_string();

filter.insert(&item);
assert!(filter.contains(&item));

let removed = filter.delete(&item);
assert!(removed);
assert!(!filter.contains(&item));

println!("has overflowed: {}", filter.has_overflowed());
```

### ScalableBloomFilter

Maintains a chain of fixed-size filter slices. When the active slice exceeds
`fill_threshold` (default 0.5), a new slice is appended with scaled capacity
and tightened FPR. The compound FPR across all slices remains bounded.

```rust
use bloomcraft::core::BloomFilter;
use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy};

let mut filter = ScalableBloomFilter::<u64>::with_strategy(
    1_000, 0.01, 0.5,
    GrowthStrategy::Adaptive {
        initial_ratio: 0.5,
        min_ratio: 0.3,
        max_ratio: 0.9,
    },
)?;

for i in 0..50_000_u64 {
    filter.insert(&i);
}

let metrics = filter.health_metrics();
println!("slices: {}", metrics.filter_count);
println!("FPR upper bound: {:.6}", metrics.max_fpr);
```

### PartitionedBloomFilter

Divides the bit array into `k` equal partitions. Each hash function probes
within one partition, keeping memory access local to a single cache line.

```rust
use bloomcraft::core::BloomFilter;
use bloomcraft::filters::PartitionedBloomFilter;

let mut filter = PartitionedBloomFilter::<String>::with_alignment(
    10_000, 0.01, 64,
)?;

filter.insert(&"item".to_string());
println!("partitions: {}", filter.partition_count());
```

### RegisterBlockedBloomFilter

Uses 512-bit register blocks so each query touches exactly one cache line.
Throughput is higher than `StandardBloomFilter` at the cost of a higher FPR
for a given memory budget (the block-aligned layout wastes bits at block
boundaries).

```rust
use bloomcraft::core::BloomFilter;
use bloomcraft::filters::RegisterBlockedBloomFilter;

let mut filter = RegisterBlockedBloomFilter::<u64>::new(100_000, 0.01)?;

filter.insert(&42);
assert!(filter.contains(&42));
```

### TreeBloomFilter

A hierarchical filter that assigns items to leaf bins in a branching tree.
Useful for location-aware lookups across tiered storage (e.g., region to
datacenter to rack).

```rust
use bloomcraft::core::BloomFilter;
use bloomcraft::filters::TreeBloomFilter;

let mut router = TreeBloomFilter::<String>::new(
    vec![4, 8], 1_000, 0.01,
)?;

let item = "session:alice".to_string();

router.insert_to_bin(&item, &[2, 5]);

assert!(router.contains_in_bin(&item, &[2, 5]));

for path in router.locate(&item) {
    println!("may be at: {:?}", path);
}
```

### AtomicScalableBloomFilter (feature: `concurrent`)

Concurrent variant of `ScalableBloomFilter` using sharded sub-filters and
CAS-based growth election. All operations take `&self`.

```rust
use bloomcraft::filters::AtomicScalableBloomFilter;
use std::sync::Arc;

let filter = Arc::new(
    AtomicScalableBloomFilter::<u64>::new(1_000, 0.01)?,
);

let f = Arc::clone(&filter);
std::thread::spawn(move || {
    f.insert(&42);
});
```

### AtomicPartitionedBloomFilter (feature: `concurrent`)

Concurrent partitioned filter using `AtomicU64` for lock-free bit operations
on cache-line-aligned partitions.

```rust
use bloomcraft::filters::AtomicPartitionedBloomFilter;
use std::sync::Arc;

let filter = Arc::new(
    AtomicPartitionedBloomFilter::<u64>::new(100_000, 0.01)?,
);

let f = Arc::clone(&filter);
std::thread::spawn(move || {
    f.insert(&42);
});
```

### ShardedBloomFilter

Distributes items across independent `StandardBloomFilter` shards. Each shard
is lock-free; shards are selected by hash. Good for high-write-throughput
workloads.

### StripedBloomFilter

A single logical filter striped into `RwLock`-protected regions. Provides
`&self` operations with finer-grained locking than a single `Mutex`.

### ClassicBitsFilter / ClassicHashFilter

Implementations of Bloom's original 1970 paper. Method 1 (`ClassicBitsFilter`)
and Method 2 (`ClassicHashFilter`). Provided as educational baselines and
research references. Not recommended for production use.

---

## Concurrency Models

BloomCraft provides three synchronization models, distinguished by the insert
method's `self` type.

### 1. Single-threaded / external lock (`&mut self`)

Standard Rust ownership. Wrap in `Arc<Mutex<T>>` for multi-threaded access.
Applies to `CountingBloomFilter`, `PartitionedBloomFilter`, etc.

### 2. Atomic operations (`&self` via `ConcurrentBloomFilter`)

Uses atomic `fetch_or` with `Ordering::Relaxed` on `AtomicU64` words. Applies
to `StandardBloomFilter` (which implements `ConcurrentBloomFilter`).

```rust
use bloomcraft::core::ConcurrentBloomFilter;
use bloomcraft::filters::StandardBloomFilter;
use std::sync::Arc;

let filter = Arc::new(StandardBloomFilter::<u64>::new(100_000, 0.01)?);

let f = Arc::clone(&filter);
std::thread::spawn(move || {
    f.insert_concurrent(&42);
});
```

### 3. Interior mutability (`&self` via `SharedBloomFilter`)

Concurrency managed within the type using atomic shards (`ShardedBloomFilter`)
or padded `RwLock` striping (`StripedBloomFilter`).

---

## Type-State Builders

Builders enforce correct parameter ordering at compile time, eliminating
runtime panics from missing or misordered configuration.

```rust
use bloomcraft::builder::StandardBloomFilterBuilder;
use bloomcraft::hash::HashStrategy;

let (filter, meta) = StandardBloomFilterBuilder::new()
    .expected_items(100_000)
    .false_positive_rate(0.001)
    .hash_strategy(HashStrategy::EnhancedDouble)
    .build_with_metadata::<String>()?;

println!("Memory footprint: {} bytes", meta.memory_bytes());
```

---

## Hash Strategies

All filters use Lemire's unbiased range reduction. The hash strategies map
two 64-bit seeds to `k` indices:

| Strategy | Formula | Notes |
|---|---|---|
| `Double` | `h(i) = h1 + i * h2` | Kirsch-Mitzenmacher (2006) |
| `EnhancedDouble` (default) | `h(i) = h1 + i * h2 + (i^2 + i) / 2` | Better uniformity |
| `Triple` | `h(i) = h1 + i * h2 + i^2 * h3` | Maximum independence |

Underlying hashers are configured via feature flags:

| Hasher | Feature | Algorithm |
|---|---|---|
| `StdHasher` | (default) | SipHash-1-3 |
| `WyHasher` | `wyhash` | WyHash |
| `XxHasher` | `xxhash` | XXH3 |

---

## Feature Flags

| Flag | Description |
|---|---|
| `serde` | `Serialize`/`Deserialize` for all filter types, plus zero-copy binary format |
| `bincode` | Bincode encoding (implies `serde`) |
| `xxhash` | `XxHasher` (XXH3) |
| `wyhash` | `WyHasher` |
| `rayon` | Parallel batch insert / query |
| `simd` | AVX2 / SSE4.1 / NEON vectorized batch hashing |
| `metrics` | `MetricsCollector`, `FalsePositiveTracker`, `LatencyHistogram` |
| `trace` | Per-query `QueryTrace` timing instrumentation |
| `concurrent` | `AtomicScalableBloomFilter`, `AtomicPartitionedBloomFilter` |
| `proptest` | Property-based test utilities |

---

## Architecture

```
src/
 core/           Traits (BloomFilter, ConcurrentBloomFilter), BitVec, math
 filters/        Core filter implementations
 sync/           ShardedBloomFilter, StripedBloomFilter
 builder/        Type-state builders
 hash/           BloomHasher trait, hash strategies, hasher impls
 metrics/        Telemetry, latency histograms, FPR tracking
 serde_support/  Serialization formats, zero-copy bindings
 error.rs        BloomCraftError enum
```

---

## Benchmarks

Benchmarks use Criterion and live in `benches/`:

```bash
cargo bench --bench standard_bench            # StandardBloomFilter throughput
cargo bench --bench counting_bench            # CountingBloomFilter operations
cargo bench --bench scalable_bench            # ScalableBloomFilter under growth
cargo bench --bench register_blocked_bench    # Register-blocked throughput & comparisons
cargo bench --bench partitioned_bench         # Partitioned filter performance
cargo bench --bench tree_bench                # TreeBloomFilter queries
cargo bench --bench sharded_bench             # ShardedBloomFilter concurrency scaling
cargo bench --bench atomic_scalable_bench     # AtomicScalableBloomFilter (requires --features concurrent)
cargo bench --bench atomic_partitioned_bench  # AtomicPartitionedBloomFilter (requires --features concurrent)
cargo bench --bench historical_bench          # Hash strategy & historical comparisons
```

### Memory efficiency reference

For a standard Bloom filter, the optimal bit count per item is
`m = -n * ln(p) / (ln 2)^2`:

| Target FPR | Bits per element | Memory for 1,000,000 items |
|---|---|---|
| 10% | ~4.8 | ~600 KB |
| 1% | ~9.6 | ~1.2 MB |
| 0.1% | ~14.4 | ~1.8 MB |
| 0.01% | ~19.2 | ~2.4 MB |

Note: `RegisterBlockedBloomFilter` and partitioned variants deviate from
optimal memory due to alignment constraints. The actual FPR for a given
capacity is slightly higher than the target. Run `cargo bench --bench
register_blocked_bench -- rbbf/fpr_targets` to measure the gap.

---

## References

The academic papers that informed BloomCraft's design are listed below and are also preserved in the [`references/`](references/) directory for convenient browsing.

1. Bloom, B. H. (1970). *Space/time trade-offs in hash coding with allowable errors*. Communications of the ACM.
2. Kirsch, A. & Mitzenmacher, M. (2006). *Less Hashing, Same Performance: Building a Better Bloom Filter*. ESA.
3. Almeida, P. et al. (2007). *Scalable Bloom Filters*. Information Processing Letters.
4. Lemire, D. (2019). *Fast Random Integer Generation in an Interval*. ACM Transactions on Modeling and Computer Simulation.

---

## Contact

- Security reports: zaudrehman@gmail.com
- General contribution questions: open a GitHub issue

---

## Contributing

Bug reports, API feedback, and pull requests are welcome.

- **Issues:** Label as `bug`, `enhancement`, or `question`.
- **Pull requests:** Target `main`. Include documentation (`///`), tests, and a `CHANGELOG.md` entry.
- **Unsafe code:** The library uses `unsafe` in limited, audited locations
  (primarily SIMD intrinsics, manual allocation in `PartitionedBloomFilter`,
  and `Send`/`Sync` impls for concurrent types). Pull requests introducing new
  `unsafe` must include a safety comment explaining invariants and
  preconditions.
- **MSRV:** Do not use features stabilized after Rust 1.73 without prior
  coordination.

---

## License

Licensed under either of [MIT License](LICENSE-MIT) or
[Apache License, Version 2.0](LICENSE-APACHE) at your option.
