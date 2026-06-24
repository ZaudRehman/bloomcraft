# BloomCraft

[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/bloomcraft.svg)](https://crates.io/crates/bloomcraft)
[![Documentation](https://docs.rs/bloomcraft/badge.svg)](https://docs.rs/bloomcraft)

A production-grade Bloom filter library for Rust. BloomCraft provides twelve filter variants, from the classical space-optimal filter to scalable, partitioned, register-blocked, and concurrent implementations, unified under a coherent trait hierarchy with type-state builders, pluggable hash strategies, and optional Serde, metrics, and SIMD support.

## Why BloomCraft?

Most Rust Bloom filter crates ship one or two variants behind a single trait. BloomCraft ships twelve, covering every practical trade-off between space, speed, deletion, scalability, and concurrency, all under a single, coherent API featuring:

* **Three distinct concurrency models:** `&mut self` with external locking, `&self` lock-free operations via `AtomicU64` CAS, and `&self` wait-free operations via interior mutability.
* **Type-state builders:** Misconfiguration is a compile-time error, not a runtime panic.
* **Pluggable hash strategies:** From standard SipHash to SIMD-accelerated WyHash and XXH3.
* **Zero Unsafe Code:** The entire library is implemented in 100% safe Rust.

If you need a filter you can delete from, one that grows without bounds, one that saturates a single cache line per query, or one that accepts concurrent writes from 64 threads without a Mutex in sight, this crate provides a specific, mathematically-verified type for your requirement rather than bolting synchronization onto a generic struct.

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
- [License](#license)

---

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
bloomcraft = "0.1"
```

To enable optional features like Serde serialization, fast hashing, or concurrency:

```toml
[dependencies]
bloomcraft = { version = "0.1", features = ["serde", "wyhash", "metrics", "concurrent"] }
```

**Minimum Supported Rust Version (MSRV):** 1.70.0

---

## Quick Start

```rust
use bloomcraft::core::BloomFilter;
use bloomcraft::filters::StandardBloomFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut filter = StandardBloomFilter::<String>::new(10_000, 0.01)?;

    filter.insert(&"hello".to_string());
    filter.insert(&"world".to_string());

    assert!(filter.contains(&"hello".to_string()));   // Definitely present
    assert!(!filter.contains(&"rust".to_string()));   // Definitely absent (with high probability)
    
    Ok(())
}
```

---

## Filter Selection

| Use Case | Filter | Feature Gate | Notes |
|---|---|---|---|
| General purpose, known capacity | `StandardBloomFilter` | *always* | Optimal space, supports union/intersect |
| Need deletion | `CountingBloomFilter` | *always* | 4–16× memory overhead, per-element counters |
| Unknown or growing dataset | `ScalableBloomFilter` | *always* | Auto-grows, overall FPR stays bounded |
| Concurrent, growing dataset | `AtomicScalableBloomFilter` | `concurrent` | Lock-free inserts, sharded internals |
| Query-heavy, cache-sensitive | `PartitionedBloomFilter` | *always* | 2–4× faster queries, cache-line aligned |
| Ultra-fast queries, FPR-tolerant| `RegisterBlockedBloomFilter`| *always* | 512-bit AVX blocks, 20–30% faster, 2–3× FPR |
| Concurrent, cache-optimized | `AtomicPartitionedBloomFilter`| `concurrent` | Wait-free inserts + cache-line partitions |
| Multi-level / location-aware | `TreeBloomFilter` | *always* | Returns which subtree contains an item |
| High-concurrency writes | `ShardedBloomFilter` | *always* | `&self` insert via lock-free atomic shards |
| High-concurrency, low-memory | `StripedBloomFilter` | *always* | Fine-grained `RwLock` striping, `&self` |
| Historical / research reference | `ClassicHashFilter` | *always* | Bloom 1970 Method 1 |
| Historical / research reference | `ClassicBitsFilter` | *always* | Bloom 1970 Method 2 |

### Concurrency Quick-Reference

| Filter | Ownership for Insert | Synchronization Mechanism |
|---|---|---|
| `StandardBloomFilter` | `&mut self` or `&self` | Atomic CAS on `AtomicU64` words |
| `CountingBloomFilter` | `&mut self` | Requires external `Mutex` |
| `ScalableBloomFilter` | `&mut self` | Requires external `Mutex` |
| `AtomicScalableBloomFilter` | `&self` | Lock-free shards + `RwLock` for growth |
| `PartitionedBloomFilter` | `&mut self` | Requires external `RwLock` |
| `RegisterBlockedBloomFilter` | `&mut self` | Requires external `Mutex` |
| `AtomicPartitionedBloomFilter`| `&self` | Wait-free `AtomicU64` `fetch_or` |
| `TreeBloomFilter` | `&mut self` | Requires external `RwLock` |
| `ShardedBloomFilter` | `&self` | Lock-free atomic shards |
| `StripedBloomFilter` | `&self` | Striped `RwLock` array |

---

## Filter Variants

### StandardBloomFilter

The classic space-optimal Bloom filter. The underlying bit array is backed by `AtomicU64` words, enabling wait-free concurrent writes via the `ConcurrentBloomFilter` extension trait without requiring an external lock.

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

Extends the standard filter with per-slot counters, enabling safe deletion. Slots can be configured to 4, 8, or 16 bits. Overflow events are tracked to allow callers to detect correctness degradation safely.

```rust
use bloomcraft::core::{BloomFilter, DeletableBloomFilter};
use bloomcraft::filters::CountingBloomFilter;

let mut filter = CountingBloomFilter::<String>::new(10_000, 0.01)?;
let item = "item".to_string();

filter.insert(&item);
assert!(filter.contains(&item));

// Returns true if the item was successfully removed
let removed = filter.delete(&item);
assert!(removed);
assert!(!filter.contains(&item));

println!("has overflowed: {}", filter.has_overflowed());
```

### ScalableBloomFilter

Maintains a chain of fixed-size filter slices. When the active slice's fill ratio exceeds `fill_threshold` (default 0.5 — mathematically proven optimal), a new slice is appended with scaled capacity and tightened FPR. The compound FPR remains bounded.

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

Divides the bit array into `k` equal partitions. Each hash function probes within a single partition, confining memory access to a single cache line. This eliminates cross-partition false sharing and delivers 2–4× higher lookup throughput.

```rust
use bloomcraft::core::BloomFilter;
use bloomcraft::filters::PartitionedBloomFilter;

let mut filter = PartitionedBloomFilter::<String>::with_alignment(10_000, 0.01, 64)?;

filter.insert(&"item".to_string());
println!("partitions: {}", filter.partition_count());
```

### TreeBloomFilter

A hierarchical filter mapping items to leaf nodes in a configurable tree. Designed for location-aware lookups across tiered storage systems (e.g., region → datacenter → rack).

```rust
use bloomcraft::core::BloomFilter;
use bloomcraft::filters::TreeBloomFilter;

// 2 levels: 4 regions × 8 datacenters = 32 leaf bins
let mut router = TreeBloomFilter::<String>::new(vec!, 1_000, 0.01)?;[1][2]
let item = "session:alice".to_string();

router.insert_to_bin(&item, &)?;[3][4]

// Query specific bin
assert!(router.contains_in_bin(&item, &));[4][3]
assert!(!router.contains_in_bin(&item, &));[5]

// Locate all possible paths
for path in router.locate(&item) {
    println!("may be at: {:?}", path); 
}
```

---

## Concurrency Models

BloomCraft ensures thread-safety at the type system level, offering three explicit synchronization paradigms.

### 1. Single-Threaded / External Lock (`&mut self`)
Standard ownership rules. Wrap in `Arc<Mutex<T>>` for multi-threaded access. Applies to `CountingBloomFilter`, `PartitionedBloomFilter`, etc.

### 2. Lock-Free Atomic (`&self` via `ConcurrentBloomFilter`)
Uses atomic `fetch_or` operations with `Ordering::Relaxed` for wait-free insertions. Zero locking overhead.

```rust
use bloomcraft::core::ConcurrentBloomFilter;
use bloomcraft::filters::StandardBloomFilter;
use std::sync::Arc;

let filter = Arc::new(StandardBloomFilter::<u64>::new(100_000, 0.01)?);

let f = Arc::clone(&filter);
std::thread::spawn(move || {
    f.insert_concurrent(&42); // Wait-free
});
```

### 3. Interior Mutability (`&self` via `SharedBloomFilter`)
Applies to `ShardedBloomFilter` and `StripedBloomFilter`. Concurrency is managed entirely within the type using atomic shards or padded `RwLock` striping (to prevent false sharing).

---

## Type-State Builders

Builders enforce correct parameter ordering and presence at compile time, eliminating runtime panics from missing configurations.

```rust
use bloomcraft::builder::StandardBloomFilterBuilder;
use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::hash::HashStrategy;

let (filter, meta) = StandardBloomFilterBuilder::new()
    .expected_items(100_000)      // Required
    .false_positive_rate(0.001)   // Required
    .hash_strategy(HashStrategy::EnhancedDouble) // Optional
    .build_with_metadata::<String>()?;

println!("Memory footprint: {} bytes", meta.memory_bytes());
```

---

## Hash Strategies

All filters use Lemire's unbiased range reduction and support pluggable strategies to map two 64-bit seeds to `k` indices. 

*   `Double`: `h(i) = h₁ + i·h₂`
*   `EnhancedDouble` (Default): `h(i) = h₁ + i·h₂ + (i²+i)/2`
*   `Triple`: `h(i) = h₁ + i·h₂ + i²·h₃`

Underlying hashers are configured via feature flags: `StdHasher` (default, SipHash-1-3), `WyHasher` (fast avalanche), and `XxHasher` (XXH3).

---

## Feature Flags

| Flag | Description |
|---|---|
| `serde` | Implements `Serialize`/`Deserialize` and provides zero-copy binary format |
| `xxhash` | Enables `XxHasher` (XXH3) |
| `wyhash` | Enables `WyHasher` |
| `rayon` | Enables parallel batch insert / query operations |
| `simd` | Enables AVX2/SSE4.1/NEON vectorized batch hashing |
| `metrics` | Enables `MetricsCollector`, `FalsePositiveTracker`, and `LatencyHistogram` |
| `trace` | Enables `tracing`-compatible span instrumentation |
| `concurrent` | Enables `AtomicScalableBloomFilter` and `AtomicPartitionedBloomFilter` |

---

## Architecture

```text
bloomcraft/
├── src/
│   ├── core/           # Traits (BloomFilter, ConcurrentBloomFilter), BitVec, Math Params
│   ├── filters/        # Core implementations (Standard, Counting, Scalable, Partitioned, etc.)
│   ├── sync/           # ShardedBloomFilter, StripedBloomFilter
│   ├── builder/        # Type-state builders
│   ├── hash/           # BloomHasher trait, HashStrategies, Hasher implementations
│   ├── metrics/        # Telemetry, latencies, FPR tracking
│   ├── serde_support/  # Serialization formats, ZeroCopy bindings
│   └── error.rs        # Centralized BloomCraftError enum
```

---

## Benchmarks

Run the benchmark suites using Criterion:

```bash
cargo bench                      # Run all suites
cargo bench comparison           # Compare filter variants
cargo bench concurrent           # Multi-threaded throughput scaling
cargo bench register_blocked     # Register-blocked vs Partitioned cache misses
```

**Memory Efficiency Reference** (`m = −n·ln(p) / (ln 2)²`):

| Target FPR | Bits / Element | Memory for 1,000,000 items |
|---|---|---|
| 10% | ~4.8 | ~600 KB |
| 1% | ~9.6 | ~1.2 MB |
| 0.1% | ~14.4 | ~1.8 MB |
| 0.01% | ~19.2 | ~2.4 MB |

---

## References

1. Bloom, B. H. (1970). *Space/time trade-offs in hash coding with allowable errors*. CACM.
2. Kirsch, A. & Mitzenmacher, M. (2006). *Less Hashing, Same Performance: Building a Better Bloom Filter*. ESA.
3. Almeida, P. et al. (2007). *Scalable Bloom Filters*. Information Processing Letters.
4. Lemire, D. (2016). *A fast alternative to the modulo reduction*.

---

## Contributing

Bug reports, API feedback, and pull requests are welcome. 

*   **Issues:** Please label your issue `bug`, `enhancement`, or `question`.
*   **Pull Requests:** Target the `main` branch. Ensure code includes documentation (`///`), unit tests, and a `CHANGELOG.md` entry.
*   **Unsafe Code:** BloomCraft strictly adheres to a **ZERO `unsafe`** engineering policy. Pull requests introducing `unsafe` code will not be accepted. All optimizations and memory layouts must be mathematically provable and expressible in 100% safe Rust.
*   **MSRV:** Do not use features stabilized after Rust 1.70.0 without prior coordination.

---

## License

Licensed under either of [MIT License](LICENSE-MIT) or [Apache License, Version 2.0](LICENSE-APACHE) at your option.
