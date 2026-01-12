# BloomCraft 🌸

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-green.svg)](https://docs.rs/bloomcraft)

**BloomCraft** is a production-grade Rust library implementing multiple variants of Bloom filters — probabilistic data structures for approximate set membership queries. Designed to be indistinguishable from infrastructure code shipped by Principal Engineers at FAANG-scale systems.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Filter Variants](#filter-variants)
- [Hash Functions](#hash-functions)
- [Concurrent Filters](#concurrent-filters)
- [Serialization](#serialization)
- [Metrics & Observability](#metrics--observability)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Testing & Benchmarking](#testing--benchmarking)
- [Use Cases](#use-cases)
- [License](#license)

---

## Overview

A **Bloom filter** is a space-efficient probabilistic data structure that tests whether an element is a member of a set. It can have false positives (saying an element is in the set when it isn't) but never false negatives (if it says an element is not in the set, it definitely isn't).

BloomCraft provides:
- **7 filter variants** for different use cases (including 2 historical implementations)
- **Thread-safe concurrent filters** with lock-free and striped-locking options
- **Pluggable hash functions** including SIMD-optimized implementations
- **Comprehensive serialization** with zero-copy support
- **Production-grade observability** with metrics collection and false positive tracking
- **Zero unsafe code** in core modules (compile-time enforced)

---

## Features

### Filter Variants

| Variant | Description | Use Case | Deletion |
|---------|-------------|----------|----------|
| `StandardBloomFilter` | Classic space-optimal implementation | General purpose | ❌ |
| `CountingBloomFilter` | Supports element deletion via 4/8-bit counters | Dynamic sets | ✅ |
| `ScalableBloomFilter` | Grows dynamically as items are added | Unknown capacity | ❌ |
| `PartitionedBloomFilter` | Cache-line optimized for better performance | High-performance | ❌ |
| `HierarchicalBloomFilter` | Multi-level filtering for tiered storage | Distributed systems | ❌ |
| `ClassicBitsFilter` | Burton Bloom's Method 2 (1970) - bit array | Educational | ❌ |
| `ClassicHashFilter` | Burton Bloom's Method 1 (1970) - hash table | Educational | ❌ |

### Concurrent Filters

| Variant | Synchronization | Best For |
|---------|-----------------|----------|
| `ShardedBloomFilter` | Lock-free atomics | Read-heavy workloads |
| `StripedBloomFilter` | Striped RwLocks | Write-heavy workloads |

### Hash Functions

| Hasher | Feature Flag | Performance | Notes |
|--------|--------------|-------------|-------|
| `StdHasher` | (built-in) | Good | DoS-resistant (SipHash) |
| `WyHasher` | `wyhash` | Excellent | Ultra-fast (~0.5 cycles/byte) |
| `XxHasher` | `xxhash` | Excellent | Industry-standard (~0.7 cycles/byte) |
| `SimdHasher` | `simd` | Best | AVX2/SSE4.1/NEON vectorized |

### Hash Strategies

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `DoubleHashing` | `h(i) = h1 + i × h2` | General purpose (fastest) |
| `EnhancedDoubleHashing` | `h(i) = h1 + i × h2 + (i² + i)/2` | High accuracy (default) |
| `TripleHashing` | `h(i) = h1 + i × h2 + i² × h3` | Research/validation |

### Feature Flags

```toml
[features]
default = []
serde = ["dep:serde", "dep:bincode", "dep:bytemuck"]  # Serialization support
xxhash = ["dep:xxhash-rust"]                           # XXHash hasher
wyhash = ["dep:wyhash"]                                # WyHash hasher
simd = []                                               # SIMD-optimized hashing
metrics = []                                            # Metrics collection
```

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
bloomcraft = "0.1"

# With all features
bloomcraft = { version = "0.1", features = ["serde", "xxhash", "wyhash", "simd", "metrics"] }
```

---

## Quick Start

### Basic Usage

```rust
use bloomcraft::prelude::*;

// Create a filter for 10,000 items with 1% false positive rate
let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(10_000, 0.01);

// Insert items
filter.insert(&"hello");
filter.insert(&"world");

// Check membership
assert!(filter.contains(&"hello"));  // true - definitely in set
assert!(filter.contains(&"world"));  // true - definitely in set
assert!(!filter.contains(&"foo"));   // probably false (could be false positive)
```

### Using the Builder Pattern

```rust
use bloomcraft::builder::StandardBloomFilterBuilder;

let filter: StandardBloomFilter<String> = StandardBloomFilterBuilder::new()
    .expected_items(100_000)
    .false_positive_rate(0.001)
    .build()
    .unwrap();
```

### Counting Filter (with Deletion)

```rust
use bloomcraft::filters::CountingBloomFilter;

let mut filter = CountingBloomFilter::<String>::new(10_000, 0.01);

filter.insert(&"item".to_string());
assert!(filter.contains(&"item".to_string()));

// Safe deletion - checks existence first
filter.delete(&"item".to_string());
assert!(!filter.contains(&"item".to_string()));
```

### Scalable Filter (Dynamic Growth)

```rust
use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy};

// Grows automatically as items are added
let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::with_strategy(
    1000,                           // Initial capacity
    0.01,                           // Target FPR
    0.5,                            // Error tightening ratio
    GrowthStrategy::Geometric(2.0)  // Double capacity each growth
);

for i in 0..10_000 {
    filter.insert(&i);
}

println!("Sub-filters: {}", filter.filter_count());
println!("Total capacity: {}", filter.total_capacity());
```

### Hierarchical Filter (Multi-Level)

```rust
use bloomcraft::filters::HierarchicalBloomFilter;

// 3-level hierarchy: 2 datacenters, 4 racks each, 8 servers each
let mut filter: HierarchicalBloomFilter<&str> = HierarchicalBloomFilter::new(
    vec![2, 4, 8],  // Branching factors
    10_000,         // Items per bin
    0.01            // FPR
);

// Insert data for datacenter 0, rack 2, server 5
filter.insert_to_bin(&"user:12345", &[0, 2, 5]).unwrap();

// Locate which bins contain the data
let locations = filter.locate(&"user:12345");
for location in locations {
    println!("Found in: DC {}, Rack {}, Server {}", 
             location[0], location[1], location[2]);
}
```

### Concurrent Filter

```rust
use bloomcraft::sync::ShardedBloomFilter;
use std::sync::Arc;
use std::thread;

let filter = Arc::new(std::sync::Mutex::new(
    ShardedBloomFilter::<String>::new(1_000_000, 0.01)
));

let handles: Vec<_> = (0..4).map(|i| {
    let f = Arc::clone(&filter);
    thread::spawn(move || {
        for j in 0..1000 {
            f.lock().unwrap().insert(&format!("item-{}-{}", i, j));
        }
    })
}).collect();

for h in handles {
    h.join().unwrap();
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         BloomCraft                               │
├─────────────────────────────────────────────────────────────────┤
│  lib.rs                                                          │
│  ├── Public API exports & prelude                                │
│  ├── Feature gate coordination                                   │
│  └── Re-exports for convenience                                  │
├─────────────────────────────────────────────────────────────────┤
│  core/                    │  hash/                               │
│  ├── filter.rs (traits)   │  ├── hasher.rs (BloomHasher trait)   │
│  ├── bitvec.rs (BitVec)   │  ├── strategies.rs (double/triple)   │
│  └── params.rs (math)     │  ├── wyhash.rs, xxhash.rs            │
│                           │  └── simd.rs (SIMD batch hashing)    │
├───────────────────────────┼─────────────────────────────────────┤
│  filters/                 │  builder/                            │
│  ├── standard.rs          │  ├── standard.rs                     │
│  ├── counting.rs          │  ├── counting.rs                     │
│  ├── scalable.rs          │  └── scalable.rs                     │
│  ├── partitioned.rs       │                                      │
│  ├── hierarchical.rs      │                                      │
│  ├── classic_bits.rs      │                                      │
│  └── classic_hash.rs      │                                      │
├───────────────────────────┼─────────────────────────────────────┤
│  sync/                    │  metrics/                            │
│  ├── sharded.rs           │  ├── collector.rs                    │
│  ├── striped.rs           │  ├── tracker.rs                      │
│  └── atomic_counter.rs    │  └── histogram.rs                    │
├───────────────────────────┴─────────────────────────────────────┤
│  serde_support/                                                  │
│  ├── standard.rs, counting.rs                                    │
│  ├── sharded.rs, striped.rs                                      │
│  └── zerocopy.rs (10-100x faster serialization)                  │
├─────────────────────────────────────────────────────────────────┤
│  error.rs - BloomCraftError enum with all error variants         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Filter Variants

### StandardBloomFilter

The classic, space-optimal Bloom filter implementation.

```rust
use bloomcraft::filters::StandardBloomFilter;

let mut filter: StandardBloomFilter<String> = StandardBloomFilter::new(10_000, 0.01);

// Core operations
filter.insert(&"item".to_string());
filter.contains(&"item".to_string());  // true
filter.clear();

// Batch operations
filter.insert_batch(&["a".to_string(), "b".to_string()]);
let results = filter.contains_batch(&["a".to_string(), "c".to_string()]);

// Statistics
println!("Size: {} bits", filter.size());
println!("Hash functions: {}", filter.hash_count());
println!("Fill rate: {:.2}%", filter.fill_rate() * 100.0);
println!("Estimated FPR: {:.4}", filter.estimate_fpr());

// Merge operations
let filter2 = StandardBloomFilter::new(10_000, 0.01);
let union = filter.union(&filter2).unwrap();
let intersection = filter.intersect(&filter2).unwrap();
```

### CountingBloomFilter

Supports element deletion via atomic counters with overflow protection.

```rust
use bloomcraft::filters::CountingBloomFilter;

// Default 8-bit counters (max value 255)
let mut filter = CountingBloomFilter::<String>::new(10_000, 0.01);

// Or specify 4-bit counters (max value 15, more space efficient)
let mut filter = CountingBloomFilter::<String>::with_counter_size(10_000, 0.01, 4);

filter.insert(&"item".to_string());

// Safe deletion (checks existence first)
let deleted = filter.delete(&"item".to_string());  // Returns true if found

// Unchecked deletion (faster but can cause false negatives)
filter.delete_unchecked(&"item".to_string());

// Counter statistics
println!("Max counter value: {}", filter.max_counter_value());
println!("Avg counter value: {:.2}", filter.avg_counter_value());
println!("Overflow count: {}", filter.overflow_count());
println!("Has overflowed: {}", filter.has_overflowed());

// Counter histogram
let histogram = filter.counter_histogram();
println!("Counters at 0: {}", histogram[0]);
```

### ScalableBloomFilter

Grows dynamically as items are added, maintaining target FPR.

```rust
use bloomcraft::filters::{ScalableBloomFilter, GrowthStrategy};

// Constant growth (all sub-filters same size)
let mut filter = ScalableBloomFilter::with_strategy(
    1000, 0.01, 0.5, GrowthStrategy::Constant
);

// Geometric growth (each filter 2x larger)
let mut filter = ScalableBloomFilter::with_strategy(
    1000, 0.01, 0.5, GrowthStrategy::Geometric(2.0)
);

// Monitor growth
println!("Sub-filters: {}", filter.filter_count());
println!("Total capacity: {}", filter.total_capacity());
println!("Fill rate: {:.2}%", filter.fill_rate() * 100.0);

// Per-filter statistics
for (capacity, fill_rate, fpr) in filter.filter_stats() {
    println!("  Capacity: {}, Fill: {:.2}%, FPR: {:.4}", 
             capacity, fill_rate * 100.0, fpr);
}
```

### PartitionedBloomFilter

Cache-line optimized for 2-4x faster queries.

```rust
use bloomcraft::filters::PartitionedBloomFilter;

// Standard creation
let mut filter: PartitionedBloomFilter<i32> = PartitionedBloomFilter::new(10_000, 0.01);

// With cache-line alignment (64 bytes)
let mut filter = PartitionedBloomFilter::with_alignment(10_000, 0.01, 64);

// Partition statistics
println!("Partitions: {}", filter.partition_count());
println!("Partition size: {} bits", filter.partition_size());

let fill_rates = filter.partition_fill_rates();
for (i, rate) in fill_rates.iter().enumerate() {
    println!("Partition {}: {:.2}% full", i, rate * 100.0);
}
```

### HierarchicalBloomFilter

Multi-level filtering for tiered storage and distributed systems.

```rust
use bloomcraft::filters::HierarchicalBloomFilter;

// Create hierarchy: 2 levels with branching [4, 8] = 32 leaf bins
let mut filter: HierarchicalBloomFilter<&str> = HierarchicalBloomFilter::new(
    vec![4, 8],  // Branching factors per level
    1000,        // Capacity per bin
    0.01         // FPR
);

// Insert to specific bin
filter.insert_to_bin(&"item", &[2, 5]).unwrap();

// Check existence anywhere
assert!(filter.contains(&"item"));

// Locate which bins contain item
let bins = filter.locate(&"item");

// Check specific bin
let in_bin = filter.contains_in_bin(&"item", &[2, 5]).unwrap();

// Statistics
println!("Depth: {}", filter.depth());
println!("Total bins: {}", filter.bin_count());
println!("Total nodes: {}", filter.node_count());

// Per-bin fill rates
let fill_rates = filter.all_bin_fill_rates();
```

### Classic Filters (Educational)

Historical implementations from Burton Bloom's 1970 paper.

```rust
use bloomcraft::filters::{ClassicBitsFilter, ClassicHashFilter};

// Method 2: Bit array (became the standard)
let mut bits_filter: ClassicBitsFilter<&str> = ClassicBitsFilter::new(10_000, 7);
bits_filter.insert(&"hello");

// Method 1: Hash table with chaining
let mut hash_filter: ClassicHashFilter<&str> = ClassicHashFilter::new(1000, 3);
hash_filter.insert(&"hello");

// Hash filter specific stats
println!("Avg chain length: {:.2}", hash_filter.avg_chain_length());
println!("Max chain length: {}", hash_filter.max_chain_length());
println!("Load factor: {:.2}", hash_filter.load_factor());
```

---

## Hash Functions

### BloomHasher Trait

All hash functions implement the `BloomHasher` trait:

```rust
pub trait BloomHasher: Send + Sync {
    fn hash_bytes(&self, bytes: &[u8]) -> u64;
    fn hash_bytes_with_seed(&self, bytes: &[u8], seed: u64) -> u64;
    fn hash_bytes_pair(&self, bytes: &[u8]) -> (u64, u64);
    fn hash_bytes_triple(&self, bytes: &[u8]) -> (u64, u64, u64);
    fn name(&self) -> &'static str;
}
```

### Using Different Hashers

```rust
use bloomcraft::hash::{StdHasher, BloomHasher, recommended_hasher, hasher_with_seed};

// Default hasher (SipHash, DoS-resistant)
let hasher = StdHasher::new();

// Platform-recommended hasher (WyHash > XXHash > StdHasher)
let hasher = recommended_hasher();

// Seeded hasher for reproducibility
let hasher = hasher_with_seed(42);

// With filters
use bloomcraft::filters::StandardBloomFilter;
let filter = StandardBloomFilter::<String>::with_hasher(10_000, 0.01, StdHasher::new());
```

### SIMD Hasher (Feature: `simd`)

Vectorized batch hashing with AVX2/SSE4.1/NEON support.

```rust
#[cfg(feature = "simd")]
{
    use bloomcraft::hash::simd::{SimdHasher, CpuFeatures};

    let hasher = SimdHasher::new();
    
    // Check CPU capabilities
    let features = CpuFeatures::detect();
    println!("AVX2: {}", features.has_avx2);
    println!("NEON: {}", features.has_neon);
    println!("SIMD available: {}", features.has_simd());

    // Batch hashing (SIMD-accelerated for batch ≥ 8)
    let values = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let hashes = hasher.hash_batch_u64(&values);
    
    // Performance characteristics:
    // - Batch < 8: Uses scalar path (SIMD overhead not worth it)
    // - Batch ≥ 8: Uses SIMD if available
    // - Speedup: 2-3.5x for large batches
}
```

### Hash Benchmarking

```rust
use bloomcraft::hash::bench::{compare_hashers, benchmark_hasher};

let items: Vec<Vec<u8>> = (0..1000)
    .map(|i| format!("item{}", i).into_bytes())
    .collect();
let item_refs: Vec<&[u8]> = items.iter().map(|v| v.as_slice()).collect();

// Compare all available hashers
let results = compare_hashers(&item_refs);
for result in &results {
    println!("{}: {:.2} ns/hash, {:.2} M hashes/sec",
        result.name,
        result.time_per_hash_ns,
        result.throughput / 1_000_000.0
    );
}
```

---

## Concurrent Filters

### ShardedBloomFilter

Lock-free concurrent filter using sharding and atomic operations.

```rust
use bloomcraft::sync::ShardedBloomFilter;
use std::sync::Arc;
use std::thread;

let filter = Arc::new(std::sync::Mutex::new(
    ShardedBloomFilter::<String>::new(1_000_000, 0.01)
));

// Concurrent inserts
let handles: Vec<_> = (0..8).map(|i| {
    let f = Arc::clone(&filter);
    thread::spawn(move || {
        for j in 0..10_000 {
            f.lock().unwrap().insert(&format!("item-{}-{}", i, j));
        }
    })
}).collect();

for h in handles { h.join().unwrap(); }

println!("Shard count: {}", filter.lock().unwrap().shard_count());
```

### StripedBloomFilter

Concurrent filter using striped RwLocks for write-heavy workloads.

```rust
use bloomcraft::sync::StripedBloomFilter;

let filter = StripedBloomFilter::<String>::new(1_000_000, 0.01);
println!("Stripe count: {}", filter.stripe_count());
```

### AtomicCounterArray

Cache-line padded atomic counters for high-concurrency scenarios.

```rust
use bloomcraft::sync::{AtomicCounterArray, CacheLinePadded};
use std::sync::atomic::AtomicU64;

// Array of counters with false-sharing prevention
let counters = AtomicCounterArray::new(16);

// Increment with overflow protection
counters.increment(5).unwrap();

// Saturating operations (never fail)
counters.increment_saturating(5);
counters.decrement_saturating(5);

// Statistics
println!("Sum: {}", counters.sum());
println!("Min: {}", counters.min());
println!("Max: {}", counters.max());

// Single padded counter
let counter = CacheLinePadded::new(AtomicU64::new(0));
counter.fetch_add(1).unwrap();
```

---

## Serialization

### Standard Serde (Feature: `serde`)

Works with any serde-compatible format.

```rust
use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::core::BloomFilter;

let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
filter.insert(&"hello");

// JSON
let json = serde_json::to_string(&filter).unwrap();
let restored: StandardBloomFilter<&str> = serde_json::from_str(&json).unwrap();

// Bincode (binary, more compact)
let bytes = bincode::serialize(&filter).unwrap();
let restored: StandardBloomFilter<&str> = bincode::deserialize(&bytes).unwrap();
```

### Zero-Copy Serialization

10-100x faster than standard serde, minimal allocations.

```rust
use bloomcraft::filters::StandardBloomFilter;
use bloomcraft::serde_support::zerocopy::ZeroCopyBloomFilter;
use bloomcraft::core::BloomFilter;

let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(10_000, 0.01);
filter.insert(&"hello");

// Serialize (safe, explicit byte-level control)
let bytes = ZeroCopyBloomFilter::serialize(&filter);

// Validate without full deserialization
ZeroCopyBloomFilter::validate(&bytes).unwrap();

// Deserialize (safe, no UB)
let restored: StandardBloomFilter<&str> = 
    ZeroCopyBloomFilter::deserialize_generic(&bytes).unwrap();

// Get serialized size
let size = ZeroCopyBloomFilter::serialized_size(&filter);
```

**Zero-Copy Format:**

```
[Header: 32 bytes]
  Magic:        4 bytes  ("BLOM")
  Version:      2 bytes  (format version)
  Filter Type:  2 bytes  (0=Standard, 1=Counting, 2=Scalable)
  Size:         8 bytes  (filter size in bits/counters)
  Num Hashes:   4 bytes  (number of hash functions)
  Hash Strategy: 1 byte  (0=Double, 1=Enhanced, 2=Triple)
  Reserved:     11 bytes (for future extensions)

[Data: Variable]
  Raw bit/counter data (little-endian u64 words)
```

### Format Comparison

| Format | Size (1M items) | Serialize | Deserialize | Use Case |
|--------|-----------------|-----------|-------------|----------|
| JSON | 3.2 MB | 450 ms | 580 ms | Human-readable, debugging |
| Bincode | 1.2 MB | 85 ms | 95 ms | General purpose |
| Zero-Copy | 1.0 MB | 8 ms | 0.5 ms | Performance-critical |

---

## Metrics & Observability

### MetricsCollector (Feature: `metrics`)

Unified metrics collection with export capabilities.

```rust
#[cfg(feature = "metrics")]
{
    use bloomcraft::metrics::{MetricsCollector, FilterMetrics};
    use std::time::Duration;

    // Basic collector
    let metrics = MetricsCollector::new();

    // With latency histogram tracking
    let metrics = MetricsCollector::with_histogram();

    // Record operations
    metrics.record_insert();
    metrics.record_query(true);   // Positive result
    metrics.record_query(false);  // Negative result
    metrics.record_remove();

    // Record with latency
    metrics.record_query_latency(true, Duration::from_micros(50));
    metrics.record_insert_latency(Duration::from_micros(30));

    // Confirmed queries (when you can verify actual membership)
    metrics.record_confirmed_query(true, true);   // True positive
    metrics.record_confirmed_query(true, false);  // False positive

    // Get snapshot
    let snapshot = metrics.snapshot();
    println!("Total inserts: {}", snapshot.total_inserts);
    println!("Total queries: {}", snapshot.total_queries);
    println!("Queries/sec: {:.2}", snapshot.queries_per_second());
    println!("FP rate: {:.4}", snapshot.fp_tracker.current_fp_rate);

    // Export to Prometheus format
    let prometheus = snapshot.to_prometheus_format("bloomcraft");
    
    // Export to JSON (requires serde feature)
    #[cfg(feature = "serde")]
    let json = snapshot.to_json().unwrap();
}
```

### FalsePositiveTracker

Real-time false positive rate tracking with sliding window analysis.

```rust
#[cfg(feature = "metrics")]
{
    use bloomcraft::metrics::{FalsePositiveTracker, FpTrackerConfig};

    // Basic tracker
    let tracker = FalsePositiveTracker::new(10_000);

    // With custom configuration
    let tracker = FalsePositiveTracker::with_config(FpTrackerConfig {
        window_size: 1000,
        expected_fp_rate: 0.01,
        alert_threshold: 1.5,  // Alert if FP rate > 1.5x expected
    });

    // Record queries
    tracker.record_positive();      // Filter said "yes"
    tracker.record_true_positive(); // Actually in set
    tracker.record_negative();      // Filter said "no"

    // Or record confirmed results
    tracker.record_confirmed(true, true);   // True positive
    tracker.record_confirmed(true, false);  // False positive

    // Get statistics
    println!("Current FP rate: {:.4}", tracker.current_fp_rate());
    println!("Window FP rate: {:.4}", tracker.window_fp_rate());
    println!("Expected FP rate: {:.4}", tracker.expected_fp_rate());
    println!("Is alert: {}", tracker.is_alert());

    // Snapshot
    let snapshot = tracker.snapshot();
    println!("Deviation: {:.2}%", snapshot.deviation_percent());
}
```

### LatencyHistogram

High-resolution latency tracking with percentile calculations.

```rust
#[cfg(feature = "metrics")]
{
    use bloomcraft::metrics::LatencyHistogram;
    use std::time::Duration;

    let histogram = LatencyHistogram::new();

    // Record latencies
    histogram.record(Duration::from_micros(50));
    histogram.record(Duration::from_micros(100));
    histogram.record(Duration::from_micros(150));

    // Get percentiles
    println!("P50: {:?}", histogram.percentile(0.50));
    println!("P90: {:?}", histogram.percentile(0.90));
    println!("P99: {:?}", histogram.percentile(0.99));

    // Get snapshot with all stats
    let stats = histogram.snapshot();
    println!("Mean: {:?}", stats.mean);
    println!("Min: {:?}", stats.min);
    println!("Max: {:?}", stats.max);
}
```

---

## API Reference

### Core Traits

#### BloomFilter<T>

The base trait all filter variants implement:

```rust
pub trait BloomFilter<T: Hash>: Send + Sync {
    fn insert(&mut self, item: &T);
    fn contains(&self, item: &T) -> bool;
    fn clear(&mut self);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn false_positive_rate(&self) -> f64;
    fn expected_items(&self) -> usize;
    fn bit_count(&self) -> usize;
    fn hash_count(&self) -> usize;
}
```

### Core Module (`src/core/`)

#### BitVec - Lock-Free Bit Vector

```rust
use bloomcraft::core::bitvec::BitVec;

let bitvec = BitVec::new(10_000).unwrap();

// Atomic operations (lock-free)
bitvec.set(42);           // Set bit
bitvec.get(42);           // Get bit
bitvec.clear_bit(42);     // Clear single bit

// Range operations
bitvec.set_range(0, 100, true);
let bits = bitvec.get_range(0, 100);

// Statistics
println!("Population count: {}", bitvec.count_ones());
println!("Memory usage: {} bytes", bitvec.memory_usage());

// Set operations
let other = BitVec::new(10_000).unwrap();
let union = bitvec.union(&other).unwrap();
let intersection = bitvec.intersect(&other).unwrap();
```

#### Parameter Calculations

```rust
use bloomcraft::core::params::{
    optimal_bit_count, optimal_hash_count, expected_fp_rate,
    calculate_filter_params, bits_per_element
};

// Calculate optimal parameters
let m = optimal_bit_count(10_000, 0.01).unwrap();      // Bits needed
let k = optimal_hash_count(m, 10_000).unwrap();        // Hash functions
let fpr = expected_fp_rate(m, 10_000, k).unwrap();     // Expected FPR

// Or calculate both at once
let (m, k) = calculate_filter_params(10_000, 0.01).unwrap();

// Bits per element for a given FPR
let bpe = bits_per_element(0.01).unwrap();  // ~9.6 bits/element for 1% FPR
```

### Error Handling

```rust
use bloomcraft::error::{BloomCraftError, Result};

// All error variants
pub enum BloomCraftError {
    InvalidParameters { message: String },
    FalsePositiveRateOutOfBounds { fp_rate: f64 },
    InvalidItemCount { count: usize },
    CapacityExceeded { capacity: usize, attempted: usize },
    UnsupportedOperation { operation: String, variant: String },
    IncompatibleFilters { reason: String },
    InvalidHashCount { count: usize, min: usize, max: usize },
    InvalidFilterSize { size: usize },
    CounterOverflow { max_value: u64 },
    CounterUnderflow { min_value: u64 },
    IndexOutOfBounds { index: usize, length: usize },
    InvalidRange { start: usize, end: usize, length: usize, reason: String },
    #[cfg(feature = "serde")]
    SerializationError { message: String },
    InternalError { message: String },
}

// Convenience constructors
let err = BloomCraftError::invalid_parameters("message");
let err = BloomCraftError::counter_overflow(255);
```

---

## Performance

### Space Efficiency

| FP Rate | Bits/Element | Memory for 1M items |
|---------|--------------|---------------------|
| 10%     | ~4.8         | ~600 KB             |
| 1%      | ~9.6         | ~1.2 MB             |
| 0.1%    | ~14.4        | ~1.8 MB             |
| 0.01%   | ~19.2        | ~2.4 MB             |

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert    | O(k)       | k = number of hash functions |
| Contains  | O(k)       | Early termination on first 0 bit |
| Clear     | O(m/64)    | m = filter size in bits |
| Delete (Counting) | O(k) | Requires counter decrements |

### Filter Comparison

| Filter | Insert | Query | Memory | Best For |
|--------|--------|-------|--------|----------|
| Standard | O(k) | O(k) | m bits | General purpose |
| Counting | O(k) | O(k) | 4-8× Standard | Dynamic sets |
| Scalable | O(k) | O(k × tiers) | Grows on demand | Unknown size |
| Partitioned | O(k) | O(k) | m bits | Cache-sensitive |
| Hierarchical | O(k × depth) | O(k × depth) | Higher | Distributed |

### Metrics Overhead

| Operation | Overhead | Thread Safety |
|-----------|----------|---------------|
| `record_insert()` | ~5ns | Lock-free |
| `record_query()` | ~8ns | Lock-free |
| `record_latency()` | ~15ns | Lock-free |
| `snapshot()` | ~100ns | Atomic read |

---

## Testing & Benchmarking

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with all features
cargo test --all-features

# Run specific test module
cargo test filters::counting::

# Run property-based tests (may take longer)
cargo test --release -- --ignored

# Run with output
cargo test -- --nocapture
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench insert
cargo bench query
cargo bench hash_functions

# Compare filter variants
cargo bench comparison

# Memory benchmarks
cargo bench memory

# Concurrent benchmarks
cargo bench concurrent
```

### Available Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `insert` | Insert throughput for all filter types |
| `query` | Query throughput and false positive rates |
| `hash_functions` | Compare hash function performance |
| `comparison` | Side-by-side filter variant comparison |
| `concurrent` | Multi-threaded performance |
| `memory` | Memory usage and allocation patterns |
| `historical` | Classic filter implementations |

---

## Use Cases

### Database Query Optimization
Avoid disk lookups for non-existent keys in LSM-trees and key-value stores.

```rust
use bloomcraft::filters::StandardBloomFilter;

// Check if key might exist before expensive disk lookup
let mut filter: StandardBloomFilter<String> = StandardBloomFilter::new(1_000_000, 0.01);

fn get_value(key: &str, filter: &StandardBloomFilter<String>) -> Option<String> {
    if !filter.contains(&key.to_string()) {
        return None;  // Definitely not in database
    }
    // Proceed with disk lookup...
    None
}
```

### Distributed Caching
Track which cache nodes might contain a key.

```rust
use bloomcraft::filters::HierarchicalBloomFilter;

// Track data across datacenters, racks, and servers
let mut filter: HierarchicalBloomFilter<String> = HierarchicalBloomFilter::new(
    vec![3, 10, 50],  // 3 DCs, 10 racks each, 50 servers each
    100_000,
    0.001
);
```

### Network Packet Deduplication
Detect duplicate packets in high-throughput network processing.

```rust
use bloomcraft::filters::CountingBloomFilter;

// Use counting filter to handle packet retransmissions
let mut seen: CountingBloomFilter<[u8; 32]> = CountingBloomFilter::new(10_000_000, 0.0001);
```

### Web Crawling
Track visited URLs efficiently.

```rust
use bloomcraft::filters::ScalableBloomFilter;

// Grows as you discover more URLs
let mut visited: ScalableBloomFilter<String> = ScalableBloomFilter::new(1_000_000, 0.001);
```

### Spell Checking
Fast dictionary membership testing.

```rust
use bloomcraft::filters::StandardBloomFilter;

let mut dictionary: StandardBloomFilter<String> = StandardBloomFilter::new(500_000, 0.01);
// Load dictionary words...

fn is_word(word: &str, dict: &StandardBloomFilter<String>) -> bool {
    dict.contains(&word.to_string())
}
```

---

## Contributing

Contributions are welcome! Please see our contributing guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/bloomcraft.git
cd bloomcraft

# Build with all features
cargo build --all-features

# Run tests
cargo test --all-features

# Run clippy
cargo clippy --all-features -- -D warnings

# Format code
cargo fmt
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

## References

- Bloom, B. H. (1970). "Space/time trade-offs in hash coding with allowable errors". Communications of the ACM.
- Kirsch & Mitzenmacher (2006). "Less Hashing, Same Performance: Building a Better Bloom Filter".
- Fan et al. (2000). "Summary cache: a scalable wide-area web cache sharing protocol".
- Almeida et al. (2007). "Scalable Bloom Filters". Information Processing Letters.
- Putze et al. (2009). "Cache-, hash- and space-efficient bloom filters". Journal of Experimental Algorithmics.
