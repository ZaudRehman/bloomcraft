# Changelog

All notable changes to BloomCraft are documented here.

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
conventions. Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `RegisterBlockedBloomFilter`: 512-bit block layout guaranteeing one cache miss
  per query; 20–30% faster than `PartitionedBloomFilter` at 2–3× FPR overhead
- `AtomicScalableBloomFilter` (feature: `concurrent`): lock-free concurrent
  scalable filter backed by sharded sub-filters and CAS-based growth election
- `AtomicPartitionedBloomFilter` (feature: `concurrent`): wait-free concurrent
  cache-line partitioned filter using `AtomicU64::fetch_or` with `Relaxed` ordering
- `GrowthStrategy::Adaptive` and `GrowthStrategy::Bounded` variants for
  `ScalableBloomFilter`
- `CapacityExhaustedBehavior` enum: `Silent`, `Error`, `Panic` (debug builds only)
- `QueryStrategy` enum: `Forward` and `Reverse` iteration order for scalable filter queries
- HyperLogLog cardinality estimation in `ScalableBloomFilter` via
  `with_cardinality_tracking()` and `estimate_unique_count()`
- `ScalableHealthMetrics` struct returned by `health_metrics()` — 13 runtime fields
  for production monitoring
- `TreeBloomFilter`: hierarchical filter with `insert_to_bin()`, `locate()`,
  and `contains_in_bin()`
- `StripedBloomFilter::with_concurrency(n)`: derives stripe count from expected
  thread count using Lemire's fast range reduction
- Type-state builders for `StandardBloomFilter`, `CountingBloomFilter`, and
  `ScalableBloomFilter`
- `serde` feature: `Serialize`/`Deserialize` for all filter types plus zero-copy
  binary format via `ZeroCopyBloomFilter`
- `metrics` feature: `MetricsCollector`, `FalsePositiveTracker`, `LatencyHistogram`,
  Prometheus text export
- `trace` feature: `QueryTrace` and `QueryTraceBuilder` for `tracing`-compatible
  span instrumentation
- `wyhash` and `xxhash` feature flags for `WyHasher` and `XxHasher`
- `simd` feature flag for AVX2/SSE4.1/NEON vectorized batch hashing

---

## [0.1.0] - 2026-01-01

### Added
- Initial release
- `StandardBloomFilter` with `AtomicU64` backing and `ConcurrentBloomFilter` trait
- `CountingBloomFilter` with 4/8/16-bit counter slots and `DeletableBloomFilter` trait
- `ScalableBloomFilter` with `Geometric(2.0)` and `Constant` growth strategies
- `PartitionedBloomFilter` with cache-line alignment
- `ShardedBloomFilter` and `StripedBloomFilter` implementing `SharedBloomFilter`
- `ClassicBitsFilter` and `ClassicHashFilter` (Bloom 1970 Method 1 and 2)
- `BloomFilter`, `ConcurrentBloomFilter`, `SharedBloomFilter`,
  `DeletableBloomFilter`, and `MergeableBloomFilter` traits
- `EnhancedDoubleHashing` as the default hash strategy
- `StdHasher` (SipHash-1-3) as the default hasher
- `BloomCraftError` typed error enum

[Unreleased]: https://github.com/ZaudRehman/BloomCraft/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ZaudRehman/BloomCraft/releases/tag/v0.1.0
