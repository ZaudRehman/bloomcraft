# Changelog

All notable changes to BloomCraft are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.1.0] - 2026-07-02

### Added

- **Initial Release:** Comprehensive suite of probabilistic data structures.
- **Core & Traits**
  - `BloomFilter`, `ConcurrentBloomFilter`, `SharedBloomFilter`, `DeletableBloomFilter`, and `MergeableBloomFilter` traits.
  - `BloomCraftError` typed error enum.
- **Filters**
  - `StandardBloomFilter`: Standard implementation with `AtomicU64` backing.
  - `CountingBloomFilter`: Deletion support with 4-/8-/16-bit counter slots.
  - `ScalableBloomFilter`: Dynamically growing filter with `Geometric(2.0)` and `Constant` growth.
  - `PartitionedBloomFilter`: Cache-line aligned partitioned filter.
  - `RegisterBlockedBloomFilter`: 512-bit block layout—one cache-line touch per query, trades FPR for throughput relative to `StandardBloomFilter`.
  - `TreeBloomFilter`: Hierarchical filter with bin-level insert/query.
  - `ClassicBitsFilter` & `ClassicHashFilter`: Bloom 1970 Method 1 and 2 (educational/historical baselines).
- **Concurrent Filters** (feature `concurrent`)
  - `ShardedBloomFilter` & `StripedBloomFilter`: High-concurrency shared filters.
  - `StripedBloomFilter::with_concurrency(n)`: Stripe count derived from expected thread count.
  - `AtomicScalableBloomFilter`: Concurrent scalable filter backed by sharded sub-filters and CAS-based growth.
  - `AtomicPartitionedBloomFilter`: Concurrent cache-aligned partitioned filter.
- **Builders**
  - Type-state builders enforcing correct parameter ordering for `StandardBloomFilter`, `CountingBloomFilter`, and `ScalableBloomFilter`.
- **Scalable Filter Features**
  - `GrowthStrategy::Adaptive`: Per-stage FPR tightening driven by observed fill rates.
  - `GrowthStrategy::Bounded`: Geometric growth capped at a per-filter size limit.
  - `CapacityExhaustedBehavior` enum: `Silent`, `Error`, `Panic` (debug-only).
  - `QueryStrategy` enum: `Forward` / `Reverse` iteration over sub-filters.
  - HyperLogLog++ cardinality estimation via `with_cardinality_tracking()` and `estimate_unique_count()`.
  - `ScalableHealthMetrics`: 13 runtime fields returned by `health_metrics()`.
- **Hashing**
  - `EnhancedDoubleHashing` (default hash strategy).
  - `StdHasher` (SipHash-1-3, default hasher).
  - `wyhash` feature: `WyHasher`.
  - `xxhash` feature: `XxHasher` (xxh3).
  - `simd` feature: AVX2 / SSE4.1 / NEON vectorized batch hashing.
- **Serialization** (feature `serde`)
  - `Serialize` / `Deserialize` for all filter types.
  - Zero-copy binary format via `ZeroCopyBloomFilter`.
- **Observability**
  - `metrics` feature: `MetricsCollector`, `FalsePositiveTracker`, `LatencyHistogram`, and Prometheus text export.
  - `trace` feature: `QueryTrace` / `QueryTraceBuilder` for per-query timing instrumentation.

[0.1.0]: https://github.com/ZaudRehman/BloomCraft/releases/tag/v0.1.0