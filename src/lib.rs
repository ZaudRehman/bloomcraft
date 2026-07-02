//! Production-grade Bloom filters for Rust.
//!
//! BloomCraft provides a comprehensive family of Bloom filter implementations
//! spanning classical algorithms, cache-optimised variants, lock-free concurrent
//! filters, and hierarchical/tree filters — all with a uniform [`BloomFilter`]
//! trait interface and type-safe builders.
//!
//! # Modules
//!
//! - [`core`] - Core traits, parameter math, and bit-vector primitives.
//! - [`filters`] - Bloom filter implementations.
//! - [`hash`] - Hashers and index-generation strategies.
//! - [`builder`] - Builders for standard, counting, and scalable filters.
//! - [`sync`] - Concurrent shared-filter variants.
//! - [`error`] - Crate-wide error and `Result` types.
//! - [`serde_support`] - Serialization helpers, available with the `serde` feature.
//! - [`metrics`] - Metrics collection, available with the `metrics` feature.
//!
//! # Quick start
//!
//! ```rust
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: StandardBloomFilter<&str> =
//!     StandardBloomFilter::new(10_000, 0.01).expect("valid params");
//! filter.insert(&"hello");
//! filter.insert(&"world");
//!
//! assert!(filter.contains(&"hello"));
//! assert!(!filter.contains(&"goodbye"));
//! ```
//!
//! # Concurrency models
//!
//! | Model | Trait | `&self` / `&mut self` | Examples |
//! |---|---|---|---|
//! | **Single-threaded** | [`BloomFilter`] | `&mut self` | `StandardBloomFilter`, `CountingBloomFilter`, `ScalableBloomFilter` |
//! | **Lock-free atomic** | [`ConcurrentBloomFilter`] | `&self` (atomic ops) | `StandardBloomFilter` |
//! | **Interior mutability** | [`SharedBloomFilter`] | `&self` (shards/stripes) | `ShardedBloomFilter`, `StripedBloomFilter` |
//!
//! ```rust
//! use bloomcraft::StandardBloomFilter;
//! use bloomcraft::core::ConcurrentBloomFilter;
//! use std::sync::Arc;
//!
//! let filter = Arc::new(StandardBloomFilter::<String>::new(10_000, 0.01).unwrap());
//! let f2 = Arc::clone(&filter);
//! std::thread::spawn(move || { f2.insert_concurrent(&"item".to_string()); }).join().unwrap();
//! assert!(filter.contains(&"item".to_string()));
//! ```
//!
//! # Choosing a filter
//!
//! | Filter | Use case | Concurrency |
//! |---|---|---|
//! | [`StandardBloomFilter`] | General-purpose, known size | Lock-free atomic |
//! | [`CountingBloomFilter`] | Dynamic sets with deletion | `Mutex`-wrapped |
//! | [`ScalableBloomFilter`] | Unknown or unbounded size | `Mutex`-wrapped |
//! | [`AtomicScalableBloomFilter`] | Concurrent, unbounded size | Lock-free atomic |
//! | [`PartitionedBloomFilter`] | Cache-optimised queries | `Mutex`-wrapped |
//! | [`AtomicPartitionedBloomFilter`] | Concurrent, cache-optimised | Lock-free atomic |
//! | [`RegisterBlockedBloomFilter`] | Ultra-fast queries (AVX2) | `Mutex`-wrapped |
//! | [`TreeBloomFilter`] | Hierarchical / location-aware | `Mutex`-wrapped |
//! | [`ShardedBloomFilter`] | High-throughput lock-free | `SharedBloomFilter` |
//! | [`StripedBloomFilter`] | Moderate concurrency, low overhead | `SharedBloomFilter` |
//! | [`ClassicBitsFilter`] | Educational / historical | `Mutex`-wrapped |
//! | [`ClassicHashFilter`] | Educational / historical | `Mutex`-wrapped |
//!
//! # Builders
//!
//! Each major filter has a type-state [`builder`] that enforces
//! correct parameter ordering at compile time:
//!
//! ```rust
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! let filter: StandardBloomFilter<&str> = StandardBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .build()
//!     .expect("valid params");
//! ```
//!
//! # Hashers
//!
//! By default filters use [`StdHasher`] (FNV-1a). Enable
//! the `wyhash` or `xxhash` features for faster alternatives, or `simd` for
//! AVX2/NEON-accelerated hashing:
//!
//! ```rust
//! # #[cfg(feature = "wyhash")]
//! # {
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::hash::WyHasher;
//!
//! let filter: StandardBloomFilter<String, WyHasher> =
//!     StandardBloomFilter::with_hasher(10_000, 0.01, WyHasher::new())
//!         .expect("valid params");
//! # }
//! ```
//!
//! # Serialization
//!
//! With the `serde` feature enabled, filters implement `Serialize` / `Deserialize`
//! for any format (JSON, Bincode, MessagePack, …):
//!
//! ```rust
//! # #[cfg(feature = "serde")]
//! # {
//! use bloomcraft::StandardBloomFilter;
//!
//! let filter = StandardBloomFilter::<String>::new(10_000, 0.01).unwrap();
//! filter.insert(&"serialize_me".to_string());
//! let json = serde_json::to_string(&filter).expect("serialization failed");
//! let restored: StandardBloomFilter<String> =
//!     serde_json::from_str(&json).expect("deserialization failed");
//! assert!(restored.contains(&"serialize_me".to_string()));
//! # }
//! ```
//!
//! See [`serde_support`] for format-specific helper types (validation, version
//! checks, and estimated sizes before round-tripping).
//!
//! # Observable metrics
//!
//! With the `metrics` feature, filters carry a [`MetricsCollector`] that tracks
//! insert/query counts, latency histograms, and false-positive rates with
//! sliding-window detection:
//!
//! ```rust
//! # #[cfg(feature = "metrics")]
//! # {
//! use bloomcraft::StandardBloomFilter;
//! use bloomcraft::metrics::{MetricsCollector, LatencyHistogram};
//!
//! let collector = MetricsCollector::with_histogram();
//! let filter = StandardBloomFilter::<String>::new(10_000, 0.01).unwrap();
//! # drop(filter);
//! # }
//! ```
//!
//! # Feature flags
//!
//! | Flag | Description | Default |
//! |---|---|---|
//! | `serde` | `Serialize`/`Deserialize` for all filter types | off |
//! | `bincode` | Binary serialisation helpers (used with `serde`) | off |
//! | `xxhash` | XXHash3 hasher (fast on large inputs) | off |
//! | `wyhash` | WyHash hasher (fast on small inputs) | off |
//! | `simd` | AVX2/NEON SIMD batch hashing | off |
//! | `metrics` | `MetricsCollector`, `LatencyHistogram`, `FalsePositiveTracker` | off |
//! | `trace` | Query tracing for `ScalableBloomFilter` | off |
//! | `concurrent` | `AtomicScalableBloomFilter`, `AtomicPartitionedBloomFilter` | off |
//!
//! # Unsafe code policy
//!
//! `unsafe` is used in five well-audited locations, each justified with an
//! explicit safety comment:
//!
//! - `sync/sharded.rs` — `AtomicPtr<Arc<BitVec>>` for lock-free reads during
//!   concurrent clear operations.
//! - `sync/striped.rs` — `UnsafeCell<Arc<BitVec>>` guarded by `parking_lot::RwLock`.
//! - `sync/striped.rs` — Thread-safety marker trait impls (`Send + Sync`) for
//!   types whose correctness follows from the `RwLock` guards.
//! - `hash/simd.rs` — AVX2 / NEON intrinsics with runtime CPUID feature detection
//!   and a scalar fallback path.
//! - `error.rs` — None (zero unsafe in error types).
//!
//! **All public methods are safe.** Unsafe internals are not exposed.
//!
//! # Notes
//!
//! The crate root re-exports the most commonly used traits and types. More
//! specialized functionality remains available through module paths.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod builder;
pub mod core;
pub mod error;
pub mod filters;
pub mod hash;
pub mod sync;
pub mod util;

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
pub mod serde_support;

#[cfg(feature = "metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "metrics")))]
pub mod metrics;

// Core error/result re-exports
pub use crate::error::{BloomCraftError, Result};

// Core trait re-exports
pub use crate::core::{
    BloomFilter,
    ConcurrentBloomFilter,
    DeletableBloomFilter,
    MergeableBloomFilter,
    MutableBloomFilter,
    ScalableBloomFilter as ScalableBloomFilterTrait,
    SharedBloomFilter,
};

// Common filter re-exports
pub use crate::filters::{
    CapacityExhaustedBehavior,
    ClassicBitsFilter,
    ClassicHashFilter,
    CounterSize,
    CountingBloomFilter,
    FilterHealth,
    GrowthStrategy,
    PartitionedBloomFilter,
    QueryStrategy,
    RegisterBlockedBloomFilter,
    ScalableBloomFilter,
    StandardBloomFilter,
    TreeBloomFilter,
};

#[cfg(feature = "trace")]
pub use crate::filters::{QueryTrace, QueryTraceBuilder};

#[cfg(feature = "concurrent")]
pub use crate::filters::{AtomicPartitionedBloomFilter, AtomicScalableBloomFilter};

// Hash re-exports
pub use crate::hash::{
    BloomHasher,
    DoubleHashing,
    EnhancedDoubleHashing,
    HashStrategy,
    HashStrategyKind,
    IndexingStrategy,
    StdHasher,
    TripleHashing,
};

#[cfg(feature = "wyhash")]
pub use crate::hash::{WyHasher, WyHasherBuilder};

#[cfg(feature = "xxhash")]
pub use crate::hash::{XxHasher, XxHasherBuilder};

#[cfg(feature = "simd")]
pub use crate::hash::SimdHasher;

// Builder re-exports
pub use crate::builder::{
    CountingBloomFilterBuilder,
    ScalableBloomFilterBuilder,
    StandardBloomFilterBuilder,
};

// Concurrent shared-filter re-exports
pub use crate::sync::{ShardedBloomFilter, StripedBloomFilter};

#[cfg(feature = "serde")]
pub use crate::serde_support as serde;

#[cfg(feature = "metrics")]
pub use crate::metrics::{
    FalsePositiveTracker,
    FilterMetrics,
    LatencyHistogram,
    LatencyStats,
    MetricsCollector,
    MetricsSnapshot,
};

/// Convenient imports for common BloomCraft use.
///
/// ```rust
/// use bloomcraft::prelude::*;
///
/// let mut filter: StandardBloomFilter<&str> =
///     StandardBloomFilter::new(1_000, 0.01).expect("valid parameters");
/// filter.insert(&"hello");
/// assert!(filter.contains(&"hello"));
/// ```
pub mod prelude {
    pub use crate::builder::{
        CountingBloomFilterBuilder,
        ScalableBloomFilterBuilder,
        StandardBloomFilterBuilder,
    };
    pub use crate::core::{
        BloomFilter,
        ConcurrentBloomFilter,
        DeletableBloomFilter,
        MergeableBloomFilter,
        MutableBloomFilter,
        SharedBloomFilter,
    };
    pub use crate::error::{BloomCraftError, Result};
    pub use crate::filters::{
        ClassicBitsFilter,
        ClassicHashFilter,
        CountingBloomFilter,
        PartitionedBloomFilter,
        RegisterBlockedBloomFilter,
        ScalableBloomFilter,
        StandardBloomFilter,
        TreeBloomFilter,
    };
    pub use crate::hash::{
        BloomHasher,
        DoubleHashing,
        EnhancedDoubleHashing,
        HashStrategy,
        HashStrategyKind,
        IndexingStrategy,
        StdHasher,
        TripleHashing,
    };
    pub use crate::sync::{ShardedBloomFilter, StripedBloomFilter};

    #[cfg(feature = "concurrent")]
    pub use crate::filters::{AtomicPartitionedBloomFilter, AtomicScalableBloomFilter};

    #[cfg(feature = "trace")]
    pub use crate::filters::{QueryTrace, QueryTraceBuilder};

    #[cfg(feature = "serde")]
    pub use crate::serde_support::prelude::*;

    #[cfg(feature = "metrics")]
    pub use crate::metrics::prelude::*;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_prelude_imports() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        filter.insert(&"test".to_string());
        assert!(filter.contains(&"test".to_string()));
    }

    #[test]
    fn test_trait_usage() {
        fn test_filter<F: BloomFilter<String>>(filter: &mut F) {
            filter.insert(&"item".to_string());
            assert!(filter.contains(&"item".to_string()));
        }
        let mut filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        test_filter(&mut filter);
    }

    #[test]
    fn test_builder() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        assert!(filter.is_empty());
    }

    #[test]
    fn test_concurrent_filter_shared_trait() {
        use std::sync::Arc;
        let filter = Arc::new(ShardedBloomFilter::<String>::new(1000, 0.01));
        filter.insert(&"concurrent".to_string());
        assert!(filter.contains(&"concurrent".to_string()));
        let f2 = Arc::clone(&filter);
        let handle = std::thread::spawn(move || {
            f2.insert(&"thread_item".to_string());
        });
        handle.join().unwrap();
        assert!(filter.contains(&"thread_item".to_string()));
    }

    #[test]
    fn test_striped_filter_shared_trait() {
        use std::sync::Arc;
        let filter = Arc::new(StripedBloomFilter::<String>::new(1000, 0.01).unwrap());
        filter.insert(&"striped".to_string());
        assert!(filter.contains(&"striped".to_string()));
    }

    #[test]
    fn test_shared_filter_clear() {
        let filter = ShardedBloomFilter::<String>::new(1000, 0.01);
        filter.insert(&"item1".to_string());
        filter.insert(&"item2".to_string());
        assert!(!filter.is_empty());
        filter.clear();
        assert!(filter.is_empty());
        assert!(!filter.contains(&"item1".to_string()));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        filter.insert(&"serialize_me".to_string());
        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<String> = bincode::deserialize(&bytes).unwrap();
        assert!(restored.contains(&"serialize_me".to_string()));
    }

    #[test]
    fn test_concurrent_bloom_filter_trait() {
        use std::sync::Arc;
        let filter = Arc::new(StandardBloomFilter::<String>::new(1000, 0.01).unwrap());
        filter.insert_concurrent(&"atomic_item".to_string());
        assert!(filter.contains(&"atomic_item".to_string()));
        let f2 = Arc::clone(&filter);
        let handle = std::thread::spawn(move || {
            f2.insert_concurrent(&"thread_atomic".to_string());
        });
        handle.join().unwrap();
        assert!(filter.contains(&"thread_atomic".to_string()));
    }

    #[test]
    fn test_tree_bloom_filter_basic() {
        let mut filter = TreeBloomFilter::<String>::new(vec![2, 3], 100, 0.01).unwrap();
        filter.insert_to_bin(&"item1".to_string(), &[0, 1]).unwrap();
        assert!(filter.contains(&"item1".to_string()));
        let locations = filter.locate(&"item1".to_string());
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0], vec![0, 1]);
    }

    #[test]
    fn test_tree_bloom_filter_hierarchy() {
        let mut router = TreeBloomFilter::<String>::new(vec![3, 10], 1000, 0.01).unwrap();
        router.insert_to_bin(&"session:alice".to_string(), &[1, 5]).unwrap();
        assert!(router.contains_in_bin(&"session:alice".to_string(), &[1, 5]).unwrap());
        assert!(!router.contains_in_bin(&"session:alice".to_string(), &[0, 3]).unwrap());
        assert!(router.contains(&"session:alice".to_string()));
    }

    #[test]
    fn test_tree_vs_standard_difference() {
        let mut tree = TreeBloomFilter::<i32>::new(vec![2, 2], 100, 0.01).unwrap();
        let standard = StandardBloomFilter::<i32>::new(100, 0.01).unwrap();
        tree.insert_to_bin(&42, &[0, 1]).unwrap();
        assert!(tree.contains_in_bin(&42, &[0, 1]).unwrap());
        assert!(!tree.contains_in_bin(&42, &[1, 0]).unwrap());
        standard.insert(&42);
        assert!(standard.contains(&42));
    }
}
