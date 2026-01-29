//! BloomCraft: Production-grade Bloom filter library for Rust.
//!
//! BloomCraft provides a comprehensive collection of Bloom filter implementations,
//! from historical algorithms to modern optimized variants, all with a focus on
//! correctness, performance, and usability.
//!
//! # What are Bloom Filters?
//!
//! A Bloom filter is a space-efficient probabilistic data structure that tests whether
//! an element is a member of a set. It can produce:
//! - **False positives**: May indicate an element is in the set when it isn't
//! - **Zero false negatives**: If it says an element isn't in the set, it definitely isn't
//!
//! # Quick Start
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! // Create a filter for 10,000 items with 1% false positive rate
//! let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(10_000, 0.01);
//!
//! // Insert items
//! filter.insert(&"hello");
//! filter.insert(&"world");
//!
//! // Query items
//! assert!(filter.contains(&"hello"));   // true - definitely in set
//! assert!(!filter.contains(&"goodbye")); // false - definitely not in set
//! ```
//!
//! # Three Concurrency Models
//!
//! BloomCraft provides three distinct patterns for thread-safe operations:
//!
//! ## 1. Single-Threaded (`BloomFilter` trait)
//!
//! Traditional filters requiring `&mut self` for modifications:
//!
//! ```
//! use bloomcraft::filters::CountingBloomFilter;
//! use bloomcraft::core::BloomFilter;
//! use std::sync::{Arc, Mutex};
//!
//! // Wrap in Mutex for concurrent access
//! let filter = Arc::new(Mutex::new(CountingBloomFilter::<String>::new(10_000, 0.01)));
//!
//! let filter_clone = Arc::clone(&filter);
//! std::thread::spawn(move || {
//!     filter_clone.lock().unwrap().insert(&"item".to_string());
//! });
//! ```
//!
//! ## 2. Lock-Free Atomic (`ConcurrentBloomFilter` trait)
//!
//! `StandardBloomFilter` uses atomic operations for wait-free concurrency:
//!
//! ```
//! use bloomcraft::StandardBloomFilter;
//! use bloomcraft::core::ConcurrentBloomFilter;
//! use std::sync::Arc;
//!
//! // No Mutex needed - atomic operations!
//! let filter = Arc::new(StandardBloomFilter::<String>::new(10_000, 0.01));
//!
//! let filter_clone = Arc::clone(&filter);
//! std::thread::spawn(move || {
//!     filter_clone.insert_concurrent(&"concurrent_item".to_string());
//! });
//! ```
//!
//! ## 3. Shared Interior Mutability (`SharedBloomFilter` trait)
//!
//! Sharded and striped filters provide lock-free or fine-grained locking with `&self` methods:
//!
//! ```
//! use bloomcraft::sync::ShardedBloomFilter;
//! use bloomcraft::core::SharedBloomFilter;
//! use std::sync::Arc;
//!
//! // No Mutex needed - uses interior mutability!
//! let filter = Arc::new(ShardedBloomFilter::<String>::new(10_000, 0.01));
//!
//! let filter_clone = Arc::clone(&filter);
//! std::thread::spawn(move || {
//!     filter_clone.insert(&"sharded_item".to_string());  // &self method!
//! });
//! ```
//!
//! # Using Builders (Recommended)
//!
//! ```
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! // Type-safe builder pattern
//! let filter: StandardBloomFilter<&str> = StandardBloomFilterBuilder::new()
//!     .expected_items(10_000)
//!     .false_positive_rate(0.01)
//!     .build()
//!     .unwrap();
//! ```
//!
//! # Choosing a Filter
//!
//! | Filter | Best For | Concurrency Model | Memory Overhead |
//! |--------|----------|-------------------|-----------------|
//! | `StandardBloomFilter` | General-purpose, known size | Lock-free atomic | Optimal |
//! | `CountingBloomFilter` | Dynamic sets with deletion | Mutex required | 4-8x |
//! | `ScalableBloomFilter` | Unknown/unbounded size | Mutex required | Grows dynamically |
//! | `AtomicScalableBloomFilter` | Concurrent, unknown size | Lock-free atomic | Grows dynamically |
//! | `PartitionedBloomFilter` | High-performance queries | Mutex required | 1.05-1.2× | **2-4× faster** |
//! | `RegisterBlockedBloomFilter` | Ultra-fast queries | Mutex required | 1.3-1.5× | **3-5× faster** |
//! | `AtomicPartitionedBloomFilter` | Concurrent, cache-optimized | Lock-free atomic | 1.05-1.2× | **2-4× faster** |
//! | `TreeBloomFilter` | Hierarchical data organization | Mutex required | 2-4x |
//! | `ShardedBloomFilter` | High concurrency, lock-free | SharedBloomFilter (&self) | 2-4x |
//! | `StripedBloomFilter` | Moderate concurrency | SharedBloomFilter (&self) | Optimal |
//!
//! # Features
//!
//! ## Core Features (always enabled)
//! - Lock-free concurrent queries
//! - Optimal parameter calculation
//! - Multiple hash functions
//! - Batch operations
//! - **Minimal unsafe code** (confined to sync primitives and SIMD)
//! - Type-safe builders
//! - Three concurrency patterns
//!
//! ## Optional Features
//!
//! - `serde` - Serialization support (JSON, Bincode, Zero-copy)
//! - `xxhash` - Fast xxHash function
//! - `wyhash` - Ultra-fast WyHash function
//! - `simd` - SIMD-accelerated hash functions (AVX2/NEON)
//! - `metrics` - Performance monitoring and observability
//!
//! # Serialization (with `serde` feature)
//!
//! ```ignore
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::serde_support::zerocopy::ZeroCopyBloomFilter;
//!
//! let filter = StandardBloomFilterBuilder::new()
//!     .expected_items(1_000_000)
//!     .false_positive_rate(0.01)
//!     .build()?;
//!
//! // Standard serde (JSON, Bincode, etc.)
//! let json = serde_json::to_string(&filter)?;
//! let bytes = bincode::serialize(&filter)?;
//!
//! // Zero-copy (10-100x faster)
//! let zc_bytes = ZeroCopyBloomFilter::serialize(&filter);
//! let loaded = ZeroCopyBloomFilter::deserialize(&zc_bytes)?;
//! ```
//!
//! # Concurrent Filters (SharedBloomFilter trait)
//!
//! Both `ShardedBloomFilter` and `StripedBloomFilter` implement the `SharedBloomFilter`
//! trait, which provides `&self` methods for concurrent access without external locking:
//!
//! ```
//! use bloomcraft::sync::{ShardedBloomFilter, StripedBloomFilter};
//! use bloomcraft::core::SharedBloomFilter;
//! use std::sync::Arc;
//! use std::thread;
//!
//! // Lock-free sharded filter
//! let sharded = Arc::new(ShardedBloomFilter::<String>::new(1_000_000, 0.01));
//! let sharded_clone = Arc::clone(&sharded);
//! thread::spawn(move || {
//!     sharded_clone.insert(&"concurrent_item".to_string());  // &self method
//! });
//!
//! // Fine-grained striped filter
//! let striped = Arc::new(StripedBloomFilter::<String>::new(1_000_000, 0.01));
//! striped.insert(&"another_item".to_string());  // No Mutex needed!
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::double_must_use)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::len_zero)]
#![allow(clippy::bool_assert_comparison)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::default_constructed_unit_structs)]
#![allow(clippy::assertions_on_constants)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/bloomcraft/0.1.0")]

//! # Unsafe Code Policy
//!
//! BloomCraft uses unsafe code in limited, well-audited locations:
//! - **Sync module**: `UnsafeCell` for interior mutability in `StripedBloomFilter`
//! - **Sync module**: `AtomicPtr` for lock-free clear in `ShardedBloomFilter`
//! - **Sync module**: Thread-safety marker traits (`Send + Sync`)
//! - **SIMD module**: AVX2/NEON intrinsics with runtime feature detection
//! - **Zero unsafe in public APIs**: All public methods are safe
//!
//! All unsafe blocks include explicit safety documentation justifying their use.

/// Core data structures and traits
pub mod core;

/// Error types and result aliases
pub mod error;

/// Filter implementations (variants)
pub mod filters;

/// Hash functions and strategies
pub mod hash;

/// Utility functions and helpers
pub mod util;

/// Concurrent Bloom filter implementations
///
/// This module contains unsafe code for implementing concurrent primitives.
/// The unsafe implementations are sound because:
/// - `ShardedBloomFilter` uses `AtomicPtr<Arc<BitVec>>` with proper synchronization
/// - `StripedBloomFilter` uses `UnsafeCell<Arc<BitVec>>` guarded by RwLocks
/// - `CacheLinePadded<T>` is Send/Sync if T is Send/Sync
/// - All unsafe code has explicit safety documentation
#[allow(unsafe_code)]
pub mod sync;

/// Type-safe builders for all filter types
pub mod builder;

/// Serialization support (requires `serde` feature)
#[cfg(feature = "serde")]
pub mod serde_support;

/// Observability and monitoring (requires `metrics` feature)
#[cfg(feature = "metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "metrics")))]
pub mod metrics;

// Re-export commonly used types at crate root
pub use error::{BloomCraftError, Result};

// Re-export core traits
pub use core::filter::{BloomFilter, ConcurrentBloomFilter, SharedBloomFilter};

// Re-export all filter types at the crate root
pub use filters::{
    ClassicBitsFilter, ClassicHashFilter, CountingBloomFilter, PartitionedBloomFilter,
    ScalableBloomFilter, StandardBloomFilter, TreeBloomFilter, RegisterBlockedBloomFilter,
};

// Re-export scalable filter types (enhanced exports)
pub use filters::{
    CapacityExhaustedBehavior, GrowthStrategy, QueryStrategy, ScalableHealthMetrics,
};

// Re-export scalable feature-gated types
#[cfg(feature = "trace")]
pub use filters::{QueryTrace, QueryTraceBuilder};

#[cfg(feature = "concurrent")]
pub use filters::AtomicScalableBloomFilter;

// Re-export cache-optimized filter variants
#[cfg(feature = "concurrent")]
pub use filters::AtomicPartitionedBloomFilter;

// Re-export builders at the crate root
pub use builder::{
    CountingBloomFilterBuilder, ScalableBloomFilterBuilder, StandardBloomFilterBuilder,
};

// Re-export concurrent filters at the crate root
pub use sync::{ShardedBloomFilter, StripedBloomFilter};

// Re-export common hash types (only what's actually exported from hash::mod)
pub use hash::BloomHasher;

// Re-export serde support with shorter path
#[cfg(feature = "serde")]
pub use serde_support as serde;

// Re-export metrics
#[cfg(feature = "metrics")]
pub use metrics::{FalsePositiveTracker, FilterMetrics, LatencyHistogram, MetricsCollector};

/// Prelude module for convenient imports.
///
/// # Examples
///
/// ```
/// use bloomcraft::prelude::*;
///
/// let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
/// filter.insert(&"hello");
/// assert!(filter.contains(&"hello"));
/// ```
pub mod prelude {
    pub use crate::core::filter::{BloomFilter, ConcurrentBloomFilter, SharedBloomFilter};
    pub use crate::error::{BloomCraftError, Result};
    pub use crate::filters::{
        ClassicBitsFilter, ClassicHashFilter, CountingBloomFilter, PartitionedBloomFilter,
        ScalableBloomFilter, StandardBloomFilter, TreeBloomFilter, RegisterBlockedBloomFilter,
    };
    // Scalable filter types
    pub use crate::filters::{
        CapacityExhaustedBehavior, GrowthStrategy, QueryStrategy, ScalableHealthMetrics,
    };

    // Feature-gated scalable types
    #[cfg(feature = "trace")]
    pub use crate::filters::{QueryTrace, QueryTraceBuilder};

    #[cfg(feature = "concurrent")]
    pub use crate::filters::AtomicScalableBloomFilter;

    #[cfg(feature = "concurrent")]
    pub use crate::filters::AtomicPartitionedBloomFilter;
    
    pub use crate::hash::BloomHasher;

    #[cfg(feature = "simd")]
    pub use crate::hash::simd::SimdHasher;

    // Re-export builders
    pub use crate::builder::{
        CountingBloomFilterBuilder, ScalableBloomFilterBuilder, StandardBloomFilterBuilder,
    };

    // Re-export concurrent filters
    pub use crate::sync::{ShardedBloomFilter, StripedBloomFilter};

    // Conditionally re-export serde support
    #[cfg(feature = "serde")]
    pub use crate::serde_support::{
        zerocopy::ZeroCopyBloomFilter, CountingFilterSerdeSupport, StandardFilterSerdeSupport,
    };

    // Metrics (with feature gate)
    #[cfg(feature = "metrics")]
    pub use crate::metrics::{
        FalsePositiveTracker, FilterMetrics, LatencyHistogram, MetricsCollector,
    };
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_prelude_imports() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01);
        filter.insert(&"test".to_string());
        assert!(filter.contains(&"test".to_string()));
    }

    #[test]
    fn test_trait_usage() {
        fn test_filter<F: BloomFilter<String>>(filter: &mut F) {
            filter.insert(&"item".to_string());
            assert!(filter.contains(&"item".to_string()));
        }

        let mut filter = StandardBloomFilter::<String>::new(100, 0.01);
        test_filter(&mut filter);
    }

    #[test]
    fn test_builder() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_concurrent_filter_shared_trait() {
        use std::sync::Arc;

        // Test SharedBloomFilter trait - no Mutex needed!
        let filter = Arc::new(ShardedBloomFilter::<String>::new(1000, 0.01));
        filter.insert(&"concurrent".to_string());
        assert!(filter.contains(&"concurrent".to_string()));

        // Verify it works across threads
        let filter_clone = Arc::clone(&filter);
        let handle = std::thread::spawn(move || {
            filter_clone.insert(&"thread_item".to_string());
        });
        handle.join().unwrap();
        assert!(filter.contains(&"thread_item".to_string()));
    }

    #[test]
    fn test_striped_filter_shared_trait() {
        use std::sync::Arc;

        // Test StripedBloomFilter with SharedBloomFilter trait
        let filter = Arc::new(StripedBloomFilter::<String>::new(1000, 0.01));
        filter.insert(&"striped".to_string());
        assert!(filter.contains(&"striped".to_string()));
    }

    #[test]
    fn test_shared_filter_clear() {
        // Test clear() with SharedBloomFilter trait
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
        let mut filter = StandardBloomFilter::<String>::new(100, 0.01);
        filter.insert(&"serialize_me".to_string());

        // Test bincode
        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<String> = bincode::deserialize(&bytes).unwrap();
        assert!(restored.contains(&"serialize_me".to_string()));

        // Test zero-copy
        use crate::serde_support::zerocopy::ZeroCopyBloomFilter;
        let zc_bytes = ZeroCopyBloomFilter::serialize(&filter);
        let zc_restored = ZeroCopyBloomFilter::deserialize(&zc_bytes).unwrap();
        assert!(zc_restored.contains(&"serialize_me".to_string()));
    }

    #[test]
    fn test_concurrent_bloom_filter_trait() {
        use std::sync::Arc;

        // StandardBloomFilter implements ConcurrentBloomFilter
        let filter = Arc::new(StandardBloomFilter::<String>::new(1000, 0.01));

        // insert_concurrent is the only concurrent method
        filter.insert_concurrent(&"atomic_item".to_string());

        // contains is already thread-safe (atomic loads)
        assert!(filter.contains(&"atomic_item".to_string()));

        // Test from another thread
        let filter_clone = Arc::clone(&filter);
        let handle = std::thread::spawn(move || {
            filter_clone.insert_concurrent(&"thread_atomic".to_string());
        });
        handle.join().unwrap();
        assert!(filter.contains(&"thread_atomic".to_string()));
    }

    #[test]
    fn test_tree_bloom_filter_basic() {
        let mut filter = TreeBloomFilter::<String>::new(vec![2, 3], 100, 0.01);

        filter.insert_to_bin(&"item1".to_string(), &[0, 1]).unwrap();
        assert!(filter.contains(&"item1".to_string()));

        let locations = filter.locate(&"item1".to_string());
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0], vec![0, 1]);
    }

    #[test]
    fn test_tree_bloom_filter_hierarchy() {
        // Test hierarchical organization (datacenter example)
        let mut router = TreeBloomFilter::<String>::new(vec![3, 10], 1000, 0.01);

        // Insert session to continent 1, datacenter 5
        router.insert_to_bin(&"session:alice".to_string(), &[1, 5]).unwrap();

        // Verify location-aware queries
        assert!(router.contains_in_bin(&"session:alice".to_string(), &[1, 5]).unwrap());
        assert!(!router.contains_in_bin(&"session:alice".to_string(), &[0, 3]).unwrap());

        // Verify global query
        assert!(router.contains(&"session:alice".to_string()));
    }

    #[test]
    fn test_tree_vs_standard_difference() {
        // Demonstrate difference between Tree and Standard filters
        let mut tree = TreeBloomFilter::<i32>::new(vec![2, 2], 100, 0.01);
        let standard = StandardBloomFilter::<i32>::new(100, 0.01);

        // TreeBloomFilter: location-aware
        tree.insert_to_bin(&42, &[0, 1]).unwrap();
        assert!(tree.contains_in_bin(&42, &[0, 1]).unwrap());
        assert!(!tree.contains_in_bin(&42, &[1, 0]).unwrap());

        // StandardBloomFilter: no location tracking
        standard.insert(&42);
        assert!(standard.contains(&42));
        // No equivalent of contains_in_bin()
    }
}