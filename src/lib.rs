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
//!
//! // Create a filter for 10,000 items with 1% false positive rate
//! let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(10_000, 0.01);
//!
//! // Insert items
//! filter.insert(&"hello");
//! filter.insert(&"world");
//!
//! // Query items
//! assert!(filter.contains(&"hello")); // true - definitely in set
//! assert!(!filter.contains(&"goodbye")); // false - definitely not in set
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
//! | Filter                 | Best For                  | Trade-offs              |
//! |------------------------|---------------------------|-------------------------|
//! | `StandardBloomFilter`  | General-purpose, known size | Optimal space, no deletion |
//! | `CountingBloomFilter`  | Dynamic sets with deletion | 4x memory overhead      |
//! | `ScalableBloomFilter`  | Unknown/unbounded size    | Grows dynamically       |
//! | `PartitionedBloomFilter` | High-performance queries | 2-4x faster queries    |
//! | `HierarchicalBloomFilter` | Multi-level data       | Location information   |
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
//! - Concurrent filters (sharded, striped)
//!
//! ## Optional Features
//!
//! - `serde` - Serialization support (JSON, Bincode, Zero-copy)
//! - `xxhash` - Fast xxHash function
//! - `wyhash` - Ultra-fast WyHash function
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
//! # Concurrent Filters
//!
//! ```
//! use bloomcraft::sync::ShardedBloomFilter;
//! use bloomcraft::core::BloomFilter;
//! use std::sync::Arc;
//!
//! let filter = Arc::new(std::sync::Mutex::new(ShardedBloomFilter::<String>::new(1_000_000, 0.01)));
//!
//! // Share across threads - use Mutex for mutable access
//! let filter_clone = Arc::clone(&filter);
//! std::thread::spawn(move || {
//!     filter_clone.lock().unwrap().insert(&"concurrent_item".to_string());
//! });
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
//! - **Sync module**: Thread-safety marker traits (`Send + Sync`)
//! - **SIMD module**: AVX2/NEON intrinsics with runtime feature detection
//! - **Zero unsafe in public APIs**: All public methods are safe
//!
//! All unsafe blocks include safety documentation justifying their use.

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
/// This module contains unsafe code for implementing Send/Sync marker traits.
/// The unsafe implementations are sound because:
/// - `ShardedBloomFilter` uses `Arc<BitVec>` which is thread-safe via atomics
/// - `StripedBloomFilter` uses `RwLock` for synchronization
/// - `CacheLinePadded<T>` is Send/Sync if T is Send/Sync
#[allow(unsafe_code)]
pub mod sync;

/// Type-safe builders for all filter types
pub mod builder;

/// Serialization support (requires `serde` feature)
#[cfg(feature = "serde")]
pub mod serde_support;

/// Observability and monitoring
#[cfg(feature = "metrics")]
pub mod metrics;

// Re-export commonly used types at crate root
pub use error::{BloomCraftError, Result};

// Re-export core trait
pub use core::filter::BloomFilter;

// Re-export all filter types at the crate root
pub use filters::{
    CountingBloomFilter, HierarchicalBloomFilter, PartitionedBloomFilter, ScalableBloomFilter,
    StandardBloomFilter, ClassicHashFilter, ClassicBitsFilter,
};

// Re-export builders at the crate root
pub use builder::{
    CountingBloomFilterBuilder, ScalableBloomFilterBuilder, StandardBloomFilterBuilder,
};

// Re-export concurrent filters at the crate root
pub use sync::{ShardedBloomFilter, StripedBloomFilter};

// Re-export common hash types
pub use hash::{BloomHasher, StdHasher};

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
    pub use crate::core::filter::BloomFilter;
    pub use crate::error::{BloomCraftError, Result};
    pub use crate::filters::{
        CountingBloomFilter, HierarchicalBloomFilter, PartitionedBloomFilter,
        ScalableBloomFilter, StandardBloomFilter, ClassicHashFilter, ClassicBitsFilter,
    };
    pub use crate::hash::{BloomHasher, StdHasher};
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

    // Metrics (always available)
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
        let mut filter = StandardBloomFilter::<String>::new(100, 0.01);
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
    fn test_concurrent_filter() {
        let mut filter = ShardedBloomFilter::<String>::new(1000, 0.01);
        filter.insert(&"concurrent".to_string());
        assert!(filter.contains(&"concurrent".to_string()));
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
}
