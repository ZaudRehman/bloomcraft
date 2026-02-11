//! Bloom filter implementations.
//!
//! This module contains all Bloom filter variants provided by BloomCraft.
//!
//! # Available Filters
//!
//! ## Production Filters
//!
//! - [`StandardBloomFilter`] - General-purpose filter with optimal space efficiency
//! - [`CountingBloomFilter`] - Supports deletion using counters instead of bits
//! - [`ScalableBloomFilter`] - Dynamically grows to accommodate unbounded items
//! - [`PartitionedBloomFilter`] - Cache-optimized with L1/L2 alignment for 2-4× faster queries
//! - [`RegisterBlockedBloomFilter`] - Ultra-fast queries (512-bit AVX blocks, 20-30% faster than partitioned)
//! - [`TreeBloomFilter`] - Hierarchical organization with location tracking
//!
//! ## Concurrent Filters (feature-gated)
//!
//! - [`AtomicScalableBloomFilter`] - Lock-free concurrent scalable filter (requires `concurrent` feature)
//! - [`AtomicPartitionedBloomFilter`] - Lock-free cache-optimized filter (requires `concurrent` feature)
//!
//! ## Historical/Educational Filters
//!
//! - [`ClassicHashFilter`] - Burton Bloom's Method 1 (1970) using hash table with chaining
//! - [`ClassicBitsFilter`] - Burton Bloom's Method 2 (1970) using bit array
//!
//! # Choosing a Filter
//!
//! | Filter | Use Case | Memory | Operations |
//! |--------|----------|--------|------------|
//! | [`StandardBloomFilter`] | Known size, no deletion | Optimal (m bits) | Insert, Query |
//! | [`CountingBloomFilter`] | Need deletion | 4-8x Standard | Insert, Delete, Query |
//! | [`ScalableBloomFilter`] | Unknown size | Grows dynamically | Insert, Query, Auto-grow |
//! | [`AtomicScalableBloomFilter`] | Concurrent, unknown size | Grows dynamically | Insert, Query (lock-free) |
//! | [`PartitionedBloomFilter`] | Query-heavy (cache-fit) | ~1.2x Standard | Insert, Query (2-4x faster) |
//! | [`RegisterBlockedBloomFilter`] | Ultra-fast queries (high FPR) | 1.3-1.5× Standard | Insert, Query (3-5× faster, 2.5× FPR) |
//! | [`AtomicPartitionedBloomFilter`] | Concurrent, query-heavy | 1.05-1.2× Standard | Insert, Query (2-4× faster, lock-free) |
//! | [`TreeBloomFilter`] | Hierarchical data (DC/rack) | k × m bits | Insert, Query, Locate |
//! | [`ClassicHashFilter`] | Educational/research | O(n) elements | Insert, Query |
//! | [`ClassicBitsFilter`] | Educational/research | m bits | Insert, Query |
//!
//! # Examples
//!
//! ## Standard Bloom Filter
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! let mut filter: StandardBloomFilter<String> = StandardBloomFilter::new(10_000, 0.01);
//! filter.insert(&"hello".to_string());
//! assert!(filter.contains(&"hello".to_string()));
//! ```
//!
//! ## Counting Bloom Filter (with deletion)
//!
//! ```
//! use bloomcraft::filters::CountingBloomFilter;
//!
//! let mut filter: CountingBloomFilter<String> = CountingBloomFilter::new(10_000, 0.01);
//! filter.insert(&"temporary".to_string());
//! assert!(filter.contains(&"temporary".to_string()));
//!
//! filter.delete(&"temporary".to_string());
//! assert!(!filter.contains(&"temporary".to_string()));
//! ```
//!
//! ## Scalable Bloom Filter (dynamic growth)
//!
//! ```
//! use bloomcraft::filters::ScalableBloomFilter;
//!
//! let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);
//!
//! // Can insert far more than initial capacity
//! for i in 0..10_000 {
//!     filter.insert(&i);
//! }
//!
//! println!("Grew to {} sub-filters", filter.filter_count());
//! ```
//!
//! ## Concurrent Scalable Bloom Filter 
//!
//! ```ignore
//! use bloomcraft::filters::AtomicScalableBloomFilter;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let filter = Arc::new(AtomicScalableBloomFilter::new(1_000, 0.01));
//!
//! let mut handles = vec![];
//! for thread_id in 0..8 {
//!     let f = Arc::clone(&filter);
//!     let h = thread::spawn(move || {
//!         for i in 0..1_000 {
//!             f.insert(&(thread_id * 1_000 + i));
//!         }
//!     });
//!     handles.push(h);
//! }
//!
//! for h in handles {
//!     h.join().unwrap();
//! }
//!
//! assert_eq!(filter.len(), 8_000);
//! ```
//!
//! ## Partitioned Bloom Filter (cache-optimized, 2-4× faster)
//!
//! ```
//! use bloomcraft::filters::PartitionedBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! // Auto-tuned for CPU cache
//! let mut filter: PartitionedBloomFilter<String> =
//!     PartitionedBloomFilter::new_cache_tuned(10_000, 0.01).unwrap();
//!
//! filter.insert(&"item".to_string());
//! assert!(filter.contains(&"item".to_string())); // 2-4× faster queries
//!
//! // Or manually specify 64-byte cache alignment
//! let mut filter: PartitionedBloomFilter<String> =
//!     PartitionedBloomFilter::with_alignment(10_000, 0.01, 64).unwrap();
//! ```
//!
//! ## Register-Blocked Bloom Filter (ultra-fast, higher FPR)
//!
//! ```
//! use bloomcraft::filters::RegisterBlockedBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! // 512-bit blocks for maximum query speed
//! let mut filter: RegisterBlockedBloomFilter<u64> =
//!     RegisterBlockedBloomFilter::new(100_000, 0.01).unwrap();
//!
//! filter.insert(&42);
//! assert!(filter.contains(&42)); // 20-30% faster than partitioned
//!
//! println!("Target FPR: {:.2}%", filter.target_fpr() * 100.0);
//! println!("Actual FPR will be ~2.5-3.0% due to blocking overhead");
//! ```
//!
//! ## Concurrent Partitioned Bloom Filter
//!
//! ```ignore
//! #[cfg(feature = "concurrent")]
//! {
//!     use bloomcraft::filters::AtomicPartitionedBloomFilter;
//!     use std::sync::Arc;
//!
//!     let filter = Arc::new(
//!         AtomicPartitionedBloomFilter::<u64>::new(1_000_000, 0.01).unwrap()
//!     );
//!
//!     // Wait-free inserts from multiple threads
//!     let handles: Vec<_> = (0..8).map(|tid| {
//!         let f = Arc::clone(&filter);
//!         std::thread::spawn(move || {
//!             for i in 0..10_000 {
//!                 f.insert_concurrent(&(tid * 10_000 + i));
//!             }
//!         })
//!     }).collect();
//!
//!     for handle in handles {
//!         handle.join().unwrap();
//!     }
//! }
//! ```
//!
//! ## Tree Bloom Filter (hierarchical organization)
//!
//! ```
//! use bloomcraft::filters::TreeBloomFilter;
//!
//! // 4 regions, 8 datacenters per region
//! let mut filter: TreeBloomFilter<String> =
//!     TreeBloomFilter::new(vec![4, 8], 1000, 0.01).unwrap();
//!
//! // Insert to specific location
//! filter.insert_to_bin(&"user:12345".to_string(), &[2, 5]).unwrap(); // Region 2, DC 5
//!
//! // Find all locations containing this item
//! let locations = filter.locate(&"user:12345".to_string());
//! for loc in locations {
//!     println!("Found at path: {:?}", loc);
//! }
//! ```

#![warn(missing_docs)]
#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Production-grade filter implementations
pub mod standard;
pub use standard::StandardBloomFilter;

pub mod counting;
pub use counting::{CounterSize, CountingBloomFilter};

pub mod scalable;
pub use scalable::{GrowthStrategy, ScalableBloomFilter, CapacityExhaustedBehavior, QueryStrategy, ScalableHealthMetrics};

// Feature-gated exports from scalable
#[cfg(feature = "trace")]
pub use scalable::{QueryTrace, QueryTraceBuilder};

#[cfg(feature = "concurrent")]
pub use scalable::AtomicScalableBloomFilter;

mod partitioned;
pub use partitioned::PartitionedBloomFilter;

// Concurrent partitioned variant (feature-gated)
#[cfg(feature = "concurrent")]
mod atomic_partitioned;

#[cfg(feature = "concurrent")]
pub use atomic_partitioned::AtomicPartitionedBloomFilter;

// Register-blocked variant (always available)
mod register_blocked;
pub use register_blocked::RegisterBlockedBloomFilter;

pub mod tree;
pub use tree::{TreeBloomFilter, TreeConfig, TreeCapacityStats, TreeStats, LocateIter, MAX_TREE_DEPTH, MAX_TOTAL_NODES};

// Historical/educational implementations
mod classic_bits;
pub use classic_bits::ClassicBitsFilter;

mod classic_hash;
pub use classic_hash::ClassicHashFilter;

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::filter::BloomFilter;

    /// Verify that all filter types are accessible and can be instantiated.
    #[test]
    fn test_all_filters_accessible() {
        // Standard filter
        let _standard: StandardBloomFilter<String> = StandardBloomFilter::new(100, 0.01);

        // Counting filter
        let _counting: CountingBloomFilter<String> = CountingBloomFilter::new(100, 0.01);

        // Scalable filter
        let _scalable: ScalableBloomFilter<String> = ScalableBloomFilter::new(100, 0.01);

        // Partitioned filter
        let _partitioned: PartitionedBloomFilter<String> = PartitionedBloomFilter::new(100, 0.01).unwrap();

        // Tree filter
        let _tree: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 3], 100, 0.01).unwrap();

        // Classic filters
        let _classic_bits: ClassicBitsFilter<String> = ClassicBitsFilter::with_fpr(100, 0.01);
        let _classic_hash: ClassicHashFilter<String> = ClassicHashFilter::with_fpr(100, 0.01);
    }

    /// Test that CounterSize enum is accessible and works correctly.
    #[test]
    fn test_counter_size_enum() {
        assert_eq!(CounterSize::FourBit.max_value(), 15);
        assert_eq!(CounterSize::EightBit.max_value(), 255);
        assert_eq!(CounterSize::SixteenBit.max_value(), 65535);

        assert_eq!(CounterSize::FourBit.bits(), 4);
        assert_eq!(CounterSize::EightBit.bits(), 8);
        assert_eq!(CounterSize::SixteenBit.bits(), 16);
    }

    /// Test that GrowthStrategy enum is accessible and works correctly.
    #[test]
    fn test_growth_strategy_enum() {
        use scalable::GrowthStrategy;

        // Test Constant variant
        match GrowthStrategy::Constant {
            GrowthStrategy::Constant => {}
            _ => panic!("Expected Constant"),
        }

        // Test Geometric variant
        match GrowthStrategy::Geometric(2.0) {
            GrowthStrategy::Geometric(scale) => {
                assert_eq!(scale, 2.0);
            }
            _ => panic!("Expected Geometric"),
        }

        // Test default
        let default_strategy = GrowthStrategy::default();
        match default_strategy {
            GrowthStrategy::Geometric(scale) => {
                assert_eq!(scale, 2.0);
            }
            _ => panic!("Expected default to be Geometric(2.0)"),
        }
    }

    /// Test new scalable enums and structs.
    #[test]
    fn test_scalable_new_types() {
        // Test CapacityExhaustedBehavior
        let _silent = CapacityExhaustedBehavior::Silent;
        let _error = CapacityExhaustedBehavior::Error;
        #[cfg(debug_assertions)]
        let _panic = CapacityExhaustedBehavior::Panic;

        assert_eq!(CapacityExhaustedBehavior::default(), CapacityExhaustedBehavior::Silent);

        // Test QueryStrategy
        let _forward = QueryStrategy::Forward;
        let _reverse = QueryStrategy::Reverse;
        assert_eq!(QueryStrategy::default(), QueryStrategy::Reverse);

        // Test ScalableHealthMetrics
        let mut filter: ScalableBloomFilter<i32> = ScalableBloomFilter::new(100, 0.01);
        for i in 0..50 {
            filter.insert(&i);
        }

        let metrics = filter.health_metrics();
        assert!(metrics.filter_count > 0);
        assert_eq!(metrics.total_items, 50);
        assert!(metrics.estimated_fpr > 0.0);
    }

    #[cfg(feature = "concurrent")]
    #[test]
    fn test_atomic_scalable_filter() {
        use std::sync::Arc;
        use std::thread;

        let filter = Arc::new(AtomicScalableBloomFilter::new(100, 0.01));

        let mut handles = vec![];
        for thread_id in 0..4 {
            let f = Arc::clone(&filter);
            let h = thread::spawn(move || {
                for i in 0..25 {
                    f.insert(&(thread_id * 25 + i));
                }
            });
            handles.push(h);
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(filter.len(), 100);

        // Verify all items present
        for i in 0..100 {
            assert!(filter.contains(&i));
        }
    }

    /// Verify that all filter types implement Send + Sync.
    #[test]
    fn test_filters_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<StandardBloomFilter<String>>();
        assert_send_sync::<CountingBloomFilter<String>>();
        assert_send_sync::<ScalableBloomFilter<String>>();
        assert_send_sync::<PartitionedBloomFilter<String>>();
        assert_send_sync::<TreeBloomFilter<String>>();
        assert_send_sync::<ClassicBitsFilter<String>>();
        assert_send_sync::<ClassicHashFilter<String>>();
    }

    /// Test that filters can be used with different types.
    #[test]
    fn test_generic_type_flexibility() {
        // Integer types
        let _i32_filter: StandardBloomFilter<i32> = StandardBloomFilter::new(100, 0.01);
        let _u64_filter: StandardBloomFilter<u64> = StandardBloomFilter::new(100, 0.01);

        // String types
        let _string_filter: StandardBloomFilter<String> = StandardBloomFilter::new(100, 0.01);
        let _str_filter: StandardBloomFilter<&str> = StandardBloomFilter::new(100, 0.01);

        // Tuple types
        let _tuple_filter: StandardBloomFilter<(i32, String)> = StandardBloomFilter::new(100, 0.01);

        // Vector types
        let _vec_filter: StandardBloomFilter<Vec<u8>> = StandardBloomFilter::new(100, 0.01);
    }

    /// Verify basic insert/contains functionality across all filters.
    #[test]
    fn test_basic_functionality_all_filters() {
        // Standard
        let standard: StandardBloomFilter<i32> = StandardBloomFilter::new(100, 0.01);
        standard.insert(&42);
        assert!(standard.contains(&42));
        assert!(!standard.contains(&43));

        // Counting
        let mut counting: CountingBloomFilter<i32> = CountingBloomFilter::new(100, 0.01);
        counting.insert(&42);
        assert!(counting.contains(&42));
        let deleted = counting.delete(&42);
        assert!(deleted, "Item should have been deleted");
        assert!(!counting.contains(&42));

        // Scalable
        let mut scalable: ScalableBloomFilter<i32> = ScalableBloomFilter::new(10, 0.01);
        for i in 0..100 {
            scalable.insert(&i);
        }
        assert!(scalable.contains(&50));

        // Partitioned
        let mut partitioned: PartitionedBloomFilter<i32> = PartitionedBloomFilter::new(100, 0.01).unwrap();
        partitioned.insert(&42);
        assert!(partitioned.contains(&42));

        // Tree
        let mut tree: TreeBloomFilter<i32> = TreeBloomFilter::new(vec![2, 3], 100, 0.01).unwrap();
        tree.insert_to_bin(&42, &[0, 1]).unwrap();
        assert!(tree.contains_in_bin(&42, &[0, 1]).unwrap());

        // Classic bits
        let mut classic_bits: ClassicBitsFilter<i32> = ClassicBitsFilter::new(1000, 7);
        classic_bits.insert(&42);
        assert!(classic_bits.contains(&42));

        // Classic hash
        let mut classic_hash: ClassicHashFilter<i32> = ClassicHashFilter::new(1000, 3);
        classic_hash.insert(&42);
        assert!(classic_hash.contains(&42));
    }

    /// Test that module documentation examples are valid.
    #[test]
    fn test_documentation_patterns() {
        // Pattern 1: Type-annotated construction
        let _filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);

        // Pattern 2: Turbofish syntax
        let _filter = StandardBloomFilter::<String>::new(1000, 0.01);

        // Pattern 3: Inferred from usage
        let filter = StandardBloomFilter::new(1000, 0.01);
        filter.insert(&"hello".to_string());
        let _: bool = filter.contains(&"hello".to_string());
    }

    /// Verify that filters can be cleared.
    #[test]
    fn test_clear_functionality() {
        let mut standard: StandardBloomFilter<i32> = StandardBloomFilter::new(100, 0.01);
        standard.insert(&42);
        assert!(standard.contains(&42));
        standard.clear();
        assert!(!standard.contains(&42));

        let mut counting: CountingBloomFilter<i32> = CountingBloomFilter::new(100, 0.01);
        counting.insert(&42);
        assert!(counting.contains(&42));
        counting.clear();
        assert!(!counting.contains(&42));
    }

    /// Test batch operations.
    #[test]
    fn test_batch_operations() {
        let filter: StandardBloomFilter<i32> = StandardBloomFilter::new(100, 0.01);

        let items = vec![1, 2, 3, 4, 5];
        filter.insert_batch(&items);

        for item in &items {
            assert!(filter.contains(item));
        }

        let queries = vec![1, 2, 3, 6, 7, 8];
        let results = filter.contains_batch(&queries);
        assert_eq!(results[0..3], [true, true, true]);
        assert_eq!(results[3..6], [false, false, false]);
    }

    /// Test TreeBloomFilter specific functionality.
    #[test]
    fn test_tree_bloom_filter_locate() {
        let mut filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 2], 100, 0.01).unwrap();

        filter.insert_to_bin(&"item1".to_string(), &[0, 1]).unwrap();
        filter.insert_to_bin(&"item2".to_string(), &[1, 0]).unwrap();

        // Verify locate finds correct bin
        let loc1 = filter.locate(&"item1".to_string());
        assert_eq!(loc1.len(), 1);
        assert_eq!(loc1[0], vec![0, 1]);

        let loc2 = filter.locate(&"item2".to_string());
        assert_eq!(loc2.len(), 1);
        assert_eq!(loc2[0], vec![1, 0]);
    }

    /// Test TreeBloomFilter batch operations.
    #[test]
    fn test_tree_bloom_filter_batch() {
        let mut filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2], 100, 0.01).unwrap();

        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let refs: Vec<&String> = items.iter().collect();

        filter.insert_batch_to_bin(&refs, &[0]).unwrap();

        for item in &items {
            assert!(filter.contains(item));
        }

        assert_eq!(filter.len(), 3);
    }

    /// Test TreeBloomFilter stats.
    #[test]
    fn test_tree_bloom_filter_stats() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 3], 100, 0.01).unwrap();

        let stats = filter.stats();
        assert_eq!(stats.depth, 2);
        assert_eq!(stats.leaf_bins, 6); // 2 × 3
        assert!(stats.total_nodes > 0);
        assert!(stats.memory_usage > 0);
    }
}
