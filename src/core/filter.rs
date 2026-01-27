//! Core Bloom filter trait definitions.
//!
//! This module defines the fundamental traits that all Bloom filter variants must implement.
//! These traits establish the contract for Bloom filter behavior, guarantees, and operations.
//!
//! # Design Principles
//!
//! 1. **No False Negatives**: If an item was inserted, `contains()` MUST return `true`
//! 2. **Bounded False Positives**: False positive rate should match configured parameters
//! 3. **Thread Safety**: All implementations must be `Send + Sync`
//! 4. **Type Safety**: Generic over item type to prevent type mismatches
//!
//! # Trait Hierarchy
//!
//! ```text
//! BloomFilter<T> (single-threaded, requires &mut self)
//!     ├── MutableBloomFilter<T> (marker for filters requiring &mut self)
//!     ├── DeletableBloomFilter<T> (supports deletion via counters)
//!     ├── MergeableBloomFilter<T> (supports union/intersection)
//!     └── ScalableBloomFilter<T> (supports dynamic growth)
//!
//! ConcurrentBloomFilter<T> (extension trait for lock-free atomic operations)
//!     └── (Only StandardBloomFilter implements this)
//!
//! SharedBloomFilter<T> (separate trait, methods take &self)
//!     └── (ShardedBloomFilter, StripedBloomFilter implement this)
//! ```
//!
//! # Three Concurrency Models
//!
//! BloomCraft provides three distinct patterns for thread-safe operations:
//!
//! ## 1. Single-Threaded (`BloomFilter` trait)
//!
//! Traditional filters requiring `&mut self` for modifications. These filters have:
//! - **Zero synchronization overhead**
//! - **Optimal single-threaded performance**
//! - **Explicit ownership semantics**
//!
//! **Implementations:**
//! - `StandardBloomFilter` (also supports `ConcurrentBloomFilter`)
//! - `CountingBloomFilter`
//! - `ScalableBloomFilter`
//! - `PartitionedBloomFilter`
//! - `TreeBloomFilter`
//! - `ClassicHashFilter`
//! - `ClassicBitsFilter`
//!
//! **Concurrent Usage:** Wrap in `Mutex` or `RwLock`
//!
//! ```ignore
//! use bloomcraft::filters::CountingBloomFilter;
//! use bloomcraft::core::BloomFilter;
//! use std::sync::{Arc, Mutex};
//!
//! let filter = Arc::new(Mutex::new(CountingBloomFilter::<String>::new(10_000, 0.01)));
//!
//! let filter_clone = Arc::clone(&filter);
//! std::thread::spawn(move || {
//!     filter_clone.lock().unwrap().insert(&"item".to_string());
//! });
//! ```
//!
//! ## 2. Lock-Free Atomic (`ConcurrentBloomFilter` extension trait)
//!
//! `StandardBloomFilter` extends `BloomFilter` with lock-free atomic operations:
//! - **Wait-free inserts** (no blocking)
//! - **Methods take `&self`** via `_concurrent` suffix
//! - **Direct `Arc` usage** without locks
//!
//! **Only** `StandardBloomFilter` implements this because it uses `AtomicU64` for all state.
//!
//! ```ignore
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
//! ## 3. Interior Mutability (`SharedBloomFilter` trait)
//!
//! Separate trait for filters using sharding, striping, or other interior mutability:
//! - **Methods take `&self`** (standard names, no suffix)
//! - **Lock-free or fine-grained locking** internally
//! - **Direct `Arc` usage** without external locks
//!
//! **Implementations:**
//! - `ShardedBloomFilter` (lock-free via independent shards)
//! - `StripedBloomFilter` (fine-grained RwLock striping)
//!
//! ```ignore
//! use bloomcraft::sync::ShardedBloomFilter;
//! use bloomcraft::core::SharedBloomFilter;
//! use std::sync::Arc;
//!
//! // No Mutex needed - sharding provides concurrency!
//! let filter = Arc::new(ShardedBloomFilter::<String>::new(10_000, 0.01));
//!
//! let filter_clone = Arc::clone(&filter);
//! std::thread::spawn(move || {
//!     filter_clone.insert(&"sharded_item".to_string());  // &self method!
//! });
//! ```
//!
//! # Why Three Patterns?
//!
//! Rust's type system distinguishes between:
//! - **Exclusive access** (`&mut T`) - "I'm the only one modifying this"
//! - **Shared access** (`&T`) - "Others may access concurrently"
//!
//! 1. **`BloomFilter`** uses `&mut self` for zero-overhead single-threaded performance
//! 2. **`ConcurrentBloomFilter`** extends `BloomFilter` with `&self` atomic methods
//! 3. **`SharedBloomFilter`** is a separate trait for filters that ONLY support `&self`
//!
//! These are fundamentally different contracts that cannot be unified.
//!
//! # Type Parameter
//!
//! All traits are generic over the item type `T`, ensuring:
//! - **Type safety**: Cannot insert `String` and query for `i32` on the same filter
//! - **Cleaner API**: No type annotations needed on every method call
//! - **Better error messages**: Compiler catches type mismatches at compile time
//!
//! # Filter Selection Guide
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │ Use Case                     │ Recommended Filter                        │
//! ├──────────────────────────────────────────────────────────────────────────┤
//! │ General-purpose, known size  │ StandardBloomFilter                       │
//! │ High-concurrency writes      │ ShardedBloomFilter (lock-free)            │
//! │ Moderate concurrency         │ StripedBloomFilter (fine-grained locks)   │
//! │ Need deletion support        │ CountingBloomFilter (4x memory)           │
//! │ Unknown/growing dataset      │ ScalableBloomFilter (dynamic growth)      │
//! │ High-performance queries     │ PartitionedBloomFilter (2-4x faster)      │
//! │ Multi-level data             │ TreeBloomFilter (location info)   │
//! │ Historical/research          │ ClassicHashFilter, ClassicBitsFilter      │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Concurrency Pattern Summary
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ Filter Type              │ Single-threaded │ Multi-threaded             │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │ StandardBloomFilter      │ &mut filter     │ Arc<filter> (no lock!)     │
//! │ ShardedBloomFilter       │ N/A             │ Arc<filter> (no lock!)     │
//! │ StripedBloomFilter       │ N/A             │ Arc<filter> (no lock!)     │
//! │ CountingBloomFilter      │ &mut filter     │ Arc<Mutex<filter>>         │
//! │ ScalableBloomFilter      │ &mut filter     │ Arc<Mutex<filter>>         │
//! │ PartitionedBloomFilter   │ &mut filter     │ Arc<RwLock<filter>>        │
//! │ TreeBloomFilter          │ &mut filter     │ Arc<RwLock<filter>>        │
//! │ ClassicHashFilter        │ &mut filter     │ Arc<Mutex<filter>>         │
//! │ ClassicBitsFilter        │ &mut filter     │ Arc<Mutex<filter>>         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use crate::error::Result;
use std::hash::Hash;

/// Single-threaded Bloom filter trait.
///
/// Defines the minimum operations that every single-threaded Bloom filter must support.
/// All implementations guarantee **no false negatives** while allowing **bounded false positives**.
///
/// Implementations of this trait require exclusive access (`&mut self`) for
/// modifications. This design provides:
/// - Zero synchronization overhead
/// - Clear ownership semantics
/// - Optimal single-threaded performance
///
/// # Type Parameter
///
/// * `T` - Type of items stored in the filter (must implement [`Hash`])
///
/// # Guarantees
///
/// ## No False Negatives
/// ```text
/// filter.insert(&x);
/// assert!(filter.contains(&x)); // MUST be true
/// ```
///
/// ## Determinism
/// Given the same items and configuration, operations produce identical results across runs.
///
/// ## Thread Safety
/// All implementations must be `Send + Sync`. Methods are safe to call from multiple
/// threads when using appropriate synchronization primitives (`Mutex`/`RwLock`).
///
/// # Mutability Contract
///
/// Requires exclusive access (`&mut self`) for `insert()` because:
/// 1. They modify internal structures (Vec growth, HashMap updates, tree mutations)
/// 2. They track metadata requiring coordinated updates (insertion count, cardinality estimates)
/// 3. They cannot use lock-free atomic operations for all state
///
/// For concurrent writes with these filters, wrap in `Arc<Mutex<_>>` or `Arc<RwLock<_>>`.
///
/// Examples: `StandardBloomFilter`, `ScalableBloomFilter`, `CountingBloomFilter`
///
/// # Examples
///
/// ```ignore
/// use bloomcraft::filters::StandardBloomFilter;
/// use bloomcraft::core::BloomFilter;
///
/// let mut filter = StandardBloomFilter::<String>::new(1000, 0.01);
/// filter.insert(&"hello".to_string());
/// assert!(filter.contains(&"hello".to_string()));
/// ```
pub trait BloomFilter<T: Hash>: Send + Sync {
    /// Insert an item into the filter.
    ///
    /// After this operation, `contains(item)` is guaranteed to return `true`.
    /// Inserting the same item multiple times is idempotent from a correctness
    /// perspective (though some implementations may update internal counters).
    ///
    /// # Arguments
    ///
    /// * `item` - Item to insert (must implement [`Hash`])
    ///
    /// # Thread Safety
    ///
    /// Requires exclusive access (`&mut self`). For concurrent writes, wrap the filter
    /// in `Arc<Mutex<_>>` or use `ConcurrentBloomFilter` implementations.
    ///
    /// # Performance
    ///
    /// * Time: O(k) where k is the number of hash functions
    /// * Space: O(1) - no allocations
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(1000, 0.01);
    /// filter.insert(&42);
    /// filter.insert(&100);
    /// assert!(filter.contains(&42));
    /// ```
    fn insert(&mut self, item: &T);

    /// Check if an item might be in the filter.
    ///
    /// # Returns
    ///
    /// * `true` - Item **might** be present (could be false positive)
    /// * `false` - Item is **definitely NOT** present (guaranteed)
    ///
    /// # Guarantees
    ///
    /// * If `insert(x)` was called, `contains(x)` MUST return `true`
    /// * If `contains(x)` returns `false`, `x` was definitely never inserted
    /// * False positives are bounded by the configured rate
    ///
    /// # Thread Safety
    ///
    /// Safe to call concurrently with other `contains()` calls. Concurrent calls
    /// with `insert()` are safe only if the implementation uses atomics.
    ///
    /// # Performance
    ///
    /// * Time: O(k) where k is the number of hash functions
    /// * Best case: O(1) if first tested bit is 0 (early termination possible)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(1000, 0.01);
    /// filter.insert(&42);
    ///
    /// assert!(filter.contains(&42));    // true - item was inserted
    /// assert!(!filter.contains(&43));   // false - item not inserted
    /// ```
    #[must_use]
    fn contains(&self, item: &T) -> bool;

    /// Clear all items from the filter.
    ///
    /// Resets the filter to its initial empty state. After this operation:
    /// * `len()` returns 0
    /// * `is_empty()` returns `true`
    /// * All `contains()` queries return `false`
    ///
    /// # Thread Safety
    ///
    /// Requires exclusive access. **NOT safe** to call concurrently with `insert()`
    /// or `contains()`. Only call when you have exclusive access to the filter.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(1000, 0.01);
    /// filter.insert(&42);
    /// filter.insert(&100);
    ///
    /// filter.clear();
    ///
    /// assert_eq!(filter.len(), 0);
    /// assert!(!filter.contains(&42));
    /// assert!(!filter.contains(&100));
    /// ```
    fn clear(&mut self);

    /// Get the approximate number of items inserted.
    ///
    /// # Returns
    ///
    /// Implementation-dependent count of items or operations.
    ///
    /// # Implementation Notes
    ///
    /// Different implementations return different metrics:
    /// * **Standard Bloom**: May return insertion count or number of set bits
    /// * **Counting Bloom**: Number of non-zero counters
    /// * **Scalable Bloom**: Total across all sub-filters
    ///
    /// This is **NOT** an accurate unique item count. For cardinality estimation,
    /// use [`estimate_count()`](Self::estimate_count).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(1000, 0.01);
    /// assert_eq!(filter.len(), 0);
    ///
    /// filter.insert(&1);
    /// filter.insert(&2);
    /// assert!(filter.len() > 0);
    /// ```
    #[must_use]
    fn len(&self) -> usize;

    /// Check if the filter is empty.
    ///
    /// # Returns
    ///
    /// `true` if no items have been inserted, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(1000, 0.01);
    /// assert!(filter.is_empty());
    ///
    /// filter.insert(&42);
    /// assert!(!filter.is_empty());
    /// ```
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimate the current false positive rate.
    ///
    /// Calculates the theoretical false positive probability based on:
    /// * Filter size (m bits)
    /// * Number of items inserted (n)
    /// * Number of hash functions (k)
    ///
    /// Formula: `(1 - e^(-kn/m))^k`
    ///
    /// # Returns
    ///
    /// Estimated false positive probability in range [0.0, 1.0]
    ///
    /// # Accuracy
    ///
    /// This is a **theoretical estimate** assuming:
    /// * Perfect uniform hash distribution
    /// * Independent hash functions
    /// * All inserted items are unique
    ///
    /// Empirical false positive rates typically fall within **10-20%** of this estimate
    /// for well-configured filters operating within their designed capacity.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(1000, 0.01);
    ///
    /// // Empty filter has 0% FP rate
    /// assert_eq!(filter.false_positive_rate(), 0.0);
    ///
    /// // Insert to capacity
    /// for i in 0..1000 {
    ///     filter.insert(&i);
    /// }
    ///
    /// // Should be close to 1%
    /// let fp_rate = filter.false_positive_rate();
    /// assert!(fp_rate >= 0.008 && fp_rate <= 0.012);
    /// ```
    #[must_use]
    fn false_positive_rate(&self) -> f64;

    /// Get the configured capacity of this filter.
    ///
    /// Returns the expected number of items the filter was designed to hold
    /// at the target false positive rate.
    ///
    /// # Returns
    ///
    /// Expected capacity in number of items
    ///
    /// # Note
    ///
    /// You can insert more items than the capacity, but the false positive
    /// rate will increase beyond the configured target.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::StandardBloomFilter;
    ///
    /// let filter = StandardBloomFilter::<String>::new(10000, 0.01);
    /// assert_eq!(filter.expected_items(), 10000);
    /// ```
    #[must_use]
    fn expected_items(&self) -> usize;

    /// Get the size of the filter in bits.
    ///
    /// Returns the total number of bits allocated for the filter's storage.
    ///
    /// # Returns
    ///
    /// Filter size in bits
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::StandardBloomFilter;
    ///
    /// let filter = StandardBloomFilter::<String>::new(10000, 0.01);
    /// let bits = filter.bit_count();
    /// println!("Filter uses {} bits ({} bytes)", bits, bits / 8);
    /// ```
    #[must_use]
    fn bit_count(&self) -> usize;

    /// Get the number of hash functions used.
    ///
    /// # Returns
    ///
    /// Number of hash functions (k)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::StandardBloomFilter;
    ///
    /// let filter = StandardBloomFilter::<String>::new(10000, 0.01);
    /// println!("Using {} hash functions", filter.hash_count());
    /// ```
    #[must_use]
    fn hash_count(&self) -> usize;

    /// Insert multiple items.
    ///
    /// Equivalent to calling `insert()` for each item. Implementations may
    /// override this for batch-specific optimizations, but the default
    /// implementation is typically sufficient.
    ///
    /// # Arguments
    ///
    /// * `items` - Iterator of items to insert
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(10000, 0.01);
    /// let items = vec!;[1][2][3][4][5]
    /// filter.insert_batch(items.iter());
    ///
    /// for item in &items {
    ///     assert!(filter.contains(item));
    /// }
    /// ```
    fn insert_batch<'a, I>(&mut self, items: I)
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        for item in items {
            self.insert(item);
        }
    }

    /// Check if multiple items might all be in the filter.
    ///
    /// Returns `true` only if **ALL** items might be present.
    ///
    /// # Arguments
    ///
    /// * `items` - Iterator of items to check
    ///
    /// # Returns
    ///
    /// `true` if all items might be present, `false` if any item is definitely absent
    ///
    /// # Performance
    ///
    /// Short-circuits on first `false` result, making best case O(k) (one hash check).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(10000, 0.01);
    /// filter.insert(&1);
    /// filter.insert(&2);
    /// filter.insert(&3);
    ///
    /// assert!(filter.contains_all(vec![&1, &2, &3].into_iter()));
    /// assert!(!filter.contains_all(vec![&1, &2, &4].into_iter()));
    /// ```
    #[must_use]
    fn contains_all<'a, I>(&self, items: I) -> bool
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        items.into_iter().all(|item| self.contains(item))
    }

    /// Check if any of the items might be in the filter.
    ///
    /// Returns `true` if **at least one** item might be present.
    ///
    /// # Arguments
    ///
    /// * `items` - Iterator of items to check
    ///
    /// # Returns
    ///
    /// `true` if any item might be present, `false` if all items are definitely absent
    ///
    /// # Performance
    ///
    /// Short-circuits on first `true` result.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(10000, 0.01);
    /// filter.insert(&1);
    ///
    /// assert!(filter.contains_any(vec![&1, &2, &3].into_iter()));
    /// assert!(!filter.contains_any(vec![&2, &3, &4].into_iter()));
    /// ```
    #[must_use]
    fn contains_any<'a, I>(&self, items: I) -> bool
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        items.into_iter().any(|item| self.contains(item))
    }

    /// Estimate the number of unique items in the filter.
    ///
    /// Uses cardinality estimation based on the number of set bits to approximate
    /// the count of **unique** items, as opposed to [`len()`](Self::len) which may
    /// count insertions or set bits.
    ///
    /// # Formula
    ///
    /// ```text
    /// n_estimated = -(m/k) × ln(1 - X/m)
    /// ```
    ///
    /// where:
    /// * `m` = total number of bits in the filter
    /// * `k` = number of hash functions
    /// * `X` = number of bits currently set to 1
    ///
    /// # Returns
    ///
    /// Estimated number of unique items
    ///
    /// # Accuracy
    ///
    /// This is an **approximation** with accuracy depending on filter saturation:
    /// * **Most accurate**: 30-70% full (typical error < 5%)
    /// * **Less accurate**: < 10% or > 90% full (error can exceed 20%)
    ///
    /// # Implementation Requirements
    ///
    /// Implementations **SHOULD** override this method with proper cardinality estimation.
    /// The default implementation simply returns `self.len()`, which is typically **NOT**
    /// an accurate unique item count.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter};
    ///
    /// let mut filter = StandardBloomFilter::<i32>::new(10000, 0.01);
    /// for i in 0..1000 {
    ///     filter.insert(&i);
    /// }
    ///
    /// let estimated = filter.estimate_count();
    /// // With proper implementation, should be within 5-10% of 1000
    /// let error = ((estimated as i32 - 1000).abs() as f64 / 1000.0) * 100.0;
    /// assert!(error < 10.0, "Estimation error: {:.1}%", error);
    /// ```
    #[must_use]
    fn estimate_count(&self) -> usize {
        // Default: return len() - implementations should override with proper estimation
        self.len()
    }
}

/// Marker trait for filters requiring exclusive access for inserts.
///
/// Some filters cannot use lock-free atomic operations due to:
/// - Mutable internal structures (trees, hash tables)
/// - Metadata tracking requiring coordinated updates
/// - Memory reallocation on insert
///
/// These filters require `Arc<Mutex<Filter>>` or `Arc<RwLock<Filter>>`
/// for concurrent access.
///
/// # Implementations
///
/// - `ScalableBloomFilter`: Requires write lock when growing
/// - `TreeBloomFilter`: Tree structure needs mutable access
/// - `ClassicHashFilter`: Hash table with chaining
///
/// # Usage
///
/// ```ignore
/// use bloomcraft::{ScalableBloomFilter, BloomFilter};
/// use std::sync::{Arc, Mutex};
/// use std::thread;
///
/// // Wrap in Mutex for concurrent access
/// let filter = Arc::new(Mutex::new(
///     ScalableBloomFilter::<String>::new(1000, 0.01)
/// ));
///
/// let handles: Vec<_> = (0..8).map(|i| {
///     let f = Arc::clone(&filter);
///     thread::spawn(move || {
///         let mut guard = f.lock().unwrap();
///         guard.insert(&format!("item_{}", i));
///     })
/// }).collect();
/// ```
pub trait MutableBloomFilter<T: Hash>: BloomFilter<T> {
    /// Insert with exclusive access (for filters that need `&mut self`).
    ///
    /// This method is automatically implemented for any type that implements
    /// `BloomFilter` but cannot use lock-free operations.
    fn insert_mut(&mut self, item: &T) {
        // Default: delegate to trait's insert() which uses &self
        self.insert(item);
    }
}

/// Extension trait for Bloom filters with lock-free concurrent operations.
///
/// Only filters where **ALL internal state uses atomic operations** can implement
/// this trait. Currently, only `StandardBloomFilter` qualifies.
///
/// # Concurrent Safety Guarantees
///
/// Implementations provide:
/// - **Wait-free inserts**: No thread blocking
/// - **Progress guarantee**: Bounded completion time
/// - **Memory safety**: Proper atomic orderings
///
/// # Performance
///
/// Expected scaling with `Arc<Filter>`:
/// - 2 threads: 1.7-2.0x throughput
/// - 4 threads: 3.0-3.6x throughput  
/// - 8 threads: 5.5-7.0x throughput
///
/// # Usage
///
/// ```ignore
/// use bloomcraft::{StandardBloomFilter, ConcurrentBloomFilter};
/// use std::sync::Arc;
/// use std::thread;
///
/// // No Mutex needed - direct Arc usage
/// let filter = Arc::new(StandardBloomFilter::<String>::new(10_000, 0.01));
///
/// let handles: Vec<_> = (0..8).map(|i| {
///     let f = Arc::clone(&filter);
///     thread::spawn(move || {
///         f.insert_concurrent(&format!("item_{}", i));
///     })
/// }).collect();
///
/// for h in handles { h.join().unwrap(); }
/// ```
pub trait ConcurrentBloomFilter<T: Hash>: BloomFilter<T> {
    /// Insert an item using lock-free atomic operations.
    ///
    /// Safe to call concurrently from multiple threads when wrapped in `Arc`.
    fn insert_concurrent(&self, item: &T);
    
    /// Batch insert using lock-free operations.
    fn insert_batch_concurrent(&self, items: &[T]) {
        for item in items {
            self.insert_concurrent(item);
        }
    }
    
    /// Batch insert by reference using lock-free operations.
    fn insert_batch_ref_concurrent(&self, items: &[&T]) {
        for item in items {
            self.insert_concurrent(item);
        }
    }
}


/// Thread-safe Bloom filter trait using interior mutability.
///
/// For filters using **sharding**, **striping**, or other interior mutability
/// patterns. These filters don't implement `BloomFilter` because they take
/// `&self` for all operations (incompatible signatures).
///
/// # Design
///
/// Methods take `&self` because synchronization is handled internally via:
/// - Lock-free sharding (multiple independent filters)
/// - Fine-grained locking (striped locks)
/// - Atomic operations per shard
///
/// # Usage
///
/// ```ignore
/// use std::sync::Arc;
/// use bloomcraft::sync::ShardedBloomFilter;
/// use bloomcraft::core::SharedBloomFilter;
///
/// let filter = Arc::new(ShardedBloomFilter::<String>::new(10_000, 0.01));
///
/// let handles: Vec<_> = (0..8).map(|i| {
///     let f = Arc::clone(&filter);
///     thread::spawn(move || {
///         f.insert(&format!("item_{}", i));  // &self method!
///     })
/// }).collect();
/// ```
pub trait SharedBloomFilter<T: Hash + Send + Sync>: Send + Sync {
    /// Insert an item (thread-safe, interior mutability).
    fn insert(&self, item: &T);

    /// Check if an item is possibly in the set (thread-safe).
    #[must_use]
    fn contains(&self, item: &T) -> bool;

    /// Remove all items (thread-safe).
    fn clear(&self);

    /// Get the number of bits set (thread-safe).
    #[must_use]
    fn len(&self) -> usize;

    /// Check if empty (thread-safe).
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the expected false positive rate.
    #[must_use]
    fn false_positive_rate(&self) -> f64;

    /// Estimate current item count.
    #[must_use]
    fn estimate_count(&self) -> usize;

    /// Get expected capacity.
    #[must_use]
    fn expected_items(&self) -> usize;

    /// Get total bit count.
    #[must_use]
    fn bit_count(&self) -> usize;

    /// Get hash function count.
    #[must_use]
    fn hash_count(&self) -> usize;

    /// Insert multiple items (thread-safe batch).
    fn insert_batch<'a, I>(&self, items: I)
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        for item in items {
            self.insert(item);
        }
    }

    /// Query multiple items (thread-safe batch).
    #[must_use]
    fn contains_batch<'a, I>(&self, items: I) -> Vec<bool>
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        items.into_iter().map(|item| self.contains(item)).collect()
    }

    /// Check if multiple items might all be in the filter (thread-safe).
    #[must_use]
    fn contains_all<'a, I>(&self, items: I) -> bool
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        items.into_iter().all(|item| self.contains(item))
    }

    /// Check if any items might be in the filter (thread-safe).
    #[must_use]
    fn contains_any<'a, I>(&self, items: I) -> bool
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        items.into_iter().any(|item| self.contains(item))
    }
}

/// Extension trait for Bloom filters that support deletion.
///
/// Standard Bloom filters cannot remove items because clearing a bit might
/// affect other items that hash to the same position. **Counting Bloom filters**
/// solve this by using counters instead of single bits.
///
/// # Guarantees
///
/// After `remove(x)`:
/// * If `x` was inserted exactly once, `contains(x)` returns `false`
/// * If `x` was inserted multiple times, `contains(x)` may still return `true`
/// * Other items remain unaffected
///
/// # Examples
///
/// ```ignore
/// use bloomcraft::{CountingBloomFilter, BloomFilter, DeletableBloomFilter};
///
/// let mut filter = CountingBloomFilter::<String>::new(10000, 0.01);
/// filter.insert(&"hello".to_string());
/// assert!(filter.contains(&"hello".to_string()));
///
/// filter.remove(&"hello".to_string())?;
/// assert!(!filter.contains(&"hello".to_string()));
/// ```
pub trait DeletableBloomFilter<T: Hash>: BloomFilter<T> {
    /// Remove an item from the filter.
    ///
    /// Decrements the counters at the positions determined by the item's hash values.
    /// If any counter would underflow, the operation fails.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to remove
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Item removed successfully
    /// * `Err(_)` - Item cannot be removed (was never inserted or counter underflow)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// * Item was never inserted (all counters must be > 0)
    /// * Counter would underflow below zero
    ///
    /// # Thread Safety
    ///
    /// Requires `&mut self`. For concurrent operations, use external synchronization.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{CountingBloomFilter, BloomFilter, DeletableBloomFilter};
    ///
    /// let mut filter = CountingBloomFilter::<i32>::new(10000, 0.01);
    /// filter.insert(&42);
    /// filter.insert(&42); // Insert twice
    ///
    /// filter.remove(&42)?; // Remove once
    /// assert!(filter.contains(&42)); // Still present (inserted twice)
    ///
    /// filter.remove(&42)?; // Remove again
    /// assert!(!filter.contains(&42)); // Now absent
    /// ```
    fn remove(&mut self, item: &T) -> Result<()>;

    /// Check if an item can be safely removed.
    ///
    /// Checks if all counters for this item are non-zero, indicating the item
    /// appears to be present and can be removed without underflow.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to check
    ///
    /// # Returns
    ///
    /// `true` if item appears to be in the filter and can be removed
    ///
    /// # Note
    ///
    /// This is subject to false positives like `contains()`. A `true` result
    /// does not guarantee the item was actually inserted.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{CountingBloomFilter, DeletableBloomFilter};
    ///
    /// let mut filter = CountingBloomFilter::<i32>::new(1000, 0.01);
    /// assert!(!filter.can_remove(&42));
    ///
    /// filter.insert(&42);
    /// assert!(filter.can_remove(&42));
    /// ```
    #[must_use]
    fn can_remove(&self, item: &T) -> bool;
}

/// Extension trait for Bloom filters that support merging.
///
/// Merging allows combining multiple filters into one, useful for distributed
/// systems where each node maintains its own filter and periodic aggregation is needed.
///
/// # Guarantees
///
/// After `union(A, B)`:
/// * Result contains all items from A
/// * Result contains all items from B
/// * False positive rate may increase (up to sum of individual rates)
///
/// # Compatibility Requirements
///
/// Filters can only be merged if they have:
/// * Same size (m bits)
/// * Same number of hash functions (k)
/// * Same hash configuration (seed, algorithm)
///
/// # Examples
///
/// ```ignore
/// use bloomcraft::{StandardBloomFilter, BloomFilter, MergeableBloomFilter};
///
/// let mut filter1 = StandardBloomFilter::<String>::new(10000, 0.01);
/// let mut filter2 = StandardBloomFilter::<String>::new(10000, 0.01);
///
/// filter1.insert(&"alice".to_string());
/// filter2.insert(&"bob".to_string());
///
/// filter1.union(&filter2);
///
/// assert!(filter1.contains(&"alice".to_string()));
/// assert!(filter1.contains(&"bob".to_string()));
/// ```
pub trait MergeableBloomFilter<T: Hash>: BloomFilter<T> {
    /// Merge another filter into this one (union operation).
    ///
    /// Performs bitwise OR of the underlying bit arrays. After this operation,
    /// the filter contains all items from both filters.
    ///
    /// # Arguments
    ///
    /// * `other` - Filter to merge (must have compatible parameters)
    ///
    /// # Panics
    ///
    /// Panics if filters have incompatible parameters (different size or hash count).
    /// Use [`is_compatible()`](Self::is_compatible) to check before merging.
    ///
    /// # Thread Safety
    ///
    /// Requires `&mut self`. Not safe for concurrent writes.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter, MergeableBloomFilter};
    ///
    /// let mut filter1 = StandardBloomFilter::<String>::new(10000, 0.01);
    /// let mut filter2 = StandardBloomFilter::<String>::new(10000, 0.01);
    ///
    /// filter1.insert(&"alice".to_string());
    /// filter2.insert(&"bob".to_string());
    ///
    /// filter1.union(&filter2);
    /// assert!(filter1.contains(&"alice".to_string()));
    /// assert!(filter1.contains(&"bob".to_string()));
    /// ```
    fn union(&mut self, other: &Self);

    /// Compute intersection with another filter.
    ///
    /// Performs bitwise AND of the underlying bit arrays. The result contains
    /// only items that might be in both filters.
    ///
    /// # Warning
    ///
    /// This operation **significantly increases** the false positive rate because
    /// the resulting filter has fewer set bits relative to the items it represents.
    /// The FP rate can increase by an order of magnitude or more.
    ///
    /// # Arguments
    ///
    /// * `other` - Filter to intersect with
    ///
    /// # Panics
    ///
    /// Panics if filters have incompatible parameters.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, BloomFilter, MergeableBloomFilter};
    ///
    /// let mut filter1 = StandardBloomFilter::<i32>::new(10000, 0.01);
    /// let mut filter2 = StandardBloomFilter::<i32>::new(10000, 0.01);
    ///
    /// filter1.insert(&1);
    /// filter1.insert(&2);
    ///
    /// filter2.insert(&2);
    /// filter2.insert(&3);
    ///
    /// filter1.intersect(&filter2);
    /// // filter1 now approximately contains items in both filters
    /// ```
    fn intersect(&mut self, other: &Self);

    /// Check if filters are compatible for merging.
    ///
    /// Returns `true` if filters have the same parameters and can be safely merged.
    ///
    /// # Arguments
    ///
    /// * `other` - Filter to check compatibility with
    ///
    /// # Returns
    ///
    /// `true` if filters can be merged, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{StandardBloomFilter, MergeableBloomFilter};
    ///
    /// let filter1 = StandardBloomFilter::<i32>::new(10000, 0.01);
    /// let filter2 = StandardBloomFilter::<i32>::new(10000, 0.01);
    /// let filter3 = StandardBloomFilter::<i32>::new(20000, 0.01);
    ///
    /// assert!(filter1.is_compatible(&filter2));
    /// assert!(!filter1.is_compatible(&filter3));
    /// ```
    #[must_use]
    fn is_compatible(&self, other: &Self) -> bool;
}

/// Extension trait for scalable Bloom filters.
///
/// Scalable Bloom filters grow dynamically as more items are added,
/// maintaining a bounded false positive rate by allocating new sub-filters
/// with progressively tighter FP rate targets.
///
/// # Growth Strategy
///
/// When capacity is exceeded:
/// 1. Allocate new sub-filter with tighter FP rate
/// 2. Subsequent inserts go to the new filter
/// 3. Queries check all sub-filters
///
/// # Examples
///
/// ```ignore
/// use bloomcraft::{ScalableBloomFilter, BloomFilter};
///
/// let mut filter = ScalableBloomFilter::<i32>::new(100, 0.01);
///
/// // Can insert far more than initial capacity
/// for i in 0..10000 {
///     filter.insert(&i);
/// }
///
/// // False positive rate remains bounded
/// assert!(filter.false_positive_rate() < 0.015);
/// println!("Grew to {} sub-filters", filter.tier_count());
/// ```
pub trait ScalableBloomFilter<T: Hash>: BloomFilter<T> {
    /// Get the current capacity before next growth.
    ///
    /// # Returns
    ///
    /// Number of items that can be inserted into current sub-filter before growth
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{ScalableBloomFilter, ScalableBloomFilter as SBF};
    ///
    /// let filter = SBF::<i32>::new(1000, 0.01);
    /// println!("Current capacity: {}", filter.current_capacity());
    /// ```
    #[must_use]
    fn current_capacity(&self) -> usize;

    /// Get the target false positive rate.
    ///
    /// Returns the overall target FP rate the filter tries to maintain across
    /// all growth tiers.
    ///
    /// # Returns
    ///
    /// Target FP rate configured at creation
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{ScalableBloomFilter, ScalableBloomFilter as SBF};
    ///
    /// let filter = SBF::<i32>::new(1000, 0.01);
    /// assert_eq!(filter.target_fp_rate(), 0.01);
    /// ```
    #[must_use]
    fn target_fp_rate(&self) -> f64;

    /// Force growth to next size tier.
    ///
    /// Allocates a new sub-filter immediately. Normally happens automatically
    /// when capacity is exceeded, but can be triggered manually for preallocation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{ScalableBloomFilter, ScalableBloomFilter as SBF};
    ///
    /// let mut filter = SBF::<i32>::new(1000, 0.01);
    /// filter.grow(); // Preallocate next tier
    /// ```
    fn grow(&mut self);

    /// Get the number of internal sub-filters.
    ///
    /// Returns the count of growth tiers currently allocated.
    ///
    /// # Returns
    ///
    /// Number of sub-filters
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use bloomcraft::{ScalableBloomFilter, ScalableBloomFilter as SBF};
    ///
    /// let mut filter = SBF::<i32>::new(1000, 0.01);
    /// assert_eq!(filter.tier_count(), 1);
    ///
    /// for i in 0..5000 {
    ///     filter.insert(&i);
    /// }
    /// assert!(filter.tier_count() > 1);
    /// ```
    #[must_use]
    fn tier_count(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::marker::PhantomData;

    // Mock implementation for testing trait methods
    struct MockBloomFilter<T> {
        items: std::sync::Mutex<HashSet<u64>>,
        _marker: PhantomData<T>,
    }

    impl<T> MockBloomFilter<T> {
        fn new() -> Self {
            Self {
                items: std::sync::Mutex::new(HashSet::new()),
                _marker: PhantomData,
            }
        }
    }

    impl<T: Hash + Send + Sync> BloomFilter<T> for MockBloomFilter<T> {
        fn insert(&mut self, item: &T) {
            use std::hash::Hasher;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            item.hash(&mut hasher);
            let hash = hasher.finish();
            self.items.lock().unwrap().insert(hash);
        }

        fn contains(&self, item: &T) -> bool {
            use std::hash::Hasher;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            item.hash(&mut hasher);
            let hash = hasher.finish();
            self.items.lock().unwrap().contains(&hash)
        }

        fn clear(&mut self) {
            self.items.lock().unwrap().clear();
        }

        fn len(&self) -> usize {
            self.items.lock().unwrap().len()
        }

        fn false_positive_rate(&self) -> f64 {
            0.01
        }

        fn expected_items(&self) -> usize {
            1000
        }

        fn bit_count(&self) -> usize {
            10000
        }

        fn hash_count(&self) -> usize {
            7
        }
    }

    #[test]
    fn test_is_empty_returns_true_for_new_filter() {
        let filter = MockBloomFilter::<i32>::new();
        assert!(filter.is_empty());
    }

    #[test]
    fn test_is_empty_returns_false_after_insert() {
        let mut filter = MockBloomFilter::<i32>::new();
        filter.insert(&42);
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_insert_batch() {
        let mut filter = MockBloomFilter::<i32>::new();
        let items = vec![1, 2, 3, 4, 5];

        filter.insert_batch(items.iter());

        assert_eq!(filter.len(), 5);
        for item in &items {
            assert!(filter.contains(item));
        }
    }

    #[test]
    fn test_contains_all_true_when_all_present() {
        let mut filter = MockBloomFilter::<i32>::new();
        filter.insert(&1);
        filter.insert(&2);
        filter.insert(&3);

        assert!(filter.contains_all(vec![&1, &2, &3].into_iter()));
    }

    #[test]
    fn test_contains_all_false_when_one_missing() {
        let mut filter = MockBloomFilter::<i32>::new();
        filter.insert(&1);
        filter.insert(&2);

        assert!(!filter.contains_all(vec![&1, &2, &4].into_iter()));
    }

    #[test]
    fn test_contains_any_true_when_one_present() {
        let mut filter = MockBloomFilter::<i32>::new();
        filter.insert(&1);

        assert!(filter.contains_any(vec![&1, &2, &3].into_iter()));
    }

    #[test]
    fn test_contains_any_false_when_none_present() {
        let mut filter = MockBloomFilter::<i32>::new();
        filter.insert(&1);

        assert!(!filter.contains_any(vec![&2, &3, &4].into_iter()));
    }

    #[test]
    fn test_clear_removes_all_items() {
        let mut filter = MockBloomFilter::<i32>::new();
        filter.insert(&1);
        filter.insert(&2);

        filter.clear();

        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert!(!filter.contains(&1));
        assert!(!filter.contains(&2));
    }

    #[test]
    fn test_estimate_count_default_implementation() {
        let mut filter = MockBloomFilter::<i32>::new();
        filter.insert(&1);
        filter.insert(&2);

        // Default implementation returns len()
        assert_eq!(filter.estimate_count(), 2);
    }

    #[test]
    fn test_false_positive_rate_returns_value() {
        let filter = MockBloomFilter::<i32>::new();
        let fp_rate = filter.false_positive_rate();
        assert!((0.0..=1.0).contains(&fp_rate));
    }

    #[test]
    fn test_insert_and_contains() {
        let mut filter = MockBloomFilter::<i32>::new();

        assert!(!filter.contains(&42));
        filter.insert(&42);
        assert!(filter.contains(&42));
    }

    #[test]
    fn test_trait_bounds() {
        fn accept_filter<T: Hash>(_f: &impl BloomFilter<T>) {}
        let filter = MockBloomFilter::<i32>::new();
        accept_filter::<i32>(&filter);
    }
}
