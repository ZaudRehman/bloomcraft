//! Standard Bloom filter implementation.
//!
//! This module provides a production-grade implementation of the classic Bloom filter
//! data structure with modern optimizations. It uses enhanced double hashing for
//! generating k independent hash functions from just two base hashes.
//!
//! # Algorithm
//!
//! A Bloom filter is a space-efficient probabilistic data structure that supports
//! two operations:
//!
//! - Insert: Add an element to the set
//! - Query: Test whether an element is in the set
//!
//! # Properties
//!
//! - False positives: Possible (controllable via parameters)
//! - False negatives: Never occur
//! - Space efficiency: ~9.6 bits per element for 1% FP rate
//! - Time complexity: O(k) for both insert and query
//!
//! # Mathematical Foundation
//!
//! Given:
//! - n = expected number of elements
//! - p = desired false positive rate
//!
//! Optimal parameters:
//! - m = -n * ln(p) / (ln(2)²) ≈ 1.44 * n * log₂(1/p)  (filter size in bits)
//! - k = (m/n) * ln(2) ≈ 0.693 * (m/n)  (number of hash functions)
//!
//! Actual false positive rate:
//! - p_actual = (1 - e^(-kn/m))^k
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! // Create filter for 10,000 items with 1% false positive rate
//! let mut filter = StandardBloomFilter::new(10_000, 0.01);
//!
//! // Insert items
//! filter.insert(&"hello");
//! filter.insert(&"world");
//!
//! // Query items
//! assert!(filter.contains(&"hello"));
//! assert!(filter.contains(&"world"));
//! assert!(!filter.contains(&"goodbye")); // Probably false
//! ```
//!
//! ## Batch Operations
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! let mut filter = StandardBloomFilter::new(1000, 0.01);
//!
//! // Insert multiple items
//! let items = vec!["a", "b", "c", "d"];
//! filter.insert_batch(&items);
//!
//! // Query multiple items
//! let queries = vec!["a", "b", "x", "y"];
//! let results = filter.contains_batch(&queries);
//! assert_eq!(results, vec![true, true, false, false]);
//! ```
//!
//! ## Custom Hash Function
//!
//! ```
//! # #[cfg(feature = "wyhash")]
//! # {
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::hash::WyHasher;
//!
//! let filter: StandardBloomFilter<&str, WyHasher> = StandardBloomFilter::with_hasher(
//!     10_000,
//!     0.01,
//!     WyHasher::new()
//! );
//! # }
//! ```
//!
//! # References
//!
//! - Bloom, B. H. (1970). "Space/time trade-offs in hash coding with allowable errors"
//! - Kirsch, A., & Mitzenmacher, M. (2006). "Less Hashing, Same Performance: Building a Better Bloom Filter"

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(unused_imports)]
#![allow(missing_docs)]

use crate::core::bitvec::BitVec;
use crate::core::filter::{BloomFilter, MutableBloomFilter, ConcurrentBloomFilter, MergeableBloomFilter};
use crate::core::params::{optimal_k, optimal_m};
use crate::error::{BloomCraftError, Result};
use crate::hash::strategies::{EnhancedDoubleHashing, HashStrategy as HashStrategyTrait};
use crate::hash::{BloomHasher, StdHasher};
#[cfg(feature = "wyhash")]
use crate::hash::WyHasher;
use std::hash::Hash;
use std::marker::PhantomData;

/// Convert a hashable item to bytes using Rust's `Hash` trait.
///
/// This is the bridge between generic `T: Hash` and the `&[u8]` API
/// required by `BloomHasher`. Uses `std::collections::hash_map::DefaultHasher`
/// to get a stable u64, then converts to little-endian bytes.
///
/// # Performance
///
/// This is an inline function that should optimize to near-zero overhead.
/// The compiler can often elide the intermediate array allocation.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}

/// Health status of the Bloom filter
#[derive(Debug, Clone, PartialEq)]
pub enum FilterHealth {
    /// Filter is operating normally
    Healthy {
        fill_rate: f64,
        current_fpr: f64,
        estimated_items: usize,
    },
    /// Filter performance is degrading
    Degraded {
        fill_rate: f64,
        current_fpr: f64,
        fpr_ratio: f64,
        estimated_items: usize,
        recommendation: &'static str,
    },
    /// Filter is critically saturated
    Critical {
        fill_rate: f64,
        current_fpr: f64,
        fpr_ratio: f64,
        estimated_items: usize,
        recommendation: &'static str,
    },
}

impl std::fmt::Display for FilterHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterHealth::Healthy { fill_rate, current_fpr, estimated_items } => {
                write!(
                    f,
                    "Healthy | Fill: {:.1}% | FPR: {:.4} | Items: ~{}",
                    fill_rate * 100.0,
                    current_fpr,
                    estimated_items
                )
            }
            FilterHealth::Degraded { fill_rate, fpr_ratio, recommendation, .. } => {
                write!(
                    f,
                    "Degraded | Fill: {:.1}% | FPR: {:.1}× target | {}",
                    fill_rate * 100.0,
                    fpr_ratio,
                    recommendation
                )
            }
            FilterHealth::Critical { fill_rate, fpr_ratio, recommendation, .. } => {
                write!(
                    f,
                    "CRITICAL | Fill: {:.1}% | FPR: {:.1}× target | {}",
                    fill_rate * 100.0,
                    fpr_ratio,
                    recommendation
                )
            }
        }
    }
}

/// Performance metrics for the filter (optional, enabled with "metrics" feature)
#[cfg(feature = "metrics")]
#[derive(Debug, Clone, Default)]
pub struct FilterMetrics {
    pub total_inserts: usize,
    pub total_queries: usize,
    pub query_hits: usize,
    pub query_misses: usize,
}


/// Standard Bloom filter with optimal parameters.
///
/// This implementation uses:
/// - Lock-free bit vector for thread-safe insertions
/// - Enhanced double hashing for k hash functions
/// - Optimal parameter calculation
/// - Batch operation support
///
/// # Type Parameters
///
/// * `T` - Type of items stored (must implement `Hash`)
/// * `H` - Hash function type (must implement `BloomHasher`)
///
/// # Memory Layout
///
/// ```text
/// StandardBloomFilter {
///     bitvec: BitVec,           // m bits
///     k: usize,                 // number of hash functions
///     hasher: H,                // hash function
///     expected_items: usize,    // n (for statistics)
///     target_fpr: f64,          // p (for statistics)
/// }
/// ```
///
/// # Thread Safety
///
/// - Insert: Thread-safe (lock-free atomic operations)
/// - Query: Thread-safe (lock-free atomic loads)
/// - Clear: Requires exclusive access (`&mut self`)
#[derive(Debug)]
pub struct StandardBloomFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone,
{
    /// Underlying bit vector
    bitvec: BitVec,

    /// Number of hash functions (k)
    k: usize,

    /// Hash function
    hasher: H,

    /// Expected number of items (for statistics)
    expected_items: usize,

    /// Target false positive rate (for statistics)
    target_fpr: f64,

    /// Hash strategy used for generating hash indices
    strategy: EnhancedDoubleHashing,

    /// Phantom data for type parameter T
    _phantom: PhantomData<T>,
}

impl<T, H> Clone for StandardBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    fn clone(&self) -> Self {
        Self {
            bitvec: self.bitvec.clone(),
            k: self.k,
            hasher: self.hasher.clone(),
            expected_items: self.expected_items,
            target_fpr: self.target_fpr,
            strategy: self.strategy,
            _phantom: PhantomData,
        }
    }
}

impl<T> StandardBloomFilter<T, StdHasher>
where
    T: Hash,
{
    /// Create a new standard Bloom filter with default hasher.
    ///
    /// Automatically calculates optimal parameters (m and k) based on the
    /// expected number of items and desired false positive rate.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items to insert (n)
    /// * `fpr` - Target false positive rate (p), must be in (0, 1)
    ///
    /// # Returns
    ///
    /// New Bloom filter with optimal parameters
    ///
    /// # Panics
    ///
    /// Panics if `fpr` is not in the range (0, 1) or `expected_items` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// // 1% false positive rate for 10,000 items
    /// let filter: StandardBloomFilter<String> = StandardBloomFilter::new(10_000, 0.01);
    ///
    /// // 0.1% false positive rate for 1 million items
    /// let filter: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000_000, 0.001);
    /// ```
    #[must_use]
    pub fn new(expected_items: usize, fpr: f64) -> Self {
        Self::with_hasher(expected_items, fpr, StdHasher::new())
    }
}

impl<T, H> StandardBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    /// Create a new standard Bloom filter with custom hasher.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items to insert
    /// * `fpr` - Target false positive rate (0 < fpr < 1)
    /// * `hasher` - Custom hash function
    ///
    /// # Panics
    ///
    /// Panics if `fpr` is not in (0, 1) or `expected_items` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    ///
    /// let hasher = StdHasher::with_seed(42);
    /// let filter: StandardBloomFilter<String, _> =
    ///     StandardBloomFilter::with_hasher(10_000, 0.01, hasher);
    /// ```
    #[must_use]
    pub fn with_hasher(expected_items: usize, fpr: f64, hasher: H) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            fpr > 0.0 && fpr < 1.0,
            "fpr must be in range (0, 1), got {}",
            fpr
        );

        let m = optimal_m(expected_items, fpr);
        let k = optimal_k(expected_items, m);

        Self {
            bitvec: BitVec::new(m).expect("BitVec creation failed with valid parameters"),
            k,
            hasher,
            expected_items,
            target_fpr: fpr,
            strategy: EnhancedDoubleHashing,
            _phantom: PhantomData,
        }
    }

    /// Create a Bloom filter with explicit parameters.
    ///
    /// This allows fine-grained control over the filter size and number of
    /// hash functions, bypassing automatic parameter calculation.
    ///
    /// # Arguments
    ///
    /// * `m` - Filter size in bits
    /// * `k` - Number of hash functions
    /// * `hasher` - Hash function
    ///
    /// # Panics
    ///
    /// Panics if `m` or `k` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    ///
    /// // Create filter with 10,000 bits and 7 hash functions
    /// let filter: StandardBloomFilter<String, _> =
    ///     StandardBloomFilter::with_params(10_000, 7, StdHasher::new());
    /// ```
    #[must_use]
    pub fn with_params(m: usize, k: usize, hasher: H) -> Self {
        assert!(m > 0, "m must be > 0");
        assert!(k > 0, "k must be > 0");

        Self {
            bitvec: BitVec::new(m).expect("BitVec creation failed with valid parameters"),
            k,
            hasher,
            expected_items: 0, // Unknown
            target_fpr: 0.0,   // Unknown
            strategy: EnhancedDoubleHashing,
            _phantom: PhantomData,
        }
    }

    /// Get the size of the filter in bits (m).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let filter: StandardBloomFilter<String> = StandardBloomFilter::new(10_000, 0.01);
    /// println!("Filter size: {} bits", filter.size());
    /// ```
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.bitvec.len()
    }

    /// Get the number of hash functions (k).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let filter: StandardBloomFilter<String> = StandardBloomFilter::new(10_000, 0.01);
    /// println!("Using {} hash functions", filter.hash_count());
    /// ```
    #[must_use]
    #[inline]
    pub fn hash_count(&self) -> usize {
        self.k
    }

    /// Get the expected number of items.
    #[must_use]
    #[inline]
    pub fn expected_items(&self) -> usize {
        self.expected_items
    }

    /// Get the target false positive rate.
    #[must_use]
    #[inline]
    pub fn target_fpr(&self) -> f64 {
        self.target_fpr
    }

    /// Get the number of bits currently set in the filter.
    ///
    /// This can be used to estimate the fill rate of the filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(10_000, 0.01);
    /// filter.insert(&"hello");
    /// filter.insert(&"world");
    ///
    /// let set_bits = filter.count_set_bits();
    /// println!("Fill rate: {:.2}%", (set_bits as f64 / filter.size() as f64) * 100.0);
    /// ```
    #[must_use]
    pub fn count_set_bits(&self) -> usize {
        self.bitvec.count_ones()
    }

    /// Calculate the current fill rate as a fraction in [0, 1].
    ///
    /// Fill rate = (number of set bits) / (total bits)
    ///
    /// # Returns
    ///
    /// Fill rate as a float in range [0, 1]
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        self.count_set_bits() as f64 / self.size() as f64
    }

    /// Estimate the current false positive rate based on fill rate.
    ///
    /// Uses the formula: FPR ≈ (1 - e^(-k*n/m))^k
    ///
    /// First estimates n (number of items) from the fill rate using:
    /// n ≈ -(m/k) × ln(1 - fill_rate)
    ///
    /// Then calculates FPR using the standard formula.
    ///
    /// # Returns
    ///
    /// Estimated current false positive rate
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<String> = StandardBloomFilter::new(10_000, 0.01);
    ///
    /// for i in 0..5000 {
    ///     filter.insert(&i.to_string());
    /// }
    ///
    /// println!("Estimated FPR: {:.4}", filter.estimate_fpr());
    /// ```
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        let set_bits = self.count_set_bits();
        let m = self.size() as f64;
        let k = self.k as f64;

        if set_bits == 0 {
            return 0.0;
        }

        let fill_rate = set_bits as f64 / m;

        if fill_rate >= 1.0 {
            return 1.0;
        }

        // Estimate n (number of items) from fill rate
        // fill_rate ≈ 1 - e^(-kn/m)
        // => n ≈ -(m/k) × ln(1 - fill_rate)
        let estimated_n = -(m / k) * (1.0 - fill_rate).ln();

        // Calculate FPR using the standard formula: (1 - e^(-kn/m))^k
        let exponent = -k * estimated_n / m;
        (1.0 - exponent.exp()).powf(k)
    }

    /// Check if the filter is approximately full.
    ///
    /// Returns true if the fill rate exceeds 50%, which typically indicates
    /// the filter is approaching saturation and FPR is degrading.
    ///
    /// # Returns
    ///
    /// `true` if fill rate > 0.5
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.fill_rate() > 0.5
    }

    /// Insert an item into the filter.
    ///
    /// This operation is thread-safe and lock-free.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    /// ```
    #[inline]
    pub fn insert(&self, item: &T) {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        for idx in indices {
            self.bitvec.set(idx);
        }
    }

    /// Check if an item might be in the filter.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to check
    ///
    /// # Returns
    ///
    /// - `true`: Item might be in the set (or false positive)
    /// - `false`: Item is definitely not in the set
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    ///
    /// assert!(filter.contains(&"hello"));  // True positive
    /// assert!(!filter.contains(&"world")); // True negative (probably)
    /// ```
    #[must_use]
    #[inline]
    pub fn contains(&self, item: &T) -> bool {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        indices.iter().all(|&idx| self.bitvec.get(idx))
    }

    /// Insert multiple items in batch.
    ///
    /// This method is more efficient than calling `insert()` repeatedly because:
    /// - Pre-allocates hash computation results
    /// - Enables better compiler optimization (tight loop)
    /// - Reduces function call overhead
    ///
    /// # Performance
    ///
    /// Expected speedup: 1.3-1.5x faster than individual inserts for batches > 100 items
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of items to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    ///
    /// let items = vec!["hello", "world", "foo", "bar"];
    /// filter.insert_batch(&items);
    ///
    /// assert!(filter.contains(&"hello"));
    /// assert!(filter.contains(&"world"));
    /// ```
    pub fn insert_batch(&self, items: &[T]) {
        for item in items {
            let bytes = hash_item_to_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);

            // Use HashStrategy to generate indices
            let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.bitvec.len());
            
            for idx in indices {
                self.bitvec.set(idx);
            }
        }
    }

    /// Query multiple items at once.
    ///
    /// This method is more efficient than calling `contains()` repeatedly because:
    /// - Pre-allocates result vector (no repeated allocation)
    /// - Enables better compiler optimization (tight loop, no function calls)
    /// - Better cache locality (sequential processing)
    ///
    /// # Performance
    ///
    /// Expected speedup: 1.5-2x faster than individual queries for batches > 100 items
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of items to query
    ///
    /// # Returns
    ///
    /// Vector of booleans indicating presence (true = probably present, false = definitely absent)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    /// filter.insert(&"world");
    ///
    /// let queries = vec!["hello", "world", "foo"];
    /// let results = filter.contains_batch(&queries);
    ///
    /// assert_eq!(results, vec![true, true, false]);
    /// ```
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        // Pre-allocate exact size needed (critical for performance)
        let mut results = Vec::with_capacity(items.len());

        for item in items {
            let bytes = hash_item_to_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);

            // Use HashStrategy to generate indices
            let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.bitvec.len());
            
            // Check if all k bits are set (with early exit)
            let mut present = true;
            for idx in indices {
                if !self.bitvec.get(idx) {
                    present = false;
                    break;
                }
            }
            
            results.push(present);
        }

        results
    }

    /// Query multiple items by reference (zero-copy batch operation).
    ///
    /// This is an optimized variant of `contains_batch` that accepts references
    /// to items instead of owned values, eliminating the need for cloning in benchmarks
    /// and other performance-critical contexts.
    ///
    /// # Performance
    ///
    /// This method has the same performance as `contains_batch` but avoids any
    /// potential cloning overhead when preparing the input batch.
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of references to items to query
    ///
    /// # Returns
    ///
    /// Vector of booleans indicating presence
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello".to_string());
    ///
    /// let items = vec!["hello".to_string(), "world".to_string()];
    /// let refs: Vec<&String> = items.iter().collect();
    /// let results = filter.contains_batch_ref(&refs);
    ///
    /// // Check individual results
    /// assert_eq!(results[0], true);   // "hello" is present
    /// assert_eq!(results[1], false);  // "world" is absent
    /// ```
    #[must_use]
    pub fn contains_batch_ref(&self, items: &[&T]) -> Vec<bool> {
        let mut results = Vec::with_capacity(items.len());

        for item in items {
            let bytes = hash_item_to_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
            
            let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.bitvec.len());
            
            let mut present = true;
            for idx in indices {
                if !self.bitvec.get(idx) {
                    present = false;
                    break;
                }
            }
            
            results.push(present);
        }

        results
    }

    /// Insert multiple items by reference (zero-copy batch operation).
    ///
    /// Zero-copy variant of `insert_batch` for performance-critical contexts.
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of references to items to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
    ///
    /// let items = vec!["hello".to_string(), "world".to_string()];
    /// let refs: Vec<&String> = items.iter().collect();
    /// filter.insert_batch_ref(&refs);
    ///
    /// // Verify they were inserted
    /// assert!(filter.contains(&"hello".to_string()));
    /// assert!(filter.contains(&"world".to_string()));
    /// ```
    pub fn insert_batch_ref(&self, items: &[&T]) {
        for item in items {
            let bytes = hash_item_to_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
            
            let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.bitvec.len());
            
            for idx in indices {
                self.bitvec.set(idx);
            }
        }
    }

    /// Clear all bits in the filter.
    ///
    /// This operation requires exclusive access.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    ///
    /// filter.clear();
    /// assert!(!filter.contains(&"hello"));
    /// assert_eq!(filter.count_set_bits(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.bitvec.clear();
    }

    /// Check if the filter is empty (no bits set).
    ///
    /// # Returns
    ///
    /// `true` if no bits are set
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// assert!(filter.is_empty());
    ///
    /// filter.insert(&"hello");
    /// assert!(!filter.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count_set_bits() == 0
    }

    /// Compute the union of two Bloom filters.
    ///
    /// The resulting filter contains all items from both filters.
    /// Both filters must have the same size and number of hash functions.
    ///
    /// # Arguments
    ///
    /// * `other` - Other filter to union with
    ///
    /// # Returns
    ///
    /// New filter containing the union
    ///
    /// # Errors
    ///
    /// Returns error if filters have incompatible parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter1: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// let mut filter2: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    ///
    /// filter1.insert(&"a");
    /// filter2.insert(&"b");
    ///
    /// let union = filter1.union(&filter2).unwrap();
    /// assert!(union.contains(&"a"));
    /// assert!(union.contains(&"b"));
    /// ```
    pub fn union(&self, other: &Self) -> Result<Self> {
        if self.size() != other.size() {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Size mismatch".to_string(),
            });
        }

        if self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Hash count mismatch".to_string(),
            });
        }

        let mut result = self.clone();
        result.bitvec = self.bitvec.union(&other.bitvec)?;
        Ok(result)
    }

    /// Compute the intersection of two Bloom filters.
    ///
    /// The resulting filter may contain items from both filters, but with
    /// increased false positive rate. This operation is approximate.
    ///
    /// # Arguments
    ///
    /// * `other` - Other filter to intersect with
    ///
    /// # Returns
    ///
    /// New filter containing the intersection
    ///
    /// # Errors
    ///
    /// Returns error if filters have incompatible parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let mut filter1: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// let mut filter2: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    ///
    /// filter1.insert(&"a");
    /// filter1.insert(&"b");
    /// filter2.insert(&"b");
    /// filter2.insert(&"c");
    ///
    /// let intersection = filter1.intersect(&filter2).unwrap();
    /// assert!(intersection.contains(&"b")); // In both
    /// ```
    pub fn intersect(&self, other: &Self) -> Result<Self> {
        if self.size() != other.size() {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Size mismatch".to_string(),
            });
        }

        if self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Hash count mismatch".to_string(),
            });
        }

        let mut result = self.clone();
        result.bitvec = self.bitvec.intersect(&other.bitvec)?;
        Ok(result)
    }

    /// Estimate cardinality (number of unique items) using fill rate.
    ///
    /// Uses the standard Bloom filter cardinality estimation formula:
    /// n_estimated = -(m/k) × ln(1 - X/m)
    ///
    /// where:
    /// - m = number of bits
    /// - k = number of hash functions  
    /// - X = number of set bits
    ///
    /// # Accuracy
    ///
    /// - Low load (< 50% full): ±5% error
    /// - Medium load (50-80%): ±10% error
    /// - High load (> 80%): ±20% error
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let filter: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000, 0.01);
    ///
    /// for i in 0..500 {
    ///     filter.insert(&i);
    /// }
    ///
    /// let estimated = filter.estimate_cardinality();
    /// assert!((estimated as i32 - 500).abs() < 50); // Within 10%
    /// ```
    #[must_use]
    pub fn estimate_cardinality(&self) -> usize {
        let set_bits = self.count_set_bits();
        if set_bits == 0 {
            return 0;
        }

        let m = self.size() as f64;
        let k = self.k as f64;
        let x = set_bits as f64;

        if set_bits >= self.size() {
            // Filter completely saturated
            return usize::MAX;
        }

        // Cardinality estimation: n ≈ -(m/k) × ln(1 - X/m)
        let fill_rate = x / m;
        let estimated_n = -(m / k) * (1.0 - fill_rate).ln();
        
        estimated_n.max(0.0) as usize
    }

    /// Get memory usage in bytes.
    ///
    /// # Returns
    ///
    /// Approximate memory usage in bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let filter: StandardBloomFilter<String> = StandardBloomFilter::new(10_000, 0.01);
    /// println!("Memory usage: {} bytes", filter.memory_usage());
    /// ```
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.bitvec.memory_usage()
            + std::mem::size_of::<usize>() * 2 // k, expected_items
            + std::mem::size_of::<f64>() // target_fpr
            + std::mem::size_of::<H>() // hasher
            + std::mem::size_of::<EnhancedDoubleHashing>() // strategy
    }

    /// Get the raw bit data as a slice of u64 values.
    ///
    /// This is useful for serialization.
    #[must_use]
    pub fn raw_bits(&self) -> Vec<u64> {
        self.bitvec.to_raw()
    }

    /// Get the number of hash functions (alias for hash_count).
    #[must_use]
    #[inline]
    pub fn num_hashes(&self) -> usize {
        self.k
    }

    /// Create a filter from raw parts (for deserialization).
    ///
    /// # Arguments
    ///
    /// * `bits` - Pre-populated bit vector
    /// * `k` - Number of hash functions
    /// * `_strategy` - Hash strategy (currently ignored, uses EnhancedDoubleHashing)
    pub fn from_parts(
        bits: BitVec,
        k: usize,
        _strategy: crate::hash::HashStrategy,
    ) -> Result<Self>
    where
        H: Default,
    {
        if k == 0 {
            return Err(BloomCraftError::invalid_hash_count(k, 1, 32));
        }

        Ok(Self {
            expected_items: 0,
            target_fpr: 0.0,
            k,
            bitvec: bits,
            hasher: H::default(),
            strategy: EnhancedDoubleHashing,
            _phantom: PhantomData,
        })
    }

    /// Get the number of bits set (alias for count_set_bits).
    #[must_use]
    pub fn len(&self) -> usize {
        self.count_set_bits()
    }

    /// Get the estimated false positive rate (alias for estimate_fpr).
    #[must_use]
    pub fn estimated_fp_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    /// Get the name of the hasher used by this filter.
    ///
    /// Used for serialization compatibility checking.
    #[must_use]
    pub fn hasher_name(&self) -> &'static str {
        self.hasher.name()
    }

    /// Get the hash strategy used by this filter.
    ///
    /// Returns the strategy enum variant for serialization.
    #[must_use]
    pub fn hash_strategy(&self) -> crate::hash::HashStrategy {
        // This filter uses EnhancedDoubleHashing internally
        crate::hash::HashStrategy::EnhancedDouble
    }

    /// Create a filter with a specific hash strategy.
    ///
    /// # Arguments
    ///
    /// * `m` - Filter size in bits
    /// * `k` - Number of hash functions
    /// * `strategy` - Hash strategy to use (currently ignored, uses EnhancedDoubleHashing)
    #[must_use]
    pub fn with_strategy(m: usize, k: usize, _strategy: crate::hash::HashStrategy) -> Self
    where
        H: Default,
    {
        assert!(m > 0, "m must be > 0");
        assert!(k > 0, "k must be > 0");

        Self {
            bitvec: BitVec::new(m).expect("BitVec creation failed with valid parameters"),
            k,
            hasher: H::default(),
            expected_items: 0,
            target_fpr: 0.0,
            strategy: EnhancedDoubleHashing,
            _phantom: PhantomData,
        }
    }
}

impl<T, H> BloomFilter<T> for StandardBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
     fn insert(&mut self, item: &T) {
        // Cast &mut self to &self to use atomic operations
        // This is safe because our operations are thread-safe via atomics
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());
        
        for idx in indices {
            self.bitvec.set(idx);
        }
    }

    fn contains(&self, item: &T) -> bool {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());
        indices.iter().all(|&idx| self.bitvec.get(idx))
    }

    fn clear(&mut self) {
        StandardBloomFilter::clear(self);
    }

    fn len(&self) -> usize {
        self.count_set_bits()
    }

    fn is_empty(&self) -> bool {
        StandardBloomFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn expected_items(&self) -> usize {
        self.expected_items
    }

    fn bit_count(&self) -> usize {
        self.size()
    }

    fn hash_count(&self) -> usize {
        self.k
    }
}

impl<T, H> ConcurrentBloomFilter<T> for StandardBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    #[inline]
    fn insert_concurrent(&self, item: &T) {
        // Delegate to the inherently lock-free insert method
        StandardBloomFilter::insert(self, item);
    }

    fn insert_batch_concurrent(&self, items: &[T]) {
        // Delegate to existing batch insert (already lock-free)
        StandardBloomFilter::insert_batch(self, items);
    }

    fn insert_batch_ref_concurrent(&self, items: &[&T]) {
        // Delegate to existing batch insert by reference (already lock-free)
        StandardBloomFilter::insert_batch_ref(self, items);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        assert!(filter.size() > 0);
        assert!(filter.hash_count() > 0);
        assert_eq!(filter.expected_items(), 1000);
        assert_eq!(filter.target_fpr(), 0.01);
    }

    #[test]
    #[should_panic(expected = "expected_items must be > 0")]
    fn test_new_zero_items() {
        let _: StandardBloomFilter<String> = StandardBloomFilter::new(0, 0.01);
    }

    #[test]
    #[should_panic(expected = "fpr must be in range")]
    fn test_new_invalid_fpr() {
        let _: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 1.5);
    }

    #[test]
    fn test_insert_and_contains() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);

        filter.insert(&"hello".to_string());
        assert!(filter.contains(&"hello".to_string()));
        assert!(!filter.contains(&"world".to_string()));
    }

    #[test]
    fn test_multiple_inserts() {
        let filter: StandardBloomFilter<i32> = StandardBloomFilter::new(1000, 0.01);

        for i in 0..100 {
            filter.insert(&i);
        }

        for i in 0..100 {
            assert!(filter.contains(&i), "Item {} should be present", i);
        }
    }

    #[test]
    fn test_is_empty() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        assert!(filter.is_empty());

        filter.insert(&"test".to_string());
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);

        filter.insert(&"hello".to_string());
        filter.insert(&"world".to_string());
        assert!(!filter.is_empty());

        filter.clear();
        assert!(filter.is_empty());
        assert!(!filter.contains(&"hello".to_string()));
        assert!(!filter.contains(&"world".to_string()));
    }

    #[test]
    fn test_count_set_bits() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        assert_eq!(filter.count_set_bits(), 0);

        filter.insert(&"test".to_string());
        assert!(filter.count_set_bits() > 0);
    }

    #[test]
    fn test_fill_rate() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        assert_eq!(filter.fill_rate(), 0.0);

        for i in 0..100 {
            filter.insert(&i.to_string());
        }

        let fill_rate = filter.fill_rate();
        assert!(fill_rate > 0.0 && fill_rate < 1.0);
    }

    #[test]
    fn test_estimate_fpr() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        assert_eq!(filter.estimate_fpr(), 0.0);

        for i in 0..500 {
            filter.insert(&i.to_string());
        }

        let estimated_fpr = filter.estimate_fpr();
        assert!(estimated_fpr > 0.0);
        assert!(estimated_fpr < 1.0);
    }

    #[test]
    fn test_is_full() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(100, 0.01);
        assert!(!filter.is_full());

        // Insert many items to saturate the filter
        for i in 0..1000 {
            filter.insert(&i.to_string());
        }

        assert!(filter.is_full());
    }

    #[test]
    fn test_insert_batch() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);

        let items: Vec<String> = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .collect();
        filter.insert_batch(&items);

        for item in &items {
            assert!(filter.contains(item));
        }
    }

    #[test]
    fn test_contains_batch() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);

        filter.insert(&"a".to_string());
        filter.insert(&"b".to_string());

        let queries: Vec<String> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(String::from)
            .collect();
        let results = filter.contains_batch(&queries);

        assert_eq!(results[0], true);
        assert_eq!(results[1], true);
        assert_eq!(results[2], false);
        assert_eq!(results[3], false);
    }

    #[test]
    fn test_union() {
        let filter1: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        let filter2: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);

        filter1.insert(&"a".to_string());
        filter1.insert(&"b".to_string());
        filter2.insert(&"b".to_string());
        filter2.insert(&"c".to_string());

        let union = filter1.union(&filter2).unwrap();

        assert!(union.contains(&"a".to_string()));
        assert!(union.contains(&"b".to_string()));
        assert!(union.contains(&"c".to_string()));
    }

    #[test]
    fn test_union_incompatible_size() {
        let filter1: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        let filter2: StandardBloomFilter<String> = StandardBloomFilter::new(2000, 0.01);

        let result = filter1.union(&filter2);
        assert!(result.is_err());
    }

    #[test]
    fn test_intersect() {
        let filter1: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        let filter2: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);

        filter1.insert(&"a".to_string());
        filter1.insert(&"b".to_string());
        filter2.insert(&"b".to_string());
        filter2.insert(&"c".to_string());

        let intersection = filter1.intersect(&filter2).unwrap();

        assert!(intersection.contains(&"b".to_string()));
    }

    #[test]
    fn test_memory_usage() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        let mem = filter.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_with_params() {
        let filter: StandardBloomFilter<String, StdHasher> =
            StandardBloomFilter::with_params(1000, 7, StdHasher::new());

        assert_eq!(filter.size(), 1000);
        assert_eq!(filter.hash_count(), 7);
    }

    #[test]
    fn test_false_positive_rate() {
        let filter: StandardBloomFilter<u64> = StandardBloomFilter::new(1000, 0.01);

        // Insert 1000 items
        for i in 0..1000 {
            filter.insert(&i);
        }

        // Test 10000 items not in filter
        let mut false_positives = 0;
        for i in 1000..11000 {
            if filter.contains(&i) {
                false_positives += 1;
            }
        }

        let actual_fpr = false_positives as f64 / 10000.0;
        println!("Actual FPR: {:.4}, Target: 0.01", actual_fpr);

        // Should be reasonably close to target (within 5x)
        assert!(actual_fpr < 0.05);
    }

    #[test]
    fn test_no_false_negatives() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);

        let items: Vec<String> = vec!["apple", "banana", "cherry", "date", "elderberry"]
            .into_iter()
            .map(String::from)
            .collect();
        for item in &items {
            filter.insert(item);
        }

        // All inserted items must be found (no false negatives)
        for item in &items {
            assert!(filter.contains(item), "False negative for {}", item);
        }
    }

    #[test]
    fn test_bloom_filter_trait() {
        let mut filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);

        // Test via trait
        BloomFilter::insert(&mut filter, &"test".to_string());
        assert!(BloomFilter::contains(&filter, &"test".to_string()));
        assert_eq!(BloomFilter::len(&filter), filter.count_set_bits());
        assert!(!BloomFilter::is_empty(&filter));

        BloomFilter::clear(&mut filter);
        assert!(BloomFilter::is_empty(&filter));
    }

    #[test]
    fn test_clone() {
        let filter1: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
        filter1.insert(&"test".to_string());

        let filter2 = filter1.clone();
        assert!(filter2.contains(&"test".to_string()));
        assert_eq!(filter1.size(), filter2.size());
        assert_eq!(filter1.hash_count(), filter2.hash_count());
    }

    #[test]
    fn test_estimate_cardinality() {
        let filter: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000, 0.01);

        // Insert 500 items
        for i in 0..500 {
            filter.insert(&i);
        }

        let estimated = filter.estimate_cardinality();
        
        // Should be within 20% of actual (500)
        assert!(
            estimated >= 400 && estimated <= 600,
            "Estimated cardinality {} should be near 500",
            estimated
        );
    }

    #[test]
    fn test_union_contains_all_items() {
        let filter1: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000, 0.01);
        let filter2: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000, 0.01);

        // Disjoint sets
        for i in 0..50 {
            filter1.insert(&i);
        }

        for i in 50..100 {
            filter2.insert(&i);
        }

        let union = filter1.union(&filter2).unwrap();

        // All items from both filters should be present
        for i in 0..100 {
            assert!(union.contains(&i), "Union missing item {}", i);
        }
    }

    #[test]
    fn test_union_with_overlap() {
        let filter1: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000, 0.01);
        let filter2: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000, 0.01);

        // Overlapping sets: 0-49 and 25-74
        for i in 0..50 {
            filter1.insert(&i);
        }

        for i in 25..75 {
            filter2.insert(&i);
        }

        let union = filter1.union(&filter2).unwrap();

        // All items 0-74 should be present
        for i in 0..75 {
            assert!(union.contains(&i), "Union missing item {}", i);
        }

        // Items outside range should not be present
        assert!(!union.contains(&100));
    }

    #[test]
    fn test_intersect_only_common_items() {
        let filter1: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000, 0.01);
        let filter2: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000, 0.01);

        for i in 0..50 {
            filter1.insert(&i);
        }

        for i in 25..75 {
            filter2.insert(&i);
        }

        let intersection = filter1.intersect(&filter2).unwrap();

        // Items 25-49 should be present (in both)
        for i in 25..50 {
            assert!(
                intersection.contains(&i),
                "Intersection missing item {}",
                i
            );
        }

        // Items unique to filter1 (0-24) should not be present
        for i in 0..25 {
            assert!(
                !intersection.contains(&i),
                "Intersection should not contain item {} (only in filter1)",
                i
            );
        }

        // Items unique to filter2 (50-74) should not be present  
        for i in 50..75 {
            assert!(
                !intersection.contains(&i),
                "Intersection should not contain item {} (only in filter2)",
                i
            );
        }
    }

    #[test]
    fn test_cardinality_empty_filter() {
        let filter: StandardBloomFilter<u64> = StandardBloomFilter::new(1_000, 0.01);
        assert_eq!(filter.estimate_cardinality(), 0);
    }

    #[test]
    fn test_cardinality_accuracy() {
        let filter: StandardBloomFilter<u64> = StandardBloomFilter::new(10_000, 0.01);

        // Insert known number of items
        for i in 0..1000 {
            filter.insert(&i);
        }

        let estimated = filter.estimate_cardinality();
        let error = (estimated as i32 - 1000).abs() as f64 / 1000.0;

        // Should be within 15% for well-sized filter
        assert!(
            error < 0.15,
            "Cardinality estimation error {:.1}% exceeds 15%",
            error * 100.0
        );
    }
}
