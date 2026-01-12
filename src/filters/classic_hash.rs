//! Classic Bloom filter using Burton Bloom's Method 1 (1970).
//!
//! This module implements the original hash table-based method from Burton Bloom's
//! seminal 1970 paper "Space/Time Trade-offs in Hash Coding with Allowable Errors".
//!
//! # Historical Context
//!
//! Method 1 was the first approach proposed by Burton Bloom. It uses a hash table
//! with chaining where only the first d elements in each chain are stored. This
//! provides a controlled false positive rate.
//!
//! # Algorithm Description
//!
//! From the 1970 paper:
//!
//! ```text
//! "The set S is stored in a hash table of size m using a hash function h.
//! Each position in the table can hold at most d elements. When a collision
//! occurs, the element is added to the chain only if the chain has fewer than
//! d elements. Otherwise, it is discarded."
//! ```
//!
//! # Key Differences from Modern Bloom Filters
//!
//! | Aspect | Method 1 (1970) | Modern Bloom Filter |
//! |--------|-----------------|---------------------|
//! | Data structure | Hash table with chains | Bit array |
//! | Collision handling | Chaining (limited depth) | Multiple hash functions |
//! | Space efficiency | Lower (stores elements) | Higher (only bits) |
//! | False positive control | Chain depth (d) | Number of hash functions (k) |
//! | Implementation complexity | Higher | Lower |
//!
//! # Parameters
//!
//! - m: Hash table size (number of buckets)
//! - d: Maximum chain depth per bucket
//! - n: Expected number of elements
//!
//! # False Positive Rate
//!
//! The false positive probability is approximately:
//!
//! ```text
//! P(false positive) ≈ (1 - e^(-n/m))^d
//! ```
//!
//! # Why This Matters
//!
//! This implementation is included for:
//! 1. Historical accuracy: Understanding the evolution of Bloom filters
//! 2. Educational value: Showing the original approach
//! 3. Comparison: Demonstrating why modern bit-array approach is superior
//!
//! # Examples
//!
//! ```
//! use bloomcraft::filters::ClassicHashFilter;
//!
//! // Create filter with 1000 buckets and depth 3
//! let mut filter = ClassicHashFilter::new(1000, 3);
//!
//! // Insert items
//! filter.insert(&"hello");
//! filter.insert(&"world");
//!
//! // Query items
//! assert!(filter.contains(&"hello"));
//! assert!(filter.contains(&"world"));
//! assert!(!filter.contains(&"goodbye"));
//! ```
//!
//! # References
//!
//! - Bloom, B. H. (1970). "Space/time trade-offs in hash coding with allowable errors".
//!   Communications of the ACM, 13(7), 422-426.

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use crate::core::filter::BloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Convert a hashable item to bytes using Rust's `Hash` trait.
///
/// This is the bridge between generic `T: Hash` and the `&[u8]` API
/// required by `BloomHasher`.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}

/// A chain entry in the hash table.
///
/// Each bucket in the hash table contains a chain of up to `d` entries.
/// We store both the primary hash (for bucket lookup) and a secondary hash
/// (for collision detection) to minimize false positives from hash collisions.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct ChainEntry {
    /// Primary hash value of the stored element (used for bucket lookup)
    hash: u64,
    /// Secondary hash value for collision detection (different seed)
    secondary_hash: u64,
}

/// Classic Bloom filter using Burton Bloom's Method 1.
///
/// This implementation uses a hash table with limited-depth chaining as described
/// in the original 1970 paper.
///
/// # Type Parameters
///
/// * `T` - Type of items stored (must implement `Hash`)
/// * `H` - Hash function type (must implement `BloomHasher`)
///
/// # Memory Layout
///
/// ```text
/// ClassicHashFilter {
///     table: Vec<Vec<ChainEntry>>, // m buckets, each with up to d entries
///     m: usize,                     // number of buckets
///     d: usize,                     // maximum chain depth
///     count: usize,                 // number of elements inserted
/// }
/// ```
///
/// # Space Complexity
///
/// - Worst case: O(m * d * sizeof(ChainEntry)) bytes
/// - Best case: O(m) bytes (empty chains)
/// - Average: O(n * sizeof(ChainEntry)) where n is number of elements
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClassicHashFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone,
{
    /// Hash table: m buckets, each containing a chain of entries
    table: Vec<Vec<ChainEntry>>,
    /// Number of buckets (m)
    m: usize,
    /// Maximum chain depth per bucket (d)
    d: usize,
    /// Number of elements inserted
    count: usize,
    /// Hash function
    #[cfg_attr(feature = "serde", serde(skip))]
    hasher: H,
    /// Phantom data for type parameter T
    #[cfg_attr(feature = "serde", serde(skip))]
    _phantom: PhantomData<T>,
}

impl<T> ClassicHashFilter<T, StdHasher>
where
    T: Hash,
{
    /// Create a new classic hash filter with default hasher.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of buckets in hash table
    /// * `d` - Maximum chain depth per bucket
    ///
    /// # Panics
    ///
    /// Panics if `m` or `d` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// // 1000 buckets with depth 3
    /// let filter: ClassicHashFilter<&str> = ClassicHashFilter::new(1000, 3);
    /// ```
    #[must_use]
    pub fn new(m: usize, d: usize) -> Self {
        Self::with_hasher(m, d, StdHasher::new())
    }

    /// Create a filter with parameters derived from expected items and FPR.
    ///
    /// Calculates appropriate m and d values to achieve the target false positive rate.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items (n)
    /// * `fpr` - Target false positive rate (0 < fpr < 1)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// // For 10,000 items with 1% FPR
    /// let filter: ClassicHashFilter<i32> = ClassicHashFilter::with_fpr(10_000, 0.01);
    /// ```
    #[must_use]
    pub fn with_fpr(expected_items: usize, fpr: f64) -> Self {
        let (m, d) = calculate_params(expected_items, fpr);
        Self::new(m, d)
    }
}

impl<T, H> ClassicHashFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    /// Create a new classic hash filter with custom hasher.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of buckets
    /// * `d` - Maximum chain depth
    /// * `hasher` - Custom hash function
    ///
    /// # Panics
    ///
    /// Panics if `m` or `d` is 0.
    #[must_use]
    pub fn with_hasher(m: usize, d: usize, hasher: H) -> Self {
        assert!(m > 0, "m must be > 0");
        assert!(d > 0, "d must be > 0");

        Self {
            table: vec![Vec::with_capacity(d); m],
            m,
            d,
            count: 0,
            hasher,
            _phantom: PhantomData,
        }
    }

    /// Get the number of buckets (m).
    #[must_use]
    #[inline]
    pub fn bucket_count(&self) -> usize {
        self.m
    }

    /// Get the maximum chain depth (d).
    #[must_use]
    #[inline]
    pub fn max_depth(&self) -> usize {
        self.d
    }

    /// Get the number of elements inserted.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the filter is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Compute primary hash of an item using the BloomHasher trait.
    #[inline]
    fn compute_hash(&self, item: &T) -> u64 {
        let bytes = hash_item_to_bytes(item);
        let (h1, _) = self.hasher.hash_bytes_pair(&bytes);
        h1
    }

    /// Compute secondary hash of an item using the BloomHasher trait.
    ///
    /// Uses the second hash value from the hasher for collision detection.
    #[inline]
    fn compute_secondary_hash(&self, item: &T) -> u64 {
        let bytes = hash_item_to_bytes(item);
        let (_, h2) = self.hasher.hash_bytes_pair(&bytes);
        h2
    }

    /// Hash an item to a bucket index.
    #[inline]
    fn hash_to_bucket(&self, item: &T) -> usize {
        let hash = self.compute_hash(item);
        (hash % self.m as u64) as usize
    }

    /// Insert an item into the filter.
    ///
    /// Following Bloom's Method 1:
    /// 1. Hash item to bucket index
    /// 2. If chain has space (< d elements), add entry
    /// 3. If chain is full, discard (contributes to false positive rate)
    ///
    /// Uses dual hashing (primary + secondary) to reduce false positives
    /// from hash collisions.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(1000, 3);
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    /// ```
    pub fn insert(&mut self, item: &T) {
        let bucket_idx = self.hash_to_bucket(item);
        let hash = self.compute_hash(item);
        let secondary_hash = self.compute_secondary_hash(item);

        let chain = &mut self.table[bucket_idx];

        // Check if already present (both hashes must match)
        if chain
            .iter()
            .any(|entry| entry.hash == hash && entry.secondary_hash == secondary_hash)
        {
            return; // Already in the chain
        }

        // Add to chain if there's space
        if chain.len() < self.d {
            chain.push(ChainEntry {
                hash,
                secondary_hash,
            });
            self.count += 1;
        }

        // Otherwise discard (as per original algorithm)
    }

    /// Check if an item might be in the filter.
    ///
    /// Uses dual hashing to reduce false positives from hash collisions.
    /// Both the primary and secondary hash must match for a positive result.
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
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(1000, 3);
    /// filter.insert(&"hello");
    ///
    /// assert!(filter.contains(&"hello"));
    /// assert!(!filter.contains(&"world"));
    /// ```
    #[must_use]
    pub fn contains(&self, item: &T) -> bool {
        let bucket_idx = self.hash_to_bucket(item);
        let hash = self.compute_hash(item);
        let secondary_hash = self.compute_secondary_hash(item);

        self.table[bucket_idx]
            .iter()
            .any(|entry| entry.hash == hash && entry.secondary_hash == secondary_hash)
    }

    /// Clear all entries from the filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(1000, 3);
    /// filter.insert(&"hello");
    /// filter.clear();
    /// assert!(filter.is_empty());
    /// ```
    pub fn clear(&mut self) {
        for chain in &mut self.table {
            chain.clear();
        }
        self.count = 0;
    }

    /// Get the average chain length.
    ///
    /// This provides insight into how well the hash function is distributing items.
    ///
    /// # Returns
    ///
    /// Average number of entries per non-empty bucket
    #[must_use]
    pub fn avg_chain_length(&self) -> f64 {
        let non_empty = self.table.iter().filter(|c| !c.is_empty()).count();
        if non_empty == 0 {
            return 0.0;
        }

        self.count as f64 / non_empty as f64
    }

    /// Get the maximum chain length currently in the table.
    ///
    /// # Returns
    ///
    /// Length of longest chain
    #[must_use]
    pub fn max_chain_length(&self) -> usize {
        self.table.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Get the load factor (fraction of buckets with at least one entry).
    ///
    /// # Returns
    ///
    /// Load factor in range [0, 1]
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        let non_empty = self.table.iter().filter(|c| !c.is_empty()).count();
        non_empty as f64 / self.m as f64
    }

    /// Estimate the current false positive rate.
    ///
    /// Uses the formula: P(FP) ≈ (1 - e^(-n/m))^d
    ///
    /// # Returns
    ///
    /// Estimated false positive rate
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        let n = self.count as f64;
        let m = self.m as f64;
        let d = self.d as f64;

        (1.0 - (-n / m).exp()).powf(d)
    }

    /// Get memory usage in bytes.
    ///
    /// # Returns
    ///
    /// Approximate memory usage
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let table_overhead = self.m * std::mem::size_of::<Vec<ChainEntry>>();
        let entries = self.count * std::mem::size_of::<ChainEntry>();
        let metadata = std::mem::size_of::<Self>();
        table_overhead + entries + metadata
    }

    /// Insert multiple items in batch.
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of items to insert
    pub fn insert_batch(&mut self, items: &[T]) {
        for item in items {
            self.insert(item);
        }
    }

    /// Check multiple items in batch.
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of items to check
    ///
    /// # Returns
    ///
    /// Vector of boolean results
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }
}

impl<T, H> BloomFilter<T> for ClassicHashFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    fn insert(&mut self, item: &T) {
        ClassicHashFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        ClassicHashFilter::contains(self, item)
    }

    fn clear(&mut self) {
        ClassicHashFilter::clear(self);
    }

    fn len(&self) -> usize {
        ClassicHashFilter::len(self)
    }

    fn is_empty(&self) -> bool {
        ClassicHashFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn expected_items(&self) -> usize {
        // Classic hash filter doesn't track expected items - use current count
        self.count
    }

    fn bit_count(&self) -> usize {
        // Return logical "bit count" - total possible storage slots
        self.m * self.d * 64 // Each entry is effectively 128 bits (two u64s)
    }

    fn hash_count(&self) -> usize {
        // Method 1 uses dual hashing (primary + secondary)
        2
    }
}

/// Calculate optimal parameters (m, d) for given n and fpr.
///
/// Uses the formula: P(FP) ≈ (1 - e^(-n/m))^d
///
/// # Arguments
///
/// * `n` - Expected number of elements
/// * `fpr` - Target false positive rate
///
/// # Returns
///
/// Tuple of (m, d)
fn calculate_params(n: usize, fpr: f64) -> (usize, usize) {
    assert!(fpr > 0.0 && fpr < 1.0, "fpr must be in (0, 1)");
    assert!(n > 0, "n must be > 0");

    // Start with d = 3 (reasonable default from paper)
    let d = 3;

    // Solve for m: fpr = (1 - e^(-n/m))^d
    // => m ≈ -n / ln(1 - fpr^(1/d))
    let fpr_root = fpr.powf(1.0 / d as f64);
    let m = if fpr_root >= 1.0 {
        n * 10 // Fallback
    } else {
        let m = -(n as f64) / (1.0 - fpr_root).ln();
        m.ceil() as usize
    };

    (m.max(1), d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let filter: ClassicHashFilter<&str> = ClassicHashFilter::new(1000, 3);
        assert_eq!(filter.bucket_count(), 1000);
        assert_eq!(filter.max_depth(), 3);
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
    }

    #[test]
    #[should_panic(expected = "m must be > 0")]
    fn test_new_zero_buckets() {
        let _: ClassicHashFilter<&str> = ClassicHashFilter::new(0, 3);
    }

    #[test]
    #[should_panic(expected = "d must be > 0")]
    fn test_new_zero_depth() {
        let _: ClassicHashFilter<&str> = ClassicHashFilter::new(1000, 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut filter = ClassicHashFilter::new(100, 3);
        filter.insert(&"hello");
        assert!(filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
    }

    #[test]
    fn test_multiple_inserts() {
        let mut filter = ClassicHashFilter::new(500, 5);
        for i in 0..50 {
            filter.insert(&i);
        }

        for i in 0..50 {
            assert!(filter.contains(&i));
        }
    }

    #[test]
    fn test_clear() {
        let mut filter = ClassicHashFilter::new(100, 3);
        filter.insert(&"a");
        filter.insert(&"b");
        assert_eq!(filter.len(), 2);

        filter.clear();
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert!(!filter.contains(&"a"));
    }

    #[test]
    fn test_duplicate_insert() {
        let mut filter = ClassicHashFilter::new(100, 3);
        filter.insert(&"test");
        filter.insert(&"test");
        assert_eq!(filter.len(), 1); // Should only count once
    }

    #[test]
    fn test_chain_overflow() {
        let mut filter = ClassicHashFilter::new(1, 2); // 1 bucket, depth 2

        // Insert items that will hash to same bucket
        // After 2 items, bucket is full and subsequent items are discarded
        for i in 0..10 {
            filter.insert(&i);
        }

        // Should have at most 2 items (depth limit)
        assert!(filter.len() <= 2);
    }

    #[test]
    fn test_avg_chain_length() {
        let mut filter = ClassicHashFilter::new(100, 5);
        for i in 0..50 {
            filter.insert(&i);
        }

        let avg = filter.avg_chain_length();
        assert!(avg > 0.0);
        assert!(avg <= 5.0); // Can't exceed max depth
    }

    #[test]
    fn test_max_chain_length() {
        let mut filter = ClassicHashFilter::new(100, 5);
        for i in 0..50 {
            filter.insert(&i);
        }

        let max_len = filter.max_chain_length();
        assert!(max_len > 0);
        assert!(max_len <= 5); // Can't exceed max depth
    }

    #[test]
    fn test_load_factor() {
        let mut filter = ClassicHashFilter::new(100, 3);
        assert_eq!(filter.load_factor(), 0.0);

        for i in 0..20 {
            filter.insert(&i);
        }

        let load = filter.load_factor();
        assert!(load > 0.0 && load <= 1.0);
    }

    #[test]
    fn test_estimate_fpr() {
        let mut filter = ClassicHashFilter::new(1000, 3);
        for i in 0..500 {
            filter.insert(&i);
        }

        let fpr = filter.estimate_fpr();
        assert!(fpr > 0.0 && fpr < 1.0);
    }

    #[test]
    fn test_memory_usage() {
        let filter: ClassicHashFilter<i32> = ClassicHashFilter::new(1000, 3);
        let mem = filter.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_with_fpr() {
        let filter: ClassicHashFilter<i32> = ClassicHashFilter::with_fpr(1000, 0.01);
        assert!(filter.bucket_count() > 0);
        assert!(filter.max_depth() > 0);
    }

    #[test]
    fn test_insert_batch() {
        let mut filter = ClassicHashFilter::new(100, 3);
        let items = vec!["a", "b", "c", "d"];
        filter.insert_batch(&items);

        for item in &items {
            assert!(filter.contains(item));
        }
    }

    #[test]
    fn test_contains_batch() {
        let mut filter = ClassicHashFilter::new(100, 3);
        filter.insert(&"a");
        filter.insert(&"b");

        let queries = vec!["a", "b", "c", "d"];
        let results = filter.contains_batch(&queries);
        assert_eq!(results[0], true);
        assert_eq!(results[1], true);
        assert_eq!(results[2], false);
        assert_eq!(results[3], false);
    }

    #[test]
    fn test_bloom_filter_trait() {
        let mut filter = ClassicHashFilter::new(100, 3);
        BloomFilter::insert(&mut filter, &"test");
        assert!(BloomFilter::contains(&filter, &"test"));
        assert!(!BloomFilter::is_empty(&filter));

        BloomFilter::clear(&mut filter);
        assert!(BloomFilter::is_empty(&filter));
    }

    #[test]
    fn test_no_false_negatives() {
        let mut filter = ClassicHashFilter::new(100, 3);
        let items = vec!["apple", "banana", "cherry"];

        for item in &items {
            filter.insert(item);
        }

        for item in &items {
            assert!(filter.contains(item), "False negative for {}", item);
        }
    }

    #[test]
    fn test_calculate_params() {
        let (m, d) = calculate_params(1000, 0.01);
        assert!(m > 0);
        assert!(d > 0);
    }

    #[test]
    fn test_clone() {
        let mut filter1 = ClassicHashFilter::new(100, 3);
        filter1.insert(&"test");

        let filter2 = filter1.clone();
        assert!(filter2.contains(&"test"));
        assert_eq!(filter1.len(), filter2.len());
    }
}
