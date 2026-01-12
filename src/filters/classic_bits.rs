//! Classic Bloom filter using Burton Bloom's Method 2 (1970).
//!
//! This module implements the bit array-based method from Burton Bloom's 1970 paper,
//! which became the foundation for all modern Bloom filter implementations.
//!
//! # Historical Context
//!
//! Method 2 was the second approach proposed by Burton Bloom. Unlike Method 1's
//! hash table with chaining, Method 2 uses a bit array with multiple hash functions.
//! This approach proved to be more space-efficient and became the standard.
//!
//! # Algorithm Description
//!
//! From the 1970 paper:
//!
//! ```text
//! "The set S is represented by an array of m bits, initially all set to 0.
//! To insert an element x, we compute k hash functions h₁(x), h₂(x), ..., hₖ(x)
//! and set the corresponding bits to 1. To test membership, we check if all k
//! bits are set to 1."
//! ```
//!
//! # Key Innovation
//!
//! Method 2's key insight was using multiple hash functions with a bit array
//! instead of storing actual elements. This provides:
//!
//! - Space efficiency: Only 1 bit per hash per element
//! - Simplicity: No chain management or collision handling
//! - Speed: Constant-time operations
//! - Scalability: Easy to parallelize
//!
//! # Differences from Modern Implementation
//!
//! | Aspect | Method 2 (1970) | Modern (StandardBloomFilter) |
//! |--------|-----------------|------------------------------|
//! | Hash generation | k independent hash functions | Enhanced double hashing (2 functions → k) |
//! | Bit operations | Simple set/test | Atomic operations for thread-safety |
//! | Parameter calculation | Manual/empirical | Optimal formulas |
//! | Memory layout | Dense bit array | Lock-free atomic bit vector |
//!
//! # Parameters
//!
//! - m: Size of bit array
//! - k: Number of hash functions
//! - n: Expected number of elements
//!
//! # False Positive Rate
//!
//! The false positive probability (as derived in the paper):
//!
//! ```text
//! P(false positive) = (1 - e^(-kn/m))^k
//! ```
//!
//! # Optimal Parameters
//!
//! The paper showed that optimal k is:
//!
//! ```text
//! k = (m/n) × ln(2) ≈ 0.693 × (m/n)
//! ```
//!
//! # Why This Implementation Exists
//!
//! 1. Historical accuracy: Pure implementation of the 1970 algorithm
//! 2. Educational value: Shows the original bit array approach
//! 3. Comparison baseline: Demonstrates evolution to modern optimizations
//! 4. Research: For studying classic algorithm behavior
//!
//! # Examples
//!
//! ```
//! use bloomcraft::filters::ClassicBitsFilter;
//!
//! // Create filter with 10,000 bits and 7 hash functions
//! let mut filter = ClassicBitsFilter::new(10_000, 7);
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
use crate::error::{BloomCraftError, Result};
use crate::hash::strategies::{EnhancedDoubleHashing, HashStrategy};
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

/// Classic Bloom filter using Burton Bloom's Method 2 (bit array).
///
/// This implementation uses enhanced double hashing to generate k hash values
/// from 2 base hashes, as described in Kirsch & Mitzenmacher (2006), providing
/// the functionality of k independent hash functions with better performance.
///
/// # Type Parameters
///
/// * `T` - Type of items stored (must implement `Hash`)
/// * `H` - Hash function type (must implement `BloomHasher`)
///
/// # Memory Layout
///
/// ```text
/// ClassicBitsFilter {
///     bits: Vec<u64>,  // Bit array (m bits packed into words)
///     m: usize,        // Total number of bits
///     k: usize,        // Number of hash functions
///     hasher: H,       // Hash function generator
/// }
/// ```
///
/// # Space Complexity
///
/// - Exactly m bits for the bit array
/// - Plus O(1) metadata
/// - Total: ⌈m/64⌉ × 8 bytes
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClassicBitsFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone,
{
    /// Bit array stored as u64 words
    bits: Vec<u64>,
    /// Total number of bits (m)
    m: usize,
    /// Number of hash functions (k)
    k: usize,
    /// Hash function
    #[cfg_attr(feature = "serde", serde(skip))]
    hasher: H,
    /// Hash strategy for generating k indices
    strategy: EnhancedDoubleHashing,
    /// Phantom data for type parameter T
    #[cfg_attr(feature = "serde", serde(skip))]
    _phantom: PhantomData<T>,
}

// Manual Clone implementation to maintain zero-cost abstraction
impl<T, H> Clone for ClassicBitsFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    fn clone(&self) -> Self {
        Self {
            bits: self.bits.clone(),
            m: self.m,
            k: self.k,
            hasher: self.hasher.clone(),
            strategy: self.strategy,
            _phantom: PhantomData,
        }
    }
}

impl<T> ClassicBitsFilter<T, StdHasher>
where
    T: Hash,
{
    /// Create a new classic bits filter with default hasher.
    ///
    /// # Arguments
    ///
    /// * `m` - Size of bit array
    /// * `k` - Number of hash functions
    ///
    /// # Panics
    ///
    /// Panics if `m` or `k` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// // 10,000 bits with 7 hash functions
    /// let filter: ClassicBitsFilter<&str> = ClassicBitsFilter::new(10_000, 7);
    /// ```
    #[must_use]
    pub fn new(m: usize, k: usize) -> Self {
        Self::with_hasher(m, k, StdHasher::new())
    }

    /// Create a filter with parameters derived from expected items and FPR.
    ///
    /// Uses the optimal formulas from Bloom's paper:
    /// - m = -n × ln(p) / (ln(2))²
    /// - k = (m/n) × ln(2)
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items (n)
    /// * `fpr` - Target false positive rate (0 < fpr < 1)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// // For 10,000 items with 1% FPR
    /// let filter: ClassicBitsFilter<i32> = ClassicBitsFilter::with_fpr(10_000, 0.01);
    /// ```
    #[must_use]
    pub fn with_fpr(expected_items: usize, fpr: f64) -> Self {
        let (m, k) = calculate_optimal_params(expected_items, fpr);
        Self::new(m, k)
    }
}

impl<T, H> ClassicBitsFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    /// Create a new classic bits filter with custom hasher.
    ///
    /// # Arguments
    ///
    /// * `m` - Size of bit array
    /// * `k` - Number of hash functions
    /// * `hasher` - Custom hash function
    ///
    /// # Panics
    ///
    /// Panics if `m` or `k` is 0.
    #[must_use]
    pub fn with_hasher(m: usize, k: usize, hasher: H) -> Self {
        assert!(m > 0, "m must be > 0");
        assert!(k > 0, "k must be > 0");

        let word_count = (m + 63) / 64;
        Self {
            bits: vec![0u64; word_count],
            m,
            k,
            hasher,
            strategy: EnhancedDoubleHashing,
            _phantom: PhantomData,
        }
    }

    /// Get the size of the bit array (m).
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.m
    }

    /// Get the number of hash functions (k).
    #[must_use]
    #[inline]
    pub fn hash_count(&self) -> usize {
        self.k
    }

    /// Set a bit at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - Bit index to set
    #[inline]
    fn set_bit(&mut self, index: usize) {
        debug_assert!(index < self.m, "Bit index out of bounds");
        let word_idx = index / 64;
        let bit_off = index % 64;
        let mask = 1u64 << bit_off;
        self.bits[word_idx] |= mask;
    }

    /// Test if a bit is set at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - Bit index to test
    ///
    /// # Returns
    ///
    /// `true` if bit is set
    #[inline]
    fn test_bit(&self, index: usize) -> bool {
        debug_assert!(index < self.m, "Bit index out of bounds");
        let word_idx = index / 64;
        let bit_off = index % 64;
        let mask = 1u64 << bit_off;
        (self.bits[word_idx] & mask) != 0
    }

    /// Insert an item into the filter.
    ///
    /// Following Bloom's Method 2:
    /// 1. Compute k hash values (using enhanced double hashing)
    /// 2. Set corresponding bits to 1
    ///
    /// # Arguments
    ///
    /// * `item` - Item to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// let mut filter = ClassicBitsFilter::new(1000, 7);
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    /// ```
    #[inline]
    pub fn insert(&mut self, item: &T) {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.m);

        for idx in indices {
            self.set_bit(idx);
        }
    }

    /// Check if an item might be in the filter.
    ///
    /// Following Bloom's Method 2:
    /// 1. Compute k hash values (using enhanced double hashing)
    /// 2. Check if all corresponding bits are 1
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
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// let mut filter = ClassicBitsFilter::new(1000, 7);
    /// filter.insert(&"hello");
    ///
    /// assert!(filter.contains(&"hello"));
    /// assert!(!filter.contains(&"world"));
    /// ```
    #[must_use]
    #[inline]
    pub fn contains(&self, item: &T) -> bool {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.m);

        indices.iter().all(|&idx| self.test_bit(idx))
    }

    /// Clear all bits in the filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// let mut filter = ClassicBitsFilter::new(1000, 7);
    /// filter.insert(&"hello");
    /// filter.clear();
    /// assert!(!filter.contains(&"hello"));
    /// ```
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
    }

    /// Check if the filter is empty (no bits set).
    ///
    /// # Returns
    ///
    /// `true` if no bits are set
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&word| word == 0)
    }

    /// Count the number of bits currently set.
    ///
    /// # Returns
    ///
    /// Number of set bits
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// let mut filter = ClassicBitsFilter::new(1000, 7);
    /// assert_eq!(filter.count_set_bits(), 0);
    ///
    /// filter.insert(&"hello");
    /// assert!(filter.count_set_bits() > 0);
    /// ```
    #[must_use]
    pub fn count_set_bits(&self) -> usize {
        self.bits.iter().map(|word| word.count_ones() as usize).sum()
    }

    /// Calculate the fill rate (fraction of bits set).
    ///
    /// # Returns
    ///
    /// Fill rate in range [0, 1]
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        self.count_set_bits() as f64 / self.m as f64
    }

    /// Estimate the current false positive rate.
    ///
    /// Uses Bloom's formula: P(FP) = (1 - e^(-kn/m))^k
    ///
    /// We estimate n from the fill rate: n ≈ -(m/k) × ln(1 - fill_rate)
    ///
    /// # Returns
    ///
    /// Estimated false positive rate
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        let fill_rate = self.fill_rate();
        if fill_rate == 0.0 {
            return 0.0;
        }

        if fill_rate >= 1.0 {
            return 1.0;
        }

        // Estimate n from fill rate: n ≈ -(m/k) × ln(1 - fill_rate)
        let m_f64 = self.m as f64;
        let k_f64 = self.k as f64;
        let estimated_n = -(m_f64 / k_f64) * (1.0 - fill_rate).ln();

        // Calculate FPR using standard formula: (1 - e^(-kn/m))^k
        let exponent = -(k_f64 * estimated_n) / m_f64;
        (1.0 - exponent.exp()).powf(k_f64)
    }

    /// Check if the filter is approximately full.
    ///
    /// Returns true if fill rate > 0.5
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.fill_rate() > 0.5
    }

    /// Get memory usage in bytes.
    ///
    /// # Returns
    ///
    /// Memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.bits.len() * std::mem::size_of::<u64>() + std::mem::size_of::<Self>()
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

    /// Compute the union of two filters.
    ///
    /// Both filters must have the same m and k.
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
    pub fn union(&self, other: &Self) -> Result<Self> {
        if self.m != other.m || self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Parameters mismatch".to_string(),
            });
        }

        let mut result = self.clone();
        for (i, word) in result.bits.iter_mut().enumerate() {
            *word |= other.bits[i];
        }

        Ok(result)
    }

    /// Compute the intersection of two filters.
    ///
    /// Both filters must have the same m and k.
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
    pub fn intersect(&self, other: &Self) -> Result<Self> {
        if self.m != other.m || self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Parameters mismatch".to_string(),
            });
        }

        let mut result = self.clone();
        for (i, word) in result.bits.iter_mut().enumerate() {
            *word &= other.bits[i];
        }

        Ok(result)
    }
}

impl<T, H> BloomFilter<T> for ClassicBitsFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    fn insert(&mut self, item: &T) {
        ClassicBitsFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        ClassicBitsFilter::contains(self, item)
    }

    fn clear(&mut self) {
        ClassicBitsFilter::clear(self);
    }

    fn len(&self) -> usize {
        self.count_set_bits()
    }

    fn is_empty(&self) -> bool {
        ClassicBitsFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn expected_items(&self) -> usize {
        // Classic filter doesn't track this - estimate from parameters
        // Using formula: n ≈ (m * ln(2)) / k
        ((self.m as f64 * std::f64::consts::LN_2) / self.k as f64) as usize
    }

    fn bit_count(&self) -> usize {
        self.m
    }

    fn hash_count(&self) -> usize {
        self.k
    }
}

/// Calculate optimal parameters (m, k) for given n and fpr.
///
/// Uses the formulas from Bloom's 1970 paper:
/// - m = -n × ln(p) / (ln(2))²
/// - k = (m/n) × ln(2)
///
/// # Arguments
///
/// * `n` - Expected number of elements
/// * `fpr` - Target false positive rate
///
/// # Returns
///
/// Tuple of (m, k)
fn calculate_optimal_params(n: usize, fpr: f64) -> (usize, usize) {
    assert!(fpr > 0.0 && fpr < 1.0, "fpr must be in (0, 1)");
    assert!(n > 0, "n must be > 0");

    let ln2 = std::f64::consts::LN_2;
    let ln2_squared = ln2 * ln2;

    // m = -n × ln(p) / (ln(2))²
    let m = (-(n as f64) * fpr.ln() / ln2_squared).ceil() as usize;

    // k = (m/n) × ln(2)
    let k = ((m as f64 / n as f64) * ln2).round() as usize;

    (m.max(1), k.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let filter: ClassicBitsFilter<&str> = ClassicBitsFilter::new(1000, 7);
        assert_eq!(filter.size(), 1000);
        assert_eq!(filter.hash_count(), 7);
        assert!(filter.is_empty());
    }

    #[test]
    #[should_panic(expected = "m must be > 0")]
    fn test_new_zero_size() {
        let _: ClassicBitsFilter<&str> = ClassicBitsFilter::new(0, 7);
    }

    #[test]
    #[should_panic(expected = "k must be > 0")]
    fn test_new_zero_k() {
        let _: ClassicBitsFilter<&str> = ClassicBitsFilter::new(1000, 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        filter.insert(&"hello");
        assert!(filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
    }

    #[test]
    fn test_multiple_inserts() {
        let mut filter = ClassicBitsFilter::new(10_000, 7);
        for i in 0..1000 {
            filter.insert(&i);
        }

        for i in 0..1000 {
            assert!(filter.contains(&i), "False negative for {}", i);
        }
    }

    #[test]
    fn test_clear() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        filter.insert(&"hello");
        filter.insert(&"world");
        assert!(!filter.is_empty());

        filter.clear();
        assert!(filter.is_empty());
        assert!(!filter.contains(&"hello"));
    }

    #[test]
    fn test_count_set_bits() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        assert_eq!(filter.count_set_bits(), 0);

        filter.insert(&"test");
        let set_bits = filter.count_set_bits();
        assert!(set_bits > 0);
        assert!(set_bits <= 7); // At most k bits set for one item
    }

    #[test]
    fn test_fill_rate() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        assert_eq!(filter.fill_rate(), 0.0);

        for i in 0..100 {
            filter.insert(&i);
        }

        let fill_rate = filter.fill_rate();
        assert!(fill_rate > 0.0 && fill_rate < 1.0);
    }

    #[test]
    fn test_estimate_fpr() {
        let mut filter = ClassicBitsFilter::new(10_000, 7);
        for i in 0..5000 {
            filter.insert(&i);
        }

        let fpr = filter.estimate_fpr();
        assert!(fpr > 0.0 && fpr < 1.0);
    }

    #[test]
    fn test_is_full() {
        let mut filter = ClassicBitsFilter::new(100, 7);
        // Saturate the filter
        for i in 0..1000 {
            filter.insert(&i);
        }

        assert!(filter.is_full());
    }

    #[test]
    fn test_memory_usage() {
        let filter: ClassicBitsFilter<i32> = ClassicBitsFilter::new(10_000, 7);
        let mem = filter.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_with_fpr() {
        let filter: ClassicBitsFilter<i32> = ClassicBitsFilter::with_fpr(10_000, 0.01);
        assert!(filter.size() > 0);
        assert!(filter.hash_count() > 0);
    }

    #[test]
    fn test_insert_batch() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        let items = vec!["a", "b", "c", "d"];
        filter.insert_batch(&items);

        for item in &items {
            assert!(filter.contains(item));
        }
    }

    #[test]
    fn test_contains_batch() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        filter.insert(&"a");
        filter.insert(&"b");

        let queries = vec!["a", "b", "c", "d"];
        let results = filter.contains_batch(&queries);
        assert_eq!(results, vec![true, true, false, false]);
    }

    #[test]
    fn test_union() {
        let mut filter1 = ClassicBitsFilter::new(1000, 7);
        let mut filter2 = ClassicBitsFilter::new(1000, 7);

        filter1.insert(&"a");
        filter1.insert(&"b");
        filter2.insert(&"b");
        filter2.insert(&"c");

        let union = filter1.union(&filter2).unwrap();
        assert!(union.contains(&"a"));
        assert!(union.contains(&"b"));
        assert!(union.contains(&"c"));
    }

    #[test]
    fn test_union_incompatible() {
        let filter1: ClassicBitsFilter<String> = ClassicBitsFilter::new(1000, 7);
        let filter2: ClassicBitsFilter<String> = ClassicBitsFilter::new(2000, 7);

        let result = filter1.union(&filter2);
        assert!(result.is_err());
    }

    #[test]
    fn test_intersect() {
        let mut filter1 = ClassicBitsFilter::new(1000, 7);
        let mut filter2 = ClassicBitsFilter::new(1000, 7);

        filter1.insert(&"a");
        filter1.insert(&"b");
        filter2.insert(&"b");
        filter2.insert(&"c");

        let intersection = filter1.intersect(&filter2).unwrap();
        assert!(intersection.contains(&"b"));
    }

    #[test]
    fn test_bloom_filter_trait() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        BloomFilter::insert(&mut filter, &"test");
        assert!(BloomFilter::contains(&filter, &"test"));
        assert!(!BloomFilter::is_empty(&filter));

        BloomFilter::clear(&mut filter);
        assert!(BloomFilter::is_empty(&filter));
    }

    #[test]
    fn test_no_false_negatives() {
        let mut filter = ClassicBitsFilter::new(10_000, 7);
        let items = vec!["apple", "banana", "cherry", "date", "elderberry"];

        for item in &items {
            filter.insert(item);
        }

        for item in &items {
            assert!(filter.contains(item), "False negative for {}", item);
        }
    }

    #[test]
    fn test_false_positive_rate() {
        let mut filter: ClassicBitsFilter<i32> = ClassicBitsFilter::with_fpr(1000, 0.01);

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
        // Should be reasonably close (within 5x)
        assert!(actual_fpr < 0.05);
    }

    #[test]
    fn test_calculate_optimal_params() {
        let (m, k) = calculate_optimal_params(10_000, 0.01);
        assert!(m > 0);
        assert!(k > 0);
        // For 1% FPR, should be around 10 bits per item and 7 hash functions
        assert!(m >= 90_000 && m <= 110_000);
        assert!(k >= 6 && k <= 8);
    }

    #[test]
    fn test_bit_operations() {
        let mut filter: ClassicBitsFilter<String> = ClassicBitsFilter::new(64, 3);

        // Test setting and getting individual bits
        filter.set_bit(0);
        assert!(filter.test_bit(0));
        assert!(!filter.test_bit(1));

        filter.set_bit(63);
        assert!(filter.test_bit(63));
    }

    #[test]
    fn test_duplicate_inserts() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        filter.insert(&"test");
        let bits_after_first = filter.count_set_bits();

        filter.insert(&"test");
        let bits_after_second = filter.count_set_bits();

        // Same number of bits should be set
        assert_eq!(bits_after_first, bits_after_second);
    }

    #[test]
    fn test_clone() {
        let mut filter1 = ClassicBitsFilter::new(1000, 7);
        filter1.insert(&"test");

        let filter2 = filter1.clone();
        assert!(filter2.contains(&"test"));
        assert_eq!(filter1.size(), filter2.size());
        assert_eq!(filter1.hash_count(), filter2.hash_count());
    }
}
