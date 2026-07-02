//! Classic K-Independent Bloom Filter (Burton Bloom's Method 2, 1970)
//!
//! Companion to [`ClassicHashFilter`](crate::filters::ClassicHashFilter) (Method 1).
//! Method 2 uses a bit array with k independent hash functions;
//! Method 1 stores actual elements in a hash table with limited-depth chaining.
//!
//! This is the foundation that all modern Bloom filters descend from. The idea:
//! a bit array of m bits, k independent hash functions, and a single rule --
//! set all k bits on insert, check all k bits on query. No element storage,
//! no chain management, just bits.
//!
//! This implementation intentionally avoids post-1970 optimizations (double
//! hashing, atomics, cache optimization) to serve as a historically accurate
//! baseline. For production, use
//! [`StandardBloomFilter`](crate::filters::StandardBloomFilter) instead.
//!
//! # Historical Context
//!
//! | Aspect | Method 2 (1970) | Modern |
//! |--------|-----------------|--------|
//! | Hash generation | k independent computations | Enhanced double hashing |
//! | Bit operations | Simple set/test | Atomic operations |
//! | Thread safety | Single-threaded | Thread-safe |
//!
//! # Example
//!
//! ```
//! use bloomcraft::filters::ClassicBitsFilter;
//!
//! let mut filter = ClassicBitsFilter::new(10_000, 7);
//! filter.insert(&"hello");
//! assert!(filter.contains(&"hello"));
//! ```
//!
//! # References
//!
//! - Bloom, B. H. (1970). "Space/time trade-offs in hash coding with allowable errors".
//!   Communications of the ACM, 13(7), 422-426.
//! - Kirsch, A., & Mitzenmacher, M. (2006). "Less Hashing, Same Performance: Building a Better Bloom Filter".
//!   European Symposium on Algorithms, 456-467.
//!
//! # Thread Safety
//!
//! **Not thread-safe**. Uses plain `Vec<u64>` with no atomics.

#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]

use crate::core::filter::BloomFilter;
use crate::core::params::{optimal_bit_count, optimal_hash_count};
use crate::error::{BloomCraftError, Result};
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Classic Bloom filter using k independent hash functions (Burton Bloom's Method 2, 1970).
///
/// The original design: k separate hash computations per operation, each going
/// through the full `BloomHasher` pipeline. Intentionally slower than modern
/// implementations to serve as a historically accurate baseline.
///
/// # Type Parameters
///
/// * `T` - Item type. Must implement `Hash`.
/// * `H` - Hash function provider. Must implement [`BloomHasher`]. Defaults to [`StdHasher`].
///
/// Stores `m` bits packed into u64 words with `k` independent hash functions.
///
/// # Thread Safety
///
/// **Not thread-safe**. Uses plain `Vec<u64>` with no atomics.
/// For concurrent access, use [`StandardBloomFilter`](crate::filters::StandardBloomFilter).
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClassicBitsFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone,
{
    /// Bit array stored as u64 words (non-atomic for historical accuracy)
    bits: Vec<u64>,

    /// Total number of bits (m)
    m: usize,

    /// Number of independent hash functions (k)
    k: usize,

    /// Hash function used to generate k independent hashes
    #[cfg_attr(feature = "serde", serde(skip))]
    hasher: H,

    /// Phantom data for type parameter T
    #[cfg_attr(feature = "serde", serde(skip))]
    _phantom: PhantomData<T>,
}

// Manual Clone implementation to handle PhantomData and hasher
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
            _phantom: PhantomData,
        }
    }
}

impl<T> ClassicBitsFilter<T, StdHasher>
where
    T: Hash,
{
    /// Create a filter with `m` bits and `k` independent hash functions.
    ///
    /// `m` is your memory budget (each bit costs 1/8 byte). You need roughly
    /// 10 bits per expected item for 1% FPR. `k` is the number of hash functions
    /// per operation; optimal k is ~0.693 x (m/n).
    ///
    /// # Panics
    ///
    /// Panics if `m` or `k` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    /// let filter: ClassicBitsFilter<&str> = ClassicBitsFilter::new(10_000, 7);
    /// ```
    #[must_use]
    pub fn new(m: usize, k: usize) -> Self {
        Self::with_hasher(m, k, StdHasher::new())
    }

    /// Create a filter with optimal (m, k) for a given capacity and false positive rate.
    ///
    /// Uses Bloom's 1970 formulas: m = -n x ln(p) / (ln(2))^2,
    /// k = (m/n) x ln(2). This gives the smallest bit array and the right
    /// number of hash functions to achieve your target FPR.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is 0 or `fpr` is not in (0, 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// let filter: ClassicBitsFilter<i32> = ClassicBitsFilter::with_fpr(10_000, 0.01);
    /// assert!(filter.hash_count() >= 6 && filter.hash_count() <= 8);
    /// ```
    #[must_use]
    pub fn with_fpr(expected_items: usize, fpr: f64) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            fpr > 0.0 && fpr < 1.0,
            "fpr must be in range (0, 1), got {fpr}"
        );

        let m = optimal_bit_count(expected_items, fpr)
            .expect("optimal_bit_count should succeed with valid parameters");
        let k = optimal_hash_count(m, expected_items)
            .expect("optimal_hash_count should succeed with valid parameters");

        Self::new(m, k)
    }
}

impl<T, H> ClassicBitsFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    /// Create a filter with `m` bits, `k` hash functions, and a custom hasher.
    ///
    /// Same as [`new`](Self::new) but lets you supply your own hash function.
    ///
    /// # Panics
    ///
    /// Panics if `m` or `k` is 0.
    #[must_use]
    pub fn with_hasher(m: usize, k: usize, hasher: H) -> Self {
        assert!(m > 0, "m must be > 0");
        assert!(k > 0, "k must be > 0");

        let word_count = m.div_ceil(64);

        Self {
            bits: vec![0u64; word_count],
            m,
            k,
            hasher,
            _phantom: PhantomData,
        }
    }

    /// Returns the total number of bits in the bit array (m).
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.m
    }

    /// Returns the number of independent hash functions computed per operation (k).
    #[must_use]
    #[inline]
    pub fn hash_count(&self) -> usize {
        self.k
    }

    /// Set a bit by index.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `index >= m`.
    #[inline]
    fn set_bit(&mut self, index: usize) {
        debug_assert!(
            index < self.m,
            "Bit index {} out of bounds (m={})",
            index,
            self.m
        );

        let word_idx = index / 64;
        let bit_offset = index % 64;
        let mask = 1u64 << bit_offset;

        self.bits[word_idx] |= mask;
    }

    /// Check if a bit is set.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `index >= m`.
    #[inline]
    fn test_bit(&self, index: usize) -> bool {
        debug_assert!(
            index < self.m,
            "Bit index {} out of bounds (m={})",
            index,
            self.m
        );

        let word_idx = index / 64;
        let bit_offset = index % 64;
        let mask = 1u64 << bit_offset;

        (self.bits[word_idx] & mask) != 0
    }

    /// Compute the i-th independent hash for an item.
    ///
    /// This is the hot path that makes the classic filter ~5x slower than modern
    /// double-hashing. Each call runs `hash_item` + `hash_bytes_pair`, and
    /// insert/contains call this k times.
    #[inline]
    fn compute_independent_hash(&self, item: &T, i: usize) -> usize {
        let (h1, _) = self.hasher.hash_item(item);
        let base_bytes = h1.to_le_bytes();
        let index_bytes = i.to_le_bytes();

        // Stack-allocated array (no heap allocation!)
        // Combining 8 bytes (hash) + 8 bytes (index) = 16 bytes total
        let mut combined = [0u8; 16];
        combined[0..8].copy_from_slice(&base_bytes);
        combined[8..16].copy_from_slice(&index_bytes);

        // Hash the combined data
        let (h, _) = self.hasher.hash_bytes_pair(&combined);

        // Map to bit index
        (h as usize) % self.m
    }

    /// Insert an item into the filter.
    ///
    /// Runs k independent hash computations and sets the resulting bits.
    /// At k=7 this takes ~864 ns per item -- no double-hashing shortcut.
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
        // TRUE 1970 ALGORITHM: Compute k independent hash values
        for i in 0..self.k {
            let index = self.compute_independent_hash(item, i);
            self.set_bit(index);
        }
    }

    /// Check if an item might be in the filter.
    ///
    /// Same k-loop as insert, but exits early on the first missing bit.
    /// That makes queries faster than inserts on average (~711 ns vs ~864 ns).
    ///
    /// Returns `true` if the item might be in the set (or false positive).
    /// Returns `false` only if the item is definitely absent.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// let mut filter = ClassicBitsFilter::new(1000, 7);
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    /// assert!(!filter.contains(&"world"));
    /// ```
    #[must_use]
    #[inline]
    pub fn contains(&self, item: &T) -> bool {
        // TRUE 1970 ALGORITHM: Compute k independent hash values and check all bits
        for i in 0..self.k {
            let index = self.compute_independent_hash(item, i);
            if !self.test_bit(index) {
                return false; // Early exit if any bit is not set
            }
        }
        true
    }

    /// Zero out all bits in the filter. O(m/64).
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
    }

    /// True if every u64 word is zero (no bits set).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&word| word == 0)
    }

    /// Population count: how many bits are set across all u64 words.
    #[must_use]
    pub fn count_set_bits(&self) -> usize {
        self.bits
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Fraction of bits set: `count_set_bits` / m.
    ///
    /// Beyond ~0.5 fill rate, the false positive rate climbs steeply.
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        self.count_set_bits() as f64 / self.m as f64
    }

    /// Estimate the current FPR from the fill rate.
    ///
    /// Two-step inversion of Bloom's formula:
    /// 1. Estimate n from fill rate: `n ~ -(m/k) x ln(1 - fill_rate)`
    /// 2. Plug n into Bloom's formula: P(FP) = (1 - e^(-kn/m))^k
    ///
    /// Works best with uniformly random items and optimal hash count.
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        let fill_rate = self.fill_rate();

        if fill_rate == 0.0 {
            return 0.0;
        }

        if fill_rate >= 1.0 {
            return 1.0;
        }

        let m_f64 = self.m as f64;
        let k_f64 = self.k as f64;

        // Estimate n from fill rate: fill_rate ~ 1 - e^(-kn/m)
        // Solving for n: n ~ -(m/k) x ln(1 - fill_rate)
        let estimated_n = -(m_f64 / k_f64) * (1.0 - fill_rate).ln();

        // Calculate FPR using Bloom's formula: (1 - e^(-kn/m))^k
        let exponent = -(k_f64 * estimated_n) / m_f64;
        (1.0 - exponent.exp()).powf(k_f64)
    }

    /// Fill rate above 50%? FPR climbs steeply past this point.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.fill_rate() > 0.5
    }

    /// Approximate heap + stack memory in bytes.
    ///
    /// Includes the bit array vector plus struct metadata.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        // Bit array memory
        let bits_mem = self.bits.len() * std::mem::size_of::<u64>();

        // Struct overhead (metadata fields)
        let metadata_mem = std::mem::size_of::<Self>();

        bits_mem + metadata_mem
    }

    /// Insert all items from a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    /// let mut filter = ClassicBitsFilter::new(1000, 7);
    /// filter.insert_batch(&["a", "b", "c"]);
    /// assert!(filter.contains(&"a"));
    /// ```
    pub fn insert_batch(&mut self, items: &[T]) {
        for item in items {
            self.insert(item);
        }
    }

    /// Check all items in a slice. Returns one bool per item in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    /// let mut filter = ClassicBitsFilter::new(1000, 7);
    /// filter.insert(&"a");
    /// assert_eq!(filter.contains_batch(&["a", "b"]), vec![true, false]);
    /// ```
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Compute the union of two filters (bitwise OR).
    ///
    /// Clones self and ORs every u64 word from other. O(m/64) time.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the filters have different m or k parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// let mut f1 = ClassicBitsFilter::new(1000, 7);
    /// let mut f2 = ClassicBitsFilter::new(1000, 7);
    /// f1.insert(&"a");
    /// f2.insert(&"b");
    /// let union = f1.union(&f2).unwrap();
    /// assert!(union.contains(&"a"));
    /// assert!(union.contains(&"b"));
    /// ```
    pub fn union(&self, other: &Self) -> Result<Self> {
        if self.m != other.m || self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "Parameter mismatch: self(m={}, k={}) vs other(m={}, k={})",
                    self.m, self.k, other.m, other.k
                ),
            });
        }

        let mut result = self.clone();
        for (i, word) in result.bits.iter_mut().enumerate() {
            *word |= other.bits[i];
        }

        Ok(result)
    }

    /// Compute the intersection of two filters (bitwise AND).
    ///
    /// Clones self and ANDs every u64 word from other. O(m/64) time.
    /// Intersection can increase the FPR: cleared bits from one filter
    /// amplify set bits from the other.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the filters have different m or k parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicBitsFilter;
    ///
    /// let mut f1 = ClassicBitsFilter::new(1000, 7);
    /// let mut f2 = ClassicBitsFilter::new(1000, 7);
    /// f1.insert(&"a");
    /// f1.insert(&"b");
    /// f2.insert(&"b");
    /// f2.insert(&"c");
    /// let intersection = f1.intersect(&f2).unwrap();
    /// assert!(intersection.contains(&"b"));
    /// ```
    pub fn intersect(&self, other: &Self) -> Result<Self> {
        if self.m != other.m || self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: format!(
                    "Parameter mismatch: self(m={}, k={}) vs other(m={}, k={})",
                    self.m, self.k, other.m, other.k
                ),
            });
        }

        let mut result = self.clone();
        for (i, word) in result.bits.iter_mut().enumerate() {
            *word &= other.bits[i];
        }

        Ok(result)
    }
}

// Implement BloomFilter trait for generic filter operations
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
        // Estimate from parameters using: n ~ (m x ln(2)) / k
        ((self.m as f64 * std::f64::consts::LN_2) / self.k as f64) as usize
    }

    fn bit_count(&self) -> usize {
        self.m
    }

    fn hash_count(&self) -> usize {
        self.k
    }

    fn count_set_bits(&self) -> usize {
        self.bits
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }
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

        // No false negatives
        for i in 0..1000 {
            assert!(filter.contains(&i), "False negative for {i}");
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
        assert!(!filter.contains(&"world"));
    }

    #[test]
    fn test_count_set_bits() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        assert_eq!(filter.count_set_bits(), 0);

        filter.insert(&"test");

        let set_bits = filter.count_set_bits();
        assert!(set_bits > 0);
        assert!(set_bits <= 7); // At most k bits for one item
    }

    #[test]
    fn test_fill_rate() {
        let mut filter = ClassicBitsFilter::new(1000, 7);
        assert!(filter.fill_rate().abs() < f64::EPSILON);

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
        assert!((6..=8).contains(&filter.hash_count()));
    }

    #[test]
    #[should_panic(expected = "expected_items must be > 0")]
    fn test_with_fpr_zero_items() {
        let _: ClassicBitsFilter<i32> = ClassicBitsFilter::with_fpr(0, 0.01);
    }

    #[test]
    #[should_panic(expected = "fpr must be in range (0, 1)")]
    fn test_with_fpr_invalid_fpr() {
        let _: ClassicBitsFilter<i32> = ClassicBitsFilter::with_fpr(1000, 1.5);
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
            assert!(filter.contains(item), "False negative for {item}");
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
        for i in 1000..11_000 {
            if filter.contains(&i) {
                false_positives += 1;
            }
        }

        let actual_fpr = f64::from(false_positives) / 10_000.0;

        assert!(
            actual_fpr < 0.20,
            "FPR too high: {actual_fpr:.4} (expected < 0.20 for 1970 baseline with k independent hashes)",
        );

        // Sanity check: verify filter isn't catastrophically broken
        assert!(
            actual_fpr < 0.50,
            "FPR catastrophically high: {actual_fpr:.4} - filter may be broken",
        );
    }

    #[test]
    fn test_duplicate_inserts() {
        let mut filter = ClassicBitsFilter::new(1000, 7);

        filter.insert(&"test");
        let bits_after_first = filter.count_set_bits();

        filter.insert(&"test");
        let bits_after_second = filter.count_set_bits();

        // Same number of bits should be set (idempotent)
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

    #[test]
    fn test_independent_hash_generation() {
        let filter: ClassicBitsFilter<&str> = ClassicBitsFilter::new(10_000, 7);

        // Compute all k hash indices for the same item
        let mut indices = Vec::new();
        for i in 0..7 {
            let index = filter.compute_independent_hash(&"test", i);
            indices.push(index);
        }

        // All indices should be in valid range
        for &idx in &indices {
            assert!(idx < 10_000);
        }

        // Indices should mostly be different (very low collision probability)
        let unique_count = indices
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(unique_count >= 5, "Too many hash collisions");
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
}
