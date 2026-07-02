//! Classic Bloom filter using Burton Bloom's Method 1 (1970).
//!
//! Companion to [`ClassicBitsFilter`](crate::filters::ClassicBitsFilter) (Method 2).
//! Method 1 stores actual elements in a hash table with limited-depth chaining;
//! Method 2 uses a bit array with k independent hash functions.
//!
//! This is the original hash table approach: it stores real `T` values in buckets
//! with limited-depth chains. You get deterministic membership, element recovery,
//! and the ability to iterate stored items. The tradeoff is memory: each element
//! costs `sizeof(T)` bytes instead of ~1.2 bytes per element for a modern Bloom filter.
//!
//! If you just need membership checks, use
//! [`StandardBloomFilter`](crate::filters::StandardBloomFilter) instead (stores bits
//! rather than full elements, making it substantially more space-efficient). Use this
//! filter when you need to retrieve stored items or study the original 1970 algorithm.
//!
//! # Historical Context
//!
//! Method 1 was the first approach proposed by Burton Bloom. Unlike bit-array filters,
//! it stores **actual set elements** in a hash table with limited-depth chaining.
//!
//! | Aspect | Method 1 (1970) | Modern |
//! |--------|-----------------|--------|
//! | Data structure | Hash table with chains | Bit array |
//! | False positives | Only from chain overflow | From bit collisions |
//! | Element recovery | Yes (iterate chains) | No |
//!
//! # Sizing Guide
//!
//! Pick `m ≈ n` and `d = 3` as a starting point. If [`ClassicHashFilter::discarded_count`] climbs
//! above 20% of total insertions, double `m`. If memory is tight, lower `d` and
//! accept a higher FPR.
//!
//! # Example
//!
//! ```
//! use bloomcraft::filters::ClassicHashFilter;
//!
//! let mut filter = ClassicHashFilter::new(1000, 3);
//! filter.insert(&"hello");
//! assert!(filter.contains(&"hello"));
//!
//! // Retrieve stored elements (unique to Method 1)
//! assert!(filter.elements().any(|&x| x == "hello"));
//! ```
//!
//! # References
//!
//! - Bloom, B. H. (1970). "Space/time trade-offs in hash coding with allowable errors".
//!   Communications of the ACM, 13(7), 422-426.
//!
//! # Thread Safety
//!
//! **Not thread-safe**. All operations need `&mut self`. Wrap in `Mutex` for concurrent use.
//!

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]

use crate::core::filter::BloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;

/// Classic hash table-based Bloom filter that stores actual elements.
///
/// A faithful implementation of Burton Bloom's 1970 Method 1. Stores real `T`
/// values in a hash table with limited-depth chains. Unlike bit-array filters,
/// this one lets you retrieve your data via [`elements`](Self::elements).
///
/// # Type Parameters
///
/// * `T` - Item type. Must implement `Hash`, `Clone`, and `Eq`.
/// * `H` - Hash function type. Must implement [`BloomHasher`](crate::hash::BloomHasher). Defaults to [`StdHasher`](crate::hash::StdHasher).
///
/// Stores actual `T` values in `m` buckets of depth `d` each.
///
/// # Thread Safety
///
/// **Not thread-safe**. All operations need `&mut self`. Wrap in `Mutex` for concurrent use.
#[derive(Debug, Clone)]
pub struct ClassicHashFilter<T, H = StdHasher>
where
    T: Hash + Clone + Eq,
    H: BloomHasher + Clone,
{
    /// Hash table: m buckets, each containing up to d elements
    table: Vec<Vec<T>>,
    
    /// Number of buckets (m)
    m: usize,
    
    /// Maximum chain depth per bucket (d)
    d: usize,
    
    /// Number of elements successfully inserted
    count: usize,
    
    /// Number of elements discarded due to full chains
    discarded: usize,
    
    /// Hash function (defaults to [`StdHasher`](crate::hash::StdHasher))
    hasher: H,
}

impl<T> ClassicHashFilter<T, StdHasher>
where
    T: Hash + Clone + Eq,
{
    /// Create a new classic hash filter with the default hasher.
    ///
    /// `m` is the number of buckets, `d` is the max chain depth per bucket.
    /// A small `m` means more collisions and higher FPR. A small `d` reduces
    /// FPR but increases discards.
    ///
    /// # Panics
    ///
    /// Panics if `m` or `d` is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    /// let filter: ClassicHashFilter<&str> = ClassicHashFilter::new(1000, 3);
    /// ```
    #[must_use]
    pub fn new(m: usize, d: usize) -> Self {
        Self::with_hasher(m, d, StdHasher::new())
    }

    /// Create a filter sized by expected items and a target false positive rate.
    ///
    /// Uses a heuristic with fixed d=3. This gives roughly the requested FPR,
    /// not a hard guarantee. For precise control, use [`new`](Self::new).
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is 0 or `fpr` is not in (0, 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    /// let filter: ClassicHashFilter<i32> = ClassicHashFilter::with_fpr(10_000, 0.01);
    /// ```
    #[must_use]
    pub fn with_fpr(expected_items: usize, fpr: f64) -> Self {
        let (m, d) = calculate_optimal_params(expected_items, fpr);
        Self::new(m, d)
    }
}

impl<T, H> ClassicHashFilter<T, H>
where
    T: Hash + Clone + Eq,
    H: BloomHasher + Clone,
{
    /// Create a filter with `m` buckets, `d` max depth, and a custom hasher.
    ///
    /// Same as [`new`](Self::new) but lets you supply your own hash function.
    ///
    /// # Panics
    ///
    /// Panics if `m` or `d` is 0.
    #[must_use]
    pub fn with_hasher(m: usize, d: usize, hasher: H) -> Self {
        assert!(m > 0, "m (bucket count) must be > 0");
        assert!(d > 0, "d (max depth) must be > 0");
        
        Self {
            table: vec![Vec::with_capacity(d.min(4)); m],
            m,
            d,
            count: 0,
            discarded: 0,
            hasher,
        }
    }

    /// Returns the number of buckets allocated at construction.
    #[must_use]
    #[inline]
    pub const fn bucket_count(&self) -> usize {
        self.m
    }

    /// Returns the maximum chain depth per bucket set at construction.
    #[must_use]
    #[inline]
    pub const fn max_depth(&self) -> usize {
        self.d
    }

    /// Returns the number of items currently stored in the filter.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns the number of items discarded because their bucket chain was full.
    ///
    /// A high discard count means the filter is saturated.
    #[must_use]
    #[inline]
    pub const fn discarded_count(&self) -> usize {
        self.discarded
    }

    /// Returns `true` if no items have been stored yet.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    // Hash item to bucket index via self.hasher modulo bucket count.
    #[inline]
    fn hash_to_bucket(&self, item: &T) -> usize {
        let (h, _) = self.hasher.hash_item(item);
        (h as usize) % self.m
    }

    /// Insert an item into the filter.
    ///
    /// Returns `true` if the item was stored, `false` if it was already present
    /// or discarded because the bucket chain was full. Discards increment
    /// [`discarded_count`](Self::discarded_count).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(100, 3);
    /// assert!(filter.insert(&"hello"));   // New item, stored
    /// assert!(!filter.insert(&"hello"));  // Already present, idempotent
    /// ```
    pub fn insert(&mut self, item: &T) -> bool {
        let bucket_idx = self.hash_to_bucket(item);
        let chain = &mut self.table[bucket_idx];

        // Check if already present (idempotent)
        if chain.contains(item) {
            return false;
        }

        // Add to chain if there's space
        if chain.len() < self.d {
            chain.push(item.clone());
            self.count += 1;
            true
        } else {
            // Chain is full - discard as per original algorithm
            self.discarded += 1;
            false
        }
    }

    /// Check whether an item might be in the filter.
    ///
    /// Unlike bit-array filters, this is **deterministic** within a bucket:
    /// if we stored it, we find it with an exact match. False positives only
    /// happen when an item was discarded during insert (bucket was full).
    ///
    /// Returns `true` if the item is stored (or was discarded, false positive).
    /// Returns `false` only when the item is definitely absent.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(1000, 3);
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));  // True positive
    /// assert!(!filter.contains(&"world")); // True negative
    /// ```
    #[must_use]
    pub fn contains(&self, item: &T) -> bool {
        let bucket_idx = self.hash_to_bucket(item);
        let chain = &self.table[bucket_idx];
        // Return true if the item is present, OR if the bucket is full (we can't
        // rule out that this item was discarded during insert). This ensures no
        // false negatives: a discarded item might still have been "inserted".
        chain.contains(item) || chain.len() == self.d
    }

    /// Remove all items and reset the discard counter.
    ///
    /// After this, the filter behaves as if no items were ever inserted.
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
    /// assert!(!filter.contains(&"hello"));
    /// ```
    pub fn clear(&mut self) {
        for chain in &mut self.table {
            chain.clear();
        }
        self.count = 0;
        self.discarded = 0;
    }

    /// Average number of items in non-empty buckets.
    ///
    /// If consistently near `d`, the filter is filling up and discards will increase.
    #[must_use]
    pub fn avg_chain_length(&self) -> f64 {
        let non_empty = self.table.iter().filter(|c| !c.is_empty()).count();
        if non_empty == 0 {
            return 0.0;
        }
        self.count as f64 / non_empty as f64
    }

    /// Length of the longest chain currently in the table.
    ///
    /// Returns 0 if the filter is empty. At most `d` (the configured max depth).
    #[must_use]
    pub fn max_chain_length(&self) -> usize {
        self.table.iter().map(Vec::len).max().unwrap_or(0)
    }

    /// Fraction of buckets that hold at least one item.
    ///
    /// A load factor near 1.0 paired with a high discard count suggests the filter
    /// is undersized.
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        let non_empty = self.table.iter().filter(|c| !c.is_empty()).count();
        non_empty as f64 / self.m as f64
    }

    /// Estimate the current FPR from the chain distribution.
    ///
    /// Counts full buckets (chain length == d) and divides by m.
    /// Returns 0.0 if the filter is empty.
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        // Count buckets at maximum depth (full chains)
        let full_buckets = self.table.iter().filter(|chain| chain.len() == self.d).count();

        // FPR ≈ probability that a query item hashes to a full bucket
        full_buckets as f64 / self.m as f64
    }

    /// Approximate memory footprint in bytes.
    ///
    /// Accounts for the bucket array, element storage, and struct metadata.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let bucket_overhead = self.m * std::mem::size_of::<Vec<T>>();
        let element_storage: usize = self
            .table
            .iter()
            .map(|chain| chain.capacity() * std::mem::size_of::<T>())
            .sum();
        let metadata = std::mem::size_of::<Self>();
        
        bucket_overhead + element_storage + metadata
    }

    /// Insert every item from a slice. Returns how many were actually stored.
    ///
    /// A return value lower than `items.len()` means some items were duplicates
    /// or discarded due to full buckets.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    /// let mut filter = ClassicHashFilter::new(100, 3);
    /// let stored = filter.insert_batch(&["a", "b", "c"]);
    /// assert_eq!(stored, 3);
    /// assert!(filter.contains(&"a"));
    /// ```
    pub fn insert_batch(&mut self, items: &[T]) -> usize {
        let mut inserted = 0;
        for item in items {
            if self.insert(item) {
                inserted += 1;
            }
        }
        inserted
    }

    /// Check every item from a slice. Returns a `Vec<bool>` in the same order.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    /// let mut filter = ClassicHashFilter::new(100, 3);
    /// filter.insert(&"a");
    /// assert_eq!(filter.contains_batch(&["a", "b"]), vec![true, false]);
    /// ```
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Iterate over every stored element in the filter.
    ///
    /// This is the headline feature of Method 1: since we store actual `T` values,
    /// we can hand them back. A [`StandardBloomFilter`](crate::filters::StandardBloomFilter)
    /// cannot do this.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(100, 3);
    /// filter.insert(&"hello");
    /// filter.insert(&"world");
    /// assert!(filter.elements().any(|&x| x == "hello"));
    /// assert!(filter.elements().any(|&x| x == "world"));
    /// ```
    pub fn elements(&self) -> impl Iterator<Item = &T> {
        self.table.iter().flat_map(|chain| chain.iter())
    }

    /// Returns true if more than 20% of insert attempts have been discarded.
    ///
    /// A saturated filter is past its useful capacity: most new items are discarded
    /// and the FPR is high.
    #[must_use]
    pub fn is_saturated(&self) -> bool {
        let total_attempts = self.count + self.discarded;
        if total_attempts == 0 {
            return false;
        }
        (self.discarded as f64 / total_attempts as f64) > 0.2
    }
}

impl<T, H> BloomFilter<T> for ClassicHashFilter<T, H>
where
    T: Hash + Clone + Eq + Send + Sync,
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
        // Return logical "bit count" - total possible storage slots in bytes
        self.m * self.d * std::mem::size_of::<T>() * 8
    }

    fn hash_count(&self) -> usize {
        // Classic hash filter uses single hash function
        1
    }

    fn count_set_bits(&self) -> usize {
        // No bit array exists. Express occupied capacity in the same unit
        // as bit_count() (slots × sizeof(T) × 8) so fill_rate() returns
        // a valid occupancy fraction: count / (m × d).
        self.count * std::mem::size_of::<T>() * 8
    }
}

/// Heuristic for picking (m, d) from expected item count and a target FPR.
///
/// This is a heuristic, not an exact Poisson solver. Fixes `d = 3` (Bloom's
/// default) and estimates `m` from there. The result is approximate. For
/// precise control, size `m` and `d` manually via [`ClassicHashFilter::new`].
///
/// # Panics
///
/// Panics if `n` is 0 or `fpr` is not in (0, 1).
fn calculate_optimal_params(n: usize, fpr: f64) -> (usize, usize) {
    assert!(fpr > 0.0 && fpr < 1.0, "fpr must be in (0, 1)");
    assert!(n > 0, "n must be > 0");

    // For classic hash filters, we balance:
    // - Larger m (more buckets) → fewer collisions, shorter chains
    // - Smaller d (lower depth limit) → lower FPR but more discards
    
    // Start with d = 3 (reasonable default from Bloom's paper)
    let d = 3;
    
    // For Poisson(λ), P(X ≥ d) ≈ fpr when λ is small
    // Using rough approximation: λ ≈ -ln(1 - fpr)
    // Then m = n / λ
    
    let lambda = if fpr < 0.01 {
        // For small FPR, use Poisson approximation
        -fpr.ln() / 2.0
    } else {
        // For larger FPR, use more conservative estimate
        (n as f64 * 0.1 / fpr).sqrt().max(1.0)
    };
    
    let m = ((n as f64) / lambda).ceil() as usize;
    
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
    #[should_panic(expected = "m (bucket count) must be > 0")]
    fn test_new_zero_buckets() {
        let _: ClassicHashFilter<&str> = ClassicHashFilter::new(0, 3);
    }

    #[test]
    #[should_panic(expected = "d (max depth) must be > 0")]
    fn test_new_zero_depth() {
        let _: ClassicHashFilter<&str> = ClassicHashFilter::new(1000, 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut filter = ClassicHashFilter::new(100, 3);
        assert!(filter.insert(&"hello"));
        assert!(filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
    }

    #[test]
    fn test_duplicate_insert() {
        let mut filter = ClassicHashFilter::new(100, 3);
        assert!(filter.insert(&"test"));
        assert!(!filter.insert(&"test")); // Should return false (already present)
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_chain_overflow() {
        let mut filter = ClassicHashFilter::new(1, 2); // 1 bucket, depth 2
        
        // First 2 items should insert successfully
        filter.insert(&0);
        filter.insert(&1);
        
        // Depending on hash collisions, may have 1-2 items
        let _initial_count = filter.len();
        let initial_discarded = filter.discarded_count();
        
        // Insert many more items - most will hash to same bucket and be discarded
        for i in 2..10 {
            filter.insert(&i);
        }
        
        // Should have at most 2 items total (depth limit)
        assert!(filter.len() <= 2);
        assert!(filter.discarded_count() >= initial_discarded);
    }

    #[test]
    fn test_overflow_contains_no_false_negative() {
        // Single bucket, depth 2 ensures all items collide
        let mut filter = ClassicHashFilter::new(1, 2);

        // Fill the bucket
        assert!(filter.insert(&1));
        assert!(filter.insert(&2));
        assert_eq!(filter.len(), 2);

        // This insert is discarded (bucket full)
        assert!(!filter.insert(&3));
        assert_eq!(filter.discarded_count(), 1);

        // contains must return true for the discarded item: we cannot
        // rule out that it was "inserted" into the now-full bucket
        assert!(filter.contains(&3));
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
        assert_eq!(filter.discarded_count(), 0);
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
        assert!(fpr >= 0.0 && fpr < 1.0);
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
        let inserted = filter.insert_batch(&items);
        
        assert_eq!(inserted, 4);
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
    fn test_elements_iterator() {
        let mut filter = ClassicHashFilter::new(100, 3);
        filter.insert(&"hello");
        filter.insert(&"world");
        filter.insert(&"rust");
        
        let elements: Vec<_> = filter.elements().collect();
        assert_eq!(elements.len(), 3);
        assert!(elements.contains(&&"hello"));
        assert!(elements.contains(&&"world"));
        assert!(elements.contains(&&"rust"));
    }

    #[test]
    fn test_is_saturated() {
        let mut filter = ClassicHashFilter::new(2, 1); // Very small filter
        
        assert!(!filter.is_saturated());
        
        // Fill it up
        for i in 0..20 {
            filter.insert(&i);
        }
        
        // Should be saturated due to high discard rate
        assert!(filter.is_saturated());
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
        let (m, d) = calculate_optimal_params(1000, 0.01);
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

    #[test]
    fn test_discarded_tracking() {
        let mut filter = ClassicHashFilter::new(1, 1); // Single bucket, depth 1
        
        filter.insert(&1);
        assert_eq!(filter.discarded_count(), 0);
        
        // These should be discarded if they hash to the same bucket
        for i in 2..10 {
            filter.insert(&i);
        }
        
        // At least some should be discarded
        assert!(filter.len() + filter.discarded_count() > 1);
    }
}
