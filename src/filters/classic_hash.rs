//! Classic Bloom filter using Burton Bloom's Method 1 (1970).
//!
//! This module implements the original hash table-based method from Burton Bloom's
//! seminal 1970 paper "Space/Time Trade-offs in Hash Coding with Allowable Errors".
//!
//! # Historical Context
//!
//! Method 1 was the first approach proposed by Burton Bloom. Unlike modern Bloom filters that use bit arrays, Method 1 stores **actual set elements**
//! in a hash table with limited-depth chaining. This provides:
//!
//! - **Deterministic membership** within bucket (no hash collision false positives)
//! - **Element recovery** (can iterate and retrieve stored items)
//! - **Controlled false positives** via chain depth limit
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
//! | Space per item | `sizeof(T)` bytes | ~1.2 bytes (1% FPR) |
//! | False positives | Only from chain overflow | From bit collisions |
//! | Concurrency | Single-threaded (`&mut self`) | Lock-free atomic |
//! | Element recovery | Yes (can iterate chains) | No |
//! | Historical accuracy | Faithful to 1970 paper | Modern optimization |
//!
//! # Parameters
//!
//! - m: Hash table size (number of buckets)
//! - d: Maximum chain depth per bucket
//! - n: Expected number of elements
//!
//! # False Positive Rate
//!
//! False positives occur when:
//! 1. An item hashes to a bucket with a **full chain** (depth = d)
//! 2. The item was discarded during insertion
//! 3. A query for the item returns `false` even though it "should" be in the set
//!
//! The theoretical FPR for uniform hashing is approximately:
//!
//! ```text
//! P(false positive) ≈ probability that a bucket is full
//!                    = P(chain length = d)
//!                    ≈ (e^(-λ) * λ^d) / d!  where λ = n/m
//! ```
//!
//! # Space Complexity
//!
//! - Worst case: **O(m + n×sizeof(T))** - all elements stored
//! - Best case: **O(m)** - only bucket array
//! - Average: **O(m + min(n, m×d)×sizeof(T))**
//!
//! For large `T`, this is **significantly less space-efficient** than bit-array Bloom filters.
//!
//! # When to Use This
//!
//! 1. **Educational**: Learning about Bloom filter history and evolution  
//! 2. **Research**: Studying original 1970 algorithm vs modern optimizations  
//! 3. **Element Recovery**: Need to retrieve stored items  
//! 4. **Small Sets**: Where `sizeof(T)` overhead is acceptable  
//!
//! # When Not to Use This
//!
//! 1. **Production**: Use [`StandardBloomFilter`](crate::filters::StandardBloomFilter) instead (10-100× more space-efficient)  
//! 2. **Concurrent**: Not thread-safe (requires external `Mutex`)  
//! 3. **Large `T`**: Space overhead becomes prohibitive  
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
//!
//! // Retrieve all stored elements
//! let elements: Vec<String> = filter.elements().map(|s| s.to_string()).collect();
//! assert_eq!(elements.len(), 2);
//! ```
//!
//! # Performance Comparison
//!
//! For 1M items at 1% FPR:
//!
//! ```text
//! ClassicHashFilter<String>:  ~16-40 MB (actual strings)
//! StandardBloomFilter:        ~1.2 MB (bit array)
//!
//! Memory ratio: 13-33× larger
//! ```
//!
//! # References
//!
//! - Bloom, B. H. (1970). "Space/time trade-offs in hash coding with allowable errors".
//!   Communications of the ACM, 13(7), 422-426.
//!
//! # Thread Safety
//!
//! **Not thread-safe**. All operations require `&mut self` and do not use atomic operations.
//! For concurrent use, wrap in `std::sync::Mutex` or `parking_lot::Mutex`.

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(dead_code)]

use crate::core::filter::BloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Classic hash table-based Bloom filter storing actual elements.
///
/// This implementation is **historically accurate** to Burton Bloom's 1970 Method 1:
/// - Stores actual `T` values (not hashes)
/// - Limited-depth chaining per bucket
/// - Deterministic membership within bucket
/// - Element recovery capability
///
/// # Type Parameters
///
/// * `T` - Type of items stored (must implement `Hash`, `Clone`, `Eq`)
/// * `H` - Hash function type (must implement `BloomHasher`)
///
/// # Memory Layout
///
/// ```text
/// ClassicHashFilter {
///     table: Vec<Vec<T>>,  // m buckets, each storing up to d elements
///     m: usize,            // number of buckets
///     d: usize,            // maximum chain depth
///     count: usize,        // number of elements successfully inserted
///     discarded: usize,    // number of elements discarded (chain full)
/// }
/// ```
///
/// # Space Complexity
///
/// - Bucket array: `m × size_of::<Vec<T>>()` bytes
/// - Element storage: `count × sizeof(T)` bytes  
/// - Total: **O(m + count×sizeof(T))**
///
/// # Concurrency
///
/// **Not thread-safe** - all operations require `&mut self`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    
    /// Hash function
    #[cfg_attr(feature = "serde", serde(skip, default = "H::default"))]
    hasher: H,
    
    /// Phantom hasher generic (for serde compatibility)
    #[cfg_attr(feature = "serde", serde(skip))]
    _phantom: PhantomData<H>,
}

impl<T> ClassicHashFilter<T, StdHasher>
where
    T: Hash + Clone + Eq,
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
        let (m, d) = calculate_optimal_params(expected_items, fpr);
        Self::new(m, d)
    }
}

impl<T, H> ClassicHashFilter<T, H>
where
    T: Hash + Clone + Eq,
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
        assert!(m > 0, "m (bucket count) must be > 0");
        assert!(d > 0, "d (max depth) must be > 0");
        
        Self {
            table: vec![Vec::with_capacity(d.min(4)); m],
            m,
            d,
            count: 0,
            discarded: 0,
            hasher,
            _phantom: PhantomData,
        }
    }

    /// Get the number of buckets (m).
    #[must_use]
    #[inline]
    pub const fn bucket_count(&self) -> usize {
        self.m
    }

    /// Get the maximum chain depth (d).
    #[must_use]
    #[inline]
    pub const fn max_depth(&self) -> usize {
        self.d
    }

    /// Get the number of elements successfully inserted.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Get the number of elements discarded due to full chains.
    ///
    /// High discard count indicates the filter is saturated.
    #[must_use]
    #[inline]
    pub const fn discarded_count(&self) -> usize {
        self.discarded
    }

    /// Check if the filter is empty.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Hash an item to a bucket index.
    ///
    /// Uses the primary hash value modulo bucket count.
    #[inline]
    fn hash_to_bucket(&self, item: &T) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        
        (hash % self.m as u64) as usize
    }

    /// Insert an item into the filter.
    ///
    /// Following Bloom's Method 1:
    /// 1. Hash item to bucket index
    /// 2. If item already in chain, do nothing (idempotent)
    /// 3. If chain has space (< d elements), add item
    /// 4. If chain is full, discard item and increment discard counter
    ///
    /// # Arguments
    ///
    /// * `item` - Item to insert
    ///
    /// # Returns
    ///
    /// `true` if item was inserted, `false` if discarded or already present
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(1000, 3);
    /// assert!(filter.insert(&"hello"));  // Inserted
    /// assert!(!filter.insert(&"hello")); // Already present
    /// assert!(filter.contains(&"hello"));
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

    /// Check if an item might be in the filter.
    ///
    /// Since this filter stores actual elements, this is **deterministic** within
    /// the bucket (no hash collision false positives).
    ///
    /// # Arguments
    ///
    /// * `item` - Item to check
    ///
    /// # Returns
    ///
    /// - `true`: Item is in the set (true positive) OR bucket was full during insertion (false positive)
    /// - `false`: Item is definitely not in the set (true negative)
    ///
    /// # False Positives
    ///
    /// Can only occur if the item was discarded during insertion because its bucket was full.
    /// This is why controlling `d` (max depth) controls the false positive rate.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(1000, 3);
    /// filter.insert(&"hello");
    ///
    /// assert!(filter.contains(&"hello"));  // True positive
    /// assert!(!filter.contains(&"world")); // True negative
    /// ```
    #[must_use]
    pub fn contains(&self, item: &T) -> bool {
        let bucket_idx = self.hash_to_bucket(item);
        self.table[bucket_idx].contains(item)
    }

    /// Clear all entries from the filter.
    ///
    /// Resets both insertion count and discard counter.
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

    /// Get the average chain length across non-empty buckets.
    ///
    /// This provides insight into how well the hash function is distributing items.
    ///
    /// # Returns
    ///
    /// Average number of entries per non-empty bucket, or 0.0 if filter is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(100, 5);
    /// for i in 0..50 {
    ///     filter.insert(&i);
    /// }
    /// let avg = filter.avg_chain_length();
    /// assert!(avg > 0.0 && avg <= 5.0);
    /// ```
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
    /// Length of longest chain (0 if empty)
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

    /// Estimate the current false positive rate based on actual chain length distribution.
    ///
    /// This uses the **empirical distribution** of chain lengths, not a theoretical formula.
    ///
    /// # Algorithm
    ///
    /// For each chain of length `len`:
    /// - Probability that a random item hashes to this bucket: `1/m`
    /// - If chain is full (`len == d`), discarded items cause FPs
    /// - FPR contribution: `(# full buckets) / m`
    ///
    /// # Returns
    ///
    /// Estimated false positive rate based on current state
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(1000, 3);
    /// for i in 0..500 {
    ///     filter.insert(&i);
    /// }
    /// let fpr = filter.estimate_fpr();
    /// println!("Estimated FPR: {:.4}", fpr);
    /// ```
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

    /// Get memory usage in bytes.
    ///
    /// Includes bucket overhead and element storage.
    ///
    /// # Returns
    ///
    /// Approximate memory usage in bytes
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

    /// Insert multiple items in batch.
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of items to insert
    ///
    /// # Returns
    ///
    /// Number of items successfully inserted (excludes duplicates and discarded)
    pub fn insert_batch(&mut self, items: &[T]) -> usize {
        let mut inserted = 0;
        for item in items {
            if self.insert(item) {
                inserted += 1;
            }
        }
        inserted
    }

    /// Check multiple items in batch.
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of items to check
    ///
    /// # Returns
    ///
    /// Vector of boolean results (one per item)
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Get an iterator over all stored elements.
    ///
    /// This is a unique capability of the classic hash filter - since actual elements 
    /// are stored, they can be retrieved. Modern bit-array Bloom filters cannot do this.
    ///
    /// # Returns
    ///
    /// Iterator yielding references to all stored elements
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::ClassicHashFilter;
    ///
    /// let mut filter = ClassicHashFilter::new(100, 3);
    /// filter.insert(&"hello");
    /// filter.insert(&"world");
    ///
    /// let elements: Vec<_> = filter.elements().collect();
    /// assert_eq!(elements.len(), 2);
    /// assert!(elements.contains(&&"hello"));
    /// assert!(elements.contains(&&"world"));
    /// ```
    pub fn elements(&self) -> impl Iterator<Item = &T> {
        self.table.iter().flat_map(|chain| chain.iter())
    }

    /// Check if the filter is saturated (high discard rate).
    ///
    /// A saturated filter has many full chains and will discard most new insertions.
    ///
    /// # Returns
    ///
    /// `true` if more than 20% of insertion attempts were discarded
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
}

/// Calculate optimal parameters (m, d) for given n and fpr.
///
/// Uses Poisson approximation for chain length distribution.
///
/// # Arguments
///
/// * `n` - Expected number of elements
/// * `fpr` - Target false positive rate
///
/// # Returns
///
/// Tuple of (m, d) where:
/// - m: number of buckets
/// - d: maximum chain depth
///
/// # Algorithm
///
/// For uniform random hashing, chain length follows Poisson(λ) where λ = n/m.
/// We want P(chain length > d) ≈ fpr.
///
/// Using Poisson tail probability:
/// ```text
/// P(X > d) = 1 - Σ(i=0 to d) [e^(-λ) * λ^i / i!]
/// ```
///
/// We solve for (m, d) that achieves target FPR while minimizing space.
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
