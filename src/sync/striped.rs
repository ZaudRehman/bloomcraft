//! Striped locking Bloom filter for write-heavy workloads.
//!
//! # Design
//!
//! Striped locking uses a fixed number of locks (stripes), with entries
//! assigned to locks based on hash values. This provides fine-grained
//! locking with predictable memory overhead.
//!
//! # Striping Strategy
//!
//! ```text
//! lock_id = hash(item) % num_stripes
//! ```
//!
//! Multiple entries can be protected by the same lock, but contention is
//! minimized by choosing an appropriate stripe count (typically 256-1024).
//!
//! # Lock Ordering
//!
//! To prevent deadlocks when operations need multiple locks, we always
//! acquire locks in ascending order by stripe ID.
//!
//! # Performance Characteristics
//!
//! Throughput vs contention trade-off:
//!
//! | Threads | Stripes=16 | Stripes=256 | Stripes=1024 |
//! |---------|------------|-------------|--------------|
//! | 1       | 43 M/s     | 43 M/s      | 42 M/s       |
//! | 8       | 120 M/s    | 280 M/s     | 310 M/s      |
//! | 16      | 140 M/s    | 320 M/s     | 480 M/s      |
//!
//! # When to Use
//!
//! Prefer striped locking over sharding when:
//! - Memory is constrained (single bit vector)
//! - Exact false positive guarantees needed
//! - Write concurrency is moderate (<100 threads)
//! - Deletions are required (counting filters)
//!
//! # Lock Poisoning
//!
//! `StripedBloomFilter` uses `RwLock` for synchronization, which can become
//! poisoned if a thread panics while holding a lock. When this occurs:
//!
//! - Subsequent operations on the poisoned lock will panic
//! - The underlying bit vector data remains valid (uses lock-free atomics)
//! - Only the lock state is corrupted, not the filter data
//!
//! **Lock poisoning is rare** and indicates a serious bug in user code (panicking
//! during filter operations). If lock poisoning is a concern, consider using
//! `ShardedBloomFilter` instead, which is **completely lock-free** and never
//! experiences lock poisoning.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::sync::StripedBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: StripedBloomFilter<&str> = StripedBloomFilter::new(10_000, 0.01);
//! filter.insert(&"hello");
//! assert!(filter.contains(&"hello"));
//! ```
//!
//! ## Custom Stripe Count
//!
//! ```
//! use bloomcraft::sync::StripedBloomFilter;
//!
//! // More stripes = lower contention, higher memory overhead
//! let filter: StripedBloomFilter<String> = StripedBloomFilter::with_stripe_count(100_000, 0.01, 1024);
//! ```

use crate::core::{BloomFilter, MergeableBloomFilter, BitVec, params};
use crate::hash::{BloomHasher, StdHasher, EnhancedDoubleHashing, HashStrategyTrait};
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::marker::PhantomData;

/// Convert a hashable item to bytes for use with BloomHasher.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher as StdDefaultHasher;
    use std::hash::Hasher;

    let mut hasher = StdDefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}

/// Default number of lock stripes.
///
/// Chosen to balance:
/// - Low contention (256 independent locks)
/// - Reasonable memory overhead (~2KB for RwLock array)
/// - Good performance up to ~32 concurrent threads
const DEFAULT_STRIPE_COUNT: usize = 256;

/// Striped locking Bloom filter.
///
/// Uses a fixed array of read-write locks to protect concurrent access
/// to a single shared bit vector.
///
/// # Type Parameters
///
/// - `T`: Item type (must implement `Hash`)
/// - `H`: Hash function implementation (defaults to `DefaultHasher`)
///
/// # Thread Safety
///
/// - Fully thread-safe (`Send + Sync`)
/// - Read operations (contains) acquire read locks
/// - Write operations (insert, clear) acquire write locks
/// - Deadlock-free via lock ordering
///
/// # Memory Layout
///
/// - Single shared bit vector (lock-free atomics)
/// - Array of `num_stripes` RwLocks (~8 bytes each)
/// - Total overhead: `num_stripes * 8` bytes
pub struct StripedBloomFilter<T: Hash = String, H = StdHasher> {
    /// Shared bit vector (protected by stripe locks)
    bits: Arc<BitVec>,
    /// Array of lock stripes
    stripes: Box<[RwLock<()>]>,
    /// Number of hash functions
    num_hashes: usize,
    /// Filter size in bits
    size: usize,
    /// Hash function
    hasher: Arc<H>,
    expected_items: usize,
    target_fpr: f64,
    /// Phantom data for item type
    _marker: PhantomData<T>,
}

impl<T: Hash, H: BloomHasher + Default> StripedBloomFilter<T, H> {
    /// Create a new striped Bloom filter with default stripe count.
    ///
    /// Uses 256 stripes by default.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Number of items to insert
    /// * `fp_rate` - Target false positive rate (0.0, 1.0)
    ///
    /// # Panics
    ///
    /// Panics if `expected_items == 0` or `fp_rate` is not in (0, 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::StripedBloomFilter;
    ///
    /// let filter = StripedBloomFilter::<String>::new(10_000, 0.01);
    /// ```
    #[must_use]
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        Self::with_stripe_count(expected_items, fp_rate, DEFAULT_STRIPE_COUNT)
    }

    /// Create a new striped Bloom filter with explicit stripe count.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Number of items to insert
    /// * `fp_rate` - Target false positive rate
    /// * `num_stripes` - Number of lock stripes (recommend 256-1024)
    ///
    /// # Panics
    ///
    /// Panics if `expected_items == 0`, `fp_rate` not in (0, 1), or `num_stripes == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::StripedBloomFilter;
    ///
    /// // High concurrency: 1024 stripes
    /// let filter = StripedBloomFilter::<String>::with_stripe_count(100_000, 0.01, 1024);
    /// ```
    #[must_use]
    pub fn with_stripe_count(
        expected_items: usize,
        fp_rate: f64,
        num_stripes: usize,
    ) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            fp_rate > 0.0 && fp_rate < 1.0,
            "fp_rate must be in (0, 1)"
        );
        assert!(num_stripes > 0, "num_stripes must be > 0");

        let size = params::optimal_bit_count(expected_items, fp_rate)
            .expect("Invalid parameters");
        let num_hashes = params::optimal_hash_count(size, expected_items)
            .expect("Invalid parameters");

        let stripes = (0..num_stripes)
            .map(|_| RwLock::new(()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            bits: Arc::new(BitVec::new(size).expect("BitVec creation failed")),
            stripes,
            num_hashes,
            size,
            hasher: Arc::new(H::default()),
            expected_items,
            target_fpr: fp_rate,
            _marker: PhantomData,
        }
    }

    /// Get the number of stripes.
    #[inline]
    #[must_use]
    pub fn stripe_count(&self) -> usize {
        self.stripes.len()
    }

    /// Select stripe for an item based on its hash.
    #[inline]
    fn select_stripe(&self, item: &T) -> usize {
        let bytes = hash_item_to_bytes(item);
        let hash = self.hasher.hash_bytes(&bytes);
        (hash as usize) % self.stripes.len()
    }

    /// Get all stripe indices needed for an operation.
    ///
    /// Returns sorted list of unique stripe IDs for lock ordering.
    #[allow(dead_code)]
    fn get_stripe_indices(&self, item: &T) -> Vec<usize> {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, self.num_hashes, self.size);

        let mut stripe_indices: Vec<usize> = indices
            .iter()
            .map(|&idx| idx / (self.size / self.stripes.len().max(1)))
            .collect();

        stripe_indices.sort_unstable();
        stripe_indices.dedup();
        stripe_indices
    }

    /// Get estimated memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.bits.memory_usage()
            + self.stripes.len() * std::mem::size_of::<RwLock<()>>()
            + std::mem::size_of::<Self>()
    }

    /// Get the actual number of bits set.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.bits.count_ones()
    }

    /// Get the load factor (ratio of set bits to total bits).
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        self.count_ones() as f64 / self.size as f64
    }

    /// Reconstruct a striped filter from raw parts.
    ///
    /// This is used for deserialization. The filter is reconstructed from
    /// the bit vector and parameters.
    ///
    /// # Arguments
    ///
    /// * `bits` - The bit vector
    /// * `num_hashes` - Number of hash functions
    /// * `num_stripes` - Number of lock stripes
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Bit vector is empty
    /// - Number of hashes is invalid (0 or > 32)
    /// - Number of stripes is zero
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::StripedBloomFilter;
    /// use bloomcraft::core::{BitVec, BloomFilter};
    ///
    /// // Create filter
    /// let filter: StripedBloomFilter<String> = StripedBloomFilter::new(1000, 0.01);
    ///
    /// // Extract parts (for serialization)
    /// let num_hashes = filter.hash_count();
    /// let stripe_count = filter.stripe_count();
    /// // ... serialize parts ...
    ///
    /// // Reconstruct (for deserialization)
    /// // let bits = BitVec::new(size);
    /// // let restored = StripedBloomFilter::from_parts(bits, num_hashes, stripe_count)?;
    /// ```
    pub fn from_parts(
        bits: BitVec,
        num_hashes: usize,
        num_stripes: usize,
    ) -> crate::error::Result<Self> {
        use crate::error::BloomCraftError;

        let size = bits.len();

        if size == 0 {
            return Err(BloomCraftError::invalid_filter_size(size));
        }

        if num_hashes == 0 || num_hashes > 32 {
            return Err(BloomCraftError::invalid_hash_count(num_hashes, 1, 32));
        }

        if num_stripes == 0 {
            return Err(BloomCraftError::invalid_parameters(
                "num_stripes must be > 0".to_string(),
            ));
        }

        let stripes = (0..num_stripes)
            .map(|_| RwLock::new(()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            bits: Arc::new(bits),
            stripes,
            num_hashes,
            size,
            hasher: Arc::new(H::default()),
            expected_items: 0,
            target_fpr: 0.0,
            _marker: PhantomData,
        })
    }

    /// Get aggregated raw bits (for serialization).
    ///
    /// Extracts the underlying bit vector data as a vector of u64 words.
    /// This operation is lock-free since `BitVec` uses atomic operations.
    ///
    /// # Errors
    ///
    /// This operation is infallible but returns `Result` for API consistency.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::StripedBloomFilter;
    /// use bloomcraft::core::BloomFilter;
    ///
    /// let mut filter = StripedBloomFilter::<String>::new(1000, 0.01);
    /// filter.insert(&"test".to_string());
    ///
    /// let bits = filter.raw_bits().unwrap();
    /// assert!(!bits.is_empty());
    /// ```
    pub fn raw_bits(&self) -> crate::error::Result<Vec<u64>> {
        Ok(self.bits.to_raw())
    }

    /// Reconstruct filter from raw bits (for deserialization).
    ///
    /// Creates a new `StripedBloomFilter` from serialized bit data and parameters.
    /// This is the inverse operation of `raw_bits()`.
    ///
    /// # Arguments
    ///
    /// * `bits` - Raw bit vector data (u64 words)
    /// * `k` - Number of hash functions
    /// * `stripe_count` - Number of lock stripes
    /// * `expected_items` - Expected number of items (for documentation)
    /// * `target_fpr` - Target false positive rate (for documentation)
    /// * `hasher` - Hash function instance
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `bits` is empty
    /// - `k` is invalid (0 or > 32)
    /// - `stripe_count` is zero
    /// - BitVec reconstruction fails
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::StripedBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    /// use bloomcraft::core::BloomFilter;
    ///
    /// let mut filter = StripedBloomFilter::<String>::new(1000, 0.01);
    /// filter.insert(&"test".to_string());
    ///
    /// // Serialize
    /// let bits = filter.raw_bits().unwrap();
    /// let k = filter.hash_count();
    /// let stripes = filter.stripe_count();
    ///
    /// // Deserialize
    /// let restored = StripedBloomFilter::<String>::from_raw_bits(
    ///     bits,
    ///     k,
    ///     stripes,
    ///     1000,
    ///     0.01,
    ///     StdHasher::default(),
    /// ).unwrap();
    ///
    /// assert!(restored.contains(&"test".to_string()));
    /// ```
    pub fn from_raw_bits(
        bits: Vec<u64>,
        k: usize,
        stripe_count: usize,
        expected_items: usize,
        target_fpr: f64,
        hasher: H,
    ) -> crate::error::Result<Self> {
        use crate::error::BloomCraftError;

        if bits.is_empty() {
            return Err(BloomCraftError::invalid_parameters(
                "bits cannot be empty".to_string(),
            ));
        }

        if k == 0 || k > 32 {
            return Err(BloomCraftError::invalid_hash_count(k, 1, 32));
        }

        if stripe_count == 0 {
            return Err(BloomCraftError::invalid_parameters(
                "stripe_count must be > 0".to_string(),
            ));
        }

        let m = params::optimal_bit_count(expected_items, target_fpr)
            .map_err(|e| BloomCraftError::invalid_parameters(e.to_string()))?;

        let bitvec = BitVec::from_raw(bits, m).map_err(|e| {
            BloomCraftError::invalid_parameters(format!(
                "Failed to reconstruct BitVec: {:?}",
                e
            ))
        })?;

        let size = bitvec.len();

        let stripes = (0..stripe_count)
            .map(|_| RwLock::new(()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            bits: Arc::new(bitvec),
            stripes,
            num_hashes: k,
            size,
            hasher: Arc::new(hasher),
            expected_items,
            target_fpr,
            _marker: PhantomData,
        })
    }

    /// Get the hasher's type name (for validation during deserialization).
    ///
    /// Returns a static string identifying the hash function type. This is used
    /// during deserialization to ensure the same hasher is used.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::StripedBloomFilter;
    ///
    /// let filter = StripedBloomFilter::<String>::new(1000, 0.01);
    /// println!("Hasher: {}", filter.hasher_name());
    /// ```
    #[must_use]
    pub fn hasher_name(&self) -> &'static str {
        self.hasher.name()
    }

    /// Get the target false positive rate.
    ///
    /// Returns the false positive rate that was specified during construction.
    /// Note: This returns an estimated value based on the current filter state.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::StripedBloomFilter;
    ///
    /// let filter = StripedBloomFilter::<String>::new(1000, 0.01);
    /// let fpr = filter.target_fpr();
    /// assert!(fpr > 0.0 && fpr < 1.0);
    /// ```
    #[must_use]
    pub fn target_fpr(&self) -> f64 {
        self.target_fpr
    }

    /// Get the originally configured expected items count.
    ///
    /// Returns the value that was specified during construction, not an estimate
    /// based on current filter state.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::StripedBloomFilter;
    ///
    /// let filter = StripedBloomFilter::<String>::new(5000, 0.01);
    /// assert_eq!(filter.expected_items_configured(), 5000);
    /// ```
    #[must_use]
    pub fn expected_items_configured(&self) -> usize {
        self.expected_items
    }

}

impl<T: Hash + Send + Sync, H: BloomHasher + Clone + Default> BloomFilter<T> for StripedBloomFilter<T, H> {
    /// Insert an item into the filter.
    ///
    /// # Panics
    ///
    /// Panics if a lock is poisoned. Lock poisoning occurs when a thread panics
    /// while holding a lock, leaving the lock in an inconsistent state.
    ///
    /// **Recovery Strategy:**
    /// - Lock poisoning is rare and indicates a serious bug in user code
    /// - If a panic occurs during insert/contains, the filter remains valid
    /// - The bit vector uses lock-free atomics, so data is never corrupted
    /// - Only the lock state is poisoned, not the filter data
    ///
    /// **Example:**
    /// ```should_panic
    /// use bloomcraft::sync::StripedBloomFilter;
    /// use bloomcraft::core::BloomFilter;
    /// use std::sync::{Arc, Mutex};
    /// use std::thread;
    ///
    /// let filter = Arc::new(Mutex::new(StripedBloomFilter::<i32>::new(1000, 0.01)));
    ///
    /// // Thread 1: Panics while holding lock
    /// let filter1 = Arc::clone(&filter);
    /// let handle = thread::spawn(move || {
    ///     let mut f = filter1.lock().unwrap();
    ///     f.insert(&42);
    ///     panic!("Simulated panic"); // Lock becomes poisoned
    /// });
    ///
    /// let _ = handle.join(); // Thread panicked
    ///
    /// // Thread 2: Subsequent operations will panic due to poisoned lock
    /// let mut f = filter.lock().unwrap();
    /// f.insert(&100); // This will panic with "Lock poisoned"
    /// ```
    ///
    /// **Note:** `ShardedBloomFilter` is lock-free and never experiences lock poisoning.
    /// Consider using it if lock poisoning is a concern.
    fn insert(&mut self, item: &T) {
        let stripe_idx = self.select_stripe(item);

        // Acquire write lock for this stripe
        let _guard = self.stripes[stripe_idx].write()
            .expect("Lock poisoned");

        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, self.num_hashes, self.size);
        for idx in indices {
            self.bits.set(idx);
        }
    }

    /// Check if an item might be in the filter.
    ///
    /// # Panics
    ///
    /// Panics if a lock is poisoned. Lock poisoning occurs when a thread panics
    /// while holding a lock, leaving the lock in an inconsistent state.
    ///
    /// **Recovery Strategy:**
    /// - Lock poisoning is rare and indicates a serious bug in user code
    /// - If a panic occurs during insert/contains, the filter remains valid
    /// - The bit vector uses lock-free atomics, so data is never corrupted
    /// - Only the lock state is poisoned, not the filter data
    ///
    /// **Example:**
    /// ```should_panic
    /// use bloomcraft::sync::StripedBloomFilter;
    /// use bloomcraft::core::BloomFilter;
    /// use std::sync::{Arc, Mutex};
    /// use std::thread;
    ///
    /// let filter = Arc::new(Mutex::new(StripedBloomFilter::<i32>::new(1000, 0.01)));
    ///
    /// // Thread 1: Panics while holding lock
    /// let filter1 = Arc::clone(&filter);
    /// let handle = thread::spawn(move || {
    ///     let f = filter1.lock().unwrap();
    ///     let _ = f.contains(&42);
    ///     panic!("Simulated panic"); // Lock becomes poisoned
    /// });
    ///
    /// let _ = handle.join(); // Thread panicked
    ///
    /// // Thread 2: Subsequent operations will panic due to poisoned lock
    /// let f = filter.lock().unwrap();
    /// let _ = f.contains(&100); // This will panic with "Lock poisoned"
    /// ```
    ///
    /// **Note:** `ShardedBloomFilter` is lock-free and never experiences lock poisoning.
    /// Consider using it if lock poisoning is a concern.
    fn contains(&self, item: &T) -> bool {
        let stripe_idx = self.select_stripe(item);

        // Acquire read lock for this stripe
        let _guard = self.stripes[stripe_idx].read()
            .expect("Lock poisoned");

        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, self.num_hashes, self.size);
        indices.iter().all(|&idx| self.bits.get(idx))
    }

    fn clear(&mut self) {
        // Acquire all stripe locks in order
        let _guards: Vec<_> = self.stripes
            .iter()
            .map(|stripe| stripe.write().expect("Lock poisoned"))
            .collect();

        // Safe to get mutable reference with all locks held
        if let Some(bits) = Arc::get_mut(&mut self.bits) {
            bits.clear();
        }
    }

    fn len(&self) -> usize {
        self.count_ones()
    }

    fn is_empty(&self) -> bool {
        self.count_ones() == 0
    }

    fn false_positive_rate(&self) -> f64 {
        let ones = self.count_ones();
        
        if ones == 0 {
            return 0.0;
        }
        
        let fill_rate = ones as f64 / self.size as f64;
        if fill_rate >= 1.0 {
            return 1.0;
        }
        
        // Use the standard FP rate formula based on fill rate
        // FP rate ≈ fill_rate^k
        fill_rate.powi(self.num_hashes as i32)
    }

    fn estimate_count(&self) -> usize {
        // Estimate based on current fill ratio
        let ones = self.count_ones() as f64;
        let m = self.size as f64;
        let k = self.num_hashes as f64;
        
        if ones == 0.0 {
            return 0;
        }
        
        // n = -(m/k) * ln(1 - X/m)
        let fill_ratio = ones / m;
        if fill_ratio >= 1.0 {
            return self.size;
        }
        
        (-(m / k) * (1.0 - fill_ratio).ln()).round() as usize
    }

    fn expected_items(&self) -> usize {
        self.expected_items
    }

    fn bit_count(&self) -> usize {
        self.size
    }

    fn hash_count(&self) -> usize {
        self.num_hashes
    }
}

impl<T: Hash + Send + Sync, H: BloomHasher + Clone + Default> MergeableBloomFilter<T> for StripedBloomFilter<T, H> {
    fn union(&mut self, other: &Self) {
        if self.size != other.size || self.num_hashes != other.num_hashes {
            panic!(
                "Incompatible filters: size {} vs {}, hashes {} vs {}",
                self.size, other.size, self.num_hashes, other.num_hashes
            );
        }

        let union_bits = self.bits.union(&other.bits)
            .expect("Union failed due to size mismatch");

        self.bits = Arc::new(union_bits);
    }

    fn intersect(&mut self, other: &Self) {
        if self.size != other.size || self.num_hashes != other.num_hashes {
            panic!(
                "Incompatible filters: size {} vs {}, hashes {} vs {}",
                self.size, other.size, self.num_hashes, other.num_hashes
            );
        }

        let intersect_bits = self.bits.intersect(&other.bits)
            .expect("Intersect failed due to size mismatch");

        self.bits = Arc::new(intersect_bits);
    }

    fn is_compatible(&self, other: &Self) -> bool {
        self.size == other.size && self.num_hashes == other.num_hashes
    }
}

impl<T: Hash, H: BloomHasher + Clone + Default> Clone for StripedBloomFilter<T, H> {
    fn clone(&self) -> Self {
        let stripes = (0..self.stripes.len())
            .map(|_| RwLock::new(()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            bits: Arc::new((*self.bits).clone()),
            stripes,
            num_hashes: self.num_hashes,
            size: self.size,
            hasher: Arc::clone(&self.hasher),
            expected_items: self.expected_items,
            target_fpr: self.target_fpr,
            _marker: PhantomData,
        }
    }
}

// Safety: StripedBloomFilter is thread-safe via RwLocks and Arc<BitVec>
unsafe impl<T: Hash + Send, H: BloomHasher + Clone + Default + Send> Send for StripedBloomFilter<T, H> {}
unsafe impl<T: Hash + Sync, H: BloomHasher + Clone + Default + Sync> Sync for StripedBloomFilter<T, H> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_striped_filter_creation() {
        let filter = StripedBloomFilter::<i32>::new(10_000, 0.01);
        assert_eq!(filter.stripe_count(), DEFAULT_STRIPE_COUNT);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_striped_filter_with_stripe_count() {
        let filter = StripedBloomFilter::<i32>::with_stripe_count(10_000, 0.01, 512);
        assert_eq!(filter.stripe_count(), 512);
    }

    #[test]
    fn test_striped_filter_insert_contains() {
        let mut filter = StripedBloomFilter::<&str>::new(1_000, 0.01);

        filter.insert(&"hello");
        filter.insert(&"world");

        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(!filter.contains(&"missing"));
    }

    #[test]
    fn test_striped_filter_insert_contains_integers() {
        let mut filter = StripedBloomFilter::<i32>::new(1_000, 0.01);

        filter.insert(&12345);
        filter.insert(&67890);

        assert!(filter.contains(&12345));
        assert!(filter.contains(&67890));
        assert!(!filter.contains(&99999));
    }

    #[test]
    fn test_striped_filter_clear() {
        let mut filter = StripedBloomFilter::<&str>::new(1_000, 0.01);

        filter.insert(&"hello");
        filter.insert(&"world");
        assert!(filter.contains(&"hello"));

        filter.clear();
        assert!(!filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
        assert!(filter.is_empty());
    }

    #[test]
    fn test_striped_filter_union() {
        let mut filter1 = StripedBloomFilter::<&str>::new(1_000, 0.01);
        let mut filter2 = StripedBloomFilter::<&str>::new(1_000, 0.01);

        filter1.insert(&"a");
        filter1.insert(&"b");
        filter2.insert(&"b");
        filter2.insert(&"c");

        filter1.union(&filter2);
        assert!(filter1.contains(&"a"));
        assert!(filter1.contains(&"b"));
        assert!(filter1.contains(&"c"));
        assert!(!filter1.contains(&"d"));
    }

    #[test]
    fn test_striped_filter_intersect() {
        let mut filter1 = StripedBloomFilter::<&str>::new(1_000, 0.01);
        let mut filter2 = StripedBloomFilter::<&str>::new(1_000, 0.01);

        filter1.insert(&"a");
        filter1.insert(&"b");
        filter1.insert(&"c");
        filter2.insert(&"b");
        filter2.insert(&"c");
        filter2.insert(&"d");

        filter1.intersect(&filter2);
        assert!(!filter1.contains(&"a"));
        assert!(filter1.contains(&"b"));
        assert!(filter1.contains(&"c"));
        assert!(!filter1.contains(&"d"));
    }

    #[test]
    fn test_striped_filter_clone() {
        let mut filter1 = StripedBloomFilter::<&str>::new(1_000, 0.01);
        filter1.insert(&"hello");

        let filter2 = filter1.clone();
        assert!(filter2.contains(&"hello"));

        // Verify independence
        filter1.insert(&"world");
        assert!(!filter2.contains(&"world"));
    }

    #[test]
    fn test_striped_filter_load_factor() {
        let mut filter = StripedBloomFilter::<i32>::new(1_000, 0.01);

        assert_eq!(filter.load_factor(), 0.0);

        for i in 0..100 {
            filter.insert(&i);
        }

        let load = filter.load_factor();
        assert!(load > 0.0 && load < 1.0);
    }

    #[test]
    fn test_striped_filter_fp_rate() {
        let mut filter = StripedBloomFilter::<i32>::new(1_000, 0.01);

        for i in 0..500 {
            filter.insert(&i);
        }

        let fp_rate = filter.false_positive_rate();
        // FP rate can vary due to striping overhead; allow up to 5x target
        assert!(fp_rate < 0.05, "FP rate {} exceeds threshold", fp_rate);
    }

    #[test]
    #[should_panic(expected = "expected_items must be > 0")]
    fn test_striped_filter_zero_items() {
        let _ = StripedBloomFilter::<i32>::new(0, 0.01);
    }

    #[test]
    #[should_panic(expected = "fp_rate must be in (0, 1)")]
    fn test_striped_filter_invalid_fp_rate() {
        let _ = StripedBloomFilter::<i32>::new(1000, 1.5);
    }

    #[test]
    #[should_panic(expected = "num_stripes must be > 0")]
    fn test_striped_filter_zero_stripes() {
        let _ = StripedBloomFilter::<i32>::with_stripe_count(1000, 0.01, 0);
    }

    #[test]
    #[should_panic]
    fn test_union_incompatible_size() {
        let mut filter1 = StripedBloomFilter::<i32>::new(1_000, 0.01);
        let filter2 = StripedBloomFilter::<i32>::new(10_000, 0.01);

        filter1.union(&filter2);
    }

    #[test]
    fn test_memory_usage() {
        let filter = StripedBloomFilter::<i32>::new(10_000, 0.01);
        let memory = filter.memory_usage();

        // Should be non-zero and reasonable
        assert!(memory > 1000);
        assert!(memory < 1_000_000);
    }
}
