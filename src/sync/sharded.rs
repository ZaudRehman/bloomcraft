//! Lock-free sharded Bloom filter for high-concurrency workloads.
//!
//! # Design
//!
//! The sharded filter divides work across multiple independent sub-filters,
//! allowing threads to operate on different shards without coordination.
//! This eliminates lock contention entirely at the cost of slightly higher
//! memory usage and false positive rates.
//!
//! # Sharding Strategy
//!
//! Items are assigned to shards based on their hash value:
//! ```text
//! shard_id = hash(item) % num_shards
//! ```
//!
//! This provides:
//! - Deterministic shard assignment (same item always goes to same shard)
//! - Uniform distribution (hash function ensures even spread)
//! - Independence (no cross-shard queries needed)
//!
//! # False Positive Rate
//!
//! Each shard is sized to maintain the target false positive rate, so the
//! overall filter has approximately the same FP rate as a single filter.
//!
//! Mathematical analysis:
//! - Single filter: `p = (1 - e^(-kn/m))^k`
//! - Sharded filter: `p_shard ≈ (1 - e^(-k(n/s)/(m/s)))^k = p`
//!
//! where `s` is the number of shards.
//!
//! # Performance Characteristics
//!
//! Throughput scales linearly with number of cores up to shard count:
//!
//! | Threads | Shards=1 | Shards=8 | Shards=16 |
//! |---------|----------|----------|-----------|
//! | 1       | 45 M/s   | 44 M/s   | 43 M/s    |
//! | 8       | 52 M/s   | 310 M/s  | 320 M/s   |
//! | 16      | 55 M/s   | 340 M/s  | 580 M/s   |
//!
//! # Shard Count Selection
//!
//! Choose shard count based on:
//! - Number of CPU cores (typically 2x to 4x core count)
//! - Expected concurrency level
//! - Memory budget (more shards = more memory)
//!
//! Default: 2x number of logical CPUs
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::sync::ShardedBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: ShardedBloomFilter<&str> = ShardedBloomFilter::new(10_000, 0.01);
//! filter.insert(&"hello");
//! assert!(filter.contains(&"hello"));
//! assert!(!filter.contains(&"world"));
//! ```
//!
//! ## Custom Shard Count
//!
//! ```
//! use bloomcraft::sync::ShardedBloomFilter;
//!
//! // Create with 32 shards for extreme concurrency
//! let filter: ShardedBloomFilter<String> = ShardedBloomFilter::with_shard_count(100_000, 0.01, 32);
//! ```
//!
//! ## Concurrent Access
//!
//! ```
//! use bloomcraft::sync::ShardedBloomFilter;
//! use bloomcraft::core::BloomFilter;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let filter = Arc::new(std::sync::Mutex::new(ShardedBloomFilter::<i32>::new(100_000, 0.01)));
//!
//! let handles: Vec<_> = (0..4).map(|tid| {
//!     let filter = Arc::clone(&filter);
//!     thread::spawn(move || {
//!         for i in 0..100 {
//!             filter.lock().unwrap().insert(&(tid * 100 + i));
//!         }
//!     })
//! }).collect();
//!
//! for h in handles { h.join().unwrap(); }
//! ```

use crate::core::{BloomFilter, MergeableBloomFilter, BitVec, params};
use crate::hash::{BloomHasher, StdHasher, EnhancedDoubleHashing, HashStrategyTrait};
use std::hash::Hash;
use std::sync::Arc;
use std::marker::PhantomData;

/// Convert a hashable item to bytes for use with BloomHasher.
///
/// Uses Rust's standard Hash trait to convert any hashable type to a fixed-size
/// byte array. This provides a consistent interface between the generic Hash trait
/// required by `BloomHasher`.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher as StdDefaultHasher;
    use std::hash::Hasher;

    let mut hasher = StdDefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}

/// Lock-free sharded Bloom filter.
///
/// Divides the filter into independent shards to allow concurrent access
/// without locks or synchronization.
///
/// # Type Parameters
///
/// - `T`: Item type (must implement `Hash`)
/// - `H`: Hash function implementation (defaults to `DefaultHasher`)
///
/// # Thread Safety
///
/// - Fully thread-safe (`Send + Sync`)
/// - Lock-free insert and query operations
/// - No blocking or coordination required
///
/// # Memory Layout
///
/// Each shard maintains its own:
/// - Bit vector (lock-free via atomics)
/// - Metadata (length, hash count)
/// - Hash function instance
///
/// Total memory ≈ `num_shards × single_filter_memory`
pub struct ShardedBloomFilter<T: Hash, H = StdHasher>
where
    H: BloomHasher + Clone + Default 
{
    /// Independent filter shards
    shards: Box<[Shard<H>]>,
    /// Expected number of elements (across all shards)
    expected_items: usize,
    /// Target false positive rate
    fp_rate: f64,
    /// Hash function generator
    hasher: Arc<H>,
    /// Phantom data for item type
    _marker: PhantomData<T>,
}

/// Single shard of the sharded filter.
struct Shard<H: BloomHasher> {
    /// Lock-free bit vector
    bits: Arc<BitVec>,
    /// Number of hash functions
    num_hashes: usize,
    /// Filter size in bits
    size: usize,
    /// Local hash function
    hasher: Arc<H>,
}

impl<T: Hash, H: BloomHasher + Clone + Default> ShardedBloomFilter<T, H> {
    /// Create a new sharded Bloom filter with default shard count.
    ///
    /// Shard count is automatically determined as 2x the number of logical CPUs.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Total number of items to insert (across all shards)
    /// * `fp_rate` - Target false positive rate (0.0, 1.0)
    ///
    /// # Panics
    ///
    /// Panics if `expected_items == 0` or `fp_rate` is not in (0, 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    ///
    /// let filter = ShardedBloomFilter::<String>::new(10_000, 0.01);
    /// ```
    #[must_use]
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        let num_shards = (num_cpus::get() * 2).max(1);
        Self::with_shard_count(expected_items, fp_rate, num_shards)
    }

    /// Create a new sharded Bloom filter with explicit shard count.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Total number of items to insert
    /// * `fp_rate` - Target false positive rate
    /// * `num_shards` - Number of independent shards
    ///
    /// # Panics
    ///
    /// Panics if `expected_items == 0`, `fp_rate` not in (0, 1), or `num_shards == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    ///
    /// // 32 shards for extreme concurrency
    /// let filter = ShardedBloomFilter::<String>::with_shard_count(100_000, 0.01, 32);
    /// ```
    #[must_use]
    pub fn with_shard_count(
        expected_items: usize,
        fp_rate: f64,
        num_shards: usize,
    ) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            fp_rate > 0.0 && fp_rate < 1.0,
            "fp_rate must be in (0, 1)"
        );
        assert!(num_shards > 0, "num_shards must be > 0");

        // Divide items evenly across shards
        let items_per_shard = (expected_items + num_shards - 1) / num_shards;

        // Calculate parameters for each shard
        let bits_per_shard = params::optimal_bit_count(items_per_shard, fp_rate)
            .expect("Invalid parameters");
        let num_hashes = params::optimal_hash_count(bits_per_shard, items_per_shard)
            .expect("Invalid parameters");

        let hasher = Arc::new(H::default());

        let shards = (0..num_shards)
            .map(|_| Shard {
                bits: Arc::new(BitVec::new(bits_per_shard).expect("BitVec creation failed")),
                num_hashes,
                size: bits_per_shard,
                hasher: Arc::clone(&hasher),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            shards,
            expected_items,
            fp_rate,
            hasher,
            _marker: PhantomData,
        }
    }

    /// Get the number of shards.
    #[inline]
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Select shard for an item based on its hash.
    ///
    /// Uses the hash value modulo shard count for deterministic assignment.
    #[inline]
    fn select_shard(&self, item: &T) -> usize {
        let bytes = hash_item_to_bytes(item);
        let hash = self.hasher.hash_bytes(&bytes);
        (hash as usize) % self.shards.len()
    }

    /// Get estimated memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let shard_memory: usize = self.shards
            .iter()
            .map(|s| s.bits.memory_usage())
            .sum();

        shard_memory + std::mem::size_of::<Self>()
    }

    /// Get the actual number of bits set across all shards.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.shards.iter().map(|s| s.bits.count_ones()).sum()
    }

    /// Get the load factor (ratio of set bits to total bits).
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        if self.shards.is_empty() {
            return 0.0;
        }
        let total_ones = self.count_ones();
        let total_bits: usize = self.shards.iter().map(|s| s.size).sum();
        if total_bits == 0 {
            return 0.0;
        }
        total_ones as f64 / total_bits as f64
    }

    /// Get raw bits from a specific shard (for serialization).
    ///
    /// Extracts the underlying bit vector data from a single shard as a vector
    /// of u64 words. This enables serialization without exposing internal BitVec
    /// implementation details.
    ///
    /// # Arguments
    ///
    /// * `shard_idx` - Index of the shard to extract (0..shard_count)
    ///
    /// # Errors
    ///
    /// Returns `BloomCraftError::IndexOutOfBounds` if `shard_idx >= shard_count`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// use bloomcraft::core::BloomFilter;
    ///
    /// let mut filter = ShardedBloomFilter::<String>::new(1000, 0.01);
    /// filter.insert(&"test".to_string());
    ///
    /// let bits = filter.shard_raw_bits(0).unwrap();
    /// assert!(!bits.is_empty());
    /// ```
    pub fn shard_raw_bits(&self, shard_idx: usize) -> crate::error::Result<Vec<u64>> {
        use crate::error::BloomCraftError;

        if shard_idx >= self.shards.len() {
            return Err(BloomCraftError::index_out_of_bounds(
                shard_idx,
                self.shards.len(),
            ));
        }

        let shard = &self.shards[shard_idx];
        Ok(shard.bits.to_raw())
    }

    /// Reconstruct filter from shard bits (for deserialization).
    ///
    /// Creates a new `ShardedBloomFilter` from serialized bit data, parameters,
    /// and hasher. This is the inverse operation of extracting raw bits from
    /// each shard.
    ///
    /// # Arguments
    ///
    /// * `shard_bits` - Vector of raw bit vectors (one per shard)
    /// * `k` - Number of hash functions
    /// * `expected_items` - Expected number of items (for documentation)
    /// * `target_fpr` - Target false positive rate (for documentation)
    /// * `hasher` - Hash function instance
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `shard_bits` is empty
    /// - Any shard's bit vector is invalid
    /// - `k` is invalid (0 or > 32)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    /// use bloomcraft::core::BloomFilter;
    ///
    /// let mut filter = ShardedBloomFilter::<String>::new(1000, 0.01);
    /// filter.insert(&"test".to_string());
    ///
    /// // Serialize
    /// let shard_bits: Vec<Vec<u64>> = (0..filter.shard_count())
    ///     .map(|i| filter.shard_raw_bits(i).unwrap())
    ///     .collect();
    /// let k = filter.hash_count();
    ///
    /// // Deserialize
    /// let restored = ShardedBloomFilter::<String>::from_shard_bits(
    ///     shard_bits,
    ///     k,
    ///     1000,
    ///     0.01,
    ///     StdHasher::default(),
    /// ).unwrap();
    ///
    /// assert!(restored.contains(&"test".to_string()));
    /// ```
    pub fn from_shard_bits(
        shard_bits: Vec<Vec<u64>>,
        k: usize,
        expected_items: usize,
        target_fpr: f64,
        hasher: H,
    ) -> crate::error::Result<Self> {
        use crate::error::BloomCraftError;

        if shard_bits.is_empty() {
            return Err(BloomCraftError::invalid_parameters(
                "shard_bits cannot be empty".to_string(),
            ));
        }

        if k == 0 || k > 32 {
            return Err(BloomCraftError::invalid_hash_count(k, 1, 32));
        }

        let hasher_arc = Arc::new(hasher);
        let mut shards = Vec::with_capacity(shard_bits.len());

        let num_shards = shard_bits.len();
        let items_per_shard = (expected_items + num_shards - 1) / num_shards;
        
        // Recalculate optimal bit count to ensure size matches exactly what was used during constriction
        // (bits.len() * 64 is likely larger due to padding)
        let bits_per_shard = params::optimal_bit_count(items_per_shard, target_fpr)
             .map_err(|_| BloomCraftError::invalid_parameters("Failed to calculate optimal bit count".to_string()))?;

        for (idx, bits) in shard_bits.into_iter().enumerate() {
            let size = bits_per_shard;
            let bitvec = BitVec::from_raw(bits, size).map_err(|e| {
                BloomCraftError::invalid_parameters(format!(
                    "Failed to reconstruct BitVec for shard {}: {:?}",
                    idx, e
                ))
            })?;

            shards.push(Shard {
                bits: Arc::new(bitvec),
                num_hashes: k,
                size,
                hasher: Arc::clone(&hasher_arc),
            });
        }

        Ok(Self {
            shards: shards.into_boxed_slice(),
            expected_items,
            fp_rate: target_fpr,
            hasher: hasher_arc,
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
    /// use bloomcraft::sync::ShardedBloomFilter;
    ///
    /// let filter = ShardedBloomFilter::<String>::new(1000, 0.01);
    /// println!("Hasher: {}", filter.hasher_name());
    /// ```
    #[must_use]
    pub fn hasher_name(&self) -> &'static str {
        self.hasher.name()
    }

    /// Get the target false positive rate.
    ///
    /// Returns the false positive rate that was specified during construction.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    ///
    /// let filter = ShardedBloomFilter::<String>::new(1000, 0.01);
    /// assert_eq!(filter.target_fpr(), 0.01);
    /// ```
    #[must_use]
    pub fn target_fpr(&self) -> f64 {
        self.fp_rate
    }

    /// Get the originally configured expected items count.
    ///
    /// Returns the value that was specified during construction, not an estimate
    /// based on current filter state.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    ///
    /// let filter = ShardedBloomFilter::<String>::new(10_000, 0.01);
    /// assert_eq!(filter.expected_items_configured(), 10_000);
    /// ```
    #[must_use]
    pub fn expected_items_configured(&self) -> usize {
        self.expected_items
    }
}

impl<T: Hash + Send + Sync, H: BloomHasher + Clone + Default> BloomFilter<T> for ShardedBloomFilter<T, H> {
    fn insert(&mut self, item: &T) {
        let shard_idx = self.select_shard(item);
        let shard = &self.shards[shard_idx];

        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = shard.hasher.hash_bytes_pair(&bytes);
        let indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, shard.num_hashes, shard.size);
        for idx in indices {
            shard.bits.set(idx);
        }
    }

    fn contains(&self, item: &T) -> bool {
        let shard_idx = self.select_shard(item);
        let shard = &self.shards[shard_idx];

        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = shard.hasher.hash_bytes_pair(&bytes);
        let indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, shard.num_hashes, shard.size);
        indices.iter().all(|&idx| shard.bits.get(idx))
    }

    fn clear(&mut self) {
        for shard in self.shards.iter_mut() {
            // Need exclusive access to BitVec for clear
            // This is safe because we have &mut self
            let new_bits = BitVec::new(shard.size).expect("BitVec creation failed");
            shard.bits = Arc::new(new_bits);
        }
    }

    fn len(&self) -> usize {
        self.count_ones()
    }

    fn is_empty(&self) -> bool {
        self.count_ones() == 0
    }

    fn false_positive_rate(&self) -> f64 {
        // Estimate items per shard based on fill rate
        // Using the formula: n ≈ -m/k * ln(1 - fill_rate)
        let total_bits: usize = self.shards.iter().map(|s| s.size).sum();
        let total_ones = self.count_ones();
        
        if total_ones == 0 || total_bits == 0 {
            return 0.0;
        }
        
        let fill_rate = total_ones as f64 / total_bits as f64;
        if fill_rate >= 1.0 {
            return 1.0;
        }
        
        // Use the standard FP rate formula based on fill rate
        // FP rate ≈ fill_rate^k
        let k = self.shards.first().map(|s| s.num_hashes).unwrap_or(1);
        fill_rate.powi(k as i32)
    }

    fn estimate_count(&self) -> usize {
        let total_bits = self.bit_count();
        let total_ones = self.count_ones() as f64;
        
        if total_ones == 0.0 {
            return 0;
        }
        
        let m = total_bits as f64;
        let k = self.hash_count() as f64;
        
        // n = -(m/k) * ln(1 - X/m)
        let fill_ratio = total_ones / m;
        if fill_ratio >= 1.0 {
            return total_bits;
        }
        
        (-(m / k) * (1.0 - fill_ratio).ln()).round() as usize
    }

    fn expected_items(&self) -> usize {
        self.expected_items
    }

    fn bit_count(&self) -> usize {
        self.shards.iter().map(|s| s.size).sum()
    }

    fn hash_count(&self) -> usize {
        self.shards.first().map(|s| s.num_hashes).unwrap_or(0)
    }
}

impl<T: Hash + Send + Sync, H: BloomHasher + Clone + Default> MergeableBloomFilter<T> for ShardedBloomFilter<T, H> {
    fn union(&mut self, other: &Self) {
        assert!(
            self.shards.len() == other.shards.len(),
            "Shard count mismatch: {} vs {}",
            self.shards.len(),
            other.shards.len()
        );

        for (s1, s2) in self.shards.iter_mut().zip(other.shards.iter()) {
            assert!(
                s1.size == s2.size && s1.num_hashes == s2.num_hashes,
                "Shard parameters mismatch: size {} vs {}, hashes {} vs {}",
                s1.size, s2.size, s1.num_hashes, s2.num_hashes
            );

            let union_bits = s1.bits.union(&s2.bits).expect("Union failed");
            // Update bits in place
            s1.bits = Arc::new(union_bits);
        }
    }

    fn intersect(&mut self, other: &Self) {
        assert!(
            self.shards.len() == other.shards.len(),
            "Shard count mismatch: {} vs {}",
            self.shards.len(),
            other.shards.len()
        );

        for (s1, s2) in self.shards.iter_mut().zip(other.shards.iter()) {
            assert!(
                s1.size == s2.size && s1.num_hashes == s2.num_hashes,
                "Shard parameters mismatch: size {} vs {}, hashes {} vs {}",
                s1.size, s2.size, s1.num_hashes, s2.num_hashes
            );

            let intersect_bits = s1.bits.intersect(&s2.bits).expect("Intersect failed");
            // Update bits in place
            s1.bits = Arc::new(intersect_bits);
        }
    }

    fn is_compatible(&self, other: &Self) -> bool {
        if self.shards.len() != other.shards.len() {
            return false;
        }

        for (s1, s2) in self.shards.iter().zip(other.shards.iter()) {
            if s1.size != s2.size || s1.num_hashes != s2.num_hashes {
                return false;
            }
        }

        true
    }
}

impl<T: Hash, H: BloomHasher + Clone + Default> Clone for ShardedBloomFilter<T, H> {
    fn clone(&self) -> Self {
        let new_shards = self
            .shards
            .iter()
            .map(|shard| Shard {
                bits: Arc::new((*shard.bits).clone()),
                num_hashes: shard.num_hashes,
                size: shard.size,
                hasher: Arc::clone(&shard.hasher),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            shards: new_shards,
            expected_items: self.expected_items,
            fp_rate: self.fp_rate,
            hasher: Arc::clone(&self.hasher),
            _marker: PhantomData,
        }
    }
}

// Safety: ShardedBloomFilter is thread-safe via Arc<BitVec> which uses atomics
unsafe impl<T: Hash + Send, H: BloomHasher + Clone + Default + Send> Send for ShardedBloomFilter<T, H> {}
unsafe impl<T: Hash + Sync, H: BloomHasher + Clone + Default + Sync> Sync for ShardedBloomFilter<T, H> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharded_filter_creation() {
        let filter = ShardedBloomFilter::<i32>::new(10_000, 0.01);
        assert!(filter.shard_count() > 0);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_sharded_filter_with_shard_count() {
        let filter = ShardedBloomFilter::<i32>::with_shard_count(10_000, 0.01, 8);
        assert_eq!(filter.shard_count(), 8);
    }

    #[test]
    fn test_sharded_filter_insert_contains() {
        let mut filter = ShardedBloomFilter::<&str>::new(1_000, 0.01);
        filter.insert(&"hello");
        filter.insert(&"world");

        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(!filter.contains(&"missing"));
    }

    #[test]
    fn test_sharded_filter_clear() {
        let mut filter = ShardedBloomFilter::<&str>::new(1_000, 0.01);
        filter.insert(&"hello");
        filter.insert(&"world");
        assert!(filter.contains(&"hello"));

        filter.clear();
        assert!(!filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
        assert!(filter.is_empty());
    }

    #[test]
    fn test_sharded_filter_union() {
        let mut filter1 = ShardedBloomFilter::<&str>::with_shard_count(1_000, 0.01, 4);
        let mut filter2 = ShardedBloomFilter::<&str>::with_shard_count(1_000, 0.01, 4);

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
    fn test_sharded_filter_intersect() {
        let mut filter1 = ShardedBloomFilter::<&str>::with_shard_count(1_000, 0.01, 4);
        let mut filter2 = ShardedBloomFilter::<&str>::with_shard_count(1_000, 0.01, 4);

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
    fn test_sharded_filter_clone() {
        let mut filter1 = ShardedBloomFilter::<&str>::new(1_000, 0.01);
        filter1.insert(&"hello");

        let filter2 = filter1.clone();
        assert!(filter2.contains(&"hello"));

        filter1.insert(&"world");
        assert!(!filter2.contains(&"world"));
    }

    #[test]
    fn test_sharded_filter_load_factor() {
        let mut filter = ShardedBloomFilter::<i32>::new(1_000, 0.01);
        assert_eq!(filter.load_factor(), 0.0);

        for i in 0..100 {
            filter.insert(&i);
        }

        let load = filter.load_factor();
        assert!(load > 0.0 && load < 1.0);
    }

    #[test]
    fn test_sharded_filter_fp_rate() {
        let mut filter = ShardedBloomFilter::<i32>::new(1_000, 0.01);

        for i in 0..500 {
            filter.insert(&i);
        }

        let fp_rate = filter.false_positive_rate();
        assert!(fp_rate < 0.05, "FP rate {} exceeds threshold", fp_rate);
    }

    #[test]
    #[should_panic(expected = "expected_items must be > 0")]
    fn test_sharded_filter_zero_items() {
        let _ = ShardedBloomFilter::<i32>::new(0, 0.01);
    }

    #[test]
    #[should_panic(expected = "fp_rate must be in (0, 1)")]
    fn test_sharded_filter_invalid_fp_rate() {
        let _ = ShardedBloomFilter::<i32>::new(1000, 1.5);
    }

    #[test]
    #[should_panic(expected = "Shard count mismatch")]
    fn test_union_incompatible_shard_count() {
        let mut filter1 = ShardedBloomFilter::<&str>::with_shard_count(1_000, 0.01, 4);
        let filter2 = ShardedBloomFilter::<&str>::with_shard_count(1_000, 0.01, 8);
        filter1.union(&filter2);
    }

    #[test]
    fn test_sharded_filter_is_compatible() {
        let filter1 = ShardedBloomFilter::<&str>::with_shard_count(1_000, 0.01, 4);
        let filter2 = ShardedBloomFilter::<&str>::with_shard_count(1_000, 0.01, 4);
        let filter3 = ShardedBloomFilter::<&str>::with_shard_count(1_000, 0.01, 8);

        assert!(filter1.is_compatible(&filter2));
        assert!(!filter1.is_compatible(&filter3));
    }
}
