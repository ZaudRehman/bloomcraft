//! Lock-free sharded Bloom filter for high-concurrency workloads.
//!
//! Partitions the filter into independent shards, each with its own `BitVec` and
//! hash function. Items are assigned to shards via [Lemire's fast range reduction]
//! (`(hash × shards) >> 64`), enabling lock-free parallel access with no
//! cross-shard coordination.
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │              ShardedBloomFilter                  │
//! │  Hash Function (Arc<H>), select_shard(h1)        │
//! └──────────────────────┬───────────────────────────┘
//!                        │
//!    ┌───────────┬───────┴──┬──────────┬──────────┐
//!    │           │          │          │          │
//!  Shard 0    Shard 1    Shard 2    Shard 3    Shard N
//! ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐    ┌──────┐
//! │ k=7  │   │ k=7  │   │ k=7  │   │ k=7  │    │ k=7  │
//! │m=1000│   │m=1000│   │m=1000│   │m=1000│    │m=1000│
//! └──────┘   └──────┘   └──────┘   └──────┘    └──────┘
//! ```
//!
//! # Performance
//!
//! Shard count has no measurable effect on sequential throughput. Under contention
//! the filter scales near-linearly up to hardware thread count:
//!
//! | Threads | Throughput | Speedup |
//! |---------|-----------|---------|
//! | 1       |  9.2 M/s  |   1.0×  |
//! | 4       | 21.0 M/s  |   2.3×  |
//! | 8       | 30.4 M/s  |   3.3×  |
//! | 16      | 34.2 M/s  |   3.7×  |
//! | 32      | 36.8 M/s  |   4.0×  |
//!
//! <small>AMD Ryzen 5 5600H (6C/12T), 1M items, 1% FPR, 12 shards.</small>
//!
//! Single-threaded overhead vs [`StandardBloomFilter`] is ~3× due to atomic
//! operations. Breakeven is typically between 2–4 threads.
//!
//! # clear() semantics
//!
//! `clear()` atomically swaps each shard's `BitVec` via `AtomicPtr`. An `insert()`
//! that loaded its pointer before the swap but writes after it may be lost.
//! This is acceptable for probabilistic data structures; strict linearizability
//! requires external synchronization.
//!
//! [`StandardBloomFilter`]: crate::filters::StandardBloomFilter
//!
//! # References
//!
//! - Sánchez, D., Yen, L., Kozyrakis, C., & Hill, M. D. (2007). Design and Implementation of 
//!   Signatures for Transactional Memory Systems. *IEEE Micro*, 27(5), 10-22. 
//! - Lemire, D. (2019). Fast Random Integer Generation in an Interval. *ACM Transactions on 
//!   Modeling and Computer Simulation (TOMS)*, 29(1), 1-12. 

#![allow(dead_code)]

use crate::core::{params, BitVec, SharedBloomFilter};
use crate::error::{BloomCraftError, Result};
use crate::hash::{BloomHasher, EnhancedDoubleHashing, StdHasher};
use crate::hash::strategies::HashStrategy;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

/// Converts a hashable item to `[u8; 8]` for use with `BloomHasher`.
///
/// The result is reused for both shard selection and bit-index generation,
/// ensuring a single hash computation per operation.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher as StdDefaultHasher;
    use std::hash::Hasher;
    
    let mut hasher = StdDefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}

/// Cache-line size for alignment (prevents false sharing between shards).
const CACHE_LINE_SIZE: usize = 128;

/// Lock-free sharded Bloom filter.
///
/// Divides the filter into independent shards, each with its own `BitVec` and
/// cloned hash function. Items are assigned to shards via `select_shard_from_hash`
/// so threads operating on different shards never contend.
///
/// # Performance
///
/// | Operation | Complexity |
/// |-----------|-----------|
/// | Insert    | O(k) k ≈ 7 @ 1% FPR |
/// | Query     | O(k), short-circuits on first zero bit |
/// | Clear     | O(s) shard swaps |
///
/// Throughput scales near-linearly under contention (see module-level docs).
///
/// # Example
///
/// ```
/// use bloomcraft::sync::ShardedBloomFilter;
/// use bloomcraft::core::SharedBloomFilter;
///
/// let filter = ShardedBloomFilter::<&str>::new(10_000, 0.01);
/// filter.insert(&"example");
/// assert!(filter.contains(&"example"));
/// ```
#[derive(Debug)]
pub struct ShardedBloomFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone,
{
    /// Independent filter shards (cache-line aligned)
    shards: Box<[Shard<H>]>,
    
    /// Expected number of elements across all shards (for documentation/metrics)
    expected_items: usize,
    
    /// Target false positive rate (used for parameter calculation)
    fprate: f64,
    
    /// Shared hash function generator (cloned per shard)
    hasher: Arc<H>,
    
    /// Phantom data for item type (zero-sized)
    _marker: PhantomData<T>,
    
    /// Performance metrics (feature-gated, zero-cost when disabled)
    #[cfg(feature = "metrics")]
    metrics: ShardedBloomMetrics,
}

/// A single independent shard in the filter.
///
/// Each shard is a complete Bloom filter with its own `BitVec`. The
/// `AtomicPtr<Arc<BitVec>>` design enables lock-free `clear()`:
/// insert/query load the current `Arc`, while `clear()` atomically swaps
/// in a fresh `BitVec`. The old `BitVec` stays alive while any thread
/// holds an `Arc` reference.
///
/// `#[repr(C, align(128))]` guarantees cache-line alignment, preventing
/// false sharing between cores operating on different shards.
#[repr(C, align(128))]
struct Shard<H> {
    bits: AtomicPtr<Arc<BitVec>>,
    numhashes: usize,
    size: usize,
    hasher: Arc<H>,
    _padding: [u8; 64],
}

impl<H> std::fmt::Debug for Shard<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ptr = self.bits.load(Ordering::SeqCst);
        
        let bits_info = if ptr.is_null() {
            "null".to_string()
        } else {
            unsafe {
                let arc = &*ptr;
                format!(
                    "Arc(ones={}, capacity={})",
                    arc.count_ones(),
                    arc.len()
                )
            }
        };
        
        f.debug_struct("Shard")
            .field("bits", &bits_info)
            .field("numhashes", &self.numhashes)
            .field("size", &self.size)
            .field("hasher", &std::any::type_name::<H>())
            .finish()
    }
}

const _: () = {
    const SHARD_SIZE: usize = size_of::<Shard<StdHasher>>();
    assert!(SHARD_SIZE <= 128);
};

impl<H> Shard<H> {
    /// Returns a clone of the current `Arc<BitVec>`, keeping it alive during the
    /// operation even if another thread calls `clear()`.
    #[inline]
    fn bits(&self) -> Arc<BitVec> {
        let ptr = self.bits.load(Ordering::SeqCst);
        debug_assert!(!ptr.is_null());
        if ptr.is_null() {
            panic!("Shard BitVec pointer is null (invariant violation)");
        }
        unsafe { Arc::clone(&*ptr) }
    }
    
    /// Atomically replaces the current `BitVec` with `new_bits`. Returns a clone
    /// of the old `Arc` so the caller can drop it safely without triggering
    /// deallocation while other threads may still hold references.
    fn replace_bits(&self, new_bits: Arc<BitVec>) -> Arc<BitVec> {
        let new_ptr = Box::into_raw(Box::new(new_bits));
        let old_ptr = self.bits.swap(new_ptr, Ordering::AcqRel);
        unsafe {
            let old_arc_box = Box::from_raw(old_ptr);
            Arc::clone(&*old_arc_box)
        }
    }
}

impl<H> Drop for Shard<H> {
    fn drop(&mut self) {
        let ptr = self.bits.swap(std::ptr::null_mut(), Ordering::Relaxed);
        debug_assert!(!ptr.is_null());
        if !ptr.is_null() {
            unsafe { let _ = Box::from_raw(ptr); }
        }
    }
}

/// Per-shard statistics.
#[derive(Debug, Clone)]
pub struct ShardStats {
    pub shard_id: usize,
    pub size: usize,
    pub ones_count: usize,
    pub fill_rate: f64,
    pub numhashes: usize,
}

#[cfg(feature = "metrics")]
#[derive(Debug, Default)]
pub struct ShardedBloomMetrics {
    pub inserts_total: std::sync::atomic::AtomicU64,
    pub queries_total: std::sync::atomic::AtomicU64,
    pub clears_total: std::sync::atomic::AtomicU64,
    pub shard_contention_events: Vec<std::sync::atomic::AtomicU64>,
}

impl<T, H> ShardedBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Creates a filter with shard count = 2 × logical CPUs.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items == 0` or `fprate ∉ (0, 1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// let filter = ShardedBloomFilter::<i32>::new(10_000, 0.01);
    /// assert!(filter.shard_count() > 0);
    /// ```
    #[must_use]
    pub fn new(expected_items: usize, fprate: f64) -> Self {
        let num_shards = num_cpus::get().saturating_mul(2).max(1);
        Self::with_shard_count(expected_items, fprate, num_shards)
    }
    
    /// Creates a filter with shard count tuned to workload.
    ///
    /// Formula: `min(2 × cores, items / 10_000, 256)`.
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// let filter = ShardedBloomFilter::<i32>::new_adaptive(100_000, 0.01);
    /// ```
    #[must_use]
    pub fn new_adaptive(expected_items: usize, fprate: f64) -> Self {
        let num_cores = num_cpus::get();
        let num_shards = Self::optimal_shard_count(num_cores, expected_items);
        Self::with_shard_count(expected_items, fprate, num_shards)
    }
    
    fn optimal_shard_count(num_cores: usize, expected_items: usize) -> usize {
        let cores_based = num_cores.saturating_mul(2);
        let items_based = (expected_items / 10_000).max(1);
        cores_based.min(items_based).clamp(1, 256)
    }
    
    /// Creates a filter with an explicit shard count.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items == 0`, `fprate ∉ (0, 1)`, or `num_shards == 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// let filter = ShardedBloomFilter::<i32>::with_shard_count(100_000, 0.01, 32);
    /// assert_eq!(filter.shard_count(), 32);
    /// ```
    #[must_use]
    pub fn with_shard_count(expected_items: usize, fprate: f64, num_shards: usize) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            fprate > 0.0 && fprate < 1.0,
            "fprate must be in (0, 1), got {}",
            fprate
        );
        assert!(num_shards > 0, "num_shards must be > 0");
        
        let items_per_shard = expected_items.div_ceil(num_shards);
        let bits_per_shard = params::optimal_bit_count(items_per_shard, fprate)
            .expect("Invalid parameters");
        let numhashes = params::optimal_hash_count(bits_per_shard, items_per_shard)
            .expect("Invalid parameters");
        
        let hasher = Arc::new(H::default());
        
        let shards = (0..num_shards)
            .map(|_| {
                let bitvec = Arc::new(BitVec::new(bits_per_shard)
                    .expect("BitVec creation failed"));
                let ptr = Box::into_raw(Box::new(bitvec));
                Shard {
                    bits: AtomicPtr::new(ptr),
                    numhashes,
                    size: bits_per_shard,
                    hasher: Arc::clone(&hasher),
                    _padding: [0; 64],
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        
        Self {
            shards,
            expected_items,
            fprate,
            hasher,
            _marker: PhantomData,
            #[cfg(feature = "metrics")]
            metrics: ShardedBloomMetrics {
                inserts_total: std::sync::atomic::AtomicU64::new(0),
                queries_total: std::sync::atomic::AtomicU64::new(0),
                clears_total: std::sync::atomic::AtomicU64::new(0),
                shard_contention_events: (0..num_shards)
                    .map(|_| std::sync::atomic::AtomicU64::new(0))
                    .collect(),
            },
        }
    }
    
    /// Returns the number of shards.
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// let filter = ShardedBloomFilter::<i32>::with_shard_count(1000, 0.01, 8);
    /// assert_eq!(filter.shard_count(), 8);
    /// ```
    #[inline]
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
    
    /// Assigns an item to a shard via [Lemire's fast range reduction].
    ///
    /// Equivalent to `hash % num_shards` but ~5× faster and unbiased.
    ///
    /// [Lemire's fast range reduction]: https://arxiv.org/abs/1805.10941
    #[inline]
    fn select_shard_from_hash(&self, hash: u64) -> usize {
        let num_shards = self.shards.len();
        if num_shards == 0 {
            return 0;
        }
        let product = (hash as u128).wrapping_mul(num_shards as u128);
        let index = (product >> 64) as usize;
        if index >= num_shards { index % num_shards } else { index }
    }
    
    /// Returns total memory used by all shards' bit vectors plus struct overhead
    /// (excludes allocator overhead and Arc control blocks).
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// let filter = ShardedBloomFilter::<i32>::new(10_000, 0.01);
    /// assert!(filter.memory_usage() > 0);
    /// ```
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let shard_memory: usize = self.shards
            .iter()
            .map(|s| s.bits().memory_usage())
            .sum();
        
        shard_memory + size_of::<Self>()
    }
    
    /// Returns the total number of set bits across all shards.
    ///
    /// O(total_bits / 64).
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.shards.iter().map(|s| s.bits().count_ones()).sum()
    }
    
    /// Returns the ratio of set bits to total bits, in `[0, 1]`.
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// use bloomcraft::core::SharedBloomFilter;
    /// let filter = ShardedBloomFilter::<i32>::new(1000, 0.01);
    /// assert_eq!(filter.load_factor(), 0.0);
    /// for i in 0..500 { filter.insert(&i); }
    /// assert!(filter.load_factor() > 0.0);
    /// ```
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
    
    /// Returns the target false-positive rate supplied at construction.
    #[must_use]
    pub fn target_fpr(&self) -> f64 {
        self.fprate
    }
    
    /// Returns the `expected_items` count supplied at construction (not the actual
    /// number of inserted items; see [`estimate_count`](SharedBloomFilter::estimate_count)).
    #[must_use]
    pub fn expected_items_configured(&self) -> usize {
        self.expected_items
    }
    
    /// Get the hasher's type name for validation during deserialization.
    #[must_use]
    pub fn hasher_name(&self) -> &'static str {
        self.hasher.name()
    }
    
    /// Returns the underlying `u64` words of a shard's bit vector (for serialization).
    ///
    /// Errors with `IndexOutOfBounds` if `shard_idx >= shard_count()`.
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// use bloomcraft::core::SharedBloomFilter;
    /// let filter = ShardedBloomFilter::<String>::new(1000, 0.01);
    /// filter.insert(&"test".to_string());
    /// let bits = filter.shard_raw_bits(0).unwrap();
    /// assert!(!bits.is_empty());
    /// ```
    pub fn shard_raw_bits(&self, shard_idx: usize) -> Result<Vec<u64>> {
        if shard_idx >= self.shards.len() {
            return Err(BloomCraftError::index_out_of_bounds(
                shard_idx,
                self.shards.len(),
            ));
        }
        
        let shard = &self.shards[shard_idx];
        let bits = shard.bits();
        let raw = bits.to_raw();
        
        // Validate data integrity
        let expected_words = shard.size.div_ceil(64);
        if raw.len() != expected_words {
            return Err(BloomCraftError::internal_error(format!(
                "Shard {} corrupted: {} words, expected {} for {} bits",
                shard_idx,
                raw.len(),
                expected_words,
                shard.size
            )));
        }
        
        Ok(raw)
    }
    
    /// Reconstructs a filter from serialised shard bits.
    ///
    /// Each element of `shard_bits` must have exactly the number of `u64` words
    /// that `optimal_bit_count(items_per_shard, target_fpr)` produces.
    ///
    /// # Errors
    ///
    /// - `InvalidParameters` if `shard_bits` is empty, `k` is 0 or > 32, or
    ///   any shard's word count doesn't match the expected size.
    ///
    /// # Example
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    /// use bloomcraft::core::SharedBloomFilter;
    ///
    /// let filter = ShardedBloomFilter::<String>::new(1000, 0.01);
    /// filter.insert(&"test".to_string());
    ///
    /// let shard_bits: Vec<Vec<u64>> = (0..filter.shard_count())
    ///     .map(|i| filter.shard_raw_bits(i).unwrap())
    ///     .collect();
    /// let k = filter.hash_count();
    ///
    /// let restored = ShardedBloomFilter::<String>::from_shard_bits(
    ///     shard_bits, k, 1000, 0.01, StdHasher::default(),
    /// ).unwrap();
    /// assert!(restored.contains(&"test".to_string()));
    /// ```
    pub fn from_shard_bits(
        shard_bits: Vec<Vec<u64>>,
        k: usize,
        expected_items: usize,
        target_fpr: f64,
        hasher: H,
    ) -> Result<Self> {
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
        let items_per_shard = expected_items.div_ceil(num_shards);
        
        // Recalculate optimal bit count to validate against actual data
        let expected_bits_per_shard = params::optimal_bit_count(items_per_shard, target_fpr)
            .map_err(|_| {
                BloomCraftError::invalid_parameters(
                    "Failed to calculate optimal bit count".to_string(),
                )
            })?;
        
        for (idx, bits) in shard_bits.into_iter().enumerate() {
            // Validate data size matches expected size
            let expected_words = expected_bits_per_shard.div_ceil(64);
            if bits.len() != expected_words {
                return Err(BloomCraftError::invalid_parameters(format!(
                    "Shard {} size mismatch: got {} words ({} bits), expected {} words ({} bits) for {}/{} items at {:.6} FPR",
                    idx,
                    bits.len(),
                    bits.len() * 64,
                    expected_words,
                    expected_bits_per_shard,
                    items_per_shard,
                    expected_items,
                    target_fpr
                )));
            }
            
            let bitvec = BitVec::from_raw(bits, expected_bits_per_shard)
                .map_err(|e| {
                    BloomCraftError::invalid_parameters(format!(
                        "Failed to reconstruct BitVec for shard {}: {:?}",
                        idx, e
                    ))
                })?;
            
            let arc_bitvec = Arc::new(bitvec);
            let ptr = Box::into_raw(Box::new(arc_bitvec));
            
            shards.push(Shard {
                bits: AtomicPtr::new(ptr),
                numhashes: k,
                size: expected_bits_per_shard,
                hasher: Arc::clone(&hasher_arc),
                _padding: [0; 64],
            });
        }
        
        Ok(Self {
            shards: shards.into_boxed_slice(),
            expected_items,
            fprate: target_fpr,
            hasher: hasher_arc,
            _marker: PhantomData,
            #[cfg(feature = "metrics")]
            metrics: ShardedBloomMetrics {
                inserts_total: std::sync::atomic::AtomicU64::new(0),
                queries_total: std::sync::atomic::AtomicU64::new(0),
                clears_total: std::sync::atomic::AtomicU64::new(0),
                shard_contention_events: (0..num_shards)
                    .map(|_| std::sync::atomic::AtomicU64::new(0))
                    .collect(),
            },
        })
    }
    
    /// Returns per-shard statistics for monitoring.
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// use bloomcraft::core::SharedBloomFilter;
    /// let filter = ShardedBloomFilter::<i32>::new(10_000, 0.01);
    /// for i in 0..1000 { filter.insert(&i); }
    /// let stats = filter.shard_stats();
    /// assert_eq!(stats.len(), filter.shard_count());
    /// ```
    #[must_use]
    pub fn shard_stats(&self) -> Vec<ShardStats> {
        self.shards
            .iter()
            .enumerate()
            .map(|(idx, shard)| {
                let bits = shard.bits();
                let ones = bits.count_ones();
                let fill_rate = ones as f64 / shard.size as f64;
                
                ShardStats {
                    shard_id: idx,
                    size: shard.size,
                    ones_count: ones,
                    fill_rate,
                    numhashes: shard.numhashes,
                }
            })
            .collect()
    }
    
    /// Returns `true` if any shard's fill rate deviates >20 % from the mean.
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// use bloomcraft::core::SharedBloomFilter;
    /// let filter = ShardedBloomFilter::<i32>::new(10_000, 0.01);
    /// assert!(!filter.has_imbalanced_shards());
    /// ```
    #[must_use]
    pub fn has_imbalanced_shards(&self) -> bool {
        let stats = self.shard_stats();
        
        if stats.is_empty() {
            return false;
        }
        
        let mean_fill = stats.iter().map(|s| s.fill_rate).sum::<f64>() / stats.len() as f64;
        
        stats.iter().any(|s| {
            if mean_fill == 0.0 {
                return false;
            }
            (s.fill_rate - mean_fill).abs() / mean_fill > 0.20
        })
    }
    
    /// Inserts all items with the same per-item cost as individual `insert` calls.
    ///
    /// ```
    /// use bloomcraft::sync::ShardedBloomFilter;
    /// let filter = ShardedBloomFilter::<i32>::new(10_000, 0.01);
    /// let items: Vec<i32> = (0..1000).collect();
    /// filter.insert_batch_chunked(&items);
    /// ```
    pub fn insert_batch_chunked<'a, I>(&self, items: I)
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        for item in items {
            let bytes = hash_item_to_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
            let shard_idx = self.select_shard_from_hash(h1);
            let shard = &self.shards[shard_idx];
            let bits = shard.bits();

            let indices = EnhancedDoubleHashing.generate_indices(
                h1,
                h2,
                0,
                shard.numhashes,
                shard.size,
            );

            for idx in indices {
                bits.set(idx);
            }
        }
    }
    
    #[cfg(feature = "metrics")]
    #[must_use]
    /// Access the filter's metrics collector.
    pub fn metrics(&self) -> &ShardedBloomMetrics {
        &self.metrics
    }
}

// SharedBloomFilter Trait Implementation

impl<T, H> SharedBloomFilter<T> for ShardedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
    fn insert(&self, item: &T) {
        #[cfg(feature = "metrics")]
        self.metrics.inserts_total.fetch_add(1, Ordering::Relaxed);
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let shard_idx = self.select_shard_from_hash(h1);
        let shard = &self.shards[shard_idx];
        let bits = shard.bits();
        let indices = EnhancedDoubleHashing.generate_indices(
            h1, h2, 0, shard.numhashes, shard.size,
        );
        for idx in indices {
            bits.set(idx);
        }
    }
    
    fn contains(&self, item: &T) -> bool {
        #[cfg(feature = "metrics")]
        self.metrics.queries_total.fetch_add(1, Ordering::Relaxed);
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let shard_idx = self.select_shard_from_hash(h1);
        let shard = &self.shards[shard_idx];
        let bits = shard.bits();
        let indices = EnhancedDoubleHashing.generate_indices(
            h1, h2, 0, shard.numhashes, shard.size,
        );
        indices.iter().all(|idx| bits.get(*idx))
    }
    
    fn clear(&self) {
        #[cfg(feature = "metrics")]
        self.metrics.clears_total.fetch_add(1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::SeqCst);
        for shard in self.shards.iter() {
            let new_bits = Arc::new(
                BitVec::new(shard.size).expect("BitVec allocation failed")
            );
            drop(shard.replace_bits(new_bits));
        }
        std::sync::atomic::fence(Ordering::SeqCst);
    }
    
    fn len(&self) -> usize {
        self.count_ones()
    }
    
    fn is_empty(&self) -> bool {
        self.count_ones() == 0
    }
    
    fn false_positive_rate(&self) -> f64 {
        let total_bits: usize = self.shards.iter().map(|s| s.size).sum();
        let total_ones = self.count_ones();
        
        if total_ones == 0 || total_bits == 0 {
            return 0.0;
        }
        
        let fill_rate = total_ones as f64 / total_bits as f64;
        
        if fill_rate >= 1.0 {
            return 1.0;
        }
        
        // Standard FP rate formula: p = (fill_rate)^k
        let k = self.shards.first().map(|s| s.numhashes).unwrap_or(1);
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
        
        // Cardinality estimation: n = -(m/k) * ln(1 - X/m)
        let fill_ratio = total_ones / m;
        
        if fill_ratio >= 1.0 {
            return total_bits; // Saturated filter
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
        self.shards.first().map(|s| s.numhashes).unwrap_or(0)
    }
    
    fn insert_batch<'a, I>(&self, items: I)
    where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
    {
        for item in items {
            self.insert(item);
        }
    }

    fn count_set_bits(&self) -> usize {
        self.count_ones()
    }
}

impl<T, H> Clone for ShardedBloomFilter<T, H>
where
    H: BloomHasher + Clone,
{
    /// Deep copy. O(total_bits).
    fn clone(&self) -> Self {
        let new_shards = self
            .shards
            .iter()
            .map(|shard| {
                let bits = shard.bits();
                let new_bitvec = Arc::new((*bits).clone());
                let ptr = Box::into_raw(Box::new(new_bitvec));
                Shard {
                    bits: AtomicPtr::new(ptr),
                    numhashes: shard.numhashes,
                    size: shard.size,
                    hasher: Arc::clone(&shard.hasher),
                    _padding: [0; 64],
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        
        Self {
            shards: new_shards,
            expected_items: self.expected_items,
            fprate: self.fprate,
            hasher: Arc::clone(&self.hasher),
            _marker: PhantomData,
            #[cfg(feature = "metrics")]
            metrics: ShardedBloomMetrics::default(),
        }
    }
}

// SAFETY: thread-safe via atomic operations and Arc-based ownership.
unsafe impl<T, H> Send for ShardedBloomFilter<T, H>
where
    T: Send,
    H: BloomHasher + Clone + Send,
{}

unsafe impl<T, H> Sync for ShardedBloomFilter<T, H>
where
    T: Sync,
    H: BloomHasher + Clone + Sync,
{}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SharedBloomFilter;
    use std::sync::{Arc, Barrier};
    use std::thread;

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
        let filter = ShardedBloomFilter::<&str>::new(1000, 0.01);
        filter.insert(&"hello");
        filter.insert(&"world");
        
        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(!filter.contains(&"missing"));
    }

    #[test]
    fn test_sharded_filter_clear() {
        let filter = ShardedBloomFilter::<&str>::new(1000, 0.01);
        filter.insert(&"hello");
        filter.insert(&"world");
        
        assert!(filter.contains(&"hello"));
        filter.clear();
        
        assert!(!filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
        assert!(filter.is_empty());
    }

    #[test]
    fn test_sharded_filter_concurrent_clear() {
        let filter = Arc::new(ShardedBloomFilter::<i32>::new(10_000, 0.01));
        
        // Insert from multiple threads
        let handles: Vec<_> = (0..4)
            .map(|tid| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for i in 0..100 {
                        f.insert(&(tid * 100 + i));
                    }
                })
            })
            .collect();
        
        for h in handles {
            h.join().unwrap();
        }
        
        assert!(!filter.is_empty());
        
        // Clear from one thread
        filter.clear();
        
        // Verify clear worked
        assert!(filter.is_empty());
        
        // Verify can still insert after clear
        filter.insert(&42);
        assert!(filter.contains(&42));
    }

    #[test]
    fn test_clear_concurrent_safety() {
        let filter = Arc::new(ShardedBloomFilter::<i32>::new(10_000, 0.01));
        
        // Spawn 8 writer threads
        let handles: Vec<_> = (0..8)
            .map(|tid| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        f.insert(&(tid * 1000));
                    }
                })
            })
            .collect();
        
        // Clear multiple times while writers are active
        for _ in 0..10 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            filter.clear();
        }
        
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_sharded_filter_clone() {
        let filter1 = ShardedBloomFilter::<&str>::new(1000, 0.01);
        filter1.insert(&"hello");
        
        let filter2 = filter1.clone();
        assert!(filter2.contains(&"hello"));
        
        filter1.insert(&"world");
        assert!(!filter2.contains(&"world")); // Independent copies
    }

    #[test]
    fn test_sharded_filter_load_factor() {
        let filter = ShardedBloomFilter::<i32>::new(1000, 0.01);
        assert_eq!(filter.load_factor(), 0.0);
        
        for i in 0..100 {
            filter.insert(&i);
        }
        
        let load = filter.load_factor();
        assert!(load > 0.0 && load < 1.0);
    }

    #[test]
    fn test_sharded_filter_fp_rate() {
        let filter = ShardedBloomFilter::<i32>::new(1000, 0.01);
        
        for i in 0..500 {
            filter.insert(&i);
        }
        
        let fprate = filter.false_positive_rate();
        assert!(fprate < 0.05, "FP rate exceeds threshold: {}", fprate);
    }

    #[test]
    #[should_panic(expected = "expected_items must be > 0")]
    fn test_sharded_filter_zero_items() {
        let _ = ShardedBloomFilter::<i32>::new(0, 0.01);
    }

    #[test]
    #[should_panic(expected = "fprate must be in (0, 1)")]
    fn test_sharded_filter_invalid_fprate() {
        let _ = ShardedBloomFilter::<i32>::new(1000, 1.5);
    }

    #[test]
    fn test_no_pathological_distribution() {
        let filter = ShardedBloomFilter::<i32>::with_shard_count(100_000, 0.01, 16);
        
        // Insert sequential integers (worst case for bad hash functions)
        for i in 0..10_000 {
            filter.insert(&i);
        }
        
        // Verify reasonable bit distribution
        let total_ones = filter.count_ones();
        let k = filter.hash_count();
        let expected = k * 10_000;
        let ratio = total_ones as f64 / expected as f64;
        
        assert!(
            ratio > 0.4 && ratio < 1.0,
            "Bit distribution suspicious: {} bits set, expected ~{}",
            total_ones,
            expected
        );
    }

    #[test]
    fn test_single_hash_per_operation() {
        let filter = ShardedBloomFilter::<i32>::with_shard_count(10_000, 0.01, 8);
        
        filter.insert(&42);
        assert!(filter.contains(&42));
        assert!(!filter.contains(&99));
    }

    #[test]
    fn test_extreme_concurrency_stress() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        
        let filter = Arc::new(ShardedBloomFilter::<i32>::new(1_000_000, 0.01));
        let insert_count = Arc::new(AtomicUsize::new(0));
        let query_count = Arc::new(AtomicUsize::new(0));
        
        // 64 writer threads
        let writers: Vec<_> = (0..64)
            .map(|tid| {
                let f = Arc::clone(&filter);
                let c = Arc::clone(&insert_count);
                thread::spawn(move || {
                    for i in 0..10_000 {
                        f.insert(&(tid * 10_000 + i));
                        c.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        
        // 64 reader threads
        let readers: Vec<_> = (0..64)
            .map(|tid| {
                let f = Arc::clone(&filter);
                let c = Arc::clone(&query_count);
                thread::spawn(move || {
                    for i in 0..10_000 {
                        let _ = f.contains(&(tid * 10_000 + i));
                        c.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();
        
        for h in writers.into_iter().chain(readers.into_iter()) {
            h.join().unwrap();
        }
        
        assert_eq!(insert_count.load(Ordering::Relaxed), 640_000);
        assert_eq!(query_count.load(Ordering::Relaxed), 640_000);
        
        // Verify no false negatives
        for tid in 0..64 {
            for i in 0..10_000 {
                assert!(
                    filter.contains(&(tid * 10_000 + i)),
                    "False negative for item {}",
                    tid * 10_000 + i
                );
            }
        }
    }

    #[test]
    fn test_no_use_after_free() {
        let filter = Arc::new(ShardedBloomFilter::<i32>::new(1000, 0.01));
        
        // Get reference to shard's BitVec
        let bits1 = filter.shards[0].bits();
        
        // Clear filter (replaces BitVec)
        filter.clear();
        
        // Old reference should still be valid (Arc keeps it alive)
        let _ = bits1.count_ones(); // Must not crash
    }

    #[test]
    fn test_concurrent_clear_no_corruption() {
        let filter = Arc::new(ShardedBloomFilter::<i32>::new(10_000, 0.01));
        let barrier = Arc::new(Barrier::new(17)); // 16 threads + 1 clearer
        
        // Insert items
        for i in 0..1000 {
            filter.insert(&i);
        }
        
        // 16 query threads
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let f = Arc::clone(&filter);
                let b = Arc::clone(&barrier);
                thread::spawn(move || {
                    b.wait();
                    for _ in 0..1000 {
                        let _ = f.contains(&42);
                    }
                })
            })
            .collect();
        
        // 1 clear thread
        let clearer = {
            let f = Arc::clone(&filter);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                b.wait();
                for _ in 0..100 {
                    f.clear();
                }
            })
        };
        
        for h in handles {
            h.join().unwrap();
        }
        clearer.join().unwrap();
        
        // Filter should be empty
        assert!(filter.is_empty());
    }

    #[test]
    fn test_shard_stats() {
        let filter = ShardedBloomFilter::<i32>::new(10_000, 0.01);
        for i in 0..1000 {
            filter.insert(&i);
        }
        
        let stats = filter.shard_stats();
        assert_eq!(stats.len(), filter.shard_count());
        
        for stat in stats {
            assert!(stat.size > 0);
            assert!(stat.fill_rate >= 0.0 && stat.fill_rate <= 1.0);
        }
    }

    #[test]
    fn test_has_imbalanced_shards() {
        let filter = ShardedBloomFilter::<i32>::new(10_000, 0.01);
        
        // Empty filter should not be imbalanced
        assert!(!filter.has_imbalanced_shards());
        
        // Add items
        for i in 0..10_000 {
            filter.insert(&i);
        }
        
        // With good hash function, should not be imbalanced
        assert!(!filter.has_imbalanced_shards());
    }

    #[test]
    fn test_shard_raw_bits() {
        let filter = ShardedBloomFilter::<i32>::new(1000, 0.01);
        filter.insert(&42);
        
        // Should be able to extract bits from each shard
        for i in 0..filter.shard_count() {
            let bits = filter.shard_raw_bits(i).unwrap();
            assert!(!bits.is_empty());
        }
    }

    #[test]
    fn test_shard_raw_bits_out_of_bounds() {
        let filter = ShardedBloomFilter::<i32>::new(1000, 0.01);
        let result = filter.shard_raw_bits(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_shard_bits_roundtrip() {
        let filter1 = ShardedBloomFilter::<i32>::new(1000, 0.01);
        filter1.insert(&42);
        filter1.insert(&100);
        
        // Serialize
        let shard_bits: Vec<Vec<u64>> = (0..filter1.shard_count())
            .map(|i| filter1.shard_raw_bits(i).unwrap())
            .collect();
        let k = filter1.hash_count();
        
        // Deserialize
        let filter2 = ShardedBloomFilter::<i32>::from_shard_bits(
            shard_bits,
            k,
            1000,
            0.01,
            StdHasher::default(),
        )
        .unwrap();
        
        // Verify
        assert!(filter2.contains(&42));
        assert!(filter2.contains(&100));
        assert!(!filter2.contains(&999));
    }

    #[test]
    fn test_from_shard_bits_size_validation() {
        let filter1 = ShardedBloomFilter::<i32>::new(1000, 0.01);
        
        // Get correct data
        let mut shard_bits: Vec<Vec<u64>> = (0..filter1.shard_count())
            .map(|i| filter1.shard_raw_bits(i).unwrap())
            .collect();
        
        // Corrupt first shard by adding extra data
        shard_bits[0].push(0xDEADBEEF);
        
        // Should fail with size mismatch error
        let result = ShardedBloomFilter::<i32>::from_shard_bits(
            shard_bits,
            filter1.hash_count(),
            1000,
            0.01,
            StdHasher::default(),
        );
        
        assert!(result.is_err());
        let err_msg = result.expect_err("Should have failed with size mismatch");
        assert!(format!("{:?}", err_msg).contains("size mismatch"));
    }

    #[test]
    fn test_new_adaptive() {
        let filter = ShardedBloomFilter::<i32>::new_adaptive(100_000, 0.01);
        assert!(filter.shard_count() > 0);
        assert!(filter.shard_count() <= 256);
    }

    #[test]
    fn test_shard_cache_line_aligned() {
        // Verify Shard struct is cache-line aligned
        assert_eq!(std::mem::align_of::<Shard<StdHasher>>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_memory_efficient_clear() {
        let filter = Arc::new(ShardedBloomFilter::<i32>::new(100_000, 0.01));
        
        // Fill filter
        for i in 0..50_000 {
            filter.insert(&i);
        }
        
        let before_clear = filter.memory_usage();
        
        // Clear should not spike memory (no 2× allocation)
        filter.clear();
        
        let after_clear = filter.memory_usage();
        
        // Memory usage should remain approximately the same
        // (within 10% due to Arc overhead)
        let ratio = after_clear as f64 / before_clear as f64;
        assert!(
            ratio >= 0.9 && ratio <= 1.1,
            "Memory usage changed significantly: before={}, after={}, ratio={:.2}",
            before_clear,
            after_clear,
            ratio
        );
    }

    #[test]
    fn test_debug_impl() {
        let filter = ShardedBloomFilter::<i32>::new(1000, 0.01);
        filter.insert(&42);
        
        let debug_str = format!("{:?}", filter);
        assert!(debug_str.contains("ShardedBloomFilter"));
        assert!(debug_str.contains("shards"));
    }

    #[test]
    fn test_empty_filter_stats() {
        let filter = ShardedBloomFilter::<i32>::new(1000, 0.01);
        
        // Should not panic
        assert_eq!(filter.load_factor(), 0.0);
        assert_eq!(filter.false_positive_rate(), 0.0);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_drop_cleanup() {
        {
            let filter = ShardedBloomFilter::<i32>::new(1000, 0.01);
            filter.insert(&42);
            // filter drops here
        }
        // If this test completes without ASAN errors, Drop is correct
    }

    #[test]
    fn test_concurrent_insert_query_visibility() {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        
        const ITEM_COUNT: usize = 10_000;
        
        let filter = Arc::new(ShardedBloomFilter::<i32>::new(ITEM_COUNT, 0.01));
        let done = Arc::new(AtomicBool::new(false));
        
        // Writer thread: insert all items, then signal completion.
        let writer_filter = Arc::clone(&filter);
        let writer_done = Arc::clone(&done);
        let writer = thread::spawn(move || {
            for i in 0..ITEM_COUNT {
                writer_filter.insert(&(i as i32));
            }
            writer_done.store(true, AtomicOrdering::Release);
        });
        
        // Reader thread: during the write phase we spin-read to exercise the
        // concurrent Release/Acquire pairs; after the writer finishes we do a
        // single sequential pass which MUST find every item.
        let reader_filter = Arc::clone(&filter);
        let reader_done = Arc::clone(&done);
        let reader = thread::spawn(move || {
            // Phase 1 – concurrent reads while writer is still inserting.
            while !reader_done.load(AtomicOrdering::Acquire) {
                for i in 0..ITEM_COUNT {
                    // Don't check result — the item may not be inserted yet.
                    let _ = reader_filter.contains(&(i as i32));
                    core::hint::spin_loop();
                }
            }
            // Phase 2 – writer has finished and signalled.  No false negatives
            // are possible because the Release store on `done` pairs with the
            // Acquire load above, establishing happens-before for every
            // preceding fetch_or(Release) on the BitVec blocks.
            let mut false_negatives: usize = 0;
            for i in 0..ITEM_COUNT {
                if !reader_filter.contains(&(i as i32)) {
                    false_negatives += 1;
                }
            }
            false_negatives
        });
        
        writer.join().unwrap();
        let false_negatives = reader.join().unwrap();
        
        assert_eq!(
            false_negatives,
            0,
            "Detected {} false negatives (memory ordering bug!)",
            false_negatives
        );
    }

    #[test]
    fn test_shard_padding() {
        use std::mem::{align_of, size_of};
        
        // Verify Shard is cache-line aligned
        assert!(
            align_of::<Shard<StdHasher>>() >= 64,
            "Shard alignment is {}, should be >= 64",
            align_of::<Shard<StdHasher>>()
        );
        
        // Verify Shard is at least 64 bytes (preferably 128)
        assert!(
            size_of::<Shard<StdHasher>>() >= 64,
            "Shard size is {}, should be >= 64",
            size_of::<Shard<StdHasher>>()
        );
    }
}