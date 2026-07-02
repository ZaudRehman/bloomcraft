//! Striped locking Bloom filter for concurrent access.
//!
//! Partitions the filter into independent lock stripes. Operations lock only the
//! stripe their hash maps to, enabling concurrent access to different stripes.
//!
//! Stripe selection uses Lemire's fast range reduction:
//! `((hash as u128 * num_stripes as u128) >> 64) as usize`.
//!
//! | Operation    | Locks Acquired      | Lock Type |
//! |--------------|--------------------|-----------|
//! | `insert()`   | Single stripe      | Write     |
//! | `contains()` | Single stripe      | Read      |
//! | `clear()`    | All stripes + bits | Write     |
//!
//! # Examples
//!
//! ```
//! use bloomcraft::sync::StripedBloomFilter;
//! use bloomcraft::core::SharedBloomFilter;
//!
//! let filter = StripedBloomFilter::<String>::new(10000, 0.01)?;
//! filter.insert(&"hello".to_string());
//! assert!(filter.contains(&"hello".to_string()));
//! # Ok::<(), bloomcraft::error::BloomCraftError>(())
//! ```
//!
//! ## Concurrent Access
//!
//! ```
//! use bloomcraft::sync::StripedBloomFilter;
//! use bloomcraft::core::SharedBloomFilter;
//! use std::sync::Arc;
//!
//! let filter = Arc::new(StripedBloomFilter::<String>::new(10000, 0.01)?);
//! let f = Arc::clone(&filter);
//! let handle = std::thread::spawn(move || f.insert(&"concurrent".to_string()));
//! handle.join().unwrap();
//! assert!(filter.contains(&"concurrent".to_string()));
//! # Ok::<(), bloomcraft::error::BloomCraftError>(())
//! ```
//!
//! ## Adaptive Stripe Count
//!
//! ```
//! use bloomcraft::sync::StripedBloomFilter;
//!
//! let filter = StripedBloomFilter::<u64>::with_concurrency(
//!     10000, 0.01, 64
//! )?;
//! assert_eq!(filter.stripe_count(), 256);
//! # Ok::<(), bloomcraft::error::BloomCraftError>(())
//! ```
//!
//! # References
//!
//! - Herlihy, M., & Shavit, N. (2012). *The Art of Multiprocessor Programming*. Morgan Kaufmann. 
//! - Lea, D. (2004). *The Concurrency Utilities ConcurrentHashMap Design*.

use crate::core::SharedBloomFilter;
use crate::core::{params, BitVec};
use crate::error::{BloomCraftError, Result};
use crate::hash::{BloomHasher, EnhancedDoubleHashing, StdHasher};
use crate::hash::strategies::HashStrategy;
use parking_lot::RwLock;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(feature = "metrics")]
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Intermediate step: hash `T` to 8 bytes, then feed to `BloomHasher::hash_bytes_pair`.
///
/// This two-phase approach decouples item serialization from the Bloom-specific
/// hashing strategy. The final index computation is performed by
/// `EnhancedDoubleHashing`.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher as StdDefaultHasher;
    use std::hash::Hasher;
    let mut hasher = StdDefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}

/// Default stripe count: balances contention vs memory (~16 KB for 256 × 64-byte locks).
const DEFAULT_STRIPE_COUNT: usize = 256;

/// Cache-line size for false-sharing prevention (x86_64 cache line).
const CACHE_LINE_SIZE: usize = 64;

/// Cache-line-padded RwLock to prevent false sharing between adjacent stripes.
#[repr(align(64))]
struct PaddedRwLock {
    lock: RwLock<()>,

    #[cfg(feature = "metrics")]
    read_count: AtomicU64,

    #[cfg(feature = "metrics")]
    write_count: AtomicU64,

    #[cfg(feature = "metrics")]
    contention_ns: AtomicU64,

    _padding: [u8; CACHE_LINE_SIZE
        - std::mem::size_of::<RwLock<()>>()
        - if cfg!(feature = "metrics") { 24 } else { 0 }],
}

/// Compile-time size check: PaddedRwLock must fit in one cache line.
const _: [(); 1] = [(); (std::mem::size_of::<PaddedRwLock>() == CACHE_LINE_SIZE) as usize];

impl PaddedRwLock {
    const fn new() -> Self {
        Self {
            lock: RwLock::new(()),
            #[cfg(feature = "metrics")]
            read_count: AtomicU64::new(0),
            #[cfg(feature = "metrics")]
            write_count: AtomicU64::new(0),
            #[cfg(feature = "metrics")]
            contention_ns: AtomicU64::new(0),
            _padding: [0; CACHE_LINE_SIZE
                - std::mem::size_of::<RwLock<()>>()
                - if cfg!(feature = "metrics") { 24 } else { 0 }],
        }
    }

    /// Record a read operation (requires `metrics` feature).
    #[cfg(feature = "metrics")]
    #[inline]
    fn record_read(&self) {
        self.read_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record a write operation (requires `metrics` feature).
    #[cfg(feature = "metrics")]
    #[inline]
    fn record_write(&self) {
        self.write_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record lock contention time (requires `metrics` feature).
    #[cfg(feature = "metrics")]
    #[inline]
    fn record_contention(&self, nanos: u64) {
        self.contention_ns
            .fetch_add(nanos, AtomicOrdering::Relaxed);
    }
}

/// Per-stripe statistics (requires `metrics` feature).
#[cfg(feature = "metrics")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StripeStats {
    pub stripe_idx: usize,
    pub read_ops: u64,
    pub write_ops: u64,
    pub contention_ns: u64,
}

/// Striped locking Bloom filter for concurrent access.
///
/// Uses lock striping to enable concurrent inserts and queries on a shared
/// bit vector. Each operation locks only the stripe its hash maps to,
/// providing memory-efficient concurrent access without a full table lock.
///
/// # Type Parameters
///
/// * `T` — Item type. Must implement `Hash + Send + Sync`.
/// * `H` — Hash function. Defaults to `StdHasher`.
pub struct StripedBloomFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone + Default,
{
    bits: RwLock<Arc<BitVec>>,
    stripes: Box<[PaddedRwLock]>,
    num_hashes: usize,
    size: usize,
    hasher: Arc<H>,
    expected_items: usize,
    target_fpr: f64,
    _marker: PhantomData<T>,
}

impl<T, H> StripedBloomFilter<T, H>
where
    H: BloomHasher + Clone + Default,
{
    /// Create a filter with the default stripe count (256).
    pub fn new(expected_items: usize, fprate: f64) -> Result<Self> {
        Self::with_stripe_count(expected_items, fprate, DEFAULT_STRIPE_COUNT)
    }

    /// Create a filter with an explicit stripe count.
    ///
    /// Recommended stripe counts by concurrency level:
    ///
    /// | Threads | Stripes |
    /// |---------|---------|
    /// | 1–4     | 16–32   |
    /// | 8–16    | 64–128  |
    /// | 32–64   | 256–512 |
    /// | 100+    | 1024–4096 |
    ///
    /// Returns an error if `expected_items == 0`, `fprate` is outside `(0, 1)`,
    /// or `num_stripes == 0`.
    pub fn with_stripe_count(
        expected_items: usize,
        fprate: f64,
        num_stripes: usize,
    ) -> Result<Self> {
        if expected_items == 0 {
            return Err(BloomCraftError::invalid_item_count(expected_items));
        }
        if fprate <= 0.0 || fprate >= 1.0 {
            return Err(BloomCraftError::fp_rate_out_of_bounds(fprate));
        }
        if num_stripes == 0 {
            return Err(BloomCraftError::invalid_parameters(
                "num_stripes must be greater than 0",
            ));
        }

        let size = params::optimal_bit_count(expected_items, fprate)?;
        let num_hashes = params::optimal_hash_count(size, expected_items)?;

        let stripes: Box<[PaddedRwLock]> = (0..num_stripes)
            .map(|_| PaddedRwLock::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let bits_vec = BitVec::new(size)?;
        let bits = RwLock::new(Arc::new(bits_vec));

        Ok(Self {
            bits,
            stripes,
            num_hashes,
            size,
            hasher: Arc::new(H::default()),
            expected_items,
            target_fpr: fprate,
            _marker: PhantomData,
        })
    }

    /// Create a filter with stripe count chosen adaptively.
    ///
    /// Uses the heuristic `clamp(next_power_of_2(threads × 4), 16, 4096)` where
    /// `threads` is the expected concurrency level.
    pub fn with_concurrency(
        expected_items: usize,
        fprate: f64,
        concurrency_level: usize,
    ) -> Result<Self> {
        let base = (concurrency_level * 4).max(16);
        let num_stripes = base.next_power_of_two().min(4096);
        Self::with_stripe_count(expected_items, fprate, num_stripes)
    }

    /// Number of lock stripes.
    #[inline]
    #[must_use]
    pub fn stripe_count(&self) -> usize {
        self.stripes.len()
    }

    /// Select a stripe via Lemire's fast range reduction: `(hash × n) >> 64`.
    #[inline]
    fn select_stripe_from_hash(&self, hash: u64) -> usize {
        let n = self.stripes.len() as u64;
        debug_assert!(n > 0);
        let idx = ((hash as u128 * n as u128) >> 64) as usize;
        debug_assert!(idx < self.stripes.len());
        idx
    }

    /// Snapshot the current BitVec (cheap Arc clone under read lock).
    #[inline]
    fn bits(&self) -> Arc<BitVec> {
        Arc::clone(&*self.bits.read())
    }

    /// Estimated memory usage in bytes (bit vector + stripes + struct overhead).
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.bits.read().memory_usage()
            + self.stripes.len() * std::mem::size_of::<PaddedRwLock>()
            + std::mem::size_of::<Self>()
    }

    /// Number of set bits. Acquires the bit-vector read lock.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.bits.read().count_ones()
    }

    /// Ratio of set bits to total bits in `[0.0, 1.0]`.
    #[must_use]
    pub fn load_factor(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        self.count_ones() as f64 / self.size as f64
    }

    /// Target FPR passed at construction.
    #[must_use]
    pub fn target_fpr(&self) -> f64 {
        self.target_fpr
    }

    /// Expected item count passed at construction.
    #[must_use]
    pub fn expected_items_configured(&self) -> usize {
        self.expected_items
    }

    /// Hasher type name.
    #[must_use]
    pub fn hasher_name(&self) -> &'static str {
        self.hasher.name()
    }

    /// Per-stripe metrics. Requires `metrics` feature.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn stripe_stats(&self) -> Vec<StripeStats> {
        self.stripes
            .iter()
            .enumerate()
            .map(|(idx, stripe)| StripeStats {
                stripe_idx: idx,
                read_ops: stripe.read_count.load(AtomicOrdering::Relaxed),
                write_ops: stripe.write_count.load(AtomicOrdering::Relaxed),
                contention_ns: stripe.contention_ns.load(AtomicOrdering::Relaxed),
            })
            .collect()
    }

    /// Returns the `top_n` most contended stripe indices (requires `metrics` feature).
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn most_contended_stripes(&self, top_n: usize) -> Vec<usize> {
        let mut stats = self.stripe_stats();
        stats.sort_by_key(|s| std::cmp::Reverse(s.contention_ns));
        stats
            .into_iter()
            .take(top_n)
            .map(|s| s.stripe_idx)
            .collect()
    }

    /// Export the full bit vector as raw `u64` words.
    ///
    /// Acquires the bit-vector read lock and returns a snapshot. The length of
    /// the returned vector is `⌈size / 64⌉`. Use this together with
    /// [`from_raw_bits`](Self::from_raw_bits) for serialization.
    #[must_use]
    pub fn raw_bits(&self) -> Vec<u64> {
        let bits = self.bits();
        bits.to_raw()
    }

    /// Reconstruct a filter from raw serialised data.
    ///
    /// # Arguments
    ///
    /// * `bits` — Raw bit-vector words produced by [`raw_bits`](Self::raw_bits).
    /// * `num_hashes` — Hash function count (*k*).
    /// * `stripe_count` — Number of lock stripes.
    /// * `expected_items` — Capacity the filter was sized for.
    /// * `target_fpr` — Target false-positive rate.
    /// * `hasher` — Hasher instance.
    ///
    /// # Errors
    ///
    /// * `InvalidParameters` — `stripe_count == 0`, or `bits` length doesn't
    ///   match the expected word count derived from `expected_items × target_fpr`.
    /// * `FalsePositiveRateOutOfBounds` — `target_fpr` not in `(0, 1)`.
    ///
    /// # Example
    ///
    /// ```
    /// use bloomcraft::sync::StripedBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    /// use bloomcraft::core::SharedBloomFilter;
    ///
    /// let filter = StripedBloomFilter::<String>::new(1000, 0.01).unwrap();
    /// filter.insert(&"test".to_string());
    ///
    /// let bits = filter.raw_bits();
    /// let k = filter.hash_count();
    /// let restored = StripedBloomFilter::<String>::from_raw_bits(
    ///     bits, k, filter.stripe_count(), 1000, 0.01, StdHasher::default(),
    /// ).unwrap();
    /// assert!(restored.contains(&"test".to_string()));
    /// ```
    pub fn from_raw_bits(
        bits: Vec<u64>,
        num_hashes: usize,
        stripe_count: usize,
        expected_items: usize,
        target_fpr: f64,
        hasher: H,
    ) -> Result<Self> {
        if stripe_count == 0 {
            return Err(BloomCraftError::invalid_parameters(
                "stripe_count must be greater than 0",
            ));
        }

        let size = params::optimal_bit_count(expected_items, target_fpr)?;

        let expected_words = size.div_ceil(64);
        if bits.len() != expected_words {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Bit vector size mismatch: got {} words, expected {} words for {} items at {:.6} FPR",
                bits.len(), expected_words, expected_items, target_fpr
            )));
        }

        let bitvec = BitVec::from_raw(bits, size)?;

        let stripes: Box<[PaddedRwLock]> = (0..stripe_count)
            .map(|_| PaddedRwLock::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            bits: RwLock::new(Arc::new(bitvec)),
            stripes,
            num_hashes,
            size,
            hasher: Arc::new(hasher),
            expected_items,
            target_fpr,
            _marker: PhantomData,
        })
    }
}

impl<T, H> SharedBloomFilter<T> for StripedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
    fn insert(&self, item: &T) {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let stripe_idx = self.select_stripe_from_hash(h1);
        let bits = self.bits();

        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        let _guard = self.stripes[stripe_idx].lock.write();
        #[cfg(feature = "metrics")]
        {
            self.stripes[stripe_idx].record_write();
            self.stripes[stripe_idx]
                .record_contention(start.elapsed().as_nanos() as u64);
        }

        let indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, self.num_hashes, self.size);
        for idx in indices {
            bits.as_ref().set(idx);
        }
    }

    fn contains(&self, item: &T) -> bool {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let stripe_idx = self.select_stripe_from_hash(h1);
        let bits = self.bits();

        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        let _guard = self.stripes[stripe_idx].lock.read();
        #[cfg(feature = "metrics")]
        {
            self.stripes[stripe_idx].record_read();
            self.stripes[stripe_idx]
                .record_contention(start.elapsed().as_nanos() as u64);
        }

        let indices = EnhancedDoubleHashing.generate_indices(h1, h2, 0, self.num_hashes, self.size);
        indices.iter().all(|idx| bits.as_ref().get(*idx))
    }

    /// Acquires all stripe locks in ascending order (deadlock-free via
    /// resource ordering), replaces the bit vector with a fresh empty one.
    fn clear(&self) {
        let _guards: Vec<_> = self
            .stripes
            .iter()
            .map(|stripe| stripe.lock.write())
            .collect();

        let new_bits = Arc::new(
            BitVec::new(self.size).expect("BitVec allocation failed in clear()")
        );
        *self.bits.write() = new_bits;
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
        let m = self.size as f64;
        let k = self.num_hashes as f64;
        let estimated_n = -(m / k) * (1.0 - fill_rate).ln();
        let exponent = -k * estimated_n / m;
        (1.0 - exponent.exp()).powf(k)
    }

    fn estimate_count(&self) -> usize {
        let ones = self.count_ones() as f64;
        let m = self.size as f64;
        let k = self.num_hashes as f64;
        if ones == 0.0 {
            return 0;
        }
        let fill_ratio = ones / m;
        if fill_ratio >= 1.0 {
            return self.size;
        }
        ((-m / k) * (1.0 - fill_ratio).ln()).round() as usize
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

impl<T, H> Clone for StripedBloomFilter<T, H>
where
    H: BloomHasher + Clone + Default,
{
    /// Deep copy: independent BitVec and stripes; shares hasher via Arc.
    fn clone(&self) -> Self {
        let stripes: Box<[PaddedRwLock]> = (0..self.stripes.len())
            .map(|_| PaddedRwLock::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let bits = {
            let guard = self.bits.read();
            Arc::new((**guard).clone())
        };

        Self {
            bits: RwLock::new(bits),
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

impl<T, H> std::fmt::Debug for StripedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StripedBloomFilter")
            .field("size", &self.size)
            .field("num_hashes", &self.num_hashes)
            .field("num_stripes", &self.stripes.len())
            .field("expected_items", &self.expected_items)
            .field("target_fpr", &self.target_fpr)
            .field("load_factor", &self.load_factor())
            .field("estimated_fpr", &self.false_positive_rate())
            .finish()
    }
}

impl<T, H> std::fmt::Display for StripedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StripedBloomFilter({} stripes, {} bits, k={}, load={:.1}%, FPR={:.4}%)",
            self.stripes.len(),
            self.size,
            self.num_hashes,
            self.load_factor() * 100.0,
            self.false_positive_rate() * 100.0
        )
    }
}

// All shared state is protected by RwLocks; BitVec uses atomics internally.
unsafe impl<T, H> Send for StripedBloomFilter<T, H> where H: BloomHasher + Clone + Default {}
unsafe impl<T, H> Sync for StripedBloomFilter<T, H> where H: BloomHasher + Clone + Default {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SharedBloomFilter;

    #[test]
    fn test_constructor_error_handling() {
        assert!(StripedBloomFilter::<u64>::new(0, 0.01).is_err());
        assert!(StripedBloomFilter::<u64>::new(1000, 0.0).is_err());
        assert!(StripedBloomFilter::<u64>::new(1000, 1.0).is_err());
        assert!(StripedBloomFilter::<u64>::new(1000, 1.5).is_err());
        assert!(StripedBloomFilter::<u64>::with_stripe_count(1000, 0.01, 0).is_err());
    }

    #[test]
    fn test_valid_construction() {
        let filter = StripedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.expected_items_configured(), 1000);
        assert_eq!(filter.target_fpr(), 0.01);
        assert_eq!(filter.stripe_count(), 256);
    }

    #[test]
    fn test_single_hash_per_operation() {
        let filter = StripedBloomFilter::<u64>::with_stripe_count(10000, 0.01, 256).unwrap();
        filter.insert(&42);
        assert!(filter.contains(&42));
        assert!(!filter.contains(&99));
    }

    #[test]
    fn test_striped_filter_insert_contains() {
        let filter = StripedBloomFilter::<&str>::new(1000, 0.01).unwrap();
        filter.insert(&"hello");
        filter.insert(&"world");
        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(!filter.contains(&"missing"));
    }

    #[test]
    fn test_striped_filter_concurrent_safety() {
        use std::thread;
        let filter = Arc::new(StripedBloomFilter::<u64>::new(10000, 0.01).unwrap());
        let handles: Vec<_> = (0..8)
            .map(|tid| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for i in 0..1000 { f.insert(&(tid * 1000 + i)); }
                })
            })
            .collect();
        for h in handles { h.join().unwrap(); }
        for tid in 0..8 {
            for i in 0..100 { assert!(filter.contains(&(tid * 1000 + i))); }
        }
    }

    #[test]
    fn test_cache_line_padding() {
        assert_eq!(std::mem::size_of::<PaddedRwLock>(), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_clear_operation() {
        let filter = StripedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        filter.insert(&42);
        filter.insert(&100);
        assert!(!filter.is_empty());
        filter.clear();
        assert!(filter.is_empty());
        assert!(!filter.contains(&42));
        assert!(!filter.contains(&100));
    }

    #[test]
    fn test_adaptive_concurrency() {
        assert_eq!(StripedBloomFilter::<u64>::with_concurrency(10000, 0.01, 1).unwrap().stripe_count(), 16);
        assert_eq!(StripedBloomFilter::<u64>::with_concurrency(10000, 0.01, 8).unwrap().stripe_count(), 32);
        assert_eq!(StripedBloomFilter::<u64>::with_concurrency(10000, 0.01, 64).unwrap().stripe_count(), 256);
        assert_eq!(StripedBloomFilter::<u64>::with_concurrency(10000, 0.01, 2048).unwrap().stripe_count(), 4096);
    }

    #[test]
    fn test_clone_independence() {
        let filter = StripedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        filter.insert(&42);
        let cloned = filter.clone();
        assert!(filter.contains(&42) && cloned.contains(&42));
        filter.insert(&100);
        assert!(filter.contains(&100) && !cloned.contains(&100));
        cloned.insert(&200);
        assert!(cloned.contains(&200) && !filter.contains(&200));
    }

    #[test]
    fn test_debug_display() {
        let filter = StripedBloomFilter::<String>::new(1000, 0.01).unwrap();
        filter.insert(&"test".to_string());
        assert!(format!("{:?}", filter).contains("StripedBloomFilter"));
        assert!(format!("{}", filter).contains("stripes"));
    }

    #[test]
    fn test_memory_usage() {
        let usage = StripedBloomFilter::<u64>::new(10000, 0.01).unwrap().memory_usage();
        assert!(usage > 1000 && usage < 1_000_000);
    }

    #[test]
    fn test_load_factor() {
        let filter = StripedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.load_factor(), 0.0);
        for i in 0..100 { filter.insert(&i); }
        assert!(filter.load_factor() > 0.0 && filter.load_factor() < 1.0);
    }

    #[test]
    fn test_false_positive_rate_estimation() {
        let filter = StripedBloomFilter::<u64>::new(10000, 0.01).unwrap();
        assert_eq!(filter.false_positive_rate(), 0.0);
        for i in 0..1000 { filter.insert(&i); }
        assert!(filter.false_positive_rate() > 0.0 && filter.false_positive_rate() < 0.05);
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_stripe_metrics() {
        let filter = StripedBloomFilter::<u64>::with_stripe_count(1000, 0.01, 16).unwrap();
        for i in 0..100 { filter.insert(&i); let _ = filter.contains(&i); }
        let stats = filter.stripe_stats();
        assert_eq!(stats.len(), 16);
        assert!(stats.iter().map(|s| s.read_ops + s.write_ops).sum::<u64>() > 0);
        assert!(filter.most_contended_stripes(5).len() <= 5);
    }

    #[test]
    fn test_concurrent_clear() {
        use std::thread;
        let filter = Arc::new(StripedBloomFilter::<u64>::new(10000, 0.01).unwrap());
        for i in 0..1000 { filter.insert(&i); }
        let handles: Vec<_> = (0..4)
            .map(|_| { let f = Arc::clone(&filter); thread::spawn(move || f.clear()) })
            .collect();
        for h in handles { h.join().unwrap(); }
        assert!(filter.is_empty());
    }

    #[test]
    fn test_no_false_negatives() {
        let filter = StripedBloomFilter::<String>::new(10000, 0.01).unwrap();
        let items: Vec<String> = (0..500).map(|i| format!("item{}", i)).collect();
        for item in &items { filter.insert(item); }
        for item in &items { assert!(filter.contains(item)); }
    }

    #[test]
    fn test_thread_safety_markers() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StripedBloomFilter<String>>();
    }
}