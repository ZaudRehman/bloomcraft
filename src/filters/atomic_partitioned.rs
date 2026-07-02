//! Concurrent partitioned Bloom filter using `AtomicU64` for lock-free operations.
//!
//! # When to Use
//!
//! | Workload | Recommendation |
//! |---|---|
//! | Read-heavy (≥90% queries) | Excellent — reads scale with thread count |
//! | Write-heavy or balanced | Prefer [`ShardedBloomFilter`](crate::sync::ShardedBloomFilter) |
//! | Build-once, query-many | Good — single-threaded build, concurrent queries |
//!
//! # Key Properties
//!
//! - Every insert touches all `k` partitions (one per hash function).
//! - Write throughput is bounded by cache-line contention on these `k` partitions
//!   and does **not** scale with thread count.
//! - Read throughput scales near-linearly (`AtomicU64::load` is read-shared).
//! - All operations use `Ordering::Relaxed` — correct for Bloom filter semantics
//!   where false positives are acceptable and bit-sets are idempotent.
//!
//! # References
//!
//! - Putze, F., Sanders, P., & Singler, J. (2009). "Cache-, Hash- and
//!   Space-Efficient Bloom Filters". *J. Experimental Algorithmics*, 14, 4.
//! - Lemire, D. (2019). "Fast Random Integer Generation in an Interval".
//!   *ACM TOMS*, 45(3).

#![cfg(feature = "concurrent")]

use crate::core::filter::{BloomFilter, ConcurrentBloomFilter};
use crate::core::params::{optimal_bit_count, optimal_hash_count, validate_params};
use crate::error::{BloomCraftError, Result};
use crate::hash::{BloomHasher, StdHasher};
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Cache line size for modern x86-64 processors (bytes).
const DEFAULT_CACHE_LINE_SIZE: usize = 64;

/// Concurrent partitioned Bloom filter with lock-free insert and query operations.
///
/// The bit array is split into `k` cache-aligned partitions of `AtomicU64`, one per
/// hash function. Each insert sets one bit in every partition; each query checks one
/// bit per partition and returns `false` on the first miss.
///
/// # Performance
///
/// | Metric | Characteristic |
/// |---|---|
/// | Read scaling | Near-linear with thread count (read-shared `AtomicU64`) |
/// | Write scaling | **Does not scale** (bounded by `fetch_or` contention) |
/// | Single-thread insert | Depends on `k` and item size |
/// | Partition count | Higher `k` = lower FPR, proportionally slower inserts |
///
/// # Caveats
///
/// - Write throughput is bounded by `fetch_or` contention on `k` shared cache lines.
///   2 threads can be *slower* than 1 due to cache-line ping-pong.
/// - Alignment (64B vs 4096B) has negligible effect — the bottleneck is not false sharing.
/// - Over-filling beyond design capacity degrades query speed and FPR exponentially.
/// - For write-heavy workloads, prefer [`ShardedBloomFilter`](crate::sync::ShardedBloomFilter).
///
/// # Memory Ordering
///
/// All operations use `Ordering::Relaxed`. This is correct because Bloom filter bits
/// are idempotent (false positives are acceptable, false negatives are impossible),
/// and no inter-thread causality is required.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "concurrent")]
/// # {
/// use bloomcraft::filters::AtomicPartitionedBloomFilter;
/// use bloomcraft::core::ConcurrentBloomFilter;
/// use std::sync::Arc;
/// use std::thread;
///
/// let filter = Arc::new(
///     AtomicPartitionedBloomFilter::<String>::new(1_000_000, 0.01).unwrap()
/// );
///
/// let handles: Vec<_> = (0..8).map(|tid| {
///     let f = Arc::clone(&filter);
///     thread::spawn(move || {
///         for i in 0..10_000 {
///             f.insert_concurrent(&format!("item_{}_{}", tid, i));
///         }
///     })
/// }).collect();
///
/// for handle in handles {
///     handle.join().unwrap();
/// }
/// # }
/// ```
#[derive(Debug)]
pub struct AtomicPartitionedBloomFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone + Default,
{
    /// Base pointer to cache-aligned allocation of `AtomicU64`.
    data: NonNull<AtomicU64>,
    /// Number of partitions (equals k, number of hash functions).
    k: usize,
    /// Size of each partition in bits.
    partition_size: usize,
    /// Stride between partitions in AtomicU64 words (includes padding).
    partition_stride: usize,
    /// Cache alignment in bytes.
    alignment: usize,
    /// Total allocated size in bytes.
    allocated_bytes: usize,
    /// Hash function instance.
    hasher: H,
    /// Expected number of items.
    expected_items: usize,
    /// Target false positive rate.
    target_fpr: f64,
    /// Actual number of items inserted (atomic).
    item_count: AtomicUsize,
    /// Phantom data for type parameter T.
    _phantom: PhantomData<T>,
}

impl<T, H> AtomicPartitionedBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Create a new atomic partitioned Bloom filter.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items (n > 0)
    /// * `fpr` - Target false positive rate (0 < fpr < 1)
    ///
    /// # Returns
    ///
    /// * `Ok(AtomicPartitionedBloomFilter)` - Lock-free concurrent filter
    /// * `Err(BloomCraftError)` - If parameters invalid or allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "concurrent")]
    /// # {
    /// use bloomcraft::filters::AtomicPartitionedBloomFilter;
    ///
    /// let filter = AtomicPartitionedBloomFilter::<u64>::new(100_000, 0.01).unwrap();
    /// # Ok::<(), bloomcraft::BloomCraftError>(())
    /// # ;
    /// # }
    /// ```
    pub fn new(expected_items: usize, fpr: f64) -> Result<Self> {
        Self::with_hasher_and_alignment(
            expected_items,
            fpr,
            H::default(),
            DEFAULT_CACHE_LINE_SIZE,
        )
    }

    /// Create with custom hasher and alignment.
    pub fn with_hasher_and_alignment(
        expected_items: usize,
        fpr: f64,
        hasher: H,
        alignment: usize,
    ) -> Result<Self> {
        // Validate inputs
        if expected_items == 0 {
            return Err(BloomCraftError::invalid_item_count(expected_items));
        }
        if fpr <= 0.0 || fpr >= 1.0 {
            return Err(BloomCraftError::fp_rate_out_of_bounds(fpr));
        }
        if !alignment.is_power_of_two() {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Alignment {} must be power of 2",
                alignment
            )));
        }

        // Calculate optimal parameters
        let m = optimal_bit_count(expected_items, fpr)?;
        let k = optimal_hash_count(m, expected_items)?;
        validate_params(m, expected_items, k)?;

        // Calculate partition size
        let base_partition_size = (m + k - 1) / k;
        let alignment_bits = alignment * 8;
        let partition_size = ((base_partition_size + alignment_bits - 1) / alignment_bits)
            * alignment_bits;

        // Calculate stride
        let partition_bytes = (partition_size + 7) / 8;
        let partition_stride_bytes = ((partition_bytes + alignment - 1) / alignment) * alignment;
        let partition_stride = partition_stride_bytes / 8; // In u64 words

        // Allocate single flat buffer for AtomicU64
        let total_bytes = partition_stride_bytes * k;

        if total_bytes == 0 || total_bytes > isize::MAX as usize {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Invalid allocation size: {}",
                total_bytes
            )));
        }

        let layout = Layout::from_size_align(total_bytes, alignment)
            .map_err(|e| BloomCraftError::invalid_parameters(format!("Invalid layout: {}", e)))?;

        // SAFETY: Layout is valid, size > 0, alignment is power of 2
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        // Zero memory
        unsafe {
            std::ptr::write_bytes(ptr, 0, total_bytes);
        }

        // Cast to AtomicU64 pointer
        let data = NonNull::new(ptr as *mut AtomicU64).expect("Allocation returned null");

        Ok(Self {
            data,
            k,
            partition_size,
            partition_stride,
            alignment,
            allocated_bytes: total_bytes,
            hasher,
            expected_items,
            target_fpr: fpr,
            item_count: AtomicUsize::new(0),
            _phantom: PhantomData,
        })
    }

    /// Get pointer to partition i's start.
    #[inline]
    fn partition_ptr(&self, partition_idx: usize) -> *const AtomicU64 {
        debug_assert!(partition_idx < self.k);
        unsafe { self.data.as_ptr().add(partition_idx * self.partition_stride) }
    }

    /// Get bit at index within partition (atomic load).
    #[inline]
    unsafe fn get_bit_atomic(&self, partition_idx: usize, bit_idx: usize) -> bool {
        debug_assert!(bit_idx < self.partition_size);
        let ptr = self.partition_ptr(partition_idx);
        let word_idx = bit_idx / 64;
        let bit_offset = bit_idx % 64;
        let word = (*ptr.add(word_idx)).load(Ordering::Relaxed);
        (word & (1u64 << bit_offset)) != 0
    }

    /// Set bit at index within partition (atomic fetch_or).
    #[inline]
    unsafe fn set_bit_atomic(&self, partition_idx: usize, bit_idx: usize) {
        debug_assert!(bit_idx < self.partition_size);
        let ptr = self.partition_ptr(partition_idx);
        let word_idx = bit_idx / 64;
        let bit_offset = bit_idx % 64;
        let mask = 1u64 << bit_offset;

        // SAFETY: Atomic fetch_or is lock-free and wait-free
        // Ordering::Relaxed is sufficient because:
        // - Bit-set operations are idempotent
        // - No inter-thread causality required
        (*ptr.add(word_idx)).fetch_or(mask, Ordering::Relaxed);
    }

    /// Unbiased hash to range using Lemire's method.
    #[inline]
    fn hash_to_range(hash: u64, range: usize) -> usize {
        ((hash as u128 * range as u128) >> 64) as usize
    }

    /// Hash item using BloomHasher trait's canonical bridge.
    #[inline]
    fn hash_item(&self, item: &T) -> (u64, u64) {
        self.hasher.hash_item(item)
    }

    /// Get partition count.
    #[inline]
    pub const fn partition_count(&self) -> usize {
        self.k
    }

    /// Get partition size in bits.
    #[inline]
    pub const fn partition_size(&self) -> usize {
        self.partition_size
    }

    /// Get alignment in bytes.
    #[inline]
    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    /// Get target FPR.
    #[inline]
    pub const fn target_fpr(&self) -> f64 {
        self.target_fpr
    }

    /// Get expected items.
    #[inline]
    pub const fn expected_items(&self) -> usize {
        self.expected_items
    }

    /// Get atomic item count.
    #[inline]
    pub fn item_count(&self) -> usize {
        self.item_count.load(Ordering::Relaxed)
    }

    /// Calculate filter saturation (0.0 to 1.0).
    pub fn saturation(&self) -> f64 {
        let mut total_set = 0;
        for partition_idx in 0..self.k {
            let ptr = self.partition_ptr(partition_idx);
            let words = (self.partition_size + 63) / 64;
            for word_idx in 0..words {
                unsafe {
                    let word = (*ptr.add(word_idx)).load(Ordering::Relaxed);
                    total_set += word.count_ones() as usize;
                }
            }
        }
        total_set as f64 / (self.k * self.partition_size) as f64
    }

    /// Estimate actual FPR based on saturation.
    pub fn estimated_fpr(&self) -> f64 {
        let n = self.item_count() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let fill_rate = 1.0 - (-n / self.partition_size as f64).exp();
        fill_rate.powi(self.k as i32)
    }
}

// Implement BloomFilter trait (for non-concurrent operations)
impl<T, H> BloomFilter<T> for AtomicPartitionedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn insert(&mut self, item: &T) {
        // Delegate to concurrent version (we have &mut self, so exclusive access)
        self.insert_concurrent(item);
    }

    fn contains(&self, item: &T) -> bool {
        let (h1, h2) = self.hash_item(item);
        for i in 0..self.k {
            let hash = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = Self::hash_to_range(hash, self.partition_size);
            if !unsafe { self.get_bit_atomic(i, bit_idx) } {
                return false;
            }
        }
        true
    }

    fn clear(&mut self) {
        // Zero all memory (exclusive access required)
        unsafe {
            std::ptr::write_bytes(self.data.as_ptr() as *mut u8, 0, self.allocated_bytes);
        }
        self.item_count.store(0, Ordering::Relaxed);
    }

    fn is_empty(&self) -> bool {
        self.item_count() == 0
    }

    fn len(&self) -> usize {
        self.item_count()
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimated_fpr()
    }

    fn expected_items(&self) -> usize {
        self.expected_items
    }

    fn bit_count(&self) -> usize {
        self.k * self.partition_size
    }

    fn hash_count(&self) -> usize {
        self.k
    }

    fn estimate_count(&self) -> usize {
        if self.saturation() < 0.01 {
            return self.item_count();
        }

        let mut total_set = 0;
        for partition_idx in 0..self.k {
            let ptr = self.partition_ptr(partition_idx);
            let words = (self.partition_size + 63) / 64;
            for word_idx in 0..words {
                let word = unsafe { (*ptr.add(word_idx)).load(Ordering::Relaxed) };
                total_set += word.count_ones() as usize;
            }
        }

        let x = total_set as f64;
        let m = (self.k * self.partition_size) as f64;
        let k = self.k as f64;
        let estimated = -(m / k) * (1.0 - x / m).ln();
        estimated.max(0.0) as usize
    }

    // In impl<T, H> BloomFilter<T> for AtomicPartitionedBloomFilter<T, H>
    fn count_set_bits(&self) -> usize {
        let mut total = 0usize;
        for partition_idx in 0..self.k {
            let ptr = self.partition_ptr(partition_idx);
            let words = (self.partition_size + 63) / 64;
            for word_idx in 0..words {
                // SAFETY: same invariants as partition_ptr documentation.
                // Relaxed ordering is sufficient — bit counts are advisory,
                // not used for synchronization decisions.
                total += unsafe {
                    (*ptr.add(word_idx)).load(Ordering::Relaxed)
                }.count_ones() as usize;
            }
        }
        total
    }
}

// Implement ConcurrentBloomFilter trait (lock-free operations)
impl<T, H> ConcurrentBloomFilter<T> for AtomicPartitionedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + Send + Sync,
{
    fn insert_concurrent(&self, item: &T) {
        let (h1, h2) = self.hash_item(item);
        for i in 0..self.k {
            let hash = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = Self::hash_to_range(hash, self.partition_size);
            unsafe {
                self.set_bit_atomic(i, bit_idx);
            }
        }
        self.item_count.fetch_add(1, Ordering::Relaxed);
    }
    
    // In impl<T, H> ConcurrentBloomFilter<T> for AtomicPartitionedBloomFilter<T, H>
    fn contains_concurrent(&self, item: &T) -> bool {
        // contains() is already lock-free (atomic loads with Relaxed ordering)
        // and safe for concurrent invocation. Delegation is semantically correct.
        self.contains(item)
    }
}

// Drop implementation
impl<T, H> Drop for AtomicPartitionedBloomFilter<T, H>
where
    H: BloomHasher + Clone + Default,
{
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(self.allocated_bytes, self.alignment)
                .expect("Drop: Layout must match allocation");
            dealloc(self.data.as_ptr() as *mut u8, layout);
        }
    }
}

// Clone implementation
impl<T, H> Clone for AtomicPartitionedBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    fn clone(&self) -> Self {
        let layout = Layout::from_size_align(self.allocated_bytes, self.alignment)
            .expect("Clone: Layout must be valid");
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.data.as_ptr() as *const u8,
                ptr,
                self.allocated_bytes,
            );
        }
        let data = NonNull::new(ptr as *mut AtomicU64).expect("Allocation returned null");

        Self {
            data,
            k: self.k,
            partition_size: self.partition_size,
            partition_stride: self.partition_stride,
            alignment: self.alignment,
            allocated_bytes: self.allocated_bytes,
            hasher: self.hasher.clone(),
            expected_items: self.expected_items,
            target_fpr: self.target_fpr,
            item_count: AtomicUsize::new(self.item_count()),
            _phantom: PhantomData,
        }
    }
}

// Thread safety markers
unsafe impl<T, H> Send for AtomicPartitionedBloomFilter<T, H>
where
    T: Send,
    H: BloomHasher + Clone + Default + Send,
{
}

unsafe impl<T, H> Sync for AtomicPartitionedBloomFilter<T, H>
where
    T: Sync,
    H: BloomHasher + Clone + Default + Sync,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_concurrent_insert() {
        let filter = AtomicPartitionedBloomFilter::<String>::new(1000, 0.01).unwrap();

        filter.insert_concurrent(&"hello".to_string());
        filter.insert_concurrent(&"world".to_string());

        assert!(filter.contains(&"hello".to_string()));
        assert!(filter.contains(&"world".to_string()));
        assert!(!filter.contains(&"goodbye".to_string()));
    }

    #[test]
    fn test_concurrent_inserts() {
        let filter = Arc::new(
            AtomicPartitionedBloomFilter::<u64>::new(10_000, 0.01).unwrap()
        );

        let handles: Vec<_> = (0..8).map(|tid| {
            let f = Arc::clone(&filter);
            thread::spawn(move || {
                for i in 0..1000 {
                    f.insert_concurrent(&(tid * 1000 + i));
                }
            })
        }).collect();

        for handle in handles {
            handle.join().unwrap();
        }

        for tid in 0..8 {
            for i in 0..1000 {
                assert!(filter.contains(&(tid * 1000 + i)));
            }
        }

        assert_eq!(filter.item_count(), 8000);
    }

    #[test]
    fn test_no_false_negatives_concurrent() {
        let filter = Arc::new(
            AtomicPartitionedBloomFilter::<u64>::new(5000, 0.01).unwrap()
        );

        let items: Vec<u64> = (0..5000).collect();

        let handles: Vec<_> = items.chunks(1000).enumerate().map(|(_tid, chunk)| {
            let f = Arc::clone(&filter);
            let chunk = chunk.to_vec();
            thread::spawn(move || {
                for &item in &chunk {
                    f.insert_concurrent(&item);
                }
            })
        }).collect();

        for handle in handles {
            handle.join().unwrap();
        }

        for &item in &items {
            assert!(filter.contains(&item), "False negative for {}", item);
        }
    }

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<AtomicPartitionedBloomFilter<u64>>();
        assert_sync::<AtomicPartitionedBloomFilter<u64>>();
    }

    #[test]
    fn test_cache_alignment() {
        let filter = AtomicPartitionedBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        let alignment = filter.alignment();
        assert!(alignment.is_power_of_two());
        let base_ptr = filter.data.as_ptr() as usize;
        assert_eq!(
            base_ptr % alignment,
            0,
            "Base pointer not {}-byte aligned",
            alignment
        );
        for i in 0..filter.partition_count() {
            let ptr = filter.partition_ptr(i) as usize;
            assert_eq!(
                ptr % alignment,
                0,
                "Partition {} not {}-byte aligned",
                i,
                alignment
            );
        }
    }

    #[test]
    fn test_parameter_validation() {
        assert!(AtomicPartitionedBloomFilter::<u64>::new(0, 0.01).is_err());
        assert!(AtomicPartitionedBloomFilter::<u64>::new(100, 0.0).is_err());
        assert!(AtomicPartitionedBloomFilter::<u64>::new(100, 1.0).is_err());
        assert!(AtomicPartitionedBloomFilter::<u64>::new(100, -0.1).is_err());
        assert!(AtomicPartitionedBloomFilter::<u64>::new(100, 1.5).is_err());

        let result = AtomicPartitionedBloomFilter::<u64>::with_hasher_and_alignment(
            100, 0.01, StdHasher::new(), 3, // not power of 2
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_saturation_and_fpr() {
        let filter = AtomicPartitionedBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        for i in 0..10_000 {
            filter.insert_concurrent(&i);
        }
        let sat = filter.saturation();
        assert!(
            sat > 0.2 && sat < 0.8,
            "Saturation {} out of expected range [0.2, 0.8]",
            sat
        );
        let estimated = filter.estimated_fpr();
        let ratio = estimated / 0.01;
        assert!(
            ratio < 5.0,
            "Estimated FPR {:.4} is too far from target 0.01 (ratio {:.2})",
            estimated,
            ratio
        );
    }

    #[test]
    fn test_clear_exclusive() {
        let mut filter = AtomicPartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        filter.insert_concurrent(&42);
        filter.insert_concurrent(&43);
        assert!(!filter.is_empty());

        filter.clear();
        assert!(filter.is_empty());
        assert_eq!(filter.len(), 0);
        assert!(!filter.contains(&42));
        assert!(!filter.contains(&43));
    }

    #[test]
    fn test_drop_safety() {
        {
            let filter = AtomicPartitionedBloomFilter::<u64>::new(10_000, 0.01).unwrap();
            for i in 0..1000 {
                filter.insert_concurrent(&i);
            }
        }
    }

    #[test]
    fn test_clone_independence() {
        let filter = AtomicPartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..50 {
            filter.insert_concurrent(&i);
        }
        let cloned = filter.clone();
        for i in 0..50 {
            assert!(cloned.contains(&i), "Clone missing item {}", i);
        }
        filter.insert_concurrent(&999);
        assert!(filter.contains(&999));
        assert!(!cloned.contains(&999));
        cloned.insert_concurrent(&888);
        assert!(cloned.contains(&888));
        assert!(!filter.contains(&888));
    }

    #[test]
    fn test_clone_many_items() {
        let filter = AtomicPartitionedBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        for i in 0..5000 {
            filter.insert_concurrent(&i);
        }
        let cloned = filter.clone();
        let mut false_negatives = 0;
        for i in 0..5000 {
            if !cloned.contains(&i) {
                false_negatives += 1;
            }
        }
        assert_eq!(false_negatives, 0, "Clone has false negatives");
    }

    #[test]
    fn test_item_count_atomic() {
        let filter = Arc::new(
            AtomicPartitionedBloomFilter::<u64>::new(10_000, 0.01).unwrap()
        );
        let thread_count = 8;
        let items_per_thread = 1000;

        let handles: Vec<_> = (0..thread_count).map(|tid| {
            let f = Arc::clone(&filter);
            thread::spawn(move || {
                for i in 0..items_per_thread {
                    f.insert_concurrent(&(tid * items_per_thread + i));
                }
            })
        }).collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let total = thread_count * items_per_thread;
        assert_eq!(filter.item_count(), total as usize);
    }

    #[test]
    fn test_bloom_filter_trait_insert() {
        let mut filter = AtomicPartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        filter.insert(&42);
        assert!(filter.contains(&42));
        assert!(!filter.contains(&99));
    }

    #[test]
    fn test_bloom_filter_trait_methods() {
        let mut filter = AtomicPartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert!(filter.is_empty());
        assert_eq!(filter.len(), 0);
        assert_eq!(filter.expected_items(), 1000);
        assert!(filter.hash_count() > 0);
        assert!(filter.bit_count() > 0);
        filter.insert(&42);
        assert!(!filter.is_empty());
        assert_eq!(filter.len(), 1);
        let fpr = filter.false_positive_rate();
        assert!(fpr >= 0.0);
    }

    #[test]
    fn test_estimate_count() {
        let filter = AtomicPartitionedBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        for i in 0..1000 {
            filter.insert_concurrent(&i);
        }
        let estimated = filter.estimate_count();
        let error = (estimated as i64 - 1000).abs() as f64 / 1000.0;
        assert!(
            error < 0.3,
            "Estimation error {:.1}% exceeds 30%",
            error * 100.0
        );
    }

    #[test]
    fn test_count_set_bits() {
        let filter = AtomicPartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.count_set_bits(), 0);
        filter.insert_concurrent(&42);
        assert!(filter.count_set_bits() > 0);
        assert!(filter.count_set_bits() <= filter.bit_count());
    }

    #[test]
    fn test_zero_item_count_after_clear() {
        let mut filter = AtomicPartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        filter.insert_concurrent(&1);
        filter.insert_concurrent(&2);
        filter.clear();
        assert_eq!(filter.item_count(), 0);
    }

    #[test]
    fn test_partition_count() {
        let filter = AtomicPartitionedBloomFilter::<u64>::new(100, 0.01).unwrap();
        assert_eq!(filter.partition_count(), filter.hash_count());
    }

    #[test]
    fn test_default_hasher_is_std_hasher() {
        let filter: AtomicPartitionedBloomFilter<u64> =
            AtomicPartitionedBloomFilter::new(100, 0.01).unwrap();
        filter.insert_concurrent(&42);
        assert!(filter.contains(&42));
    }

    #[test]
    fn test_multiple_drops() {
        let filters: Vec<_> = (0..10)
            .map(|_| AtomicPartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap())
            .collect();
        drop(filters);
    }
}
