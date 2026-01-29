//! Lock-free concurrent variant of PartitionedBloomFilter
//!
//! `AtomicPartitionedBloomFilter` provides wait-free concurrent inserts and queries
//! without requiring external synchronization (Mutex/RwLock).

#![cfg(feature = "concurrent")]
#![allow(clippy::pedantic)]

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

/// Atomic partitioned Bloom filter with lock-free concurrent operations.
///
/// This variant uses `AtomicU64` for all bit storage, enabling:
/// - **Wait-free inserts**: No thread blocks, ever
/// - **Lock-free queries**: Atomic loads only
/// - **Memory safety**: Proper ordering guarantees
///
/// # Concurrency Model
///
/// - `insert_concurrent(&self, ...)`: Lock-free, can be called from multiple threads
/// - `contains(&self, ...)`: Lock-free, thread-safe
/// - `clear(&mut self)`: Requires exclusive access (rare operation)
///
/// # Performance
///
/// Expected scaling with `Arc<AtomicPartitionedBloomFilter>`:
/// - 2 threads: 1.8-2.0× throughput
/// - 4 threads: 3.5-3.8× throughput
/// - 8 threads: 6.5-7.5× throughput
/// - 16 threads: 12-14× throughput
///
/// # Memory Ordering
///
/// All operations use `Ordering::Relaxed` because:
/// - False positives are acceptable (Bloom filter semantics)
/// - No inter-thread causality needed
/// - Bit-set operations are idempotent
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
    /// let filter = AtomicPartitionedBloomFilter::<u64>::new(100_000, 0.01)?;
    /// # Ok::<(), bloomcraft::BloomCraftError>(())
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

    /// Hash item using BloomHasher trait for two independent values.
    #[inline]
    fn hash_item(&self, item: &T) -> (u64, u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut h = DefaultHasher::new();
        item.hash(&mut h);
        let item_hash = h.finish();
        let bytes = item_hash.to_le_bytes();
        self.hasher.hash_bytes_pair(&bytes)
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

        // Verify all items present (no false negatives)
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

        // Concurrent insert
        let handles: Vec<_> = items.chunks(1000).enumerate().map(|(tid, chunk)| {
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

        // Verify no false negatives
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
}
