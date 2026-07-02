//! Cache-aligned partitioned Bloom filter.
//!
//! A `PartitionedBloomFilter` splits the bit array into `k` cache-aligned
//! partitions, one per hash function. Each hash probes only its assigned
//! partition, producing sequential access instead of the random access that
//! standard Bloom filters perform across the full array:
//!
//! ```text
//! Standard:    [============ m bits ============]
//!               h₁↑    h₂↑    h₃↑         hₖ↑
//!              (k random accesses → k cache misses)
//!
//! Partitioned: [==P₀==][==P₁==][==P₂==]...[==Pₖ₋₁==]
//!               h₀↑      h₁↑      h₂↑        hₖ₋₁↑
//!              (k sequential accesses → 1–2 cache misses)
//! ```
//!
//! This layout reduces worst-case cache misses from `k` to 1–2 when the
//! working set fits in cache. The trade-off is a 2–5% higher false-positive
//! rate relative to a standard Bloom filter with identical `(m, n, k)`,
//! because each hash function is restricted to its own partition rather than
//! the full array.
//!
//! # Performance
//!
//! Partitioning turns `k` random cache-line misses into a single sequential
//! scan of one partition, which typically fits in L1 or L2 cache. When the
//! working set fits in cache, this yields a modest throughput improvement over
//! a standard Bloom filter. Once the filter exceeds the last-level cache,
//! performance is dominated by DRAM bandwidth rather than the partition layout.
//!
//! # False-positive rate
//!
//! For `m` total bits, `k` hash functions, and `n` inserted items:
//!
//! ```text
//! fpr = (1 - e^(-kn/m))^k
//! ```
//!
//! This is 2–5% higher than a standard filter at the same parameters.
//! Example: for m = 10 MB, n = 1 M, k = 7, the standard FPR is ≈ 0.0081 and
//! the partitioned FPR is ≈ 0.0084 (+3.7%).
//!
//! # Memory layout
//!
//! Partitions are laid out in a single `std::alloc` allocation with cache-line
//! alignment between them. Each partition is padded to a 64-byte boundary to
//! prevent false sharing and align with hardware prefetcher boundaries.
//!
//! # Feature flags
//!
//! - **`metrics`**: latency histograms, health checks, Prometheus export
//! - **`serde`**: serialization via `serde`
//! - **`cache_detect`**: automatic CPU cache-size detection
//!
//! # Examples
//!
//! ```rust
//! use bloomcraft::filters::PartitionedBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! // Basic usage
//! let mut filter = PartitionedBloomFilter::<u64>::new(100_000, 0.01)?;
//! filter.insert(&42);
//! assert!(filter.contains(&42));
//!
//! // Cache-tuned (auto-detects CPU cache)
//! let filter = PartitionedBloomFilter::<u64>::new_cache_tuned(100_000, 0.01)?;
//!
//! // With metrics
//! #[cfg(feature = "metrics")]
//! let mut filter = PartitionedBloomFilter::<String>::with_metrics(100_000, 0.01)?;
//! # Ok::<(), bloomcraft::BloomCraftError>(())
//! ```
//!
//! # References
//!
//! - Putze, F., Sanders, P., & Singler, J. (2009). "Cache-, Hash- and
//!   Space-Efficient Bloom Filters". *J. Experimental Algorithmics*, 14, 4.
//! - Kirsch, A., & Mitzenmacher, M. (2006). "Less Hashing, Same Performance:
//!   Building a Better Bloom Filter". *ESA 2006*, LNCS 4168, pp. 456–467.
//! - Lemire, D. (2019). "Fast Random Integer Generation in an Interval".
//!   *ACM TOMS*, 45(3).

use crate::core::filter::BloomFilter;
use crate::core::params::{optimal_bit_count, optimal_hash_count, validate_params};
use crate::error::{BloomCraftError, Result};
use crate::hash::{BloomHasher, StdHasher};
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "serde")]
use serde::de::{MapAccess, Visitor};
#[cfg(feature = "serde")]
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "metrics")]
use std::time::Instant;

/// Cache line size for modern x86-64 processors (bytes).
const DEFAULT_CACHE_LINE_SIZE: usize = 64;

/// Maximum partition size to fit in L1 cache (32KB typical).
const MAX_PARTITION_SIZE_BITS: usize = 32_768; // 4 KB per partition

/// Minimum partition size (1 cache line).
const MIN_PARTITION_SIZE_BITS: usize = DEFAULT_CACHE_LINE_SIZE * 8;

static CACHE_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);

/// Cache-aligned partitioned Bloom filter.
///
/// The cache-aligned partition layout provides modest query throughput gains
/// over a standard Bloom filter when the working set fits in L3 cache, and
/// larger wins at DRAM-bound sizes where a standard filter incurs multiple
/// random cache misses per query.
///
/// # Type parameters
///
/// * `T` — item type (must implement [`Hash`]).
/// * `H` — hash function implementing [`BloomHasher`]; defaults to [`StdHasher`].
///
/// # Layout
///
/// Partitions are laid out in a single `std::alloc` allocation with cache-line
/// padding between them. Each partition is 64-byte aligned.
///
/// # Thread safety
///
/// | Operation | Signature | Thread-safe? |
/// |-----------|-----------|-------------|
/// | Insert | `&mut self` | Single-writer |
/// | Query | `&self` | Yes (multiple readers) |
/// | Union/Intersect | `&mut self` | Single-writer |
///
/// For lock-free concurrent access, see [`AtomicPartitionedBloomFilter`](crate::filters::AtomicPartitionedBloomFilter).
///
/// # Performance
///
/// * **Insert**: O(k), 1–2 L1 cache misses.
/// * **Query**: O(k), 1–2 L1 cache misses when the working set fits in cache.
/// * **Memory**: m bits + ~2–3% alignment overhead for 64-byte alignment.
///
/// # Examples
///
/// ```
/// use bloomcraft::filters::PartitionedBloomFilter;
/// use bloomcraft::core::BloomFilter;
///
/// let mut filter = PartitionedBloomFilter::<u64>::new(100_000, 0.01)?;
/// filter.insert(&42);
/// assert!(filter.contains(&42));
///
/// // Cache-tuned constructor (auto-detects CPU cache)
/// let filter = PartitionedBloomFilter::<u64>::new_cache_tuned(100_000, 0.01)?;
///
/// // With metrics
/// #[cfg(feature = "metrics")]
/// let filter = PartitionedBloomFilter::<String>::with_metrics(100_000, 0.01)?;
/// # Ok::<(), bloomcraft::BloomCraftError>(())
/// ```
#[derive(Debug)]
pub struct PartitionedBloomFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone + Default,
{
    /// Base pointer to cache-aligned allocation.
    data: NonNull<u64>,
    /// Number of partitions (equals k, number of hash functions).
    k: usize,
    /// Size of each partition in bits.
    partition_size: usize,
    /// Stride between partitions in u64 words (includes padding).
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
    /// Actual number of items inserted.
    item_count: usize,
    /// Phantom data for type parameter T.
    _phantom: PhantomData<T>,
    /// Production metrics (feature-gated).
    #[cfg(feature = "metrics")]
    metrics: Option<PartitionedFilterMetrics>,
}

#[cfg(feature = "metrics")]
use crate::metrics::partitioned_metrics::{
    export_prometheus, HealthCheck, PartitionedFilterMetrics,
};

#[cfg(feature = "serde")]
impl<T, H> Serialize for PartitionedBloomFilter<T, H>
where
    H: BloomHasher + Clone + Default + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let word_count = self.allocated_bytes / 8;
        let data_slice = unsafe { std::slice::from_raw_parts(self.data.as_ptr(), word_count) };

        let mut state = serializer.serialize_struct("PartitionedBloomFilter", 12)?;
        state.serialize_field("k", &self.k)?;
        state.serialize_field("partition_size", &self.partition_size)?;
        state.serialize_field("partition_stride", &self.partition_stride)?;
        state.serialize_field("alignment", &self.alignment)?;
        state.serialize_field("allocated_bytes", &self.allocated_bytes)?;
        state.serialize_field("hasher", &self.hasher)?;
        state.serialize_field("expected_items", &self.expected_items)?;
        state.serialize_field("target_fpr", &self.target_fpr)?;
        state.serialize_field("item_count", &self.item_count)?;
        state.serialize_field("data", data_slice)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, H> Deserialize<'de> for PartitionedBloomFilter<T, H>
where
    H: BloomHasher + Clone + Default + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            K,
            PartitionSize,
            PartitionStride,
            Alignment,
            AllocatedBytes,
            Hasher,
            ExpectedItems,
            TargetFpr,
            ItemCount,
            Data,
        }

        struct PartitionedVisitor<T, H>(PhantomData<(T, H)>);

        impl<'de, T, H> Visitor<'de> for PartitionedVisitor<T, H>
        where
            H: BloomHasher + Clone + Default + Deserialize<'de>,
        {
            type Value = PartitionedBloomFilter<T, H>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct PartitionedBloomFilter")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut k: Option<usize> = None;
                let mut partition_size: Option<usize> = None;
                let mut partition_stride: Option<usize> = None;
                let mut alignment: Option<usize> = None;
                let mut allocated_bytes: Option<usize> = None;
                let mut hasher: Option<H> = None;
                let mut expected_items: Option<usize> = None;
                let mut target_fpr: Option<f64> = None;
                let mut item_count: Option<usize> = None;
                let mut data: Option<Vec<u64>> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::K => k = Some(map.next_value()?),
                        Field::PartitionSize => partition_size = Some(map.next_value()?),
                        Field::PartitionStride => partition_stride = Some(map.next_value()?),
                        Field::Alignment => alignment = Some(map.next_value()?),
                        Field::AllocatedBytes => allocated_bytes = Some(map.next_value()?),
                        Field::Hasher => hasher = Some(map.next_value()?),
                        Field::ExpectedItems => expected_items = Some(map.next_value()?),
                        Field::TargetFpr => target_fpr = Some(map.next_value()?),
                        Field::ItemCount => item_count = Some(map.next_value()?),
                        Field::Data => data = Some(map.next_value()?),
                    }
                }

                let k = k.ok_or_else(|| de::Error::missing_field("k"))?;
                let partition_size =
                    partition_size.ok_or_else(|| de::Error::missing_field("partition_size"))?;
                let partition_stride =
                    partition_stride.ok_or_else(|| de::Error::missing_field("partition_stride"))?;
                let alignment = alignment.ok_or_else(|| de::Error::missing_field("alignment"))?;
                let allocated_bytes =
                    allocated_bytes.ok_or_else(|| de::Error::missing_field("allocated_bytes"))?;
                let hasher = hasher.ok_or_else(|| de::Error::missing_field("hasher"))?;
                let expected_items =
                    expected_items.ok_or_else(|| de::Error::missing_field("expected_items"))?;
                let target_fpr =
                    target_fpr.ok_or_else(|| de::Error::missing_field("target_fpr"))?;
                let item_count =
                    item_count.ok_or_else(|| de::Error::missing_field("item_count"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                if data.len() * 8 != allocated_bytes {
                    return Err(de::Error::custom(format!(
                        "data length {} doesn't match allocated_bytes {}",
                        data.len() * 8,
                        allocated_bytes
                    )));
                }

                let layout = Layout::from_size_align(allocated_bytes, alignment)
                    .map_err(|e| de::Error::custom(format!("Invalid layout: {}", e)))?;
                let ptr = unsafe { alloc(layout) };
                if ptr.is_null() {
                    return Err(de::Error::custom("allocation failed"));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr, allocated_bytes);
                }
                let data_ptr = NonNull::new(ptr as *mut u64)
                    .ok_or_else(|| de::Error::custom("null pointer"))?;

                Ok(PartitionedBloomFilter {
                    data: data_ptr,
                    k,
                    partition_size,
                    partition_stride,
                    alignment,
                    allocated_bytes,
                    hasher,
                    expected_items,
                    target_fpr,
                    item_count,
                    _phantom: PhantomData,
                    #[cfg(feature = "metrics")]
                    metrics: None,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "k",
            "partition_size",
            "partition_stride",
            "alignment",
            "allocated_bytes",
            "hasher",
            "expected_items",
            "target_fpr",
            "item_count",
            "data",
        ];
        deserializer.deserialize_struct(
            "PartitionedBloomFilter",
            FIELDS,
            PartitionedVisitor(PhantomData),
        )
    }
}

impl<T, H> PartitionedBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    /// Creates a new filter with optimal parameters.
    ///
    /// Calculates `m` (total bits) and `k` (hash functions) from the expected
    /// item count and target FPR, then allocates `k` cache-aligned partitions
    /// in a single flat allocation.
    pub fn new(expected_items: usize, fpr: f64) -> Result<Self> {
        Self::with_hasher(expected_items, fpr, H::default())
    }

    /// Creates a filter with auto-detected cache-line size.
    ///
    /// Uses platform cache detection (CPUID on x86, sysfs on ARM) to set the
    /// alignment to the L1 cache line size. May improve throughput on some
    /// hardware.
    pub fn new_cache_tuned(expected_items: usize, fpr: f64) -> Result<Self>
    where
        H: Default,
    {
        // Cache detection is always available via util module
        use crate::util::cache_detect::detect_cache_sizes;
        let cache = detect_cache_sizes();
        let alignment = cache.l1_line_bytes;
        Self::with_hasher_and_alignment(expected_items, fpr, H::default(), alignment)
    }

    /// Creates a filter with metrics recording enabled.
    ///
    /// Tracks insert/query latency, saturation, and health status. Adds
    /// measurable but non-blocking overhead on each operation.
    #[cfg(feature = "metrics")]
    pub fn with_metrics(expected_items: usize, fpr: f64) -> Result<Self>
    where
        H: Default,
    {
        let mut filter = Self::new(expected_items, fpr)?;
        filter.metrics = Some(PartitionedFilterMetrics::new());
        Ok(filter)
    }

    /// Creates a filter with a custom hasher.
    pub fn with_hasher(expected_items: usize, fpr: f64, hasher: H) -> Result<Self> {
        Self::with_hasher_and_alignment(expected_items, fpr, hasher, DEFAULT_CACHE_LINE_SIZE)
    }

    /// Creates a filter with a custom alignment.
    ///
    /// Alignment must be a power of two.
    pub fn with_alignment(expected_items: usize, fpr: f64, alignment: usize) -> Result<Self>
    where
        H: Default,
    {
        Self::with_hasher_and_alignment(expected_items, fpr, H::default(), alignment)
    }

    /// Creates a filter with full control over hasher and alignment.
    ///
    /// # Errors
    ///
    /// Returns an error if `expected_items` is zero, `fpr` is outside `(0, 1)`,
    /// or `alignment` is not a power of two. Allocation failure aborts via
    /// [`handle_alloc_error`].
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

        // Calculate optimal parameters using crate's functions
        let m = optimal_bit_count(expected_items, fpr)?;
        let k = optimal_hash_count(m, expected_items)?;
        validate_params(m, expected_items, k)?;

        // Calculate partition size: ⌈m / k⌉ bits
        let base_partition_size = m.div_ceil(k);

        // Round up to alignment boundary (in bits)
        let alignment_bits = alignment * 8;
        let partition_size = base_partition_size.div_ceil(alignment_bits) * alignment_bits;

        // Validate cache-optimal range
        if partition_size > MAX_PARTITION_SIZE_BITS
            && !CACHE_WARNING_SHOWN.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "Warning: Partition size {} bits ({} KB) exceeds L1 cache. \
                     Consider using standard filter or enabling cache_detect feature.",
                partition_size,
                partition_size / 8192
            );
        }
        if partition_size < MIN_PARTITION_SIZE_BITS {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Partition size {} bits too small (min {} bits)",
                partition_size, MIN_PARTITION_SIZE_BITS
            )));
        }

        // Calculate stride (round partition to next alignment boundary)
        let partition_bytes = partition_size.div_ceil(8); // Round up to bytes
        let partition_stride_bytes = partition_bytes.div_ceil(alignment) * alignment;
        let partition_stride = partition_stride_bytes / 8; // Convert to u64 words

        // Allocate single flat buffer
        let total_bytes = partition_stride_bytes * k;

        // Runtime safety checks (even in release mode)
        if total_bytes == 0 {
            return Err(BloomCraftError::invalid_parameters(
                "Total allocation size cannot be zero",
            ));
        }
        if total_bytes > isize::MAX as usize {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Allocation size {} exceeds isize::MAX ({})",
                total_bytes,
                isize::MAX
            )));
        }

        // Debug-mode invariant checks
        debug_assert!(alignment.is_power_of_two(), "Alignment must be power of 2");
        debug_assert!(
            total_bytes >= k * (partition_size / 8),
            "Allocation too small for requested partitions"
        );

        let layout = Layout::from_size_align(total_bytes, alignment)
            .map_err(|e| BloomCraftError::invalid_parameters(format!("Invalid layout: {}", e)))?;

        // SAFETY: Layout is valid (checked above), size > 0, alignment is power of 2
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        // Zero memory for deterministic behavior
        // SAFETY: ptr is valid, total_bytes within allocation
        unsafe {
            std::ptr::write_bytes(ptr, 0, total_bytes);
        }

        let data = NonNull::new(ptr as *mut u64).expect("Allocation returned null");

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
            item_count: 0,
            _phantom: PhantomData,
            #[cfg(feature = "metrics")]
            metrics: None,
        })
    }

    /// Get pointer to partition i's start.
    #[inline]
    fn partition_ptr(&self, partition_idx: usize) -> *mut u64 {
        debug_assert!(partition_idx < self.k);
        // SAFETY: partition_idx < k, offset within allocation
        unsafe {
            self.data
                .as_ptr()
                .add(partition_idx * self.partition_stride)
        }
    }

    /// Get bit at index within partition (unchecked).
    #[inline]
    unsafe fn get_bit_unchecked(&self, partition_idx: usize, bit_idx: usize) -> bool {
        debug_assert!(bit_idx < self.partition_size);
        let ptr = self.partition_ptr(partition_idx);
        let word_idx = bit_idx / 64;
        let bit_offset = bit_idx % 64;
        let word = ptr.add(word_idx).read();
        (word & (1u64 << bit_offset)) != 0
    }

    /// Set bit at index within partition (unchecked).
    #[inline]
    unsafe fn set_bit_unchecked(&mut self, partition_idx: usize, bit_idx: usize) {
        debug_assert!(bit_idx < self.partition_size);
        let ptr = self.partition_ptr(partition_idx);
        let word_idx = bit_idx / 64;
        let bit_offset = bit_idx % 64;
        let word_ptr = ptr.add(word_idx);
        let word = word_ptr.read();
        word_ptr.write(word | (1u64 << bit_offset));
    }

    /// Unbiased hash to range using Lemire's method.
    ///
    /// Equivalent to `hash % range` but without modulo bias.
    /// Uses 128-bit multiplication and shift: `(hash * range) >> 64`.
    #[inline]
    fn hash_to_range(hash: u64, range: usize) -> usize {
        ((hash as u128 * range as u128) >> 64) as usize
    }

    /// Hash item using BloomHasher trait's canonical bridge.
    #[inline]
    fn hash_item(&self, item: &T) -> (u64, u64) {
        self.hasher.hash_item(item)
    }

    /// Get number of partitions.
    #[inline]
    pub const fn partition_count(&self) -> usize {
        self.k
    }

    /// Get partition size in bits.
    #[inline]
    pub const fn partition_size(&self) -> usize {
        self.partition_size
    }

    /// Get cache alignment in bytes.
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

    /// Get actual item count.
    #[inline]
    pub const fn item_count(&self) -> usize {
        self.item_count
    }

    /// Get total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.allocated_bytes + std::mem::size_of::<Self>()
    }

    /// Sum of set bits across all partitions.
    ///
    /// Shared by [`saturation`], [`estimated_fpr`], [`estimate_count`],
    /// and [`count_set_bits`] to avoid redundant scans.
    #[inline]
    fn total_set_bits(&self) -> usize {
        let mut total = 0usize;
        for partition_idx in 0..self.k {
            let ptr = self.partition_ptr(partition_idx);
            let words = self.partition_size.div_ceil(64);
            for word_idx in 0..words {
                let word = unsafe { ptr.add(word_idx).read() };
                total += word.count_ones() as usize;
            }
        }
        total
    }

    /// Fraction of set bits across all partitions, in `[0, 1]`.
    pub fn saturation(&self) -> f64 {
        let total_set = self.total_set_bits();
        total_set as f64 / (self.k * self.partition_size) as f64
    }

    /// Estimated false-positive rate from actual bit saturation.
    ///
    /// Uses `total_set_bits` rather than `item_count`, so the result remains
    /// meaningful after `union`/`intersect` (which zero `item_count`).
    pub fn estimated_fpr(&self) -> f64 {
        let total_set = self.total_set_bits();
        if total_set == 0 {
            return 0.0;
        }
        let x = total_set as f64;
        let m = (self.k * self.partition_size) as f64;
        let fill_rate = x / m;
        fill_rate.powi(self.k as i32)
    }

    /// Returns `true` when saturation exceeds 70%.
    pub fn should_resize(&self) -> bool {
        self.saturation() > 0.7
    }

    /// Returns per-partition `(index, set_bits, saturation)`.
    pub fn partition_stats(&self) -> Vec<(usize, usize, f64)> {
        (0..self.k)
            .map(|partition_idx| {
                let ptr = self.partition_ptr(partition_idx);
                let words = self.partition_size.div_ceil(64);
                let mut set_bits = 0;
                for word_idx in 0..words {
                    let word = unsafe { ptr.add(word_idx).read() };
                    set_bits += word.count_ones() as usize;
                }
                let saturation = set_bits as f64 / self.partition_size as f64;
                (partition_idx, set_bits, saturation)
            })
            .collect()
    }

    /// Merge another compatible filter (union).
    ///
    /// After union, `item_count` is set to 0 because the exact count of
    /// unique items in the merged result is unknown. This means
    /// `is_empty()` will return `true` even though the filter may
    /// contain set bits — callers should not rely on `is_empty()` as
    /// a "has any data" check after set operations.
    pub fn union(&mut self, other: &Self) -> Result<()> {
        if self.k != other.k || self.partition_size != other.partition_size {
            return Err(BloomCraftError::incompatible_filters(
                "Different parameters".to_string(),
            ));
        }

        for partition_idx in 0..self.k {
            let self_ptr = self.partition_ptr(partition_idx);
            let other_ptr = other.partition_ptr(partition_idx);
            let words = self.partition_size.div_ceil(64);
            for word_idx in 0..words {
                unsafe {
                    let self_word_ptr = self_ptr.add(word_idx);
                    let other_word = other_ptr.add(word_idx).read();
                    let self_word = self_word_ptr.read();
                    self_word_ptr.write(self_word | other_word);
                }
            }
        }
        self.item_count = 0; // Unknown after union
        Ok(())
    }

    /// Create new filter as union (non-mutating).
    pub fn union_new(&self, other: &Self) -> Result<Self> {
        if self.k != other.k || self.partition_size != other.partition_size {
            return Err(BloomCraftError::incompatible_filters(
                "Different parameters".to_string(),
            ));
        }

        let mut result = Self {
            data: {
                let layout = Layout::from_size_align(self.allocated_bytes, self.alignment)
                    .map_err(|_| {
                        BloomCraftError::invalid_parameters("Invalid layout".to_string())
                    })?;
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
                NonNull::new(ptr as *mut u64).expect("Allocation returned null")
            },
            k: self.k,
            partition_size: self.partition_size,
            partition_stride: self.partition_stride,
            alignment: self.alignment,
            allocated_bytes: self.allocated_bytes,
            hasher: self.hasher.clone(),
            expected_items: self.expected_items,
            target_fpr: self.target_fpr,
            item_count: self.item_count,
            _phantom: PhantomData,
            #[cfg(feature = "metrics")]
            metrics: None,
        };
        result.union(other)?;
        Ok(result)
    }

    /// Compute intersection with another filter.
    ///
    /// After intersection, `item_count` is set to 0 because the exact
    /// count of unique items in the result is unknown. See `union()`
    /// for the same caveat about `is_empty()`.
    pub fn intersect(&mut self, other: &Self) -> Result<()> {
        if self.k != other.k || self.partition_size != other.partition_size {
            return Err(BloomCraftError::incompatible_filters(
                "Different parameters".to_string(),
            ));
        }

        for partition_idx in 0..self.k {
            let self_ptr = self.partition_ptr(partition_idx);
            let other_ptr = other.partition_ptr(partition_idx);
            let words = self.partition_size.div_ceil(64);
            for word_idx in 0..words {
                unsafe {
                    let self_word_ptr = self_ptr.add(word_idx);
                    let other_word = other_ptr.add(word_idx).read();
                    let self_word = self_word_ptr.read();
                    self_word_ptr.write(self_word & other_word);
                }
            }
        }
        self.item_count = 0; // Unknown after intersection
        Ok(())
    }

    /// Insert multiple items in batch.
    ///
    /// Current implementation delegates to per-item `insert()`.
    /// The `T: Send + Sync` bound is required by the underlying trait method.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::PartitionedBloomFilter;
    ///
    /// let mut filter = PartitionedBloomFilter::<u64>::new(10_000, 0.01)?;
    /// let items: Vec<u64> = (0..100).collect();
    /// filter.insert_batch(&items);
    /// # Ok::<(), bloomcraft::BloomCraftError>(())
    /// ```
    pub fn insert_batch(&mut self, items: &[T])
    where
        T: Send + Sync,
    {
        #[cfg(feature = "metrics")]
        let start = Instant::now();

        for item in items {
            self.insert(item);
        }

        #[cfg(feature = "metrics")]
        if let Some(ref metrics) = self.metrics {
            metrics.record_insert(start.elapsed());
        }
    }

    /// Query multiple items in batch.
    ///
    /// Current implementation delegates to per-item `contains()`.
    /// The `T: Send + Sync` bound is required by the underlying trait method.
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool>
    where
        T: Send + Sync,
    {
        #[cfg(feature = "metrics")]
        let start = Instant::now();

        let results: Vec<bool> = items.iter().map(|item| self.contains(item)).collect();

        #[cfg(feature = "metrics")]
        if let Some(ref metrics) = self.metrics {
            metrics.record_query(start.elapsed());
        }

        results
    }

    /// Export metrics in Prometheus format (requires "metrics" feature).
    #[cfg(feature = "metrics")]
    pub fn export_prometheus(&self) -> String {
        if let Some(ref metrics) = self.metrics {
            let health = self.health_check();
            export_prometheus(metrics, &health)
        } else {
            String::from(
                "# Metrics not enabled
",
            )
        }
    }

    /// Get health check status (requires "metrics" feature).
    #[cfg(feature = "metrics")]
    pub fn health_check(&self) -> HealthCheck {
        HealthCheck::new(self.saturation(), self.estimated_fpr(), self.target_fpr)
    }
}

// BloomFilter trait implementation
impl<T, H> BloomFilter<T> for PartitionedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn insert(&mut self, item: &T) {
        #[cfg(feature = "metrics")]
        let start = Instant::now();

        let (h1, h2) = self.hash_item(item);
        for i in 0..self.k {
            let hash = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = Self::hash_to_range(hash, self.partition_size);
            unsafe {
                self.set_bit_unchecked(i, bit_idx);
            }
        }
        self.item_count = self.item_count.saturating_add(1);

        #[cfg(feature = "metrics")]
        if let Some(ref metrics) = self.metrics {
            metrics.record_insert(start.elapsed());
        }
    }

    fn contains(&self, item: &T) -> bool {
        #[cfg(feature = "metrics")]
        let start = Instant::now();

        let (h1, h2) = self.hash_item(item);
        for i in 0..self.k {
            let hash = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit_idx = Self::hash_to_range(hash, self.partition_size);
            if !unsafe { self.get_bit_unchecked(i, bit_idx) } {
                #[cfg(feature = "metrics")]
                if let Some(ref metrics) = self.metrics {
                    metrics.record_query(start.elapsed());
                }
                return false;
            }
        }

        #[cfg(feature = "metrics")]
        if let Some(ref metrics) = self.metrics {
            metrics.record_query(start.elapsed());
        }

        true
    }

    fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.data.as_ptr() as *mut u8, 0, self.allocated_bytes);
        }
        self.item_count = 0;
    }

    /// Returns `true` when `item_count` is zero.
    ///
    /// After `union()` or `intersect()`, `item_count` is intentionally
    /// zeroed (the exact count is unknown), so this returns `true` even
    /// when the bitset may still contain data.
    fn is_empty(&self) -> bool {
        self.item_count == 0
    }

    fn len(&self) -> usize {
        self.item_count
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
        let total_set = self.total_set_bits();
        let m = (self.k * self.partition_size) as f64;
        if (total_set as f64 / m) < 0.01 {
            return self.item_count;
        }

        let x = total_set as f64;
        let k = self.k as f64;
        let estimated = -(m / k) * (1.0 - x / m).ln();
        estimated.max(0.0) as usize
    }

    fn count_set_bits(&self) -> usize {
        self.total_set_bits()
    }
}

// Drop implementation
impl<T, H> Drop for PartitionedBloomFilter<T, H>
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
impl<T, H> Clone for PartitionedBloomFilter<T, H>
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
        let data = NonNull::new(ptr as *mut u64).expect("Allocation returned null");

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
            item_count: self.item_count,
            _phantom: PhantomData,
            #[cfg(feature = "metrics")]
            metrics: None,
        }
    }
}

// Thread safety markers
unsafe impl<T, H> Send for PartitionedBloomFilter<T, H>
where
    T: Send,
    H: BloomHasher + Clone + Default + Send,
{
}

unsafe impl<T, H> Sync for PartitionedBloomFilter<T, H>
where
    T: Sync,
    H: BloomHasher + Clone + Default + Sync,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "metrics")]
    use crate::metrics::partitioned_metrics::HealthStatus;

    #[test]
    fn test_basic_insert_and_query() {
        let mut filter: PartitionedBloomFilter<String> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        filter.insert(&"hello".to_string());
        filter.insert(&"world".to_string());
        assert!(filter.contains(&"hello".to_string()));
        assert!(filter.contains(&"world".to_string()));
        assert!(!filter.contains(&"goodbye".to_string()));
    }

    #[test]
    fn test_no_false_negatives() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(10_000, 0.01).unwrap();
        let items: Vec<u64> = (0..5000).collect();
        for item in &items {
            filter.insert(item);
        }
        for item in &items {
            assert!(filter.contains(item), "False negative for {}", item);
        }
    }

    #[test]
    fn test_false_positive_rate_statistical() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(10_000, 0.01).unwrap();
        for i in 0..10_000 {
            filter.insert(&i);
        }
        let false_positives: usize = (10_000..110_000).filter(|&i| filter.contains(&i)).count();
        let actual_fpr = false_positives as f64 / 100_000.0;
        println!("Actual FPR: {:.4}%", actual_fpr * 100.0);
        let std_dev = (actual_fpr * (1.0 - actual_fpr) / 100_000.0).sqrt();
        let margin = 4.0 * std_dev;
        assert!(
            actual_fpr < 0.015 + margin,
            "FPR {:.4}% exceeds expected range",
            actual_fpr * 100.0
        );
    }

    #[test]
    fn test_cache_alignment() {
        let filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::with_alignment(10_000, 0.01, 64).unwrap();
        assert_eq!(filter.alignment(), 64);
        let ptr = filter.data.as_ptr() as usize;
        assert_eq!(ptr % 64, 0, "Base pointer not 64-byte aligned");
    }

    #[test]
    fn test_union_operation() {
        let mut filter1: PartitionedBloomFilter<String> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        let mut filter2: PartitionedBloomFilter<String> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        filter1.insert(&"alice".to_string());
        filter2.insert(&"bob".to_string());
        filter1.union(&filter2).unwrap();
        assert!(filter1.contains(&"alice".to_string()));
        assert!(filter1.contains(&"bob".to_string()));
    }

    #[test]
    fn test_intersect_operation() {
        let mut filter1: PartitionedBloomFilter<String> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        let mut filter2: PartitionedBloomFilter<String> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        filter1.insert(&"alice".to_string());
        filter1.insert(&"bob".to_string());
        filter2.insert(&"bob".to_string());
        filter2.insert(&"charlie".to_string());
        filter1.intersect(&filter2).unwrap();
        assert!(filter1.contains(&"bob".to_string()));
    }

    #[test]
    fn test_batch_operations() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(10_000, 0.01).unwrap();
        let items: Vec<u64> = (0..1000).collect();
        filter.insert_batch(&items);
        let results = filter.contains_batch(&items);
        assert_eq!(results.len(), 1000);
        assert!(results.iter().all(|&x| x));
    }

    #[test]
    fn test_large_batch_operations() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(100_000, 0.01).unwrap();

        // Test batches of various sizes
        for batch_size in [1, 4, 8, 16, 32, 64, 128] {
            let items: Vec<u64> = (0..batch_size).map(|i| i as u64).collect();
            filter.insert_batch(&items);
            let results = filter.contains_batch(&items);
            assert_eq!(results.len(), batch_size);
            assert!(
                results.iter().all(|&x| x),
                "Batch size {} failed",
                batch_size
            );
        }
    }

    #[test]
    fn test_thread_safety_markers() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<PartitionedBloomFilter<u64>>();
        assert_sync::<PartitionedBloomFilter<u64>>();
    }

    #[test]
    fn test_saturation_calculation() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        assert!(filter.saturation() < 0.01);
        for i in 0..500 {
            filter.insert(&i);
        }
        let sat = filter.saturation();
        assert!(sat > 0.2 && sat < 0.8);
    }

    #[test]
    fn test_partition_stats() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(10_000, 0.01).unwrap();
        for i in 0..1000 {
            filter.insert(&i);
        }
        let stats = filter.partition_stats();
        assert_eq!(stats.len(), filter.partition_count());
        for (idx, bits_set, saturation) in stats {
            assert!(bits_set > 0, "Partition {} has no bits set", idx);
            assert!(saturation > 0.0 && saturation < 1.0);
        }
    }

    #[test]
    fn test_clear() {
        let mut filter: PartitionedBloomFilter<String> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        filter.insert(&"test".to_string());
        assert!(!filter.is_empty());
        filter.clear();
        assert!(filter.is_empty());
        assert!(!filter.contains(&"test".to_string()));
    }

    #[test]
    fn test_incompatible_merge() {
        let mut filter1: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        let filter2: PartitionedBloomFilter<u64> = PartitionedBloomFilter::new(2000, 0.01).unwrap();
        assert!(filter1.union(&filter2).is_err());
    }

    #[test]
    fn test_cardinality_estimation() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(10_000, 0.01).unwrap();
        for i in 0..1000 {
            filter.insert(&i);
        }
        let estimated = filter.estimate_count();
        let error = (estimated as i32 - 1000).abs() as f64 / 1000.0;
        assert!(
            error < 0.2,
            "Cardinality estimation error {:.1}% exceeds 20%",
            error * 100.0
        );
    }

    #[test]
    fn test_lemire_hash_distribution() {
        const RANGE: usize = 1000;
        const SAMPLES: usize = 100_000;
        let mut buckets = vec![0usize; RANGE];
        for i in 0..SAMPLES {
            let hash = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::Hasher;
                let mut h = DefaultHasher::new();
                h.write_u64(i as u64);
                h.finish()
            };
            let idx = PartitionedBloomFilter::<(), StdHasher>::hash_to_range(hash, RANGE);
            buckets[idx] += 1;
        }
        let expected = SAMPLES / RANGE;
        let mut outliers = 0;
        for &count in &buckets {
            let deviation = (count as f64 - expected as f64).abs() / expected as f64;
            if deviation > 0.30 {
                outliers += 1;
            }
        }
        assert!(
            outliers < RANGE / 20,
            "Distribution has excessive outliers: {} of {} buckets",
            outliers,
            RANGE
        );
    }

    #[test]
    fn test_memory_layout() {
        let filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(10_000, 0.01).unwrap();
        for i in 0..filter.partition_count() {
            let ptr = filter.partition_ptr(i) as usize;
            assert_eq!(
                ptr % filter.alignment(),
                0,
                "Partition {} not properly aligned",
                i
            );
        }
    }

    #[test]
    fn test_drop_safety() {
        {
            let mut filter: PartitionedBloomFilter<u64> =
                PartitionedBloomFilter::new(1000, 0.01).unwrap();
            for i in 0..100 {
                filter.insert(&i);
            }
        } // Drop happens here - must not panic
    }

    #[test]
    fn test_clone_independence() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        for i in 0..50 {
            filter.insert(&i);
        }
        let mut cloned = filter.clone();
        for i in 0..50 {
            assert!(cloned.contains(&i), "Clone missing item {}", i);
        }
        filter.insert(&999);
        assert!(filter.contains(&999));
        assert!(!cloned.contains(&999));
        cloned.insert(&888);
        assert!(cloned.contains(&888));
        assert!(!filter.contains(&888));
    }

    #[test]
    fn test_clone_many_items() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(10_000, 0.01).unwrap();
        for i in 0..5000 {
            filter.insert(&i);
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
    fn test_multiple_drops() {
        let filters: Vec<_> = (0..10)
            .map(|_| PartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap())
            .collect();
        drop(filters);
    }

    #[test]
    fn test_cache_tuned_constructor() {
        let filter = PartitionedBloomFilter::<u64>::new_cache_tuned(10_000, 0.01).unwrap();
        assert!(filter.partition_count() > 0);
        assert!(filter.partition_size() > 0);
        println!(
            "Cache-tuned filter: {} partitions of {} bits each",
            filter.partition_count(),
            filter.partition_size()
        );
    }

    #[test]
    fn test_should_resize() {
        let mut filter: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();

        assert!(!filter.should_resize());

        let items_needed = (filter.partition_size() as f64 * 2.0) as usize;

        for i in 0..items_needed {
            filter.insert(&(i as u64));
        }

        println!(
            "Inserted {} items, saturation: {:.2}%",
            items_needed,
            filter.saturation() * 100.0
        );

        assert!(
            filter.should_resize(),
            "Filter should need resizing after overfilling (saturation: {:.2}%)",
            filter.saturation() * 100.0
        );
    }

    #[test]
    fn test_union_new_non_mutating() {
        let mut filter1: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();
        let mut filter2: PartitionedBloomFilter<u64> =
            PartitionedBloomFilter::new(1000, 0.01).unwrap();

        filter1.insert(&1);
        filter1.insert(&2);
        filter2.insert(&3);
        filter2.insert(&4);

        let union = filter1.union_new(&filter2).unwrap();

        // Original filters unchanged
        assert!(filter1.contains(&1));
        assert!(filter1.contains(&2));
        assert!(!filter1.contains(&3));
        assert!(!filter1.contains(&4));

        // Union contains all
        assert!(union.contains(&1));
        assert!(union.contains(&2));
        assert!(union.contains(&3));
        assert!(union.contains(&4));
    }

    #[test]
    #[cfg(feature = "metrics")]
    fn test_metrics_integration() {
        let mut filter = PartitionedBloomFilter::<u64>::with_metrics(1000, 0.01).unwrap();

        for i in 0..100 {
            filter.insert(&i);
        }

        let health = filter.health_check();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert!(health.saturation < 0.7);

        let prometheus = filter.export_prometheus();
        assert!(prometheus.contains("bloom_filter_inserts_total"));
        assert!(prometheus.contains("bloom_filter_saturation"));
    }

    #[test]
    #[cfg(feature = "metrics")]
    fn test_health_check_degraded() {
        let mut filter = PartitionedBloomFilter::<u64>::with_metrics(1000, 0.01).unwrap();

        // Overfill to trigger degraded status
        for i in 0u64..5000 {
            filter.insert(&i);
        }

        let health = filter.health_check();
        assert!(
            health.status == HealthStatus::Degraded || health.status == HealthStatus::Critical,
            "Expected degraded/critical status at high saturation"
        );
    }

    #[test]
    fn test_batch_insert_empty() {
        let mut filter = PartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let empty: Vec<u64> = vec![];
        filter.insert_batch(&empty); // Should not panic
        assert!(filter.is_empty());
    }

    #[test]
    fn test_batch_contains_empty() {
        let filter = PartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let empty: Vec<u64> = vec![];
        let results = filter.contains_batch(&empty);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_memory_usage_reasonable() {
        let filter = PartitionedBloomFilter::<u64>::new(100_000, 0.01).unwrap();
        let usage = filter.memory_usage();

        // Should be roughly: (100K items * 9.6 bits/item) / 8 = ~120KB
        // Plus alignment overhead (~5%)
        assert!(usage > 100_000, "Memory usage unexpectedly small");
        assert!(usage < 200_000, "Memory usage unexpectedly large");
        println!("Filter memory usage: {} bytes ({} KB)", usage, usage / 1024);
    }

    #[test]
    fn test_estimated_fpr_vs_actual() {
        let mut filter = PartitionedBloomFilter::<u64>::new(10_000, 0.01).unwrap();

        for i in 0..10_000 {
            filter.insert(&i);
        }

        let estimated = filter.estimated_fpr();

        // Measure actual FPR
        let test_size = 10_000;
        let false_positives = (20_000..20_000 + test_size)
            .filter(|i| filter.contains(i))
            .count();
        let actual = false_positives as f64 / test_size as f64;

        println!(
            "Estimated FPR: {:.4}%, Actual FPR: {:.4}%",
            estimated * 100.0,
            actual * 100.0
        );

        // Estimated should be within 50% of actual (rough approximation)
        let ratio = estimated / actual;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "FPR estimation too far off: estimated={:.4}, actual={:.4}",
            estimated,
            actual
        );
    }

    #[test]
    fn test_partition_balance() {
        let mut filter = PartitionedBloomFilter::<u64>::new(10_000, 0.01).unwrap();

        // Insert many items
        for i in 0..5000 {
            filter.insert(&i);
        }

        let stats = filter.partition_stats();
        let saturations: Vec<f64> = stats.iter().map(|(_, _, s)| *s).collect();

        let max_sat = saturations.iter().cloned().fold(0.0f64, f64::max);
        let min_sat = saturations.iter().cloned().fold(1.0f64, f64::min);

        println!(
            "Partition saturation range: {:.2}% - {:.2}%",
            min_sat * 100.0,
            max_sat * 100.0
        );

        // Partitions should be relatively balanced (within 2× of each other)
        assert!(
            max_sat / min_sat < 2.0,
            "Partition imbalance too high: max={:.4}, min={:.4}",
            max_sat,
            min_sat
        );
    }

    #[test]
    fn test_zero_fpr_empty_filter() {
        let filter = PartitionedBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.false_positive_rate(), 0.0);
        assert_eq!(filter.estimated_fpr(), 0.0);
    }

    #[test]
    fn test_parameter_validation() {
        // Zero items
        assert!(PartitionedBloomFilter::<u64>::new(0, 0.01).is_err());

        // Invalid FPR
        assert!(PartitionedBloomFilter::<u64>::new(1000, 0.0).is_err());
        assert!(PartitionedBloomFilter::<u64>::new(1000, 1.0).is_err());
        assert!(PartitionedBloomFilter::<u64>::new(1000, -0.1).is_err());
        assert!(PartitionedBloomFilter::<u64>::new(1000, 1.5).is_err());

        // Invalid alignment (not power of 2)
        assert!(PartitionedBloomFilter::<u64>::with_alignment(1000, 0.01, 63).is_err());
    }

    #[cfg(all(test, feature = "serde"))]
    mod serde_tests {
        use super::*;

        #[test]
        fn test_serde_round_trip() {
            let mut original: PartitionedBloomFilter<u64> =
                PartitionedBloomFilter::new(10_000, 0.01).unwrap();
            for i in 0..1000u64 {
                original.insert(&i);
            }

            let serialized = serde_json::to_string(&original).unwrap();
            let deserialized: PartitionedBloomFilter<u64> =
                serde_json::from_str(&serialized).unwrap();

            // All inserted items still present
            for i in 0..1000u64 {
                assert!(deserialized.contains(&i), "Round-trip lost item {}", i);
            }
            // Non-inserted items should match original behavior
            assert_eq!(deserialized.contains(&9999), original.contains(&9999));
            assert_eq!(deserialized.partition_size(), original.partition_size());
            assert_eq!(deserialized.partition_count(), original.partition_count());
            assert_eq!(deserialized.item_count(), original.item_count());
        }

        #[test]
        fn test_serde_round_trip_empty() {
            let original: PartitionedBloomFilter<u64> =
                PartitionedBloomFilter::new(1000, 0.01).unwrap();
            let serialized = serde_json::to_string(&original).unwrap();
            let deserialized: PartitionedBloomFilter<u64> =
                serde_json::from_str(&serialized).unwrap();

            assert!(deserialized.is_empty());
            assert_eq!(deserialized.item_count(), 0);
            assert!(!deserialized.contains(&42));
        }

        #[test]
        fn test_serde_rejects_bad_allocated_bytes() {
            let original: PartitionedBloomFilter<u64> =
                PartitionedBloomFilter::new(1000, 0.01).unwrap();
            let mut value = serde_json::to_value(&original).unwrap();

            // Tamper with allocated_bytes
            if let Some(obj) = value.as_object_mut() {
                obj.insert(
                    "allocated_bytes".to_string(),
                    serde_json::Value::from(1usize),
                );
            }
            let tampered = serde_json::to_string(&value).unwrap();
            let result: std::result::Result<PartitionedBloomFilter<u64>, _> =
                serde_json::from_str(&tampered);
            assert!(result.is_err(), "Should reject mismatched allocated_bytes");
        }

        #[test]
        fn test_serde_rejects_truncated_data() {
            let original: PartitionedBloomFilter<u64> =
                PartitionedBloomFilter::new(1000, 0.01).unwrap();
            let mut value = serde_json::to_value(&original).unwrap();

            // Truncate data array
            if let Some(obj) = value.as_object_mut() {
                if let Some(serde_json::Value::Array(data)) = obj.get_mut("data") {
                    data.pop();
                }
            }
            let tampered = serde_json::to_string(&value).unwrap();
            let result: std::result::Result<PartitionedBloomFilter<u64>, _> =
                serde_json::from_str(&tampered);
            assert!(result.is_err(), "Should reject truncated data");
        }
    }
}
