//! Counting Bloom filter — supports insertion, query, and deletion via multi-bit counters.
//!
//! Each bit position in a standard Bloom filter is replaced by a small counter
//! (4, 8, or 16 bits).  Insertion increments the counters at an item's hash
//! positions; deletion decrements them.  An item is considered present when all
//! `k` counters at its hashed positions are non-zero.
//!
//! # Deletion safety
//!
//! Deletion uses a two-phase protocol:
//! 1. Verify all `k` counters are > 0 (item is present).
//! 2. Decrement each counter.
//!
//! This prevents underflow from hash collisions.  Deleting an item that was
//! never inserted (or was already deleted) is a safe no-op.
//!
//! # Thread safety
//!
//! - **Reads** (`contains`): lock-free, relaxed atomic loads, safe to share.
//! - **Writes** (`insert`, `delete`): require `&mut self` or external sync.
//!
//! Wrap in `Arc<RwLock<CountingBloomFilter>>` for concurrent access.
//!
//! # Examples
//!
//! ```
//! use bloomcraft::filters::CountingBloomFilter;
//!
//! let mut filter = CountingBloomFilter::new(100_000, 0.01);
//!
//! filter.insert(&"user:12345");
//! assert!(filter.contains(&"user:12345"));
//!
//! assert!(filter.delete(&"user:12345"));
//! assert!(!filter.contains(&"user:12345"));
//!
//! // Deleting a non-existent item is safe
//! assert!(!filter.delete(&"never_inserted"));
//! ```
//!
//! # Reference
//!
//! - Fan, L., Cao, P., Almeida, J., & Broder, A. Z. (2000). Summary Cache: A Scalable Wide-Area Web Cache Sharing Protocol.
//!   IEEE/ACM Transactions on Networking, 8(3), 281-293.

use crate::core::filter::{BloomFilter, DeletableBloomFilter, MutableBloomFilter};
use crate::core::params::{optimal_bit_count, optimal_hash_count};
use crate::error::{BloomCraftError, Result};
use crate::hash::{BloomHasher, IndexingStrategy, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU16, AtomicU8, AtomicUsize, Ordering};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Constants and Type Definitions

/// Minimum filter size to avoid degenerate cases
const MIN_FILTER_SIZE: usize = 64;

/// Memory ordering for counter reads (relaxed is safe for queries)
const COUNTER_READ_ORDERING: Ordering = Ordering::Relaxed;

/// Memory ordering for counter updates (acquire for CAS load)
const COUNTER_UPDATE_LOAD: Ordering = Ordering::Acquire;

/// Memory ordering for counter updates (release for CAS store)
const COUNTER_UPDATE_STORE: Ordering = Ordering::Release;

/// Maximum allowed counter size (4-bit counters).
const MAX_COUNTER_4BIT: u8 = 15;

/// Maximum allowed counter size (8-bit counters).
const MAX_COUNTER_8BIT: u8 = 255;

/// Maximum allowed counter size (16-bit counters).
const MAX_COUNTER_16BIT: u16 = 65535;

// Public Types

/// Bit-width of each counter.
///
/// Trades memory against overflow headroom:
///
/// | Variant    | Max value | Bytes/counter | Typical use            |
/// |------------|-----------|---------------|------------------------|
/// | `FourBit`  | 15        | 0.5           | Low load, k ≤ 7       |
/// | `EightBit` | 255       | 1.0           | Most production loads  |
/// | `SixteenBit` | 65535   | 2.0           | High skew, λ > 0.8    |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CounterSize {
    #[default]
    /// 4-bit counters (max value: 15)
    ///
    /// - Memory: 0.5 bytes per counter
    /// - Good for: λ < 0.3, standard distributions
    /// - Overflow risk: <10⁻⁹ for properly sized filters
    FourBit,

    /// 8-bit counters (max value: 255)
    ///
    /// - Memory: 1 byte per counter
    /// - Good for: Most production use cases
    /// - Overflow risk: <10⁻¹⁵ for reasonable parameters
    EightBit,

    /// 16-bit counters (max value: 65535)
    ///
    /// - Memory: 2 bytes per counter
    /// - Good for: High-skew distributions, λ > 0.8
    /// - Overflow risk: Negligible for all practical workloads
    SixteenBit,
}

impl CounterSize {
    /// Get the maximum value this counter size can represent.
    #[must_use]
    #[inline]
    pub const fn max_value(self) -> usize {
        match self {
            Self::FourBit => MAX_COUNTER_4BIT as usize,
            Self::EightBit => MAX_COUNTER_8BIT as usize,
            Self::SixteenBit => MAX_COUNTER_16BIT as usize,
        }
    }

    /// Get the number of bits per counter.
    #[must_use]
    #[inline]
    pub const fn bits(self) -> usize {
        match self {
            Self::FourBit => 4,
            Self::EightBit => 8,
            Self::SixteenBit => 16,
        }
    }

    /// Get memory bytes per counter (physical storage).
    #[must_use]
    #[inline]
    pub const fn bytes_per_counter(self) -> usize {
        match self {
            Self::FourBit => 1, // Stored as u8 with logical limit
            Self::EightBit => 1,
            Self::SixteenBit => 2,
        }
    }

    /// Map a maximum counter value to the appropriate CounterSize variant.
    ///
    /// Values 1-15 → FourBit, 16-255 → EightBit, 256+ → SixteenBit.
    #[inline]
    #[must_use]
    pub const fn from_max_count(max: u16) -> Self {
        if max <= MAX_COUNTER_4BIT as u16 {
            Self::FourBit
        } else if max <= MAX_COUNTER_8BIT as u16 {
            Self::EightBit
        } else {
            Self::SixteenBit
        }
    }

    /// Get bit mask for extracting counter value (for packed storage).
    #[inline]
    #[must_use]
    pub const fn mask(self) -> u64 {
        match self {
            Self::FourBit => 0xF,
            Self::EightBit => 0xFF,
            Self::SixteenBit => 0xFFFF,
        }
    }
}

/// Runtime health statistics for a counting filter.
#[derive(Debug, Clone, PartialEq)]
pub struct HealthMetrics {
    /// Fraction of non-zero counters `[0, 1]`.
    pub fill_rate: f64,
    /// Estimated false positive rate.
    pub estimated_fpr: f64,
    /// Configured target false positive rate.
    pub target_fpr: f64,
    /// Maximum value across all counters.
    pub max_counter_value: usize,
    /// Mean value of non-zero counters.
    pub avg_counter_value: f64,
    /// Counters at or near saturation (≥90 % of max).
    pub saturated_count: usize,
    /// Total overflow events recorded.
    pub overflow_events: usize,
    /// Estimated overflow risk `[0, 1]`.
    pub overflow_risk: f64,
    /// Memory used by counters + struct overhead (bytes).
    pub memory_bytes: usize,
    /// Number of non-zero counters.
    pub active_counters: usize,
    /// Total counter array length.
    pub total_counters: usize,
    /// Current load factor (inserts / expected_capacity).
    pub load_factor: f64,
    /// Estimated item count.
    pub estimated_item_count: usize,
    /// Fraction of counters at maximum value.
    pub saturation_rate: f64,
    /// Memory ratio vs standard Bloom filter.
    pub memory_overhead: f64,
    /// Counter distribution `(min, max, mean, stddev)`.
    pub distribution: (f64, f64, f64, f64),
    /// Fraction of zero counters.
    pub zero_rate: f64,
    /// Coefficient of variation of counter values (capped at 1.0).
    pub fragmentation: f64,
}

// Helper Functions

// Main Implementation: CountingBloomFilter

/// Counting Bloom filter with 4/8/16-bit atomic counters.
///
/// # Type parameters
///
/// * `T` — item type (must implement `Hash + Send + Sync`)
/// * `H` — hash function (defaults to `StdHasher`)
#[derive(Debug)]
pub struct CountingBloomFilter<T, H = StdHasher>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    counters_4bit: Option<Box<[AtomicU8]>>, // 4-bit, two per byte
    counters_8bit: Option<Box<[AtomicU8]>>, // 8-bit, one per byte
    counters_16bit: Option<Box<[AtomicU16]>>, // 16-bit, native
    counter_size: CounterSize,
    num_counters: usize, // m
    k: usize,            // hash functions
    hasher: H,
    strategy: IndexingStrategy,
    expected_items: usize,
    target_fpr: f64,
    item_count: AtomicUsize,
    overflow_count: AtomicUsize,
    _phantom: PhantomData<T>,
}

// Constructors

impl<T: Hash + Send + Sync> CountingBloomFilter<T, StdHasher> {
    /// Create a filter with optimal parameters for `expected_items` and target false positive rate `fpr`.
    #[must_use]
    pub fn new(expected_items: usize, fpr: f64) -> Self {
        Self::with_hasher(expected_items, fpr, StdHasher::new())
    }

    /// Create a filter with a specific counter size.
    #[must_use]
    pub fn with_size(expected_items: usize, fpr: f64, counter_size: CounterSize) -> Self {
        Self::with_counter_size_and_hasher(expected_items, fpr, counter_size, StdHasher::new())
    }

    /// Create a filter from explicit `m` and `k` parameters.
    #[must_use]
    pub fn with_params(m: usize, k: usize) -> Self {
        Self::with_params_and_hasher(m, k, StdHasher::new())
    }
}

impl<T, H> CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Send + Sync,
{
    /// Create a filter with a custom hasher.
    #[must_use]
    pub fn with_hasher(expected_items: usize, fpr: f64, hasher: H) -> Self {
        Self::with_counter_size_and_hasher(expected_items, fpr, CounterSize::FourBit, hasher)
    }

    /// Create a filter with a specific counter size and custom hasher.
    #[must_use]
    pub fn with_counter_size_and_hasher(
        expected_items: usize,
        fpr: f64,
        counter_size: CounterSize,
        hasher: H,
    ) -> Self {
        assert!(
            expected_items > 0,
            "expected_items must be positive, got {}",
            expected_items
        );
        assert!(
            fpr > 0.0 && fpr < 1.0,
            "fpr must be in range (0, 1), got {}",
            fpr
        );

        #[cfg(debug_assertions)]
        if counter_size == CounterSize::FourBit {
            eprintln!(
                "[CountingBloomFilter] WARNING: Using 4-bit counters (max 15).                  Monitor overflow_risk via health_metrics().                  Consider 8-bit for production."
            );
        }

        let m = optimal_bit_count(expected_items, fpr)
            .unwrap_or_else(|_| {
                panic!(
                    "Failed to calculate optimal size for n={}, fpr={}",
                    expected_items, fpr
                )
            })
            .max(MIN_FILTER_SIZE);

        let k = optimal_hash_count(m, expected_items).unwrap_or_else(|_| {
            panic!(
                "Failed to calculate optimal k for m={}, n={}",
                m, expected_items
            )
        });

        Self::with_params_hasher_and_counter_size(m, k, hasher, counter_size, expected_items, fpr)
    }

    /// Create a filter with explicit counter count `m`, hash count `k`, and custom hasher.
    #[must_use]
    pub fn with_params_and_hasher(m: usize, k: usize, hasher: H) -> Self {
        Self::with_params_hasher_and_counter_size(m, k, hasher, CounterSize::FourBit, 0, 0.0)
    }

    /// Internal constructor with full parameter control.
    fn with_params_hasher_and_counter_size(
        m: usize,
        k: usize,
        hasher: H,
        counter_size: CounterSize,
        expected_items: usize,
        target_fpr: f64,
    ) -> Self {
        assert!(m > 0, "Filter size (m) must be positive, got {}", m);
        assert!(k > 0, "Hash count (k) must be positive, got {}", k);
        assert!(k <= 32, "Hash count (k) must be <= 32, got {}", k);

        // Initialize the correct counter storage based on size
        let (counters_4bit, counters_8bit, counters_16bit, num_counters) = match counter_size {
            CounterSize::FourBit => {
                let byte_len = m.div_ceil(2);
                let counters: Box<[AtomicU8]> = (0..byte_len).map(|_| AtomicU8::new(0)).collect();
                (Some(counters), None, None, m)
            }
            CounterSize::EightBit => {
                let counters: Box<[AtomicU8]> = (0..m).map(|_| AtomicU8::new(0)).collect();
                (None, Some(counters), None, m)
            }
            CounterSize::SixteenBit => {
                let counters: Box<[AtomicU16]> = (0..m).map(|_| AtomicU16::new(0)).collect();
                (None, None, Some(counters), m)
            }
        };

        Self {
            counters_4bit,
            counters_8bit,
            counters_16bit,
            counter_size,
            num_counters,
            k,
            hasher,
            strategy: IndexingStrategy::EnhancedDouble,
            expected_items,
            target_fpr,
            item_count: AtomicUsize::new(0),
            overflow_count: AtomicUsize::new(0),
            _phantom: PhantomData,
        }
    }

    /// Create a filter from raw counter data (for deserialization).
    ///
    /// # Arguments
    ///
    /// * `size` - Number of counters
    /// * `k` - Number of hash functions
    /// * `counter_size` - Size of each counter
    /// * `counters` - Raw counter values
    /// * `expected_items` - Expected number of items
    /// * `target_fpr` - Target false positive rate
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid or counter data length mismatches.
    pub fn from_raw(
        size: usize,
        k: usize,
        counter_size: CounterSize,
        counters: &[u8],
        expected_items: usize,
        target_fpr: f64,
    ) -> Result<Self>
    where
        H: Default,
    {
        if size == 0 {
            return Err(BloomCraftError::invalid_filter_size(size));
        }
        if k == 0 || k > 32 {
            return Err(BloomCraftError::invalid_hash_count(k, 1, 32));
        }

        let (counters_4bit, counters_8bit, counters_16bit) = match counter_size {
            CounterSize::FourBit => {
                if counters.len() != size {
                    return Err(BloomCraftError::invalid_parameters(format!(
                        "Expected {} bytes for 4-bit counters (1 byte per counter), got {}",
                        size,
                        counters.len()
                    )));
                }
                if counters.iter().any(|&v| v > 15) {
                    return Err(BloomCraftError::invalid_parameters(
                        "4-bit counter values must be in [0, 15]".to_string(),
                    ));
                }
                let byte_len = size.div_ceil(2);
                let atomic_counters: Box<[AtomicU8]> = (0..byte_len)
                    .map(|i| {
                        let lo = counters[i * 2];
                        let hi = if i * 2 + 1 < size {
                            counters[i * 2 + 1]
                        } else {
                            0
                        };
                        AtomicU8::new((hi << 4) | (lo & 0x0F))
                    })
                    .collect();
                (Some(atomic_counters), None, None)
            }
            CounterSize::EightBit => {
                if counters.len() != size {
                    return Err(BloomCraftError::invalid_parameters(format!(
                        "Expected {} counters, got {}",
                        size,
                        counters.len()
                    )));
                }
                let atomic_counters: Box<[AtomicU8]> =
                    counters.iter().map(|&v| AtomicU8::new(v)).collect();
                (None, Some(atomic_counters), None)
            }
            CounterSize::SixteenBit => {
                if counters.len() != size * 2 {
                    return Err(BloomCraftError::invalid_parameters(format!(
                        "Expected {} bytes for 16-bit counters, got {}",
                        size * 2,
                        counters.len()
                    )));
                }
                let atomic_counters: Box<[AtomicU16]> = counters
                    .chunks_exact(2)
                    .map(|chunk| AtomicU16::new(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect();
                (None, None, Some(atomic_counters))
            }
        };

        Ok(Self {
            counters_4bit,
            counters_8bit,
            counters_16bit,
            counter_size,
            num_counters: size,
            k,
            hasher: H::default(),
            strategy: IndexingStrategy::EnhancedDouble,
            expected_items,
            target_fpr,
            item_count: AtomicUsize::new(0),
            overflow_count: AtomicUsize::new(0),
            _phantom: PhantomData,
        })
    }

    /// Create with full parameter control (used by the builder pattern).
    #[must_use]
    pub fn with_full_params(
        m: usize,
        k: usize,
        counter_size: CounterSize,
        expected_items: usize,
        target_fpr: f64,
    ) -> Self
    where
        H: Default,
    {
        assert!(m > 0, "Filter size m must be positive, got {}", m);
        assert!(k > 0, "Hash count k must be positive, got {}", k);
        assert!(k <= 32, "Hash count k must be <= 32, got {}", k);

        let (counters_4bit, counters_8bit, counters_16bit, num_counters) = match counter_size {
            CounterSize::FourBit => {
                let byte_len = m.div_ceil(2);
                let counters: Box<[AtomicU8]> = (0..byte_len).map(|_| AtomicU8::new(0)).collect();
                (Some(counters), None, None, m)
            }
            CounterSize::EightBit => {
                let counters: Box<[AtomicU8]> = (0..m).map(|_| AtomicU8::new(0)).collect();
                (None, Some(counters), None, m)
            }
            CounterSize::SixteenBit => {
                let counters: Box<[AtomicU16]> = (0..m).map(|_| AtomicU16::new(0)).collect();
                (None, None, Some(counters), m)
            }
        };

        Self {
            counters_4bit,
            counters_8bit,
            counters_16bit,
            counter_size,
            num_counters,
            k,
            hasher: H::default(),
            strategy: IndexingStrategy::EnhancedDouble,
            expected_items,
            target_fpr,
            item_count: AtomicUsize::new(0),
            overflow_count: AtomicUsize::new(0),
            _phantom: PhantomData,
        }
    }
}

// Core counter ops

impl<T, H> CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Send + Sync,
{
    /// Read counter at `idx` (relaxed atomic load).
    #[inline]
    fn get_counter(&self, idx: usize) -> usize {
        match self.counter_size {
            CounterSize::FourBit => {
                let byte_idx = idx / 2;
                let is_high = idx % 2 == 1;
                let byte =
                    self.counters_4bit.as_ref().unwrap()[byte_idx].load(COUNTER_READ_ORDERING);
                if is_high {
                    (byte >> 4) as usize
                } else {
                    (byte & 0x0F) as usize
                }
            }
            CounterSize::EightBit => {
                self.counters_8bit.as_ref().unwrap()[idx].load(COUNTER_READ_ORDERING) as usize
            }
            CounterSize::SixteenBit => {
                self.counters_16bit.as_ref().unwrap()[idx].load(COUNTER_READ_ORDERING) as usize
            }
        }
    }

    /// Increment counter at `idx` (saturating CAS loop).
    ///
    /// Returns `false` if already at maximum.
    #[inline]
    fn increment_counter(&self, idx: usize) -> bool {
        match self.counter_size {
            CounterSize::FourBit => {
                let byte_idx = idx / 2;
                let is_high = idx % 2 == 1;
                let atomic_byte = &self.counters_4bit.as_ref().unwrap()[byte_idx];

                loop {
                    let current = atomic_byte.load(COUNTER_UPDATE_LOAD);
                    let nibble = if is_high {
                        current >> 4
                    } else {
                        current & 0x0F
                    };

                    if nibble >= MAX_COUNTER_4BIT {
                        self.overflow_count.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }

                    let new_nibble = nibble + 1;
                    let new_byte = if is_high {
                        (current & 0x0F) | (new_nibble << 4)
                    } else {
                        (current & 0xF0) | new_nibble
                    };

                    if atomic_byte
                        .compare_exchange_weak(
                            current,
                            new_byte,
                            COUNTER_UPDATE_STORE,
                            COUNTER_READ_ORDERING,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                }
            }
            CounterSize::EightBit => {
                let atomic_counter = &self.counters_8bit.as_ref().unwrap()[idx];
                loop {
                    let current = atomic_counter.load(COUNTER_UPDATE_LOAD);
                    if current == MAX_COUNTER_8BIT {
                        self.overflow_count.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    if atomic_counter
                        .compare_exchange_weak(
                            current,
                            current + 1,
                            COUNTER_UPDATE_STORE,
                            COUNTER_READ_ORDERING,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                }
            }
            CounterSize::SixteenBit => {
                let atomic_counter = &self.counters_16bit.as_ref().unwrap()[idx];
                loop {
                    let current = atomic_counter.load(COUNTER_UPDATE_LOAD);
                    if current == MAX_COUNTER_16BIT {
                        self.overflow_count.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    if atomic_counter
                        .compare_exchange_weak(
                            current,
                            current + 1,
                            COUNTER_UPDATE_STORE,
                            COUNTER_READ_ORDERING,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                }
            }
        }
    }

    /// Decrement counter at `idx` (CAS loop).
    ///
    /// Returns `false` if counter is already zero.
    #[inline]
    fn decrement_counter(&self, idx: usize) -> bool {
        match self.counter_size {
            CounterSize::FourBit => {
                let byte_idx = idx / 2;
                let is_high = idx % 2 == 1;
                let atomic_byte = &self.counters_4bit.as_ref().unwrap()[byte_idx];

                loop {
                    let current = atomic_byte.load(COUNTER_UPDATE_LOAD);
                    let nibble = if is_high {
                        current >> 4
                    } else {
                        current & 0x0F
                    };

                    if nibble == 0 {
                        return false;
                    }

                    let new_nibble = nibble - 1;
                    let new_byte = if is_high {
                        (current & 0x0F) | (new_nibble << 4)
                    } else {
                        (current & 0xF0) | new_nibble
                    };

                    if atomic_byte
                        .compare_exchange_weak(
                            current,
                            new_byte,
                            COUNTER_UPDATE_STORE,
                            COUNTER_READ_ORDERING,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                }
            }
            CounterSize::EightBit => {
                let atomic_counter = &self.counters_8bit.as_ref().unwrap()[idx];
                loop {
                    let current = atomic_counter.load(COUNTER_UPDATE_LOAD);
                    if current == 0 {
                        return false;
                    }
                    if atomic_counter
                        .compare_exchange_weak(
                            current,
                            current - 1,
                            COUNTER_UPDATE_STORE,
                            COUNTER_READ_ORDERING,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                }
            }
            CounterSize::SixteenBit => {
                let atomic_counter = &self.counters_16bit.as_ref().unwrap()[idx];
                loop {
                    let current = atomic_counter.load(COUNTER_UPDATE_LOAD);
                    if current == 0 {
                        return false;
                    }
                    if atomic_counter
                        .compare_exchange_weak(
                            current,
                            current - 1,
                            COUNTER_UPDATE_STORE,
                            COUNTER_READ_ORDERING,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                }
            }
        }
    }
}

// Core operations

impl<T, H> CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Send + Sync,
{
    /// Insert an item.
    ///
    /// Increments the `k` counters at the item's hash positions.
    /// Counters saturate at their maximum value.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter = CountingBloomFilter::new(1000, 0.01);
    /// filter.insert(&"hello");
    /// assert!(filter.contains(&"hello"));
    /// ```
    #[inline]
    pub fn insert(&mut self, item: &T) {
        let (h1, h2) = self.hasher.hash_item(item);
        let indices = self
            .strategy
            .generate_indices(h1, h2, 0, self.k, self.num_counters);

        let mut any_incremented = false;
        for idx in indices {
            if self.increment_counter(idx) {
                any_incremented = true;
            }
        }

        if any_incremented {
            self.item_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Insert without updating the internal item counter.
    ///
    /// Avoids an atomic increment on `item_count` at the cost of
    /// `len()` and `estimate_fpr()` accuracy.
    #[inline]
    pub fn insert_fast(&self, item: &T) {
        let (h1, h2) = self.hasher.hash_item(item);
        let indices = self
            .strategy
            .generate_indices(h1, h2, 0, self.k, self.num_counters);

        for idx in indices {
            self.increment_counter(idx);
        }
    }

    /// Check whether an item **might** be in the set.
    ///
    /// Returns `true` if all `k` counters are non-zero (may be a false positive),
    /// `false` if any counter is zero (definitely absent).
    #[must_use]
    #[inline]
    pub fn contains(&self, item: &T) -> bool {
        let (h1, h2) = self.hasher.hash_item(item);
        let indices = self
            .strategy
            .generate_indices(h1, h2, 0, self.k, self.num_counters);

        for idx in indices {
            if self.get_counter(idx) == 0 {
                return false;
            }
        }
        true
    }

    /// Delete an item.
    ///
    /// Returns `true` if the item was present and counters were decremented,
    /// `false` if the item was definitely absent.
    ///
    /// Uses a two-phase protocol: check all `k` counters are > 0 before
    /// decrementing any of them, preventing underflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter = CountingBloomFilter::new(1000, 0.01);
    /// filter.insert(&"temp");
    /// assert!(filter.delete(&"temp"));
    /// assert!(!filter.contains(&"temp"));
    ///
    /// // Safe to delete non-existent items
    /// assert!(!filter.delete(&"never_inserted"));
    /// ```
    #[inline]
    pub fn delete(&mut self, item: &T) -> bool {
        let (h1, h2) = self.hasher.hash_item(item);
        let indices = self
            .strategy
            .generate_indices(h1, h2, 0, self.k, self.num_counters);

        // Phase 1: verify all counters are > 0
        for &idx in &indices {
            if self.get_counter(idx) == 0 {
                return false;
            }
        }

        // Phase 2: decrement
        let mut all_decremented = true;
        for &idx in &indices {
            if !self.decrement_counter(idx) {
                all_decremented = false;
            }
        }

        if all_decremented {
            self.item_count.fetch_sub(1, Ordering::Relaxed);
        }

        all_decremented
    }

    /// Delete an item without the pre-check.
    ///
    /// Skips the two-phase verification; call only when the caller knows the
    /// item is present.  Returns `true` if counters were decremented.
    pub fn delete_unchecked(&mut self, item: &T) -> bool {
        let (h1, h2) = self.hasher.hash_item(item);
        let indices = self
            .strategy
            .generate_indices(h1, h2, 0, self.k, self.num_counters);

        let mut all_decremented = true;
        for idx in indices {
            if !self.decrement_counter(idx) {
                all_decremented = false;
            }
        }

        if all_decremented {
            self.item_count.fetch_sub(1, Ordering::Relaxed);
        }

        all_decremented
    }

    /// Insert, returning an error if all `k` counters are saturated.
    ///
    /// # Errors
    ///
    /// Returns `BloomCraftError::CapacityExceeded` when every counter at the
    /// item's hash positions has reached its maximum value.
    pub fn insert_checked(&mut self, item: &T) -> Result<()> {
        let (h1, h2) = self.hasher.hash_item(item);
        let indices = self
            .strategy
            .generate_indices(h1, h2, 0, self.k, self.num_counters);

        let mut all_saturated = true;
        for idx in indices {
            if self.increment_counter(idx) {
                all_saturated = false;
            }
        }

        if all_saturated {
            return Err(BloomCraftError::capacity_exceeded(
                self.num_counters,
                self.count_nonzero(),
            ));
        }

        self.item_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Reset all counters to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let mut filter = CountingBloomFilter::new(1000, 0.01);
    /// filter.insert(&"data");
    /// filter.clear();
    /// assert!(filter.is_empty());
    /// ```
    pub fn clear(&mut self) {
        match self.counter_size {
            CounterSize::FourBit => {
                for counter in self.counters_4bit.as_ref().unwrap().iter() {
                    counter.store(0, Ordering::Release);
                }
            }
            CounterSize::EightBit => {
                for counter in self.counters_8bit.as_ref().unwrap().iter() {
                    counter.store(0, Ordering::Release);
                }
            }
            CounterSize::SixteenBit => {
                for counter in self.counters_16bit.as_ref().unwrap().iter() {
                    counter.store(0, Ordering::Release);
                }
            }
        }
        self.item_count.store(0, Ordering::Release);
        self.overflow_count.store(0, Ordering::Release);
    }
}
// Batch operations

impl<T, H> CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Send + Sync,
{
    /// Insert all items in a slice.
    pub fn insert_batch(&mut self, items: &[T]) {
        for item in items {
            self.insert(item);
        }
    }

    /// Query multiple items.
    ///
    /// Returns a `Vec<bool>` with one element per input item.
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Check whether **all** items are present.
    ///
    /// Returns `false` on the first absent item.
    #[must_use]
    pub fn contains_all(&self, items: &[T]) -> bool {
        for item in items {
            if !self.contains(item) {
                return false; // Short-circuit
            }
        }
        true
    }

    /// Check whether **any** item is present.
    ///
    /// Returns `true` on the first present item.
    #[must_use]
    pub fn contains_any(&self, items: &[T]) -> bool {
        for item in items {
            if self.contains(item) {
                return true; // Short-circuit
            }
        }
        false
    }

    /// Delete multiple items.
    ///
    /// Returns the number of items successfully removed.
    pub fn delete_batch(&mut self, items: &[T]) -> usize {
        let mut count = 0;
        for item in items {
            if self.delete(item) {
                count += 1;
            }
        }
        count
    }

    /// Delete all items or none (transactional).
    ///
    /// Returns `Ok(n)` on success, `Err(i)` on first failure at index `i`.
    /// On failure no items are modified.
    pub fn delete_all_or_none(&mut self, items: &[T]) -> std::result::Result<usize, usize> {
        for (i, item) in items.iter().enumerate() {
            if !self.delete(item) {
                return Err(i); // Failed at index i
            }
        }
        Ok(items.len())
    }
}

// Introspection

impl<T, H> CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Send + Sync,
{
    /// Get number of non-zero counters.
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        match self.counter_size {
            CounterSize::FourBit => {
                let mut count = 0;
                for i in 0..self.num_counters {
                    if self.get_counter(i) > 0 {
                        count += 1;
                    }
                }
                count
            }
            CounterSize::EightBit => self
                .counters_8bit
                .as_ref()
                .unwrap()
                .iter()
                .filter(|c| c.load(Ordering::Relaxed) > 0)
                .count(),
            CounterSize::SixteenBit => self
                .counters_16bit
                .as_ref()
                .unwrap()
                .iter()
                .filter(|c| c.load(Ordering::Relaxed) > 0)
                .count(),
        }
    }

    /// Get maximum counter value in filter.
    #[must_use]
    pub fn max_counter_value(&self) -> usize {
        let mut max = 0usize;
        for i in 0..self.num_counters {
            let val = self.get_counter(i);
            if val > max {
                max = val;
            }
        }
        max
    }

    /// Get average value of non-zero counters.
    #[must_use]
    pub fn avg_counter_value(&self) -> f64 {
        let mut sum = 0usize;
        let mut count = 0usize;

        for i in 0..self.num_counters {
            let val = self.get_counter(i);
            if val > 0 {
                sum += val;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            sum as f64 / count as f64
        }
    }

    /// Get counter value distribution (min, max, mean, stddev).
    #[must_use]
    pub fn counter_distribution(&self) -> (f64, f64, f64, f64) {
        let mut min = usize::MAX;
        let mut max = 0;
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..self.num_counters {
            let val = self.get_counter(i);
            min = min.min(val);
            max = max.max(val);
            sum += val as f64;
            count += 1;
        }

        let mean = sum / count as f64;

        // Calculate standard deviation
        let mut variance_sum = 0.0;
        for i in 0..self.num_counters {
            let val = self.get_counter(i) as f64;
            let diff = val - mean;
            variance_sum += diff * diff;
        }
        let stddev = (variance_sum / count as f64).sqrt();

        (min as f64, max as f64, mean, stddev)
    }

    /// Get histogram of counter values.
    ///
    /// Returns vector where `result[i]` is the count of counters with value `i`.
    #[must_use]
    pub fn counter_histogram(&self) -> Vec<usize> {
        let max_val = self.max_counter_value();
        let mut histogram = vec![0; max_val + 1];

        for i in 0..self.num_counters {
            let val = self.get_counter(i).min(max_val);
            histogram[val] += 1;
        }

        histogram
    }

    /// Count saturated counters (>90% of max value).
    #[must_use]
    pub fn saturated_counter_count(&self) -> usize {
        let max_val = self.counter_size.max_value();
        let threshold = (max_val as f64 * 0.9) as usize;

        let mut count = 0;
        for i in 0..self.num_counters {
            if self.get_counter(i) >= threshold {
                count += 1;
            }
        }
        count
    }

    /// Find N counters with highest values (hot spots).
    ///
    /// Returns vector of (index, value) tuples sorted by value descending.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::CountingBloomFilter;
    ///
    /// let filter = CountingBloomFilter::<i32>::new(1000, 0.01);
    /// let hot = filter.hotspots(10);
    /// for (idx, val) in hot {
    ///     println!("Counter {} has value {}", idx, val);
    /// }
    /// ```
    #[must_use]
    pub fn hotspots(&self, n: usize) -> Vec<(usize, usize)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut heap = BinaryHeap::new();

        for i in 0..self.num_counters {
            let val = self.get_counter(i);
            if val > 0 {
                if heap.len() < n {
                    heap.push(Reverse((val, i)));
                } else if let Some(&Reverse((min_val, _))) = heap.peek() {
                    if val > min_val {
                        heap.pop();
                        heap.push(Reverse((val, i)));
                    }
                }
            }
        }

        let mut result: Vec<_> = heap
            .into_iter()
            .map(|Reverse((val, idx))| (idx, val))
            .collect();
        result.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending
        result
    }

    /// Get memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let counter_bytes = match self.counter_size {
            CounterSize::FourBit => self.num_counters.div_ceil(2),
            CounterSize::EightBit => self.num_counters,
            CounterSize::SixteenBit => self.num_counters * 2,
        };
        counter_bytes + std::mem::size_of::<Self>()
    }

    /// Calculate memory overhead vs standard Bloom filter.
    ///
    /// Returns ratio: (counting memory) / (standard memory).
    /// Typical values: 4× (4-bit), 8× (8-bit), 16× (16-bit).
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        let counting_bytes = self.memory_usage();
        let standard_bytes = self.num_counters.div_ceil(8); // Bits → bytes
        counting_bytes as f64 / standard_bytes as f64
    }

    /// Estimate false positive rate.
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        let n = self.item_count.load(Ordering::Relaxed) as f64;
        if n == 0.0 {
            return self.target_fpr;
        }
        let m = self.num_counters as f64;
        let k = self.k as f64;
        let exponent = -k * n / m;
        (1.0 - exponent.exp()).powf(k)
    }

    /// Collect comprehensive health metrics.
    #[must_use]
    pub fn health_metrics(&self) -> HealthMetrics {
        let active_counters = self.count_nonzero();
        let total_counters = self.num_counters;
        let fill_rate = active_counters as f64 / total_counters as f64;

        let max_counter = self.max_counter_value();
        let avg_counter = self.avg_counter_value();
        let saturated = self.saturated_counter_count();

        let max_possible = self.counter_size.max_value();
        let overflow_risk = if max_counter >= max_possible {
            1.0
        } else {
            (max_counter as f64 / max_possible as f64).powf(2.0)
        };

        let load_factor = if self.expected_items > 0 {
            self.item_count.load(Ordering::Relaxed) as f64 / self.expected_items as f64
        } else {
            0.0
        };

        let saturation_rate = saturated as f64 / total_counters as f64;
        let memory_overhead = self.compression_ratio();
        let distribution = self.counter_distribution();
        let zero_rate = 1.0 - fill_rate;

        let (_, _, mean, stddev) = distribution;
        let fragmentation = if mean > 0.0 {
            (stddev / mean).min(1.0)
        } else {
            0.0
        };

        HealthMetrics {
            fill_rate,
            estimated_fpr: self.estimate_fpr(),
            target_fpr: self.target_fpr,
            max_counter_value: max_counter,
            avg_counter_value: avg_counter,
            saturated_count: saturated,
            overflow_events: self.overflow_count.load(Ordering::Relaxed),
            overflow_risk,
            memory_bytes: self.memory_usage(),
            active_counters,
            total_counters,
            load_factor,
            estimated_item_count: self.item_count.load(Ordering::Relaxed),
            saturation_rate,
            memory_overhead,
            distribution,
            zero_rate,
            fragmentation,
        }
    }

    /// Get raw counter values as byte vector.
    #[must_use]
    pub fn raw_counters(&self) -> Vec<u8> {
        match self.counter_size {
            CounterSize::FourBit => {
                let mut result = Vec::with_capacity(self.num_counters);
                for i in 0..self.num_counters {
                    result.push(self.get_counter(i) as u8);
                }
                result
            }
            CounterSize::EightBit => self
                .counters_8bit
                .as_ref()
                .unwrap()
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect(),
            CounterSize::SixteenBit => self
                .counters_16bit
                .as_ref()
                .unwrap()
                .iter()
                .flat_map(|c| {
                    let v: u16 = c.load(Ordering::Relaxed);
                    v.to_le_bytes()
                })
                .collect(),
        }
    }
}

// Accessors

impl<T, H> CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Send + Sync,
{
    /// Get filter size (number of counters).
    #[must_use]
    #[inline]
    pub const fn size(&self) -> usize {
        self.num_counters
    }

    /// Get number of hash functions.
    #[must_use]
    #[inline]
    pub const fn hash_count(&self) -> usize {
        self.k
    }

    /// Get counter size configuration.
    #[must_use]
    #[inline]
    pub const fn counter_size(&self) -> CounterSize {
        self.counter_size
    }

    /// Get maximum counter value for current configuration.
    #[must_use]
    pub fn max_count(&self) -> usize {
        self.counter_size.max_value()
    }

    /// Get number of items (estimate).
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.item_count.load(Ordering::Relaxed)
    }

    /// Check if filter is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get total overflow events.
    #[must_use]
    #[inline]
    pub fn overflow_count(&self) -> usize {
        self.overflow_count.load(Ordering::Relaxed)
    }

    /// Check if any overflow occurred.
    #[must_use]
    #[inline]
    pub fn has_overflowed(&self) -> bool {
        self.overflow_count() > 0
    }

    /// Get expected items.
    #[must_use]
    #[inline]
    pub const fn expected_items(&self) -> usize {
        self.expected_items
    }

    /// Get target false positive rate.
    #[must_use]
    #[inline]
    pub fn target_fpr(&self) -> f64 {
        self.target_fpr
    }

    /// Get counter bit width.
    #[must_use]
    #[inline]
    pub fn counter_bits(&self) -> u8 {
        self.counter_size.bits() as u8
    }

    /// Get number of hash functions.
    #[must_use]
    #[inline]
    pub fn num_hashes(&self) -> usize {
        self.k
    }
}
// Trait Implementations

impl<T, H> BloomFilter<T> for CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn insert(&mut self, item: &T) {
        CountingBloomFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        CountingBloomFilter::contains(self, item)
    }

    fn clear(&mut self) {
        CountingBloomFilter::clear(self);
    }

    fn len(&self) -> usize {
        CountingBloomFilter::len(self)
    }

    fn is_empty(&self) -> bool {
        CountingBloomFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn expected_items(&self) -> usize {
        self.expected_items
    }

    fn bit_count(&self) -> usize {
        self.num_counters
    }

    fn hash_count(&self) -> usize {
        self.k
    }

    fn count_set_bits(&self) -> usize {
        self.count_nonzero()
    }
}

impl<T, H> DeletableBloomFilter<T> for CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn remove(&mut self, item: &T) -> crate::error::Result<()> {
        if CountingBloomFilter::delete(self, item) {
            Ok(())
        } else {
            Err(crate::error::BloomCraftError::InvalidParameters {
                message: "Item not in filter or counter underflow".to_string(),
            })
        }
    }

    fn can_remove(&self, item: &T) -> bool {
        CountingBloomFilter::contains(self, item)
    }
}

impl<T, H> MutableBloomFilter<T> for CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn insert_mut(&mut self, item: &T) {
        CountingBloomFilter::insert(self, item);
    }
}

impl<T, H> Clone for CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Send + Sync,
{
    fn clone(&self) -> Self {
        let (counters_4bit, counters_8bit, counters_16bit) = match self.counter_size {
            CounterSize::FourBit => {
                let counters: Box<[AtomicU8]> = self
                    .counters_4bit
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|c| AtomicU8::new(c.load(Ordering::Acquire)))
                    .collect();
                (Some(counters), None, None)
            }
            CounterSize::EightBit => {
                let counters: Box<[AtomicU8]> = self
                    .counters_8bit
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|c| AtomicU8::new(c.load(Ordering::Acquire)))
                    .collect();
                (None, Some(counters), None)
            }
            CounterSize::SixteenBit => {
                let counters: Box<[AtomicU16]> = self
                    .counters_16bit
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|c| AtomicU16::new(c.load(Ordering::Acquire)))
                    .collect();
                (None, None, Some(counters))
            }
        };

        Self {
            counters_4bit,
            counters_8bit,
            counters_16bit,
            counter_size: self.counter_size,
            num_counters: self.num_counters,
            k: self.k,
            hasher: self.hasher.clone(),
            strategy: self.strategy,
            expected_items: self.expected_items,
            target_fpr: self.target_fpr,
            item_count: AtomicUsize::new(self.item_count.load(Ordering::Acquire)),
            overflow_count: AtomicUsize::new(self.overflow_count.load(Ordering::Acquire)),
            _phantom: PhantomData,
        }
    }
}

impl<T, H> PartialEq for CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Send + Sync,
{
    fn eq(&self, other: &Self) -> bool {
        if self.num_counters != other.num_counters
            || self.k != other.k
            || self.counter_size != other.counter_size
        {
            return false;
        }

        for i in 0..self.num_counters {
            if self.get_counter(i) != other.get_counter(i) {
                return false;
            }
        }

        true
    }
}

impl<T, H> Default for CountingBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Send + Sync + Default,
{
    fn default() -> Self {
        Self::with_hasher(1000, 0.01, H::default())
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_query() {
        let mut filter = CountingBloomFilter::<String>::new(100, 0.01);

        filter.insert(&"hello".to_string());
        assert!(filter.contains(&"hello".to_string()));
        assert!(!filter.contains(&"world".to_string()));
    }

    #[test]
    fn test_delete_safety_no_underflow() {
        let mut filter = CountingBloomFilter::<String>::new(100, 0.01);

        // Insert item A
        filter.insert(&"A".to_string());
        assert!(filter.contains(&"A".to_string()));

        // Try to delete non-existent item B
        // This should NOT affect item A (no underflow)
        assert!(!filter.delete(&"B".to_string()));

        // A should still be present
        assert!(
            filter.contains(&"A".to_string()),
            "Delete of non-existent item should not cause false negatives"
        );
    }

    #[test]
    fn test_delete_with_collisions() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.1); // High collision

        // Insert many items to force collisions
        for i in 0..50 {
            filter.insert(&i);
        }

        // Delete half of them
        for i in 0..25 {
            assert!(filter.delete(&i), "Should delete item {}", i);
        }

        // Remaining items should still be present
        for i in 25..50 {
            assert!(
                filter.contains(&i),
                "Item {} should still be present after deleting others",
                i
            );
        }
    }

    #[test]
    fn test_delete_twice_returns_false() {
        let mut filter = CountingBloomFilter::<String>::new(100, 0.01);

        filter.insert(&"item".to_string());
        assert!(filter.contains(&"item".to_string()));

        // First delete succeeds
        let deleted = filter.delete(&"item".to_string());
        assert!(deleted);
        assert!(!filter.contains(&"item".to_string()));

        // Second delete returns false
        let deleted_again = filter.delete(&"item".to_string());
        assert!(!deleted_again, "Should not delete non-existent item");
    }

    #[test]
    fn test_overflow_tracking_4bit() {
        let mut filter = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::FourBit);

        // Insert same item many times to force overflow
        for _ in 0..20 {
            filter.insert(&42);
        }

        let metrics = filter.health_metrics();
        assert!(metrics.overflow_events > 0, "Should record overflow events");
        assert!(
            metrics.overflow_risk > 0.5,
            "Should detect high overflow risk"
        );
    }

    #[test]
    fn test_overflow_tracking_8bit() {
        let mut filter = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::EightBit);

        // 8-bit counters should not overflow easily
        for _ in 0..100 {
            filter.insert(&42);
        }

        let metrics = filter.health_metrics();
        // Should have very low or zero overflow risk
        assert!(metrics.overflow_risk < 0.9);
    }

    #[test]
    fn test_clear() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.01);

        for i in 0..10 {
            filter.insert(&i);
        }

        assert!(!filter.is_empty());
        filter.clear();
        assert!(filter.is_empty());

        for i in 0..10 {
            assert!(!filter.contains(&i));
        }
    }

    #[test]
    fn test_batch_operations() {
        let mut filter = CountingBloomFilter::<String>::new(1000, 0.01);

        let items: Vec<String> = (0..100).map(|i| format!("item{}", i)).collect();

        filter.insert_batch(&items);

        let results = filter.contains_batch(&items);
        assert_eq!(results.iter().filter(|&&x| x).count(), 100);
    }

    #[test]
    fn test_contains_all() {
        let mut filter = CountingBloomFilter::<i32>::new(1000, 0.01);

        filter.insert(&1);
        filter.insert(&2);
        filter.insert(&3);

        assert!(filter.contains_all(&[1, 2, 3]));
        assert!(!filter.contains_all(&[1, 2, 3, 4]));
    }

    #[test]
    fn test_contains_any() {
        let mut filter = CountingBloomFilter::<i32>::new(1000, 0.01);

        filter.insert(&42);

        assert!(filter.contains_any(&[1, 42, 99]));
        assert!(!filter.contains_any(&[1, 2, 3]));
    }

    #[test]
    fn test_delete_all_or_none_success() {
        let mut filter = CountingBloomFilter::<i32>::new(1000, 0.01);

        filter.insert(&1);
        filter.insert(&2);
        filter.insert(&3);

        match filter.delete_all_or_none(&[1, 2, 3]) {
            Ok(n) => assert_eq!(n, 3),
            Err(_) => panic!("Should succeed"),
        }

        assert!(!filter.contains(&1));
        assert!(!filter.contains(&2));
        assert!(!filter.contains(&3));
    }

    #[test]
    fn test_delete_all_or_none_failure() {
        let mut filter = CountingBloomFilter::<i32>::new(1000, 0.01);

        filter.insert(&1);
        filter.insert(&2);
        // Don't insert 3

        match filter.delete_all_or_none(&[1, 2, 3]) {
            Ok(_) => panic!("Should fail at index 2"),
            Err(i) => assert_eq!(i, 2),
        }
    }

    #[test]
    fn test_counter_statistics() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.01);

        for i in 0..50 {
            filter.insert(&i);
        }

        let metrics = filter.health_metrics();
        assert!(metrics.fill_rate > 0.0);
        assert!(metrics.fill_rate <= 1.0);
        assert!(metrics.avg_counter_value > 0.0);
        assert!(metrics.max_counter_value > 0);
    }

    #[test]
    fn test_hotspots() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.1);

        // Insert items to create some hot spots
        for i in 0..20 {
            filter.insert(&i);
        }

        let hot = filter.hotspots(5);
        assert!(hot.len() <= 5);

        // Verify sorted descending
        for i in 1..hot.len() {
            assert!(hot[i - 1].1 >= hot[i].1);
        }
    }

    #[test]
    fn test_memory_usage() {
        let filter_4bit = CountingBloomFilter::<i32>::with_size(1000, 0.01, CounterSize::FourBit);
        let filter_8bit = CountingBloomFilter::<i32>::with_size(1000, 0.01, CounterSize::EightBit);

        let mem_4bit = filter_4bit.memory_usage();
        let mem_8bit = filter_8bit.memory_usage();

        // 8-bit should use approximately 2x memory of 4-bit
        assert!(mem_8bit > mem_4bit);
    }

    #[test]
    fn test_compression_ratio() {
        let filter = CountingBloomFilter::<i32>::with_size(1000, 0.01, CounterSize::FourBit);

        let ratio = filter.compression_ratio();
        // 4-bit counters should be approximately 4x standard Bloom
        assert!((3.0..=5.0).contains(&ratio));
    }

    #[test]
    fn test_clone() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.01);

        filter.insert(&42);
        filter.insert(&99);

        let cloned = filter.clone();

        assert_eq!(filter, cloned);
        assert!(cloned.contains(&42));
        assert!(cloned.contains(&99));
    }

    #[test]
    fn test_from_raw_4bit() {
        let size = 100;
        let k = 7;
        let counter_size = CounterSize::FourBit;
        let counters = vec![0u8; size];

        let filter =
            CountingBloomFilter::<i32>::from_raw(size, k, counter_size, &counters, 1000, 0.01)
                .unwrap();

        assert_eq!(filter.size(), size);
        assert_eq!(filter.hash_count(), k);
    }

    #[test]
    fn test_from_raw_8bit() {
        let size = 100;
        let k = 7;
        let counter_size = CounterSize::EightBit;
        let counters = vec![0u8; size];

        let filter =
            CountingBloomFilter::<i32>::from_raw(size, k, counter_size, &counters, 1000, 0.01)
                .unwrap();

        assert_eq!(filter.size(), size);
        assert_eq!(filter.hash_count(), k);
    }

    #[test]
    fn test_concurrent_insert() {
        use std::sync::{Arc, RwLock};
        use std::thread;

        let filter = Arc::new(RwLock::new(CountingBloomFilter::<i32>::new(10_000, 0.01)));

        let handles: Vec<_> = (0..4)
            .map(|thread_id| {
                let filter = Arc::clone(&filter);
                thread::spawn(move || {
                    for i in 0..1000 {
                        let key = thread_id * 1000 + i;
                        filter.write().unwrap().insert(&key);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let filter = filter.read().unwrap();
        // Allow some loss due to collisions
        assert!(filter.len() >= 3800);
    }

    #[test]
    fn test_insert_checked_overflow() {
        let mut filter = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::FourBit);

        // Fill to capacity
        for _ in 0..20 {
            let _ = filter.insert_checked(&42);
        }

        // Should eventually return capacity error
        let result = filter.insert_checked(&42);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_metrics_comprehensive() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.01);

        for i in 0..50 {
            filter.insert(&i);
        }

        let metrics = filter.health_metrics();

        assert!(metrics.fill_rate > 0.0);
        assert!(metrics.estimated_fpr >= 0.0);
        assert!(metrics.load_factor > 0.0);
        assert!(metrics.memory_bytes > 0);
        assert!(metrics.active_counters > 0);
        assert_eq!(metrics.total_counters, filter.size());
        assert!(metrics.fragmentation >= 0.0);
        assert!(metrics.memory_overhead > 1.0);
    }

    #[test]
    fn test_counter_distribution() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.01);

        for i in 0..20 {
            filter.insert(&i);
        }

        let (min, max, mean, stddev) = filter.counter_distribution();

        assert!(min >= 0.0);
        assert!(max >= min);
        assert!(mean >= min && mean <= max);
        assert!(stddev >= 0.0);
    }

    #[test]
    fn test_raw_counters() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.01);

        filter.insert(&42);

        let counters = filter.raw_counters();
        assert_eq!(counters.len(), filter.size());

        // At least some counters should be non-zero
        let nonzero = counters.iter().filter(|&&c| c > 0).count();
        assert!(nonzero > 0);
    }

    #[test]
    fn test_from_raw_16bit() {
        let size = 100;
        let k = 7;
        let counter_size = CounterSize::SixteenBit;
        let counters = vec![0u8; size * 2];

        let filter =
            CountingBloomFilter::<i32>::from_raw(size, k, counter_size, &counters, 1000, 0.01)
                .unwrap();

        assert_eq!(filter.size(), size);
        assert_eq!(filter.hash_count(), k);
    }

    #[test]
    fn test_raw_counters_roundtrip_all_sizes() {
        for counter_size in [
            CounterSize::FourBit,
            CounterSize::EightBit,
            CounterSize::SixteenBit,
        ] {
            let mut filter = CountingBloomFilter::<i32>::with_size(1000, 0.01, counter_size);
            for i in 0..50 {
                filter.insert(&i);
            }

            let raw = filter.raw_counters();
            let restored = CountingBloomFilter::<i32>::from_raw(
                filter.size(),
                filter.hash_count(),
                counter_size,
                &raw,
                1000,
                0.01,
            )
            .unwrap();

            assert_eq!(restored, filter, "round-trip failed for {:?}", counter_size);
        }
    }

    #[test]
    fn test_16bit_high_count_health_metrics() {
        use crate::filters::CounterSize;
        let mut filter = CountingBloomFilter::<i32>::with_size(100, 0.5, CounterSize::SixteenBit);

        for _ in 0..500 {
            filter.insert(&42);
        }

        let raw = filter.raw_counters();
        let max_val = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .max()
            .unwrap_or(0);
        assert!(max_val > 255, "16-bit counters should exceed 255");

        let metrics = filter.health_metrics();
        assert!(
            metrics.overflow_risk < 0.5,
            "16-bit overflow_risk should stay low with counters far below 65535"
        );
        assert!(
            metrics.max_counter_value > 255,
            "max_counter_value should report true max"
        );
    }

    // --- insert_fast / delete_unchecked ---

    #[test]
    fn test_insert_fast_basic() {
        let filter = CountingBloomFilter::<i32>::new(1000, 0.01);
        filter.insert_fast(&42);
        assert!(filter.contains(&42));
    }

    #[test]
    fn test_insert_fast_no_item_count() {
        let filter = CountingBloomFilter::<i32>::new(1000, 0.01);
        filter.insert_fast(&1);
        filter.insert_fast(&2);
        filter.insert_fast(&3);
        assert_eq!(filter.len(), 0, "insert_fast must not increment item_count");
    }

    #[test]
    fn test_delete_unchecked_basic() {
        let mut filter = CountingBloomFilter::<i32>::new(1000, 0.01);
        filter.insert(&99);
        assert!(filter.contains(&99));
        let deleted = filter.delete_unchecked(&99);
        assert!(deleted);
        assert!(!filter.contains(&99));
    }

    #[test]
    fn test_delete_unchecked_absent_item() {
        let mut filter = CountingBloomFilter::<i32>::new(1000, 0.01);
        filter.insert(&1);
        let result = filter.delete_unchecked(&999);
        // absent item: some counters may be zero → decrement returns false
        // but the two-phase check is skipped, so partial decrement is possible.
        // This is the documented risk of delete_unchecked.
        assert!(!result);
    }

    #[test]
    fn test_insert_fast_saturates_at_max() {
        let filter = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::FourBit);
        // Saturate all counters for key 42: 15 is max for 4bit
        for _ in 0..30 {
            filter.insert_fast(&42);
        }
        // Verify no counter exceeds 15
        let raw = filter.raw_counters();
        let max_val = raw.iter().copied().max().unwrap_or(0);
        assert!(
            max_val <= 15,
            "4-bit counters must saturate at 15, got {}",
            max_val
        );
    }

    // --- Concurrent stress tests ---

    #[test]
    fn test_concurrent_insert_fast_same_key() {
        use std::sync::Arc;
        use std::thread;

        let filter = Arc::new(CountingBloomFilter::<i32>::with_size(
            10,
            0.5,
            CounterSize::EightBit,
        ));
        let num_threads = 8;
        let ops_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for _ in 0..ops_per_thread {
                        f.insert_fast(&42);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread panicked");
        }

        // Same key from 8 threads × 100 = 800 insert_fast on 8-bit counters (max 255)
        // Counters should saturate at 255, no wrapping, no crash
        let raw = filter.raw_counters();
        let max_val: usize = raw.iter().copied().map(|v| v as usize).max().unwrap_or(0);
        assert!(
            max_val <= 255,
            "Counters must not exceed 255, got {}",
            max_val
        );
    }

    #[test]
    fn test_concurrent_insert_fast_different_keys() {
        use std::sync::Arc;
        use std::thread;

        let filter = Arc::new(CountingBloomFilter::<i32>::with_size(
            100_000,
            0.01,
            CounterSize::EightBit,
        ));
        let num_threads = 4;
        let keys_per_thread = 500;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for i in 0..keys_per_thread {
                        let key = tid * keys_per_thread + i;
                        f.insert_fast(&key);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread panicked");
        }

        // Verify all keys are visible despite no item_count tracking
        for tid in 0..num_threads {
            for i in 0..keys_per_thread {
                let key = tid * keys_per_thread + i;
                assert!(
                    filter.contains(&key),
                    "Key {} missing after concurrent insert_fast",
                    key
                );
            }
        }
    }

    #[test]
    fn test_concurrent_insert_fast_16bit() {
        use std::sync::Arc;
        use std::thread;

        let filter = Arc::new(CountingBloomFilter::<i32>::with_size(
            100,
            0.5,
            CounterSize::SixteenBit,
        ));
        let num_threads = 8;
        let ops_per_thread = 500;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for _ in 0..ops_per_thread {
                        f.insert_fast(&42);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread panicked");
        }

        // 4000 insert_fast on 16-bit counters (max 65535) — should still be below cap
        let raw = filter.raw_counters();
        let max_val: u64 = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u64)
            .max()
            .unwrap_or(0);
        assert!(max_val > 0, "Counters should be non-zero after insert_fast");
        assert!(
            max_val <= 65535,
            "Counters must not exceed 65535, got {}",
            max_val
        );

        // Verify byte-order correctness
        let expected_len = filter.size() * 2;
        assert_eq!(
            raw.len(),
            expected_len,
            "16-bit raw_counters must be size*2 bytes ({} != {})",
            raw.len(),
            expected_len
        );
    }

    #[test]
    fn test_concurrent_mixed_operations() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let filter = Arc::new(Mutex::new(CountingBloomFilter::<i32>::with_size(
            10_000,
            0.01,
            CounterSize::EightBit,
        )));
        let num_threads = 4;
        let ops_per_thread = 200;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for i in 0..ops_per_thread {
                        let key = tid * ops_per_thread + i;
                        let mut guard = f.lock().unwrap();
                        guard.insert(&key);
                        // Immediately verify
                        assert!(guard.contains(&key));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread panicked");
        }

        let guard = filter.lock().unwrap();
        for tid in 0..num_threads {
            for i in 0..ops_per_thread {
                let key = tid * ops_per_thread + i;
                assert!(
                    guard.contains(&key),
                    "Key {} missing after concurrent insert",
                    key
                );
            }
        }
    }

    #[test]
    fn test_concurrent_insert_delete_cycle() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let filter = Arc::new(Mutex::new(CountingBloomFilter::<i32>::with_size(
            10_000,
            0.01,
            CounterSize::SixteenBit,
        )));
        let num_threads = 4;
        let cycles = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for _ in 0..cycles {
                        let key = tid * 10_000;
                        {
                            let mut guard = f.lock().unwrap();
                            guard.insert(&key);
                        }
                        {
                            let mut guard = f.lock().unwrap();
                            let deleted = guard.delete(&key);
                            assert!(deleted, "delete should succeed after insert");
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread panicked");
        }

        // All items should be gone after paired insert/delete
        let guard = filter.lock().unwrap();
        for tid in 0..num_threads {
            let key = tid * 10_000;
            assert!(
                !guard.contains(&key),
                "Key should be absent after full delete cycle"
            );
        }
    }

    // --- 16-bit saturation ---

    #[test]
    fn test_16bit_saturate_at_max() {
        let mut filter = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::SixteenBit);

        // Insert 70000 times — should saturate all k counters at 65535
        for _ in 0..70_000 {
            let _ = filter.insert_checked(&42);
        }

        let raw = filter.raw_counters();
        let max_val = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_val, 65535,
            "16-bit counters must saturate at 65535, got {}",
            max_val
        );

        // insert_checked should now error (all counters saturated)
        let result = filter.insert_checked(&42);
        assert!(
            result.is_err(),
            "insert_checked should error when all counters saturated"
        );
    }

    #[test]
    fn test_16bit_overflow_events_tracked() {
        let mut filter = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::SixteenBit);

        // Saturate
        for _ in 0..70_000 {
            let _ = filter.insert_checked(&42);
        }

        let metrics = filter.health_metrics();
        assert!(metrics.overflow_events > 0, "Should track overflow events");
        assert!(metrics.overflow_risk > 0.0, "Should detect overflow risk");
    }

    // --- Edge cases and invariants ---

    #[test]
    fn test_from_raw_error_zero_size() {
        let result =
            CountingBloomFilter::<i32>::from_raw(0, 7, CounterSize::EightBit, &[], 1000, 0.01);
        assert!(result.is_err(), "Zero size should error");
    }

    #[test]
    fn test_from_raw_error_zero_k() {
        let result = CountingBloomFilter::<i32>::from_raw(
            100,
            0,
            CounterSize::EightBit,
            &[0u8; 100],
            1000,
            0.01,
        );
        assert!(result.is_err(), "Zero hash count should error");
    }

    #[test]
    fn test_from_raw_error_mismatched_counter_bytes() {
        // 8-bit requires exactly `size` bytes
        let result = CountingBloomFilter::<i32>::from_raw(
            100,
            7,
            CounterSize::EightBit,
            &[0u8; 50],
            1000,
            0.01,
        );
        assert!(result.is_err(), "Wrong byte count for 8-bit should error");
    }

    #[test]
    fn test_from_raw_error_mismatched_16bit_bytes() {
        // 16-bit requires exactly `size * 2` bytes
        let result = CountingBloomFilter::<i32>::from_raw(
            100,
            7,
            CounterSize::SixteenBit,
            &[0u8; 150],
            1000,
            0.01,
        );
        assert!(result.is_err(), "Wrong byte count for 16-bit should error");
    }

    #[test]
    fn test_len_empty() {
        let filter = CountingBloomFilter::<i32>::new(100, 0.01);
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_len_after_insert() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.01);
        filter.insert(&1);
        assert_eq!(filter.len(), 1);
        assert!(!filter.is_empty());
        filter.insert(&2);
        assert_eq!(filter.len(), 2);
    }

    #[test]
    fn test_len_after_delete() {
        let mut filter = CountingBloomFilter::<i32>::new(100, 0.01);
        filter.insert(&1);
        filter.insert(&2);
        assert_eq!(filter.len(), 2);
        filter.delete(&1);
        assert_eq!(filter.len(), 1);
        filter.delete(&2);
        assert_eq!(filter.len(), 0);
    }

    #[test]
    fn test_counter_bits_accessor() {
        let f4 = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::FourBit);
        assert_eq!(f4.counter_bits(), 4);

        let f8 = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::EightBit);
        assert_eq!(f8.counter_bits(), 8);

        let f16 = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::SixteenBit);
        assert_eq!(f16.counter_bits(), 16);
    }

    #[test]
    fn test_num_hashes_accessor() {
        let filter = CountingBloomFilter::<i32>::new(100, 0.01);
        assert_eq!(filter.num_hashes(), filter.hash_count());
        assert!(filter.num_hashes() >= 1);
    }

    #[test]
    fn test_partial_eq_different_filters() {
        let mut a = CountingBloomFilter::<i32>::new(100, 0.01);
        let mut b = CountingBloomFilter::<i32>::new(100, 0.01);

        // Same inserts → equal
        a.insert(&1);
        b.insert(&1);
        assert_eq!(a, b);

        // Different inserts → not equal
        a.insert(&2);
        assert_ne!(a, b);
    }

    #[test]
    fn test_partial_eq_different_sizes() {
        let a = CountingBloomFilter::<i32>::with_size(100, 0.01, CounterSize::EightBit);
        let b = CountingBloomFilter::<i32>::with_size(200, 0.01, CounterSize::EightBit);
        assert_ne!(a, b);
    }

    #[test]
    fn test_default_impl() {
        let filter = CountingBloomFilter::<i32>::default();
        assert!(!filter.contains(&42));
        assert_eq!(filter.len(), 0);
    }

    #[test]
    fn test_send_sync_compile_check() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<CountingBloomFilter<i32>>();
        assert_sync::<CountingBloomFilter<i32>>();
    }

    #[test]
    fn test_insert_at_max_count_errors() {
        use crate::BloomCraftError;
        let mut filter = CountingBloomFilter::<i32>::with_size(100, 0.5, CounterSize::FourBit);

        // Insert until all k counters for key 42 are saturated
        for _ in 0..20 {
            let _ = filter.insert_checked(&42);
        }

        // insert() should still work (saturates silently)
        // insert_checked should error
        let result = filter.insert_checked(&42);
        assert!(matches!(
            result,
            Err(BloomCraftError::CapacityExceeded { .. })
        ));
    }

    #[test]
    fn test_insert_fast_saturates_silently() {
        let filter = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::FourBit);

        // insert_fast never errors, even at saturation
        for _ in 0..100 {
            filter.insert_fast(&42);
        }

        // Verify counters are capped, not wrapped
        let raw = filter.raw_counters();
        let max_val = raw.iter().copied().max().unwrap_or(0);
        assert!(max_val <= 15);
    }

    #[test]
    fn test_clear_all_sizes() {
        for cs in [
            CounterSize::FourBit,
            CounterSize::EightBit,
            CounterSize::SixteenBit,
        ] {
            let mut filter = CountingBloomFilter::<i32>::with_size(100, 0.01, cs);
            for i in 0..10 {
                filter.insert(&i);
            }
            assert!(!filter.is_empty());
            filter.clear();
            assert!(filter.is_empty());
            for i in 0..10 {
                assert!(
                    !filter.contains(&i),
                    "Item {} still present after clear ({:?})",
                    i,
                    cs
                );
            }
        }
    }

    #[test]
    fn test_raw_counters_16bit_high_values() {
        let mut filter = CountingBloomFilter::<i32>::with_size(100, 0.5, CounterSize::SixteenBit);

        // Push 16-bit counters well past 255
        for _ in 0..1_000 {
            filter.insert(&42);
        }

        let raw = filter.raw_counters();
        let expected_len = filter.size() * 2;
        assert_eq!(
            raw.len(),
            expected_len,
            "16-bit raw_counters must be size*2 bytes"
        );

        let max_val = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .max()
            .unwrap_or(0);
        assert!(
            max_val > 255,
            "16-bit raw_counters must encode values > 255"
        );

        let restored = CountingBloomFilter::<i32>::from_raw(
            filter.size(),
            filter.hash_count(),
            CounterSize::SixteenBit,
            &raw,
            filter.expected_items(),
            filter.target_fpr(),
        )
        .unwrap();

        assert_eq!(
            restored, filter,
            "16-bit round-trip with high values failed"
        );
    }

    #[test]
    fn test_from_raw_4bit_all_counter_values() {
        let size = 100;
        let k = 7;
        // Build counters with all 16 possible 4-bit values
        let mut counters = vec![0u8; size];
        for (i, counter) in counters.iter_mut().enumerate().take(size.min(16)) {
            *counter = i as u8; // values 0-15
        }

        let filter = CountingBloomFilter::<i32>::from_raw(
            size,
            k,
            CounterSize::FourBit,
            &counters,
            1000,
            0.01,
        )
        .unwrap();

        for i in 0..size.min(16) {
            assert_eq!(
                filter.get_counter(i),
                i,
                "Counter {} should be {}, got {}",
                i,
                i,
                filter.get_counter(i)
            );
        }
    }

    #[test]
    fn test_concurrent_insert_fast_4bit_saturation_stress() {
        use std::sync::Arc;
        use std::thread;

        let filter = Arc::new(CountingBloomFilter::<i32>::with_size(
            10,
            0.5,
            CounterSize::FourBit,
        ));
        let num_threads = 16;
        let ops_per_thread = 50;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for _ in 0..ops_per_thread {
                        f.insert_fast(&42);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread panicked");
        }

        // 16 × 50 = 800 insert_fast on 4-bit (max 15)
        let raw = filter.raw_counters();
        let max_val: usize = raw.iter().copied().map(|v| v as usize).max().unwrap_or(0);
        assert!(
            max_val <= 15,
            "4-bit counters must saturate at 15 under concurrent stress, got {}",
            max_val
        );
    }

    #[test]
    fn test_insert_does_not_wrap() {
        // 4-bit: 1000 inserts → saturates at 15
        let mut f4 = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::FourBit);
        for _ in 0..1000 {
            f4.insert(&42);
        }
        let max_4 = f4.raw_counters().iter().copied().max().unwrap_or(0);
        assert_eq!(max_4, 15, "4-bit must saturate at 15, got {}", max_4);

        // 8-bit: 1000 inserts → saturates at 255
        let mut f8 = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::EightBit);
        for _ in 0..1000 {
            f8.insert(&42);
        }
        let max_8 = f8.raw_counters().iter().copied().max().unwrap_or(0);
        assert_eq!(max_8, 255, "8-bit must saturate at 255, got {}", max_8);

        // 16-bit: 1000 inserts → ~1000/k per counter, well below 65535, no wrap
        let mut f16 = CountingBloomFilter::<i32>::with_size(10, 0.5, CounterSize::SixteenBit);
        for _ in 0..1000 {
            f16.insert(&42);
        }
        let raw = f16.raw_counters();
        let max_16: u64 = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u64)
            .max()
            .unwrap_or(0);
        assert!(
            max_16 > 0,
            "16-bit counters should be non-zero after inserts"
        );
        assert!(max_16 <= 65535, "16-bit must not wrap, got {}", max_16);
        // Not yet at saturation: 1000 << 65535
        assert!(
            max_16 < 65535,
            "16-bit should not be saturated after 1000 inserts"
        );
    }

    #[test]
    fn test_estimate_fpr() {
        let filter = CountingBloomFilter::<i32>::new(100, 0.01);
        let fpr = filter.estimate_fpr();
        assert!((0.0..=1.0).contains(&fpr));
    }
}
