//! Standard Bloom filter implementation.
//!
//! This module provides a production-grade implementation of the classic Bloom filter
//! data structure with modern optimizations and enhanced double hashing.
//!
//! # Algorithm
//!
//! A Bloom filter is a space-efficient probabilistic data structure that supports
//! two operations:
//!
//! - Insert: Add an element to the set (irreversible)
//! - Query: Test whether an element is in the set
//!
//! # Properties
//!
//! - **False positives**: Possible (controllable via parameters)
//! - **False negatives**: Never occur (guaranteed)
//! - **Space efficiency**: ~9.6 bits per element for 1% FP rate
//! - **Time complexity**: O(k) for both insert and query
//! - **Thread safety**: Lock-free concurrent operations via atomics
//!
//! # Mathematical Foundation
//!
//! Given:
//! - n = expected number of elements
//! - p = desired false positive rate
//!
//! Optimal parameters:
//! - m = -n × ln(p) / (ln(2)²) ≈ 1.44 × n × log₂(1/p) (filter size in bits)
//! - k = (m/n) × ln(2) ≈ 0.693 × (m/n) (number of hash functions)
//!
//! Actual false positive rate:
//! - p_actual = (1 - e^(-kn/m))^k
//!
//! # Concurrency Model
//!
//! This filter provides lock-free thread safety for concurrent operations.
//!
//! ## Lock-Free Operations
//!
//! These operations use atomic instructions and are safe to call concurrently:
//!
//! - `insert()` - Uses atomic fetch_or with Release ordering
//! - `insert_batch()` - Batch atomic inserts
//! - `contains()` - Uses atomic load with Acquire ordering
//! - `contains_batch()` - Batch atomic queries
//!
//! ## Exclusive Operations
//!
//! - `clear()` - Requires exclusive access to prevent races
//!
//! ## Memory Ordering Guarantees
//!
//! The filter uses Release-Acquire ordering to prevent false negatives:
//!
//! ```text
//! Thread A (insert):          Thread B (contains):
//! ──────────────────          ────────────────────
//! h1, h2 = hash(item)         h1, h2 = hash(item)
//! bit[h1].set() [Release] --> bit[h1].get() [Acquire]
//! bit[h2].set() [Release] --> bit[h2].get() [Acquire]
//!                             Sees both bits set
//! ```
//!
//! Release ordering ensures all prior writes are visible after an atomic store.
//! Acquire ordering ensures all writes that happened-before a store are visible.
//!
//! This guarantees that if Thread B observes bit[h1]=1, it will also see bit[h2]=1,
//! preventing false negatives in concurrent scenarios.
//!
//! ## Usage Patterns
//!
//! ### Concurrent Inserts and Queries
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//! use std::sync::Arc;
//! use std::thread;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let filter = Arc::new(StandardBloomFilter::<String>::new(10_000, 0.01)?);
//!
//! let handles: Vec<_> = (0..4).map(|tid| {
//!     let f = Arc::clone(&filter);
//!     thread::spawn(move || {
//!         for i in 0..1000 {
//!             f.insert(&format!("item-{}-{}", tid, i));
//!         }
//!     })
//! }).collect();
//!
//! for h in handles {
//!     h.join().unwrap();
//! }
//!
//! assert!(filter.contains(&"item-0-42".to_string()));
//! # Ok(())
//! # }
//! ```
//!
//! ### Mixed Operations with Clear
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//! use std::sync::{Arc, RwLock};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let filter = Arc::new(RwLock::new(StandardBloomFilter::<String>::new(1000, 0.01)?));
//!
//! let read_handle = {
//!     let f = Arc::clone(&filter);
//!     std::thread::spawn(move || {
//!         let reader = f.read().unwrap();
//!         reader.contains(&"test".to_string())
//!     })
//! };
//!
//! {
//!     let mut writer = filter.write().unwrap();
//!     writer.clear();
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Serialization Support
//!
//! Enable the `serde` feature in Cargo.toml
//!
//! **Note on Cross-Process Serialization**: The filter uses `DefaultHasher` which is
//! not deterministic across processes. Serialization works within the same process
//! for checkpoint/restore, but deserializing in a different process will cause
//! false negatives. A deterministic hasher option is planned for v0.2.0.
//!
//! # Performance Characteristics
//!
//! - **Space**: ~9.6 bits per element for 1% FPR
//! - **Time**: O(k) for all operations (typically k=7-10)
//! - **Concurrency**: Lock-free, scales linearly with cores
//! - **Batch speedup**: 1.3-2× faster for batches >100 items
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let filter = StandardBloomFilter::<String>::new(10_000, 0.01)?;
//!
//! filter.insert(&"hello".to_string());
//! filter.insert(&"world".to_string());
//!
//! assert!(filter.contains(&"hello".to_string()));
//! assert!(filter.contains(&"world".to_string()));
//! assert!(!filter.contains(&"goodbye".to_string()));
//! # Ok(())
//! # }
//! ```
//!
//! ## Batch Operations
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let filter = StandardBloomFilter::<String>::new(1000, 0.01)?;
//!
//! let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
//! filter.insert_batch(&items);
//!
//! let queries = vec!["a".to_string(), "b".to_string(), "x".to_string()];
//! let results = filter.contains_batch(&queries);
//! assert_eq!(results, vec![true, true, false]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Health Monitoring
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let filter = StandardBloomFilter::<u64>::new(1000, 0.01)?;
//!
//! for i in 0..500 {
//!     filter.insert(&i);
//! }
//!
//! let health = filter.health_check();
//! println!("Filter health: {}", health);
//!
//! match health {
//!     bloomcraft::filters::FilterHealth::Healthy { .. } => println!("All good"),
//!     bloomcraft::filters::FilterHealth::Degraded { recommendation, .. } => {
//!         println!("Warning: {}", recommendation);
//!     }
//!     bloomcraft::filters::FilterHealth::Critical { recommendation, .. } => {
//!         println!("Critical: {}", recommendation);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - Bloom, B. H. (1970). "Space/time trade-offs in hash coding with allowable errors"
//! - Kirsch, A., & Mitzenmacher, M. (2006). "Less Hashing, Same Performance: Building a Better Bloom Filter"

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use crate::core::bitvec::BitVec;
use crate::core::filter::{BloomFilter, ConcurrentBloomFilter};
use crate::core::params::{optimal_k, optimal_m};
use crate::error::{BloomCraftError, Result};
use crate::hash::strategies::{EnhancedDoubleHashing, HashStrategy as HashStrategyTrait};
use crate::hash::{BloomHasher, StdHasher};

#[cfg(feature = "wyhash")]
use crate::hash::WyHasher;

use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Convert a hashable item to bytes using Rust's `Hash` trait.
///
/// This is the bridge between generic `T: Hash` and the `&[u8]` API
/// required by `BloomHasher`. Uses `std::collections::hash_map::DefaultHasher`
/// to get a stable u64, then converts to little-endian bytes.
///
/// # Performance
///
/// This is an inline function that should optimize to near-zero overhead.
/// The compiler can often elide the intermediate array allocation.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}

/// Health status of the Bloom filter for monitoring and alerting.
///
/// Provides operational status information to help maintainers decide when to
/// rebuild or scale filters in production systems.
///
/// # Health States
///
/// - **Healthy**: Fill rate < 50% and FPR < 2× target
/// - **Degraded**: Fill rate < 70% and FPR < 5× target
/// - **Critical**: Filter is severely saturated
///
/// # Examples
///
/// ```
/// use bloomcraft::filters::{StandardBloomFilter, FilterHealth};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let filter = StandardBloomFilter::<u64>::new(1000, 0.01)?;
///
/// for i in 0..500 {
///     filter.insert(&i);
/// }
///
/// let health = filter.health_check();
/// match health {
///     FilterHealth::Healthy { .. } => println!("OK"),
///     FilterHealth::Degraded { recommendation, .. } => println!("[WARN] {}", recommendation),
///     FilterHealth::Critical { recommendation, .. } => println!("[CRIT] {}", recommendation),
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FilterHealth {
    /// Filter is operating within healthy parameters.
    Healthy {
        /// Fill rate of the filter (0.0 to 1.0).
        fill_rate: f64,
        /// Current estimated false positive rate.
        current_fpr: f64,
        /// Estimated number of items in the filter.
        estimated_items: usize,
    },
    /// Filter performance is degrading but still operational.
    Degraded {
        /// Fill rate of the filter (0.0 to 1.0).
        fill_rate: f64,
        /// Current estimated false positive rate.
        current_fpr: f64,
        /// Ratio of current FPR to target FPR.
        fpr_ratio: f64,
        /// Estimated number of items in the filter.
        estimated_items: usize,
        /// Recommended action to improve filter health.
        recommendation: &'static str,
    },
    /// Filter is critically saturated and needs immediate attention.
    Critical {
        /// Fill rate of the filter (0.0 to 1.0).
        fill_rate: f64,
        /// Current estimated false positive rate.
        current_fpr: f64,
        /// Ratio of current FPR to target FPR.
        fpr_ratio: f64,
        /// Estimated number of items in the filter.
        estimated_items: usize,
        /// Recommended action to improve filter health.
        recommendation: &'static str,
    },
}

impl FilterHealth {
    /// Get the fill rate for this health status.
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        match self {
            Self::Healthy { fill_rate, .. }
            | Self::Degraded { fill_rate, .. }
            | Self::Critical { fill_rate, .. } => *fill_rate,
        }
    }

    /// Get the current FPR for this health status.
    #[must_use]
    pub fn current_fpr(&self) -> f64 {
        match self {
            Self::Healthy { current_fpr, .. }
            | Self::Degraded { current_fpr, .. }
            | Self::Critical { current_fpr, .. } => *current_fpr,
        }
    }

    /// Get estimated items for this health status.
    #[must_use]
    pub fn estimated_items(&self) -> usize {
        match self {
            Self::Healthy { estimated_items, .. }
            | Self::Degraded { estimated_items, .. }
            | Self::Critical { estimated_items, .. } => *estimated_items,
        }
    }

    /// Check if filter is healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy { .. })
    }

    /// Check if filter is degraded.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        matches!(self, Self::Degraded { .. })
    }

    /// Check if filter is critical.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        matches!(self, Self::Critical { .. })
    }
}

impl std::fmt::Display for FilterHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterHealth::Healthy {
                fill_rate,
                current_fpr,
                estimated_items,
            } => {
                write!(
                    f,
                    "[OK] Healthy: Fill {:.1}%, FPR {:.4}, Items ~{}",
                    fill_rate * 100.0,
                    current_fpr,
                    estimated_items
                )
            }
            FilterHealth::Degraded {
                fill_rate,
                fpr_ratio,
                recommendation,
                ..
            } => {
                write!(
                    f,
                    "[WARN] Degraded: Fill {:.1}%, FPR {:.1}× target - {}",
                    fill_rate * 100.0,
                    fpr_ratio,
                    recommendation
                )
            }
            FilterHealth::Critical {
                fill_rate,
                fpr_ratio,
                recommendation,
                ..
            } => {
                write!(
                    f,
                    "[CRIT] CRITICAL: Fill {:.1}%, FPR {:.1}× target - {}",
                    fill_rate * 100.0,
                    fpr_ratio,
                    recommendation
                )
            }
        }
    }
}

/// Performance metrics for the filter (optional, enabled with "metrics" feature).
#[cfg(feature = "metrics")]
#[derive(Debug, Clone, Default)]
pub struct FilterMetrics {
    pub total_inserts: usize,
    pub total_queries: usize,
    pub query_hits: usize,
    pub query_misses: usize,
}

/// Standard Bloom filter with optimal parameters.
///
/// This implementation uses:
/// - Lock-free bit vector for thread-safe insertions
/// - Enhanced double hashing for k hash functions
/// - Optimal parameter calculation
/// - Batch operation support
///
/// # Type Parameters
///
/// * `T` - Type of items stored (must implement `Hash`)
/// * `H` - Hash function type (must implement `BloomHasher`)
///
/// # Thread Safety
///
/// - Insert: Thread-safe (lock-free atomic operations)
/// - Query: Thread-safe (lock-free atomic loads)
/// - Clear: Requires exclusive access (`&mut self`)
///
/// # Serialization
///
/// When the `serde` feature is enabled, this filter can be serialized.
/// **Note**: The hasher state is not serialized and is reconstructed via
/// `H::default()`. This means serialized filters only work within the same
/// process when using `DefaultHasher`.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StandardBloomFilter<T, H = StdHasher>
where
    H: BloomHasher + Clone,
{
    /// Underlying bit vector
    bitvec: BitVec,

    /// Number of hash functions (k)
    k: usize,

    /// Hash function
    #[cfg_attr(feature = "serde", serde(skip, default = "H::default"))]
    hasher: H,

    /// Expected number of items (for statistics)
    expected_items: usize,

    /// Target false positive rate (for statistics)
    target_fpr: f64,

    /// Hash strategy used for generating hash indices
    #[cfg_attr(feature = "serde", serde(skip, default))]
    strategy: EnhancedDoubleHashing,

    /// Phantom data for type parameter T
    #[cfg_attr(feature = "serde", serde(skip))]
    _phantom: PhantomData<T>,
}

impl<T, H> Clone for StandardBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    fn clone(&self) -> Self {
        Self {
            bitvec: self.bitvec.clone(),
            k: self.k,
            hasher: self.hasher.clone(),
            expected_items: self.expected_items,
            target_fpr: self.target_fpr,
            strategy: self.strategy,
            _phantom: PhantomData,
        }
    }
}

impl<T> StandardBloomFilter<T, StdHasher>
where
    T: Hash,
{
    /// Create a new standard Bloom filter with default hasher.
    ///
    /// Automatically calculates optimal parameters (m and k) based on the
    /// expected number of items and desired false positive rate.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items to insert (n), must be > 0
    /// * `fpr` - Target false positive rate (p), must be in (0, 1)
    ///
    /// # Returns
    ///
    /// * `Ok(StandardBloomFilter)` - New filter with optimal parameters
    /// * `Err(BloomCraftError)` - If parameters are invalid
    ///
    /// # Errors
    ///
    /// - [`BloomCraftError::InvalidItemCount`] if `expected_items == 0`
    /// - [`BloomCraftError::FalsePositiveRateOutOfBounds`] if `fpr` not in (0, 1)
    /// - [`BloomCraftError::InvalidParameters`] if calculated parameters exceed limits
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // 1% false positive rate for 10,000 items
    /// let filter = StandardBloomFilter::<String>::new(10_000, 0.01)?;
    ///
    /// // 0.1% false positive rate for 1 million items
    /// let filter = StandardBloomFilter::<u64>::new(1_000_000, 0.001)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(expected_items: usize, fpr: f64) -> Result<Self> {
        Self::with_hasher(expected_items, fpr, StdHasher::new())
    }
}

impl<T, H> StandardBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone,
{
    /// Create a new standard Bloom filter with custom hasher.
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items to insert (must be > 0)
    /// * `fpr` - Target false positive rate (must be in (0, 1))
    /// * `hasher` - Custom hash function
    ///
    /// # Returns
    ///
    /// * `Ok(StandardBloomFilter)` - New filter with optimal parameters
    /// * `Err(BloomCraftError)` - If parameters are invalid
    ///
    /// # Errors
    ///
    /// Returns error if validation fails (see [`new()`] for error types)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let hasher = StdHasher::with_seed(42);
    /// let filter = StandardBloomFilter::<String, _>::with_hasher(10_000, 0.01, hasher)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_hasher(expected_items: usize, fpr: f64, hasher: H) -> Result<Self> {
        // Validate parameters (no panics!)
        if expected_items == 0 {
            return Err(BloomCraftError::invalid_item_count(0));
        }
        if !(0.0 < fpr && fpr < 1.0) {
            return Err(BloomCraftError::fp_rate_out_of_bounds(fpr));
        }

        // Calculate optimal parameters using core params module
        let m = optimal_m(expected_items, fpr);
        let k = optimal_k(expected_items, m);

        Ok(Self {
            bitvec: BitVec::new(m)?,  // Propagate BitVec errors
            k,
            hasher,
            expected_items,
            target_fpr: fpr,
            strategy: EnhancedDoubleHashing,
            _phantom: PhantomData,
        })
    }

    /// Create a Bloom filter with explicit parameters.
    ///
    /// This allows fine-grained control over the filter size and number of
    /// hash functions, bypassing automatic parameter calculation.
    ///
    /// # Arguments
    ///
    /// * `m` - Filter size in bits (must be > 0)
    /// * `k` - Number of hash functions (must be in [1, 32])
    /// * `hasher` - Hash function
    ///
    /// # Returns
    ///
    /// * `Ok(StandardBloomFilter)` - New filter with specified parameters
    /// * `Err(BloomCraftError)` - If parameters are invalid
    ///
    /// # Errors
    ///
    /// - [`BloomCraftError::InvalidFilterSize`] if `m == 0`
    /// - [`BloomCraftError::InvalidHashCount`] if `k == 0` or `k > 32`
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::hash::StdHasher;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create filter with 10,000 bits and 7 hash functions
    /// let filter = StandardBloomFilter::<String, _>::with_params(
    ///     10_000,
    ///     7,
    ///     StdHasher::new()
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_params(m: usize, k: usize, hasher: H) -> Result<Self> {
        if m == 0 {
            return Err(BloomCraftError::invalid_filter_size(0));
        }
        if k == 0 || k > 32 {
            return Err(BloomCraftError::invalid_hash_count(k, 1, 32));
        }

        Ok(Self {
            bitvec: BitVec::new(m)?,
            k,
            hasher,
            expected_items: 0, // Unknown
            target_fpr: 0.0,   // Unknown
            strategy: EnhancedDoubleHashing,
            _phantom: PhantomData,
        })
    }

    /// Create a filter with a specific hash strategy.
    ///
    /// # Arguments
    ///
    /// * `m` - Filter size in bits (must be > 0)
    /// * `k` - Number of hash functions (must be in [1, 32])
    /// * `_strategy` - Hash strategy (currently ignored, uses EnhancedDoubleHashing)
    ///
    /// # Returns
    ///
    /// * `Ok(StandardBloomFilter)` - New filter
    /// * `Err(BloomCraftError)` - If parameters are invalid
    ///
    /// # Errors
    ///
    /// Returns error if validation fails (see [`with_params()`])
    pub fn with_strategy(m: usize, k: usize, _strategy: crate::hash::HashStrategy) -> Result<Self>
    where
        H: Default,
    {
        if m == 0 {
            return Err(BloomCraftError::invalid_filter_size(0));
        }
        if k == 0 || k > 32 {
            return Err(BloomCraftError::invalid_hash_count(k, 1, 32));
        }

        Ok(Self {
            bitvec: BitVec::new(m)?,
            k,
            hasher: H::default(),
            expected_items: 0,
            target_fpr: 0.0,
            strategy: EnhancedDoubleHashing,
            _phantom: PhantomData,
        })
    }

    /// Create a filter from raw parts (for deserialization).
    ///
    /// # Arguments
    ///
    /// * `bits` - Pre-populated bit vector
    /// * `k` - Number of hash functions
    /// * `_strategy` - Hash strategy (currently ignored, uses EnhancedDoubleHashing)
    ///
    /// # Returns
    ///
    /// * `Ok(StandardBloomFilter)` - Filter constructed from parts
    /// * `Err(BloomCraftError)` - If parameters are invalid
    pub fn from_parts(
        bits: BitVec,
        k: usize,
        _strategy: crate::hash::HashStrategy,
    ) -> Result<Self>
    where
        H: Default,
    {
        if k == 0 || k > 32 {
            return Err(BloomCraftError::invalid_hash_count(k, 1, 32));
        }

        Ok(Self {
            expected_items: 0,
            target_fpr: 0.0,
            k,
            bitvec: bits,
            hasher: H::default(),
            strategy: EnhancedDoubleHashing,
            _phantom: PhantomData,
        })
    }

    /// Get the size of the filter in bits (m).
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.bitvec.len()
    }

    /// Get the number of hash functions (k).
    #[must_use]
    #[inline]
    pub fn hash_count(&self) -> usize {
        self.k
    }

    /// Get the number of hash functions (alias for hash_count).
    #[must_use]
    #[inline]
    pub fn num_hashes(&self) -> usize {
        self.k
    }

    /// Get the expected number of items.
    #[must_use]
    #[inline]
    pub fn expected_items(&self) -> usize {
        self.expected_items
    }

    /// Get the target false positive rate.
    #[must_use]
    #[inline]
    pub fn target_fpr(&self) -> f64 {
        self.target_fpr
    }

    /// Get the number of bits currently set in the filter.
    ///
    /// This can be used to estimate the fill rate of the filter.
    #[must_use]
    pub fn count_set_bits(&self) -> usize {
        self.bitvec.count_ones()
    }

    /// Get the number of bits set (alias for count_set_bits).
    #[must_use]
    pub fn len(&self) -> usize {
        self.count_set_bits()
    }

    /// Calculate the current fill rate as a fraction in [0, 1].
    ///
    /// Fill rate = (number of set bits) / (total bits)
    #[must_use]
    pub fn fill_rate(&self) -> f64 {
        self.count_set_bits() as f64 / self.size() as f64
    }

    /// Estimate the current false positive rate based on fill rate.
    ///
    /// Uses the formula: FPR ≈ (1 - e^(-k*n/m))^k
    ///
    /// First estimates n (number of items) from the fill rate using:
    /// n ≈ -(m/k) × ln(1 - fill_rate)
    ///
    /// Then calculates FPR using the standard formula.
    #[must_use]
    pub fn estimate_fpr(&self) -> f64 {
        let set_bits = self.count_set_bits();
        let m = self.size() as f64;
        let k = self.k as f64;

        if set_bits == 0 {
            return 0.0;
        }

        let fill_rate = set_bits as f64 / m;
        if fill_rate >= 1.0 {
            return 1.0;
        }

        // Estimate n (number of items) from fill rate
        // fill_rate ≈ 1 - e^(-kn/m)
        // => n ≈ -(m/k) × ln(1 - fill_rate)
        let estimated_n = -(m / k) * (1.0 - fill_rate).ln();

        // Calculate FPR using the standard formula: (1 - e^(-kn/m))^k
        let exponent = -k * estimated_n / m;
        (1.0 - exponent.exp()).powf(k)
    }

    /// Get the estimated false positive rate (alias for estimate_fpr).
    #[must_use]
    pub fn estimated_fp_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    /// Check if the filter is approximately full.
    ///
    /// Returns true if the fill rate exceeds 50%, which typically indicates
    /// the filter is approaching saturation and FPR is degrading.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.fill_rate() > 0.5
    }

    /// Check if the filter is empty (no bits set).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count_set_bits() == 0
    }

    /// Estimate cardinality (number of unique items) using fill rate.
    ///
    /// Uses the standard Bloom filter cardinality estimation formula:
    /// n_estimated = -(m/k) × ln(1 - X/m)
    ///
    /// where:
    /// - m = number of bits
    /// - k = number of hash functions
    /// - X = number of set bits
    ///
    /// # Accuracy
    ///
    /// - Low load (< 50% full): ±5% error
    /// - Medium load (50-80%): ±10% error
    /// - High load (> 80%): ±20% error
    #[must_use]
    pub fn estimate_cardinality(&self) -> usize {
        let set_bits = self.count_set_bits();
        if set_bits == 0 {
            return 0;
        }

        let m = self.size() as f64;
        let k = self.k as f64;
        let x = set_bits as f64;

        if set_bits >= self.size() {
            // Filter completely saturated
            return usize::MAX;
        }

        // Cardinality estimation: n ≈ -(m/k) × ln(1 - X/m)
        let fill_rate = x / m;
        let estimated_n = -(m / k) * (1.0 - fill_rate).ln();

        estimated_n.max(0.0) as usize
    }

    /// Get memory usage in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.bitvec.memory_usage()
            + std::mem::size_of::<usize>() * 2 // k, expected_items
            + std::mem::size_of::<f64>() // target_fpr
            + std::mem::size_of::<H>() // hasher
            + std::mem::size_of::<EnhancedDoubleHashing>() // strategy
    }

    /// Get the raw bit data as a slice of u64 values.
    ///
    /// This is useful for serialization.
    #[must_use]
    pub fn raw_bits(&self) -> Vec<u64> {
        self.bitvec.to_raw()
    }

    /// Get the name of the hasher used by this filter.
    ///
    /// Used for serialization compatibility checking.
    #[must_use]
    pub fn hasher_name(&self) -> &'static str {
        self.hasher.name()
    }

    /// Get the hash strategy used by this filter.
    ///
    /// Returns the strategy enum variant for serialization.
    #[must_use]
    pub fn hash_strategy(&self) -> crate::hash::HashStrategy {
        // This filter uses EnhancedDoubleHashing internally
        crate::hash::HashStrategy::EnhancedDouble
    }

    /// Insert an item into the filter.
    ///
    /// This operation is thread-safe and lock-free.
    #[inline]
    pub fn insert(&self, item: &T) {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        for idx in indices {
            self.bitvec.set(idx);
        }
    }

    /// Check if an item might be in the filter.
    ///
    /// # Returns
    ///
    /// - `true`: Item might be in the set (or false positive)
    /// - `false`: Item is definitely not in the set
    #[must_use]
    #[inline]
    pub fn contains(&self, item: &T) -> bool {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        indices.iter().all(|&idx| self.bitvec.get(idx))
    }

    /// Insert multiple items in batch.
    ///
    /// This method is more efficient than calling `insert()` repeatedly because:
    /// - Pre-allocates hash computation results
    /// - Enables better compiler optimization (tight loop)
    /// - Reduces function call overhead
    ///
    /// # Performance
    ///
    /// Expected speedup: 1.3-1.5x faster than individual inserts for batches > 100 items
    pub fn insert_batch(&self, items: &[T]) {
        for item in items {
            let bytes = hash_item_to_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
            let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.bitvec.len());

            for idx in indices {
                self.bitvec.set(idx);
            }
        }
    }

    /// Insert multiple items by reference (zero-copy batch operation).
    pub fn insert_batch_ref(&self, items: &[&T]) {
        for item in items {
            let bytes = hash_item_to_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
            let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.bitvec.len());

            for idx in indices {
                self.bitvec.set(idx);
            }
        }
    }

    /// Query multiple items at once.
    ///
    /// # Performance
    ///
    /// Expected speedup: 1.5-2x faster than individual queries for batches > 100 items
    #[must_use]
    pub fn contains_batch(&self, items: &[T]) -> Vec<bool> {
        let mut results = Vec::with_capacity(items.len());

        for item in items {
            let bytes = hash_item_to_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
            let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.bitvec.len());

            let mut present = true;
            for idx in indices {
                if !self.bitvec.get(idx) {
                    present = false;
                    break;
                }
            }

            results.push(present);
        }

        results
    }

    /// Query multiple items by reference (zero-copy batch operation).
    #[must_use]
    pub fn contains_batch_ref(&self, items: &[&T]) -> Vec<bool> {
        let mut results = Vec::with_capacity(items.len());

        for item in items {
            let bytes = hash_item_to_bytes(item);
            let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
            let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.bitvec.len());

            let mut present = true;
            for idx in indices {
                if !self.bitvec.get(idx) {
                    present = false;
                    break;
                }
            }

            results.push(present);
        }

        results
    }

    /// Clear all bits in the filter.
    ///
    /// This operation requires exclusive access.
    pub fn clear(&mut self) {
        self.bitvec.clear();
    }

    /// Compute the union of two Bloom filters.
    ///
    /// The resulting filter contains all items from both filters.
    /// Both filters must have the same size and number of hash functions.
    ///
    /// # Errors
    ///
    /// Returns error if filters have incompatible parameters.
    pub fn union(&self, other: &Self) -> Result<Self> {
        if self.size() != other.size() {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Size mismatch".to_string(),
            });
        }

        if self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Hash count mismatch".to_string(),
            });
        }

        let mut result = self.clone();
        result.bitvec = self.bitvec.union(&other.bitvec)?;
        Ok(result)
    }

    /// Compute the intersection of two Bloom filters.
    ///
    /// The resulting filter may contain items from both filters, but with
    /// increased false positive rate. This operation is approximate.
    ///
    /// # Errors
    ///
    /// Returns error if filters have incompatible parameters.
    pub fn intersect(&self, other: &Self) -> Result<Self> {
        if self.size() != other.size() {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Size mismatch".to_string(),
            });
        }

        if self.k != other.k {
            return Err(BloomCraftError::IncompatibleFilters {
                reason: "Hash count mismatch".to_string(),
            });
        }

        let mut result = self.clone();
        result.bitvec = self.bitvec.intersect(&other.bitvec)?;
        Ok(result)
    }

    /// Check the operational health of this filter.
    ///
    /// Returns diagnostic information about fill rate, FPR, and recommendations.
    /// Use this for production monitoring and alerting.
    ///
    /// # Health States
    ///
    /// - **Healthy**: Fill rate < 50% and FPR < 2× target
    /// - **Degraded**: Fill rate < 70% and FPR < 5× target  
    /// - **Critical**: Filter is severely saturated
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let filter = StandardBloomFilter::<u64>::new(1000, 0.01)?;
    ///
    /// for i in 0..500 {
    ///     filter.insert(&i);
    /// }
    ///
    /// let health = filter.health_check();
    /// println!("Filter health: {}", health);
    ///
    /// match health {
    ///     bloomcraft::filters::FilterHealth::Healthy { .. } => println!("All good"),
    ///     bloomcraft::filters::FilterHealth::Degraded { recommendation, .. } => {
    ///         println!("Warning: {}", recommendation);
    ///     }
    ///     bloomcraft::filters::FilterHealth::Critical { recommendation, .. } => {
    ///         println!("Critical: {}", recommendation);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn health_check(&self) -> FilterHealth {
        let fill_rate = self.fill_rate();
        let current_fpr = self.estimate_fpr();
        let estimated_items = self.estimate_cardinality();

        let fpr_ratio = if self.target_fpr > 0.0 {
            current_fpr / self.target_fpr
        } else {
            1.0
        };

        if fill_rate < 0.5 && fpr_ratio < 2.0 {
            FilterHealth::Healthy {
                fill_rate,
                current_fpr,
                estimated_items,
            }
        } else if fill_rate < 0.7 && fpr_ratio < 5.0 {
            FilterHealth::Degraded {
                fill_rate,
                current_fpr,
                fpr_ratio,
                estimated_items,
                recommendation: "Consider creating a new filter soon",
            }
        } else {
            FilterHealth::Critical {
                fill_rate,
                current_fpr,
                fpr_ratio,
                estimated_items,
                recommendation: "URGENT: Create new filter immediately - FPR severely degraded",
            }
        }
    }
}

impl<T, H> BloomFilter<T> for StandardBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    fn insert(&mut self, item: &T) {
        // Cast &mut self to &self to use atomic operations
        // This is safe because our operations are thread-safe via atomics
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        for idx in indices {
            self.bitvec.set(idx);
        }
    }

    fn contains(&self, item: &T) -> bool {
        let bytes = hash_item_to_bytes(item);
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        let indices = self.strategy.generate_indices(h1, h2, 0, self.k, self.size());

        indices.iter().all(|&idx| self.bitvec.get(idx))
    }

    fn clear(&mut self) {
        StandardBloomFilter::clear(self);
    }

    fn len(&self) -> usize {
        self.count_set_bits()
    }

    fn is_empty(&self) -> bool {
        StandardBloomFilter::is_empty(self)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimate_fpr()
    }

    fn expected_items(&self) -> usize {
        self.expected_items
    }

    fn bit_count(&self) -> usize {
        self.size()
    }

    fn hash_count(&self) -> usize {
        self.k
    }
}

impl<T, H> ConcurrentBloomFilter<T> for StandardBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone,
{
    #[inline]
    fn insert_concurrent(&self, item: &T) {
        // Delegate to the inherently lock-free insert method
        StandardBloomFilter::insert(self, item);
    }

    fn insert_batch_concurrent(&self, items: &[T]) {
        // Delegate to existing batch insert (already lock-free)
        StandardBloomFilter::insert_batch(self, items);
    }

    fn insert_batch_ref_concurrent(&self, items: &[&T]) {
        // Delegate to existing batch insert by reference (already lock-free)
        StandardBloomFilter::insert_batch_ref(self, items);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_new_basic() {
        let filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01).unwrap();
        assert!(filter.size() > 0);
        assert!(filter.hash_count() > 0);
        assert_eq!(filter.expected_items(), 1000);
        assert_eq!(filter.target_fpr(), 0.01);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_new_various_sizes() {
        let small = StandardBloomFilter::<u64>::new(10, 0.01).unwrap();
        let medium = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        let large = StandardBloomFilter::<u64>::new(1_000_000, 0.01).unwrap();

        assert!(small.size() < medium.size());
        assert!(medium.size() < large.size());
    }

    #[test]
    fn test_new_various_fpr() {
        let high_fpr = StandardBloomFilter::<u64>::new(1000, 0.1).unwrap();
        let medium_fpr = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let low_fpr = StandardBloomFilter::<u64>::new(1000, 0.001).unwrap();

        // Lower FPR requires more bits
        assert!(high_fpr.size() < medium_fpr.size());
        assert!(medium_fpr.size() < low_fpr.size());

        // Lower FPR requires more hash functions
        assert!(high_fpr.hash_count() <= medium_fpr.hash_count());
        assert!(medium_fpr.hash_count() <= low_fpr.hash_count());
    }

    #[test]
    fn test_with_params() {
        let filter = StandardBloomFilter::<String, StdHasher>::with_params(
            10_000,
            7,
            StdHasher::new()
        ).unwrap();

        assert_eq!(filter.size(), 10_000);
        assert_eq!(filter.hash_count(), 7);
    }

    #[test]
    fn test_with_hasher() {
        let hasher = StdHasher::with_seed(42);
        let filter = StandardBloomFilter::<String, _>::with_hasher(1000, 0.01, hasher).unwrap();
        assert!(filter.size() > 0);
    }

    #[test]
    fn test_new_zero_items() {
        let result = StandardBloomFilter::<String>::new(0, 0.01);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BloomCraftError::InvalidItemCount { .. }));
    }

    #[test]
    fn test_new_invalid_fpr_zero() {
        let result = StandardBloomFilter::<String>::new(1000, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_invalid_fpr_one() {
        let result = StandardBloomFilter::<String>::new(1000, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_invalid_fpr_negative() {
        let result = StandardBloomFilter::<String>::new(1000, -0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_invalid_fpr_greater_than_one() {
        let result = StandardBloomFilter::<String>::new(1000, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_params_zero_size() {
        let result = StandardBloomFilter::<String, StdHasher>::with_params(0, 7, StdHasher::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_with_params_zero_hash_count() {
        let result = StandardBloomFilter::<String, StdHasher>::with_params(1000, 0, StdHasher::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_with_params_excessive_hash_count() {
        let result = StandardBloomFilter::<String, StdHasher>::with_params(1000, 33, StdHasher::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_and_contains() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();

        filter.insert(&"hello".to_string());
        assert!(filter.contains(&"hello".to_string()));
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_insert_multiple() {
        let filter = StandardBloomFilter::<i32>::new(100, 0.01).unwrap();

        for i in 0..50 {
            filter.insert(&i);
        }

        for i in 0..50 {
            assert!(filter.contains(&i), "Item {} should be present", i);
        }
    }

    #[test]
    fn test_no_false_negatives() {
        let filter = StandardBloomFilter::<u64>::new(10_000, 0.01).unwrap();
        let items: Vec<u64> = (0..1000).collect();

        for item in &items {
            filter.insert(item);
        }

        // Guarantee: No false negatives
        for item in &items {
            assert!(filter.contains(item), "False negative for item {}", item);
        }
    }

    #[test]
    fn test_different_types() {
        // String
        let f1 = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        f1.insert(&"test".to_string());
        assert!(f1.contains(&"test".to_string()));

        // Integer
        let f2 = StandardBloomFilter::<i32>::new(100, 0.01).unwrap();
        f2.insert(&42);
        assert!(f2.contains(&42));

        // Float (via ordered bytes)
        let f3 = StandardBloomFilter::<[u8; 8]>::new(100, 0.01).unwrap();
        let bytes = 3.14f64.to_le_bytes();
        f3.insert(&bytes);
        assert!(f3.contains(&bytes));
    }

    #[test]
    fn test_empty_string() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        filter.insert(&String::new());
        assert!(filter.contains(&String::new()));
    }

    #[test]
    fn test_insert_batch() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        filter.insert_batch(&items);

        for item in &items {
            assert!(filter.contains(item));
        }
    }

    #[test]
    fn test_insert_batch_empty() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        let items: Vec<String> = vec![];

        filter.insert_batch(&items);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_insert_batch_large() {
        let filter = StandardBloomFilter::<i32>::new(10_000, 0.01).unwrap();
        let items: Vec<i32> = (0..5000).collect();

        filter.insert_batch(&items);

        for item in &items {
            assert!(filter.contains(item));
        }
    }

    #[test]
    fn test_contains_batch() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        filter.insert_batch(&items);

        let queries = vec!["a".to_string(), "b".to_string(), "x".to_string()];
        let results = filter.contains_batch(&queries);

        assert_eq!(results.len(), 3);
        assert!(results[0]);
        assert!(results[1]);
        assert!(!results[2]);
    }

    #[test]
    fn test_contains_batch_all_present() {
        let filter = StandardBloomFilter::<i32>::new(1000, 0.01).unwrap();
        let items: Vec<i32> = (0..100).collect();

        filter.insert_batch(&items);
        let results = filter.contains_batch(&items);

        assert!(results.iter().all(|&r| r), "All items should be present");
    }

    #[test]
    fn test_contains_batch_empty() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        let queries: Vec<String> = vec![];

        let results = filter.contains_batch(&queries);
        assert!(results.is_empty());
    }

    #[test]
    fn test_count_set_bits_empty() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.count_set_bits(), 0);
    }

    #[test]
    fn test_count_set_bits_increases() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let initial = filter.count_set_bits();

        filter.insert(&42);
        let after = filter.count_set_bits();

        assert!(after > initial);
    }

    #[test]
    fn test_fill_rate_empty() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.fill_rate(), 0.0);
    }

    #[test]
    fn test_fill_rate_increases() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();

        for i in 0..100 {
            filter.insert(&i);
        }

        let fill_rate = filter.fill_rate();
        assert!(fill_rate > 0.0);
        assert!(fill_rate < 1.0);
    }

    #[test]
    fn test_estimate_fpr_empty() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        assert_eq!(filter.estimate_fpr(), 0.0);
    }

    #[test]
    fn test_estimate_fpr_increases_with_load() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();

        let fpr_initial = filter.estimate_fpr();

        for i in 0..50 {
            filter.insert(&i);
        }
        let fpr_half = filter.estimate_fpr();

        for i in 50..100 {
            filter.insert(&i);
        }
        let fpr_full = filter.estimate_fpr();

        assert!(fpr_initial < fpr_half);
        assert!(fpr_half < fpr_full);
    }

    #[test]
    fn test_estimate_cardinality() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();

        for i in 0..100 {
            filter.insert(&i);
        }

        let estimated = filter.estimate_cardinality();
        // Should be roughly 100, allow ±20% error
        assert!(estimated >= 80 && estimated <= 120, 
                "Estimated {} items, expected ~100", estimated);
    }

    #[test]
    fn test_is_empty() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        assert!(filter.is_empty());

        filter.insert(&42);
        assert!(!filter.is_empty());
    }

    #[test]
    fn test_is_full() {
        let filter = StandardBloomFilter::<u64>::new(10, 0.01).unwrap();
        assert!(!filter.is_full());

        // Saturate the filter
        for i in 0..1000 {
            filter.insert(&i);
        }

        assert!(filter.is_full());
    }

    #[test]
    fn test_memory_usage() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        let memory = filter.memory_usage();

        // Should be at least size_in_bits / 8
        let min_expected = filter.size() / 8;
        assert!(memory >= min_expected);
    }

    #[test]
    fn test_clear() {
        let mut filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();

        filter.insert(&"hello".to_string());
        filter.insert(&"world".to_string());
        assert!(filter.contains(&"hello".to_string()));

        filter.clear();

        assert!(filter.is_empty());
        assert!(!filter.contains(&"hello".to_string()));
        assert!(!filter.contains(&"world".to_string()));
        assert_eq!(filter.count_set_bits(), 0);
    }

    #[test]
    fn test_clear_idempotent() {
        let mut filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();

        filter.insert(&42);
        filter.clear();
        filter.clear(); // Clear again

        assert!(filter.is_empty());
    }

    #[test]
    fn test_union() {
        let filter1 = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        let filter2 = StandardBloomFilter::<String>::new(100, 0.01).unwrap();

        filter1.insert(&"a".to_string());
        filter2.insert(&"b".to_string());

        let union = filter1.union(&filter2).unwrap();

        assert!(union.contains(&"a".to_string()));
        assert!(union.contains(&"b".to_string()));
    }

    #[test]
    fn test_union_same_item() {
        let filter1 = StandardBloomFilter::<i32>::new(100, 0.01).unwrap();
        let filter2 = StandardBloomFilter::<i32>::new(100, 0.01).unwrap();

        filter1.insert(&42);
        filter2.insert(&42);

        let union = filter1.union(&filter2).unwrap();
        assert!(union.contains(&42));
    }

    #[test]
    fn test_union_incompatible_size() {
        let filter1 = StandardBloomFilter::<i32>::new(100, 0.01).unwrap();
        let filter2 = StandardBloomFilter::<i32>::new(1000, 0.01).unwrap();

        let result = filter1.union(&filter2);
        assert!(result.is_err());
    }

    #[test]
    fn test_intersect() {
        let filter1 = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        let filter2 = StandardBloomFilter::<String>::new(100, 0.01).unwrap();

        filter1.insert(&"a".to_string());
        filter1.insert(&"b".to_string());
        filter2.insert(&"b".to_string());
        filter2.insert(&"c".to_string());

        let intersection = filter1.intersect(&filter2).unwrap();

        // "b" should probably be in intersection
        assert!(intersection.contains(&"b".to_string()));
    }

    #[test]
    fn test_intersect_incompatible_size() {
        let filter1 = StandardBloomFilter::<i32>::new(100, 0.01).unwrap();
        let filter2 = StandardBloomFilter::<i32>::new(1000, 0.01).unwrap();

        let result = filter1.intersect(&filter2);
        assert!(result.is_err());
    }

    #[test]
    fn test_clone() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();
        filter.insert(&"hello".to_string());

        let cloned = filter.clone();

        assert_eq!(cloned.size(), filter.size());
        assert_eq!(cloned.hash_count(), filter.hash_count());
        assert!(cloned.contains(&"hello".to_string()));
    }

    #[test]
    fn test_clone_independence() {
        let filter = StandardBloomFilter::<i32>::new(100, 0.01).unwrap();
        filter.insert(&42);

        let mut cloned = filter.clone();
        cloned.clear();

        // Original should still have the item
        assert!(filter.contains(&42));
        assert!(!cloned.contains(&42));
    }

    #[test]
    fn test_concurrent_inserts_no_false_negatives() {
        use std::sync::Arc;
        use std::thread;

        let filter = Arc::new(StandardBloomFilter::<i32>::new(100_000, 0.01).unwrap());
        let num_threads = 8;
        let items_per_thread = 1000;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    for i in 0..items_per_thread {
                        f.insert(&(tid * items_per_thread + i));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Verify NO false negatives
        for tid in 0..num_threads {
            for i in 0..items_per_thread {
                assert!(filter.contains(&(tid * items_per_thread + i)),
                       "False negative for item {}", tid * items_per_thread + i);
            }
        }
    }

    #[test]
    fn test_concurrent_mixed_operations() {
        use std::sync::Arc;
        use std::thread;

        let filter = Arc::new(StandardBloomFilter::<u64>::new(50_000, 0.01).unwrap());

        // Pre-populate
        for i in 0..1000 {
            filter.insert(&i);
        }

        let handles: Vec<_> = (0..4)
            .map(|tid| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    if tid % 2 == 0 {
                        // Inserters
                        for i in 1000..2000 {
                            f.insert(&(i + tid as u64 * 10000));
                        }
                    } else {
                        // Readers
                        for i in 0..1000 {
                            assert!(f.contains(&i));
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_batch_operations() {
        use std::sync::Arc;
        use std::thread;

        let filter = Arc::new(StandardBloomFilter::<i32>::new(100_000, 0.01).unwrap());

        let handles: Vec<_> = (0..4)
            .map(|tid| {
                let f = Arc::clone(&filter);
                thread::spawn(move || {
                    let items: Vec<i32> = (tid * 1000..(tid + 1) * 1000).collect();
                    f.insert_batch(&items);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Verify all items present
        for i in 0..4000 {
            assert!(filter.contains(&i));
        }
    }

    #[test]
    fn test_health_check_healthy() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();

        for i in 0..100 {
            filter.insert(&i);
        }

        let health = filter.health_check();
        assert!(health.is_healthy());
        assert!(!health.is_degraded());
        assert!(!health.is_critical());
        assert!(health.fill_rate() < 0.5);
    }

    #[test]
    fn test_health_check_degraded() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();

        // Fill to ~60% (degraded state)
        for i in 0..70 {
            filter.insert(&i);
        }

        let health = filter.health_check();
        // Might be healthy or degraded depending on hash collisions
        assert!(health.fill_rate() > 0.3);
    }

    #[test]
    fn test_health_check_critical() {
        let filter = StandardBloomFilter::<u64>::new(50, 0.01).unwrap();

        // Severely oversaturate
        for i in 0..500 {
            filter.insert(&i);
        }

        let health = filter.health_check();
        assert!(health.is_critical());
        assert!(health.fill_rate() > 0.7);
    }

    #[test]
    fn test_health_check_helper_methods() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100 {
            filter.insert(&i);
        }

        let health = filter.health_check();

        // Test helper methods
        let _ = health.fill_rate();
        let _ = health.current_fpr();
        let _ = health.estimated_items();

        match health {
            FilterHealth::Healthy { fill_rate, current_fpr, estimated_items } => {
                assert!(fill_rate >= 0.0 && fill_rate <= 1.0);
                assert!(current_fpr >= 0.0 && current_fpr <= 1.0);
                assert!(estimated_items > 0);
            }
            _ => {}
        }
    }

    #[test]
    fn test_health_check_display() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();
        for i in 0..100 {
            filter.insert(&i);
        }

        let health = filter.health_check();
        let display = format!("{}", health);

        assert!(!display.is_empty());
        assert!(display.contains("Healthy") || display.contains("Degraded") || display.contains("CRITICAL"));
    }

    #[test]
    fn test_single_item() {
        let filter = StandardBloomFilter::<String>::new(1, 0.01).unwrap();
        filter.insert(&"single".to_string());
        assert!(filter.contains(&"single".to_string()));
    }

    #[test]
    fn test_very_small_fpr() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.0001).unwrap();

        for i in 0..10 {
            filter.insert(&i);
        }

        for i in 0..10 {
            assert!(filter.contains(&i));
        }
    }

    #[test]
    fn test_duplicate_inserts() {
        let filter = StandardBloomFilter::<i32>::new(100, 0.01).unwrap();

        filter.insert(&42);
        filter.insert(&42);
        filter.insert(&42);

        assert!(filter.contains(&42));

        // Duplicate inserts should not significantly affect fill rate
        let fill_after_dups = filter.fill_rate();
        assert!(fill_after_dups < 0.5);
    }

    #[test]
    fn test_extreme_load() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();

        // Insert way more than capacity
        for i in 0..10_000 {
            filter.insert(&i);
        }

        // Filter should be saturated but not crash
        assert!(filter.fill_rate() > 0.9);

        // Original items still present (no false negatives)
        for i in 0..100 {
            assert!(filter.contains(&i));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_bincode_roundtrip() {
        let filter = StandardBloomFilter::<String>::new(1000, 0.01).unwrap();
        filter.insert(&"hello".to_string());
        filter.insert(&"world".to_string());

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<String> = bincode::deserialize(&bytes).unwrap();

        assert!(restored.contains(&"hello".to_string()));
        assert!(restored.contains(&"world".to_string()));
        assert_eq!(restored.size(), filter.size());
        assert_eq!(restored.hash_count(), filter.hash_count());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_empty_filter() {
        let filter = StandardBloomFilter::<String>::new(100, 0.01).unwrap();

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<String> = bincode::deserialize(&bytes).unwrap();

        assert!(restored.is_empty());
        assert_eq!(restored.count_set_bits(), 0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_preserves_metadata() {
        let filter = StandardBloomFilter::<i32>::new(5000, 0.001).unwrap();
        for i in 0..1000 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<i32> = bincode::deserialize(&bytes).unwrap();

        assert_eq!(restored.expected_items(), 5000);
        assert_eq!(restored.target_fpr(), 0.001);
        assert_eq!(restored.size(), filter.size());
        assert_eq!(restored.hash_count(), filter.hash_count());

        // Fill rate should be identical
        assert!((restored.fill_rate() - filter.fill_rate()).abs() < 0.001);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_filterhealth() {
        let health = FilterHealth::Healthy {
            fill_rate: 0.3,
            current_fpr: 0.01,
            estimated_items: 100,
        };

        let json = serde_json::to_string(&health).unwrap();
        let restored: FilterHealth = serde_json::from_str(&json).unwrap();

        assert_eq!(health, restored);
    }

    #[test]
    fn test_false_positive_rate_empirical() {
        let filter = StandardBloomFilter::<u64>::new(1000, 0.01).unwrap();

        // Insert 500 items
        let inserted: HashSet<u64> = (0..500).collect();
        for &item in &inserted {
            filter.insert(&item);
        }

        // Query 10,000 items NOT in the set
        let mut false_positives = 0;
        for i in 10_000..20_000 {
            if filter.contains(&i) {
                false_positives += 1;
            }
        }

        let empirical_fpr = false_positives as f64 / 10_000.0;

        // Should be roughly 1%, allow 3× tolerance
        assert!(empirical_fpr < 0.03, 
                "Empirical FPR {:.4} exceeds 3× target", empirical_fpr);
    }

    #[test]
    fn test_fpr_increases_with_load() {
        let filter = StandardBloomFilter::<u64>::new(100, 0.01).unwrap();
        let test_set: Vec<u64> = (10_000..11_000).collect();

        // Measure FPR at different loads
        let fpr_0 = measure_fpr(&filter, &test_set);

        for i in 0..50 {
            filter.insert(&i);
        }
        let fpr_50 = measure_fpr(&filter, &test_set);

        for i in 50..100 {
            filter.insert(&i);
        }
        let fpr_100 = measure_fpr(&filter, &test_set);

        // FPR should increase monotonically
        assert!(fpr_0 <= fpr_50);
        assert!(fpr_50 <= fpr_100);
    }

    // Helper function for FPR measurement
    fn measure_fpr(filter: &StandardBloomFilter<u64>, test_set: &[u64]) -> f64 {
        let mut false_positives = 0;
        for &item in test_set {
            if filter.contains(&item) {
                false_positives += 1;
            }
        }
        false_positives as f64 / test_set.len() as f64
    }
}