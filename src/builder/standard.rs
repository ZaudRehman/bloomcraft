//! Type-state builder for [`StandardBloomFilter`].
//!
//! Enforces at compile time that both required parameters — `expected_items`
//! and `false_positive_rate` — are provided before [`build`] can be called.
//! Omitting either is a type error, not a runtime panic.
//!
//! # State Machine
//!
//! ```text
//! StandardBloomFilterBuilder::new()   →  Builder<Initial,   StdHasher>
//!     .expected_items(n)              →  Builder<WithItems,  StdHasher>
//!     .false_positive_rate(p)         →  Builder<Complete,   StdHasher>
//!     .build()                        →  Result<StandardBloomFilter<T, StdHasher>>
//! ```
//!
//! `.hasher(h)` is available at every state and transitions the `H` type
//! parameter while preserving all accumulated state.
//!
//! # Examples
//!
//! Minimal build with the default hasher:
//!
//! ```rust
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! let filter: StandardBloomFilter<u64> = StandardBloomFilterBuilder::new()
//!     .expected_items(100_000)
//!     .false_positive_rate(0.01)
//!     .build()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! With a seeded hasher for deterministic cross-run behaviour:
//!
//! ```rust
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::hash::StdHasher;
//!
//! let filter: StandardBloomFilter<u64, StdHasher> = StandardBloomFilterBuilder::new()
//!     .expected_items(100_000)
//!     .false_positive_rate(0.01)
//!     .hasher(StdHasher::with_seed(0xDEAD_BEEF))
//!     .build()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! With construction-time diagnostics:
//!
//! ```rust
//! use bloomcraft::builder::StandardBloomFilterBuilder;
//! use bloomcraft::builder::standard::FilterMetadata;
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! let (filter, meta): (StandardBloomFilter<u64>, FilterMetadata) =
//!     StandardBloomFilterBuilder::new()
//!         .expected_items(100_000)
//!         .false_positive_rate(0.01)
//!         .build_with_metadata()?;
//!
//! println!("{:.2} KB, {} hashes", meta.memory_kb(), meta.num_hashes);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`build`]: StandardBloomFilterBuilder::build

use crate::error::Result;
use crate::filters::standard::StandardBloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;

// ─────────────────────────────────────────────────────────────────────────────
// Type states
//
// Sealed inside a private module so that external code cannot name or construct
// them. This guarantees the only valid progression is through the builder's
// own transition methods.
// ─────────────────────────────────────────────────────────────────────────────

mod state {
    /// No parameters have been set.
    pub struct Initial;

    /// `expected_items` has been provided; `false_positive_rate` has not.
    pub struct WithItems {
        pub(super) expected_items: usize,
    }

    /// Both required parameters are present. [`build`] is only callable here.
    ///
    /// [`build`]: super::StandardBloomFilterBuilder::build
    pub struct Complete {
        pub(super) expected_items: usize,
        pub(super) fp_rate: f64,
    }
}

use state::{Complete, Initial, WithItems};

// ─────────────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Type-state builder for [`StandardBloomFilter`].
///
/// Construct via [`StandardBloomFilterBuilder::new`]. See the
/// [module documentation](self) for the full progression and examples.
pub struct StandardBloomFilterBuilder<State, H = StdHasher> {
    state: State,
    hasher: H,
    _phantom: PhantomData<H>,
}

// ── Initial ───────────────────────────────────────────────────────────────────

impl StandardBloomFilterBuilder<Initial, StdHasher> {
    /// Creates a builder in the `Initial` state using [`StdHasher`] as the
    /// default hasher.
    ///
    /// Call `.expected_items(n)` next to advance to `WithItems`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Initial,
            hasher: StdHasher::new(),
            _phantom: PhantomData,
        }
    }
}

impl Default for StandardBloomFilterBuilder<Initial, StdHasher> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Initial → WithItems ───────────────────────────────────────────────────────

impl<H: BloomHasher + Clone> StandardBloomFilterBuilder<Initial, H> {
    /// Sets the expected number of distinct items and advances to `WithItems`.
    ///
    /// Passing `0` here is not rejected immediately; validation is deferred to
    /// [`build`] so that all parameter errors surface from a single location.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    ///
    /// let b = StandardBloomFilterBuilder::new().expected_items(100_000);
    /// ```
    ///
    /// [`build`]: StandardBloomFilterBuilder::build
    #[must_use]
    pub fn expected_items(self, n: usize) -> StandardBloomFilterBuilder<WithItems, H> {
        StandardBloomFilterBuilder {
            state: WithItems { expected_items: n },
            hasher: self.hasher,
            _phantom: PhantomData,
        }
    }

    /// Replaces the hasher instance, transitioning the `H` type parameter.
    ///
    /// The current state (`Initial`) is preserved. This method is available
    /// at every state; see also the `WithItems` and `Complete` variants.
    #[must_use]
    pub fn hasher<H2: BloomHasher + Clone>(
        self,
        hasher: H2,
    ) -> StandardBloomFilterBuilder<Initial, H2> {
        StandardBloomFilterBuilder {
            state: Initial,
            hasher,
            _phantom: PhantomData,
        }
    }
}

// ── WithItems → Complete ──────────────────────────────────────────────────────

impl<H: BloomHasher + Clone> StandardBloomFilterBuilder<WithItems, H> {
    /// Sets the target false positive rate and advances to `Complete`.
    ///
    /// `p` must be in the open interval (0, 1). Out-of-range values are not
    /// rejected immediately; validation is deferred to [`build`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    ///
    /// let b = StandardBloomFilterBuilder::new()
    ///     .expected_items(100_000)
    ///     .false_positive_rate(0.01);
    /// ```
    ///
    /// [`build`]: StandardBloomFilterBuilder::build
    #[must_use]
    pub fn false_positive_rate(self, p: f64) -> StandardBloomFilterBuilder<Complete, H> {
        StandardBloomFilterBuilder {
            state: Complete {
                expected_items: self.state.expected_items,
                fp_rate: p,
            },
            hasher: self.hasher,
            _phantom: PhantomData,
        }
    }

    /// Replaces the hasher instance. The accumulated `expected_items` value is
    /// preserved.
    #[must_use]
    pub fn hasher<H2: BloomHasher + Clone>(
        self,
        hasher: H2,
    ) -> StandardBloomFilterBuilder<WithItems, H2> {
        StandardBloomFilterBuilder {
            state: WithItems {
                expected_items: self.state.expected_items,
            },
            hasher,
            _phantom: PhantomData,
        }
    }
}

// ── Complete ──────────────────────────────────────────────────────────────────

impl<H: BloomHasher + Clone> StandardBloomFilterBuilder<Complete, H> {
    /// Replaces the hasher instance. Both accumulated parameters are preserved.
    #[must_use]
    pub fn hasher<H2: BloomHasher + Clone>(
        self,
        hasher: H2,
    ) -> StandardBloomFilterBuilder<Complete, H2> {
        StandardBloomFilterBuilder {
            state: Complete {
                expected_items: self.state.expected_items,
                fp_rate: self.state.fp_rate,
            },
            hasher,
            _phantom: PhantomData,
        }
    }

    /// Constructs the filter.
    ///
    /// Delegates to [`StandardBloomFilter::with_hasher`], which is the
    /// authoritative site for parameter validation and optimal *m*/*k*
    /// derivation.
    ///
    /// # Errors
    ///
    /// - [`BloomCraftError::InvalidItemCount`] — `expected_items == 0`.
    /// - [`BloomCraftError::FalsePositiveRateOutOfBounds`] — `fp_rate` ∉ (0, 1).
    /// - [`BloomCraftError::InvalidParameters`] — derived *m* or *k* exceed
    ///   implementation limits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let filter: StandardBloomFilter<u64> = StandardBloomFilterBuilder::new()
    ///     .expected_items(100_000)
    ///     .false_positive_rate(0.01)
    ///     .build()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn build<T: Hash>(self) -> Result<StandardBloomFilter<T, H>> {
        StandardBloomFilter::with_hasher(
            self.state.expected_items,
            self.state.fp_rate,
            self.hasher,
        )
    }

    /// Constructs the filter and returns a [`FilterMetadata`] snapshot alongside it.
    ///
    /// The metadata is always consistent with the returned filter:
    /// `meta.filter_size == filter.size()` and `meta.num_hashes == filter.hash_count()`.
    /// Useful for logging filter dimensions and memory cost at construction time.
    ///
    /// # Errors
    ///
    /// Same conditions as [`build`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::builder::StandardBloomFilterBuilder;
    /// use bloomcraft::filters::StandardBloomFilter;
    ///
    /// let (filter, meta): (StandardBloomFilter<u64>, _) =
    ///     StandardBloomFilterBuilder::new()
    ///         .expected_items(100_000)
    ///         .false_positive_rate(0.01)
    ///         .build_with_metadata()?;
    ///
    /// println!("m = {} bits, k = {}, {:.2} KB", meta.filter_size, meta.num_hashes, meta.memory_kb());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// [`build`]: Self::build
    pub fn build_with_metadata<T: Hash>(
        self,
    ) -> Result<(StandardBloomFilter<T, H>, FilterMetadata)> {
        let n = self.state.expected_items;
        let p = self.state.fp_rate;
        let filter = StandardBloomFilter::with_hasher(n, p, self.hasher)?;
        let meta = FilterMetadata {
            expected_items: n,
            fp_rate: p,
            filter_size: filter.size(),
            num_hashes: filter.hash_count(),
            // Bits-per-item ≈ 9.6 at 1% FPR (Bloom 1970).
            bytes_per_item: filter.size() as f64 / 8.0 / n as f64,
        };
        Ok((filter, meta))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FilterMetadata
// ─────────────────────────────────────────────────────────────────────────────

/// Construction-time parameter and memory profile for a [`StandardBloomFilter`].
///
/// Returned by [`StandardBloomFilterBuilder::build_with_metadata`]. The values
/// are always consistent with the filter produced in the same call:
///
/// - `metadata.filter_size == filter.size()`
/// - `metadata.num_hashes  == filter.hash_count()`
///
/// # Examples
///
/// ```rust
/// use bloomcraft::builder::StandardBloomFilterBuilder;
/// use bloomcraft::filters::StandardBloomFilter;
///
/// let (_, meta): (StandardBloomFilter<u64>, _) = StandardBloomFilterBuilder::new()
///     .expected_items(1_000_000)
///     .false_positive_rate(0.01)
///     .build_with_metadata()?;
///
/// println!("{} bits, {} hashes, {:.1} MB",
///     meta.filter_size, meta.num_hashes, meta.memory_mb());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct FilterMetadata {
    /// *n* — expected item count supplied to the builder.
    pub expected_items: usize,
    /// *p* — target false positive rate supplied to the builder.
    pub fp_rate: f64,
    /// *m* — actual filter size in bits as derived by the optimal-*m* formula.
    pub filter_size: usize,
    /// *k* — number of hash functions as derived by the optimal-*k* formula.
    pub num_hashes: usize,
    /// Bits allocated per expected item (*m* / *n* / 8). Approximately 1.2 at
    /// 1% FPR (≈ 9.6 bits per item).
    pub bytes_per_item: f64,
}

impl FilterMetadata {
    /// Heap memory consumed by the bit array, in bytes.
    ///
    /// Computed as `⌈filter_size / 64⌉ × 8` — one `AtomicU64` word (8 bytes)
    /// per 64 bits, rounded up. Matches the allocation performed by `BitVec::new`
    /// exactly and agrees with `StandardBloomFilter::memory_usage()`.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.filter_size.div_ceil(64) * 8
    }

    /// Heap memory consumed by the bit array, in kibibytes (1 KiB = 1024 bytes).
    #[must_use]
    pub fn memory_kb(&self) -> f64 {
        self.memory_bytes() as f64 / 1024.0
    }

    /// Heap memory consumed by the bit array, in mebibytes (1 MiB = 1024 KiB).
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::StandardBloomFilter;
    use crate::hash::StdHasher;

    #[test]
    fn minimal_build() {
        let f: StandardBloomFilter<u64> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();
        assert!(f.is_empty());
        assert_eq!(f.expected_items(), 10_000);
        assert!((f.target_fpr() - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn seeded_hasher_injection() {
        let f: StandardBloomFilter<u64, StdHasher> = StandardBloomFilterBuilder::new()
            .hasher(StdHasher::with_seed(0xCAFE))
            .expected_items(1_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();
        f.insert(&42u64);
        assert!(f.contains(&42u64));
    }

    #[test]
    fn hasher_set_between_items_and_fpr() {
        let f: StandardBloomFilter<u64, StdHasher> = StandardBloomFilterBuilder::new()
            .expected_items(1_000)
            .hasher(StdHasher::with_seed(99))
            .false_positive_rate(0.01)
            .build()
            .unwrap();
        assert!(f.is_empty());
    }

    #[test]
    fn hasher_set_after_fpr() {
        let f: StandardBloomFilter<u64, StdHasher> = StandardBloomFilterBuilder::new()
            .expected_items(1_000)
            .false_positive_rate(0.01)
            .hasher(StdHasher::with_seed(7))
            .build()
            .unwrap();
        assert!(f.is_empty());
    }

    #[test]
    fn build_with_metadata_consistent_with_filter() {
        let (f, meta): (StandardBloomFilter<u64>, _) = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build_with_metadata()
            .unwrap();
        assert_eq!(f.size(), meta.filter_size);
        assert_eq!(f.hash_count(), meta.num_hashes);
        assert_eq!(meta.expected_items, 10_000);
        assert!((meta.fp_rate - 0.01).abs() < f64::EPSILON);
        assert!(meta.bytes_per_item > 0.0);
    }

    #[test]
    fn metadata_memory_units_consistent() {
        let (f, meta): (StandardBloomFilter<u64>, _) = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build_with_metadata()
            .unwrap();

        assert_eq!(
            meta.memory_bytes(),
            meta.filter_size.div_ceil(64) * 8,
            "memory_bytes() must equal the bit-array heap allocation exactly"
        );

        assert!(
            meta.memory_bytes() <= f.memory_usage(),
            "memory_bytes() must not exceed the filter's total memory_usage()"
        );

        assert!((meta.memory_kb() - meta.memory_bytes() as f64 / 1024.0).abs() < 1e-9);
        assert!((meta.memory_mb() - meta.memory_kb() / 1024.0).abs() < 1e-9);
    }

    #[test]
    fn zero_items_rejected() {
        assert!(build(0, 0.01).is_err());
    }

    #[test]
    fn invalid_fpr_zero() {
        assert!(build(1_000, 0.0).is_err());
    }

    #[test]
    fn invalid_fpr_one() {
        assert!(build(1_000, 1.0).is_err());
    }

    #[test]
    fn invalid_fpr_negative() {
        assert!(build(1_000, -0.1).is_err());
    }

    #[test]
    fn invalid_fpr_above_one() {
        assert!(build(1_000, 1.5).is_err());
    }

    #[test]
    fn no_false_negatives() {
        let f: StandardBloomFilter<u64> = StandardBloomFilterBuilder::new()
            .expected_items(10_000)
            .false_positive_rate(0.01)
            .build()
            .unwrap();
        for i in 0..1_000u64 {
            f.insert(&i);
        }
        for i in 0..1_000u64 {
            assert!(f.contains(&i), "false negative at {i}");
        }
    }

    #[test]
    fn large_filter_builds_successfully() {
        let f: StandardBloomFilter<u64> = StandardBloomFilterBuilder::new()
            .expected_items(1_000_000)
            .false_positive_rate(0.001)
            .build()
            .unwrap();
        assert!(f.size() > 0);
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    fn build(n: usize, p: f64) -> Result<StandardBloomFilter<u64>> {
        StandardBloomFilterBuilder::new()
            .expected_items(n)
            .false_positive_rate(p)
            .build()
    }
}
