//! Serialization support for Bloom filters.
//!
//! This module is only available when the `serde` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! bloomcraft = { version = "0.1", features = ["serde"] }
//! ```
//!
//! # Development Status
//!
//! Submodules are enabled one at a time as they are fixed and verified.
//! To enable the next module, uncomment the corresponding `pub mod` and
//! `pub use` lines below.

// ── Active ────────────────────────────────────────────────────────────────────
pub mod standard;
pub use standard::StandardFilterSerdeSupport;

// ── Pending fixes (uncomment one at a time) ───────────────────────────────────
// pub mod counting;
// pub use counting::CountingFilterSerdeSupport;

// pub mod zerocopy;
// pub use zerocopy::{ZeroCopyBloomFilter, ZeroCopyError};

// pub mod sharded;
// pub use sharded::ShardedFilterSerdeSupport;

// pub mod striped;
// pub use striped::StripedFilterSerdeSupport;

use crate::error::Result;

/// Serialization version for standard format.
///
/// Increment this when making breaking changes to serialization format.
pub const SERIALIZATION_VERSION: u16 = 1;

/// Magic bytes for zero-copy format identification.
pub const ZEROCOPY_MAGIC: &[u8; 4] = b"BLOM";

/// Zero-copy format version.
///
/// Increment this when making breaking changes to zero-copy format.
pub const ZEROCOPY_VERSION: u16 = 1;

/// Marker trait for serializable Bloom filters.
pub trait SerializableFilter {
    /// Get the filter type identifier for versioning.
    fn filter_type() -> &'static str;

    /// Get the current serialization format version.
    fn format_version() -> u16 {
        SERIALIZATION_VERSION
    }
}

/// Marker trait for zero-copy serializable filters.
///
/// Only filters with contiguous memory layout can implement this.
pub trait ZeroCopySerializable: SerializableFilter {
    /// Get the raw bit data for zero-copy serialization.
    fn raw_data(&self) -> &[u8];

    /// Get filter parameters for reconstruction.
    fn parameters(&self) -> FilterParameters;

    /// Reconstruct filter from raw data and parameters.
    fn from_raw(data: &[u8], params: FilterParameters) -> Result<Self>
    where
        Self: Sized;
}

/// Parameters needed to reconstruct a filter.
#[derive(Debug, Clone, Copy)]
pub struct FilterParameters {
    /// Filter size in bits.
    pub size: usize,
    /// Number of hash functions.
    pub num_hashes: usize,
    /// Hash strategy identifier.
    pub hash_strategy: u8,
    /// Additional filter-specific parameters.
    pub extra: [u64; 4],
}

impl FilterParameters {
    /// Create new parameters.
    pub const fn new(size: usize, num_hashes: usize, hash_strategy: u8) -> Self {
        Self {
            size,
            num_hashes,
            hash_strategy,
            extra: [0; 4],
        }
    }

    /// Set an extra parameter at the given index. No-op if index >= 4.
    pub fn with_extra(mut self, index: usize, value: u64) -> Self {
        if index < 4 {
            self.extra[index] = value;
        }
        self
    }
}

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        SerializableFilter,
        StandardFilterSerdeSupport,
        ZeroCopySerializable,
        // Uncomment as modules are fixed:
        // CountingFilterSerdeSupport,
        // ZeroCopyBloomFilter,
        // ShardedFilterSerdeSupport,
        // StripedFilterSerdeSupport,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_parameters() {
        let params = FilterParameters::new(1000, 7, 1)
            .with_extra(0, 42)
            .with_extra(1, 100);

        assert_eq!(params.size, 1000);
        assert_eq!(params.num_hashes, 7);
        assert_eq!(params.hash_strategy, 1);
        assert_eq!(params.extra[0], 42);
        assert_eq!(params.extra[1], 100);
        assert_eq!(params.extra[2], 0);
    }

    #[test]
    fn test_serialization_constants() {
        assert_eq!(ZEROCOPY_MAGIC, b"BLOM");
        assert!(SERIALIZATION_VERSION > 0);
        assert!(ZEROCOPY_VERSION > 0);
    }
}
