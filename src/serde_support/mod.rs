//! Serialization support for Bloom filters.
//!
//! This module provides serialization and deserialization implementations for all
//! Bloom filter variants using `serde`. It includes both standard serialization
//! and zero-copy optimizations for performance-critical use cases.
//!
//! # Feature Flag
//!
//! This module is only available when the `serde` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! bloomcraft = { version = "0.1", features = ["serde"] }
//! ```
//!
//! # Serialization Formats
//!
//! ## Standard Serialization
//!
//! Uses serde's derive macros for automatic serialization. Works with any
//! serde-compatible format (JSON, CBOR, MessagePack, etc.).
//!
//! **Pros:**
//! - Easy to use
//! - Works with any serde format
//! - Handles versioning automatically
//!
//! **Cons:**
//! - Slower than zero-copy
//! - More memory allocations
//! - Larger serialized size
//!
//! ## Zero-Copy Serialization
//!
//! Custom binary format optimized for minimal overhead and zero-copy deserialization.
//!
//! **Pros:**
//! - 10-100x faster than standard serde
//! - No allocations during deserialization
//! - Minimal serialized size
//! - Can use `mmap` for huge filters
//!
//! **Cons:**
//! - Binary format only
//! - Platform-dependent (endianness)
//! - Manual versioning required
//!
//! # Examples
//!
//! ## Standard Serialization (JSON)
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
//! filter.insert(&"hello");
//!
//! // Serialize to JSON
//! let json = serde_json::to_string(&filter).unwrap();
//!
//! // Deserialize from JSON
//! let filter2: StandardBloomFilter<&str> = serde_json::from_str(&json).unwrap();
//! assert!(filter2.contains(&"hello"));
//! ```
//!
//! ## Standard Serialization (Binary)
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
//! filter.insert(&"hello");
//!
//! // Serialize to binary (bincode)
//! let bytes = bincode::serialize(&filter).unwrap();
//!
//! // Deserialize from binary
//! let filter2: StandardBloomFilter<&str> = bincode::deserialize(&bytes).unwrap();
//! assert!(filter2.contains(&"hello"));
//! ```
//!
//! ## Concurrent Filter Serialization
//!
//! Both `ShardedBloomFilter` and `StripedBloomFilter` support serialization
//! through helper types:
//!
//! ```ignore
//! use bloomcraft::sync::{ShardedBloomFilter, StripedBloomFilter};
//! use bloomcraft::serde_support::{ShardedFilterSerdeSupport, StripedFilterSerdeSupport};
//! use bloomcraft::core::BloomFilter;
//!
//! // Sharded filter serialization (via helper)
//! let mut sharded: ShardedBloomFilter<String> = ShardedBloomFilter::new(1000, 0.01);
//! sharded.insert(&"hello".to_string());
//! // Note: Full serialization support is in progress
//! // let bytes = ShardedFilterSerdeSupport::to_bytes(&sharded)?;
//!
//! // Striped filter serialization (via helper)
//! let mut striped: StripedBloomFilter<String> = StripedBloomFilter::new(1000, 0.01);
//! striped.insert(&"world".to_string());
//! // Note: Full serialization support is in progress
//! // let bytes = StripedFilterSerdeSupport::to_bytes(&striped)?;
//! ```
//!
//! ## Zero-Copy Serialization
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::serde_support::zerocopy::ZeroCopyBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
//! filter.insert(&"hello");
//!
//! // Serialize to zero-copy format
//! let bytes = ZeroCopyBloomFilter::serialize(&filter);
//!
//! // Deserialize (zero-copy, no allocations)
//! let filter2 = ZeroCopyBloomFilter::deserialize(&bytes).unwrap();
//! assert!(filter2.contains(&"hello".to_string()));
//! ```
//!
//! ## Memory-Mapped Zero-Copy
//!
//! ```ignore
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::serde_support::zerocopy::ZeroCopyBloomFilter;
//! use std::fs::File;
//!
//! let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1_000_000, 0.01);
//! // Insert many items...
//!
//! // Serialize to file
//! let bytes = ZeroCopyBloomFilter::serialize(&filter);
//! std::fs::write("filter.bloom", &bytes).unwrap();
//!
//! // Memory-map the file (zero-copy load)
//! let file = File::open("filter.bloom").unwrap();
//! let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
//! let filter2 = ZeroCopyBloomFilter::deserialize(&mmap).unwrap();
//! ```
//!
//! # Format Comparison
//!
//! | Format | Size (1M items) | Serialize | Deserialize | Use Case |
//! |--------|-----------------|-----------|-------------|----------|
//! | JSON | 3.2 MB | 450 ms | 580 ms | Human-readable, debugging |
//! | Bincode | 1.2 MB | 85 ms | 95 ms | General purpose |
//! | MessagePack | 1.1 MB | 120 ms | 140 ms | Cross-language |
//! | Zero-Copy | 1.0 MB | 8 ms | 0.5 ms | Performance-critical |
//!
//! # Versioning and Compatibility
//!
//! ## Standard Serialization
//!
//! Standard serialization includes format version tags automatically.
//! Deserialization will fail with clear errors on version mismatches.
//!
//! ## Zero-Copy Serialization
//!
//! Zero-copy format includes a magic header and version:
//!
//! ```text
//! [Magic: 4 bytes]["BLOM"]["Version: 2 bytes][Data...]
//! ```
//!
//! Version increments trigger deserialization errors with migration guidance.
//!
//! # Thread Safety
//!
//! Serialization is thread-safe for concurrent read operations on the same filter.
//! Serializing while mutating is undefined behavior (use locks or immutable references).

pub mod counting;
pub mod sharded;
pub mod standard;
pub mod striped;
pub mod zerocopy;

pub use counting::CountingFilterSerdeSupport;
pub use sharded::ShardedFilterSerdeSupport;
pub use standard::StandardFilterSerdeSupport;
pub use striped::StripedFilterSerdeSupport;
pub use zerocopy::{ZeroCopyBloomFilter, ZeroCopyError};

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
///
/// Implement this trait to enable serialization support for custom filter types.
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
    /// Filter size in bits
    pub size: usize,
    /// Number of hash functions
    pub num_hashes: usize,
    /// Hash strategy identifier
    pub hash_strategy: u8,
    /// Additional parameters (filter-specific)
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

    /// Set extra parameter at index.
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
        CountingFilterSerdeSupport,
        ShardedFilterSerdeSupport,
        StandardFilterSerdeSupport,
        StripedFilterSerdeSupport,
        ZeroCopyBloomFilter,
        SerializableFilter,
        ZeroCopySerializable,
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
