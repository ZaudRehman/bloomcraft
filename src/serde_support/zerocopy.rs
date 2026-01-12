//! Zero-copy serialization for Bloom filters.
//!
//! Provides ultra-fast serialization and deserialization with minimal overhead.
//! Designed for performance-critical applications and memory-mapped file support.
//!
//! # Performance
//!
//! Compared to standard serde (bincode):
//! - **Serialization**: Explicit byte-level control, platform-independent
//! - **Deserialization**: Safe manual parsing, no alignment requirements
//! - **Size**: Compact binary format with 32-byte header
//!
//! # Format
//!
//! The zero-copy format is a carefully designed binary layout:
//!
//! ```text
//! [Header: 32 bytes]
//!   Magic:        4 bytes  ("BLOM")
//!   Version:      2 bytes  (format version)
//!   Filter Type:  2 bytes  (0=Standard, 1=Counting, 2=Scalable)
//!   Size:         8 bytes  (filter size in bits/counters)
//!   Num Hashes:   4 bytes  (number of hash functions)
//!   Hash Strategy: 1 byte  (0=Double, 1=Enhanced, 2=Triple)
//!   Reserved:     11 bytes (for future extensions)
//!
//! [Data: Variable]
//!   Raw bit/counter data (little-endian u64 words)
//! ```
//!
//! # Safety
//!
//! **This implementation contains ZERO unsafe code.**
//!
//! All serialization/deserialization is done through explicit byte-level
//! operations with manual endianness conversion. This ensures:
//! - No undefined behavior from unaligned access
//! - No uninitialized padding bytes
//! - Platform-independent binary format
//! - Safe on ARM, RISC-V, and all architectures
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use bloomcraft::filters::standard::StandardBloomFilter;
//! use bloomcraft::serde_support::zerocopy::ZeroCopyBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(10_000, 0.01);
//! filter.insert(&"hello");
//!
//! // Serialize (safe, explicit)
//! let bytes = ZeroCopyBloomFilter::serialize(&filter);
//!
//! // Deserialize (safe, no UB)
//! let filter2: StandardBloomFilter<&str> = ZeroCopyBloomFilter::deserialize_generic(&bytes).unwrap();
//! assert!(filter2.contains(&"hello"));
//! ```

use crate::core::bitvec::BitVec;
use crate::error::{BloomCraftError, Result};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::hasher::{BloomHasher, StdHasher};
use crate::hash::HashStrategy;

/// Magic bytes for format identification.
const MAGIC: &[u8; 4] = b"BLOM";

/// Current format version.
const VERSION: u16 = 1;

/// Header size in bytes.
const HEADER_SIZE: usize = 32;

/// Filter type identifiers.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Standard Bloom filter
    Standard = 0,
    /// Counting Bloom filter
    Counting = 1,
    /// Scalable Bloom filter
    Scalable = 2,
}

/// Zero-copy serialization errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ZeroCopyError {
    /// Invalid magic bytes in header
    #[error("Invalid magic bytes (expected 'BLOM')")]
    InvalidMagic,

    /// Unsupported format version
    #[error("Unsupported format version: {0} (expected {})", VERSION)]
    UnsupportedVersion(u16),

    /// Unknown filter type identifier
    #[error("Unknown filter type: {0}")]
    UnknownFilterType(u16),

    /// Invalid filter size (zero)
    #[error("Invalid filter size: {0}")]
    InvalidSize(usize),

    /// Invalid hash function count
    #[error("Invalid hash count: {0}")]
    InvalidHashCount(usize),

    /// Invalid hash strategy identifier
    #[error("Invalid hash strategy: {0}")]
    InvalidHashStrategy(u8),

    /// Buffer too small for deserialization
    #[error("Buffer too small: expected at least {expected} bytes, got {actual}")]
    BufferTooSmall {
        /// Expected minimum buffer size
        expected: usize,
        /// Actual buffer size provided
        actual: usize,
    },
}

impl From<ZeroCopyError> for BloomCraftError {
    fn from(err: ZeroCopyError) -> Self {
        BloomCraftError::serialization_error(err.to_string())
    }
}

/// Convert hash strategy to identifier.
fn strategy_to_id(strategy: HashStrategy) -> u8 {
    match strategy {
        HashStrategy::Double => 0,
        HashStrategy::EnhancedDouble => 1,
        HashStrategy::Triple => 2,
    }
}

/// Convert identifier to hash strategy.
fn id_to_strategy(id: u8) -> Result<HashStrategy> {
    match id {
        0 => Ok(HashStrategy::Double),
        1 => Ok(HashStrategy::EnhancedDouble),
        2 => Ok(HashStrategy::Triple),
        _ => Err(ZeroCopyError::InvalidHashStrategy(id).into()),
    }
}

/// Zero-copy Bloom filter wrapper.
///
/// Provides safe serialization and deserialization for standard Bloom filters.
pub struct ZeroCopyBloomFilter;

impl ZeroCopyBloomFilter {
    /// Serialize a standard Bloom filter to bytes.
    ///
    /// **Safety:** This function uses NO unsafe code. All serialization is
    /// done through explicit byte-level operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::standard::StandardBloomFilter;
    /// use bloomcraft::serde_support::zerocopy::ZeroCopyBloomFilter;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// let bytes = ZeroCopyBloomFilter::serialize(&filter);
    /// ```
    pub fn serialize<T: std::hash::Hash, H: BloomHasher + Clone>(
        filter: &StandardBloomFilter<T, H>,
    ) -> Vec<u8> {
        let filter_size = filter.size();
        let num_hashes = filter.num_hashes();
        let hash_strategy = filter.hash_strategy();
        let raw_bits = filter.raw_bits();

        // Calculate total size
        let data_size = raw_bits.len() * 8; // Each u64 is 8 bytes
        let total_size = HEADER_SIZE + data_size;

        let mut bytes = Vec::with_capacity(total_size);

        // ✅ SAFE SERIALIZATION: Explicit field-by-field writing (C2 FIX)
        // No padding bytes - every byte is explicitly written

        // Magic (4 bytes)
        bytes.extend_from_slice(MAGIC);

        // Version (2 bytes, little-endian)
        bytes.extend_from_slice(&VERSION.to_le_bytes());

        // Filter Type (2 bytes, little-endian)
        bytes.extend_from_slice(&(FilterType::Standard as u16).to_le_bytes());

        // Size (8 bytes, little-endian)
        bytes.extend_from_slice(&(filter_size as u64).to_le_bytes());

        // Num Hashes (4 bytes, little-endian)
        bytes.extend_from_slice(&(num_hashes as u32).to_le_bytes());

        // Hash Strategy (1 byte)
        bytes.push(strategy_to_id(hash_strategy));

        // Reserved (11 bytes) - explicit zeros
        bytes.extend_from_slice(&[0u8; 11]);

        debug_assert_eq!(bytes.len(), HEADER_SIZE);

        // Write data (bit vector as little-endian u64 words)
        for &word in &raw_bits {
            bytes.extend_from_slice(&word.to_le_bytes());
        }

        debug_assert_eq!(bytes.len(), total_size);
        bytes
    }

    /// Deserialize a standard Bloom filter from bytes.
    ///
    /// **Safety:** This function uses NO unsafe code. All deserialization is
    /// done through safe byte-by-byte parsing with explicit endianness.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Buffer is too small
    /// - Magic bytes don't match
    /// - Version is incompatible
    /// - Data is corrupted
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::standard::StandardBloomFilter;
    /// use bloomcraft::serde_support::zerocopy::ZeroCopyBloomFilter;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// let bytes = ZeroCopyBloomFilter::serialize(&filter);
    /// let restored = ZeroCopyBloomFilter::deserialize(&bytes).unwrap();
    /// ```
    pub fn deserialize(bytes: &[u8]) -> Result<StandardBloomFilter<String, StdHasher>> {
        Self::deserialize_generic(bytes)
    }

    /// Deserialize with custom type and hasher.
    pub fn deserialize_generic<T: std::hash::Hash, H: BloomHasher + Default + Clone>(
        bytes: &[u8],
    ) -> Result<StandardBloomFilter<T, H>> {
        // ✅ BOUNDS CHECK: Ensure minimum header size
        if bytes.len() < HEADER_SIZE {
            return Err(ZeroCopyError::BufferTooSmall {
                expected: HEADER_SIZE,
                actual: bytes.len(),
            }
            .into());
        }

        // ✅ SAFE PARSING: Manual field extraction with explicit endianness (C1 FIX)
        // No alignment requirements - we read byte-by-byte

        // Magic (4 bytes)
        let magic = &bytes[0..4];
        if magic != MAGIC {
            return Err(ZeroCopyError::InvalidMagic.into());
        }

        // Version (2 bytes, little-endian)
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(ZeroCopyError::UnsupportedVersion(version).into());
        }

        // Filter Type (2 bytes, little-endian)
        let filter_type = u16::from_le_bytes([bytes[6], bytes[7]]);
        if filter_type != FilterType::Standard as u16 {
            return Err(ZeroCopyError::UnknownFilterType(filter_type).into());
        }

        // Size (8 bytes, little-endian)
        let size = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);

        // Num Hashes (4 bytes, little-endian)
        let num_hashes = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);

        // Hash Strategy (1 byte)
        let hash_strategy_id = bytes[20];

        // Reserved (11 bytes) - ignored for forward compatibility

        // ✅ VALIDATE: Parameters
        if size == 0 {
            return Err(ZeroCopyError::InvalidSize(size as usize).into());
        }
        if num_hashes == 0 || num_hashes > 32 {
            return Err(ZeroCopyError::InvalidHashCount(num_hashes as usize).into());
        }

        let hash_strategy = id_to_strategy(hash_strategy_id)?;

        // Calculate expected data size
        let num_words = ((size + 63) / 64) as usize;
        let data_size = num_words * 8;
        let expected_total = HEADER_SIZE + data_size;

        // ✅ BOUNDS CHECK: Ensure buffer contains full data
        if bytes.len() < expected_total {
            return Err(ZeroCopyError::BufferTooSmall {
                expected: expected_total,
                actual: bytes.len(),
            }
            .into());
        }

        // ✅ SAFE PARSING: Extract bit data with explicit endianness
        let data_bytes = &bytes[HEADER_SIZE..expected_total];
        let raw_bits: Vec<u64> = data_bytes
            .chunks_exact(8)
            .map(|chunk| {
                u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();

        // Reconstruct BitVec
        let bitvec = BitVec::from_raw(raw_bits, size as usize)?;

        // Reconstruct filter
        StandardBloomFilter::<T, H>::from_parts(bitvec, num_hashes as usize, hash_strategy)
    }

    /// Get serialized size for a filter.
    pub fn serialized_size<T: std::hash::Hash, H: BloomHasher + Clone>(
        filter: &StandardBloomFilter<T, H>,
    ) -> usize {
        let num_words = (filter.size() + 63) / 64;
        HEADER_SIZE + num_words * 8
    }

    /// Validate zero-copy format without full deserialization.
    ///
    /// Useful for checking if a buffer contains a valid filter before use.
    pub fn validate(bytes: &[u8]) -> Result<()> {
        if bytes.len() < HEADER_SIZE {
            return Err(ZeroCopyError::BufferTooSmall {
                expected: HEADER_SIZE,
                actual: bytes.len(),
            }
            .into());
        }

        // Validate magic
        let magic = &bytes[0..4];
        if magic != MAGIC {
            return Err(ZeroCopyError::InvalidMagic.into());
        }

        // Validate version
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(ZeroCopyError::UnsupportedVersion(version).into());
        }

        // Validate filter type
        let filter_type = u16::from_le_bytes([bytes[6], bytes[7]]);
        if filter_type > FilterType::Scalable as u16 {
            return Err(ZeroCopyError::UnknownFilterType(filter_type).into());
        }

        // Validate size
        let size = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        if size == 0 {
            return Err(ZeroCopyError::InvalidSize(size as usize).into());
        }

        // Validate num_hashes
        let num_hashes = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        if num_hashes == 0 || num_hashes > 32 {
            return Err(ZeroCopyError::InvalidHashCount(num_hashes as usize).into());
        }

        // Validate hash strategy
        let hash_strategy_id = bytes[20];
        id_to_strategy(hash_strategy_id)?;

        // Validate buffer size
        let num_words = ((size + 63) / 64) as usize;
        let data_size = num_words * 8;
        let expected_total = HEADER_SIZE + data_size;

        if bytes.len() < expected_total {
            return Err(ZeroCopyError::BufferTooSmall {
                expected: expected_total,
                actual: bytes.len(),
            }
            .into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize() {
        let mut filter = StandardBloomFilter::<&str, StdHasher>::new(1000, 0.01);
        filter.insert(&"hello");
        filter.insert(&"world");

        // Serialize
        let bytes = ZeroCopyBloomFilter::serialize(&filter);
        assert!(bytes.len() > HEADER_SIZE);

        // Deserialize
        let restored: StandardBloomFilter<&str, StdHasher> =
            ZeroCopyBloomFilter::deserialize_generic(&bytes).unwrap();

        assert!(restored.contains(&"hello"));
        assert!(restored.contains(&"world"));
        assert!(!restored.contains(&"missing"));
    }

    #[test]
    fn test_empty_filter() {
        let filter = StandardBloomFilter::<String, StdHasher>::new(1000, 0.01);

        let bytes = ZeroCopyBloomFilter::serialize(&filter);
        let restored = ZeroCopyBloomFilter::deserialize(&bytes).unwrap();

        assert!(restored.is_empty());
    }

    #[test]
    fn test_validate() {
        let filter = StandardBloomFilter::<String, StdHasher>::new(1000, 0.01);
        let bytes = ZeroCopyBloomFilter::serialize(&filter);

        assert!(ZeroCopyBloomFilter::validate(&bytes).is_ok());
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = vec![0u8; HEADER_SIZE + 128];
        bytes[0..4].copy_from_slice(b"XXXX"); // Wrong magic

        assert!(ZeroCopyBloomFilter::validate(&bytes).is_err());
    }

    #[test]
    fn test_buffer_too_small() {
        let bytes = vec![0u8; HEADER_SIZE - 1];
        assert!(ZeroCopyBloomFilter::validate(&bytes).is_err());
    }

    #[test]
    fn test_serialized_size() {
        let filter = StandardBloomFilter::<String, StdHasher>::new(1000, 0.01);
        let bytes = ZeroCopyBloomFilter::serialize(&filter);
        let calculated = ZeroCopyBloomFilter::serialized_size(&filter);

        assert_eq!(bytes.len(), calculated);
    }

    #[test]
    fn test_large_filter() {
        let mut filter = StandardBloomFilter::<i32, StdHasher>::new(10_000, 0.01);
        for i in 0..1_000 {
            filter.insert(&i);
        }

        let bytes = ZeroCopyBloomFilter::serialize(&filter);
        println!("Zero-copy size: {} bytes", bytes.len());

        let restored: StandardBloomFilter<i32, StdHasher> =
            ZeroCopyBloomFilter::deserialize_generic(&bytes).unwrap();

        for i in 0..1_000 {
            assert!(restored.contains(&i));
        }
    }

    /// ⭐ CRITICAL TEST: Alignment independence (C1 fix verification)
    #[test]
    fn test_alignment_independence() {
        let mut filter = StandardBloomFilter::<&str, StdHasher>::new(100, 0.01);
        filter.insert(&"item");

        let bytes = ZeroCopyBloomFilter::serialize(&filter);

        // Test with various offsets (simulating misaligned buffers)
        for offset in 0..8 {
            let mut padded = vec![0xFFu8; offset];
            padded.extend_from_slice(&bytes);

            let result = ZeroCopyBloomFilter::deserialize_generic::<&str, StdHasher>(&padded[offset..]);
            assert!(result.is_ok(), "Failed at offset {}", offset);

            let restored = result.unwrap();
            assert!(restored.contains(&"item"));
        }
    }

    #[test]
    fn test_zero_size_rejected() {
        let mut bytes = vec![0u8; HEADER_SIZE + 64];
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&(FilterType::Standard as u16).to_le_bytes());
        bytes[8..16].copy_from_slice(&0u64.to_le_bytes()); // Invalid: size = 0

        assert!(ZeroCopyBloomFilter::validate(&bytes).is_err());
    }

    #[test]
    fn test_invalid_hash_count() {
        let mut bytes = vec![0u8; HEADER_SIZE + 64];
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&(FilterType::Standard as u16).to_le_bytes());
        bytes[8..16].copy_from_slice(&1000u64.to_le_bytes());
        bytes[16..20].copy_from_slice(&0u32.to_le_bytes()); // Invalid: 0 hash functions

        assert!(ZeroCopyBloomFilter::validate(&bytes).is_err());
    }
}
