//! Standard serialization for standard Bloom filters.
//!
//! Implements serde Serialize/Deserialize for standard Bloom filters,
//! allowing use with any serde-compatible format (JSON, CBOR, MessagePack, etc.).
//!
//! # Format
//!
//! The serialization format includes:
//! - Format version (for compatibility checking)
//! - Filter parameters (size, hash count, strategy)
//! - **Hasher type identifier** (prevents data corruption)
//! - Bit vector data
//!
//! # Safety
//!
//! Deserialization validates that the hasher type matches. Attempting to
//! deserialize with a different hasher will fail with a clear error message.
//!
//! # Examples
//!
//! ## JSON Serialization
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
//! filter.insert(&"hello");
//! filter.insert(&"world");
//!
//! // Serialize to JSON
//! let json = serde_json::to_string_pretty(&filter).unwrap();
//! println!("{}", json);
//!
//! // Deserialize
//! let restored: StandardBloomFilter<&str> = serde_json::from_str(&json).unwrap();
//! assert!(restored.contains(&"hello"));
//! assert!(restored.contains(&"world"));
//! ```
//!
//! ## Binary Serialization (Bincode)
//!
//! ```
//! use bloomcraft::filters::StandardBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: StandardBloomFilter<i32> = StandardBloomFilter::new(10_000, 0.01);
//! for i in 0..1000 {
//!     filter.insert(&i);
//! }
//!
//! // Serialize to binary
//! let bytes = bincode::serialize(&filter).unwrap();
//! println!("Serialized size: {} bytes", bytes.len());
//!
//! // Deserialize
//! let restored: StandardBloomFilter<i32> = bincode::deserialize(&bytes).unwrap();
//! for i in 0..1000 {
//!     assert!(restored.contains(&i));
//! }
//! ```

use crate::core::bitvec::BitVec;
use crate::error::{BloomCraftError, Result};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::hasher::BloomHasher;
use crate::hash::HashStrategy;
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serialization format version.
const FORMAT_VERSION: u16 = 1;

/// Serializable representation of a standard Bloom filter.
///
/// This intermediate format is used for serialization to allow
/// custom serialization logic and versioning.
#[derive(Serialize, Deserialize)]
struct StandardBloomFilterSerde {
    /// Format version for compatibility checking
    version: u16,
    /// Filter size in bits
    size: usize,
    /// Number of hash functions
    num_hashes: usize,
    /// Hash strategy identifier (0=Double, 1=EnhancedDouble, 2=Triple)
    hash_strategy: u8,
    /// **Hasher type identifier** (e.g., "StdHasher", "WyHasher", "XxHasher")
    hasher_type: String,
    /// Raw bit data (stored as u64 words)
    bits: Vec<u64>,
    /// Metadata: number of set bits (for quick statistics)
    num_bits_set: Option<usize>,
}

impl StandardBloomFilterSerde {
    /// Convert from StandardBloomFilter to serializable format.
    fn from_filter<T: std::hash::Hash, H: BloomHasher + Clone>(
        filter: &StandardBloomFilter<T, H>,
    ) -> Self {
        let raw_data = filter.raw_bits();
        let num_bits_set = Some(filter.count_set_bits());

        Self {
            version: FORMAT_VERSION,
            size: filter.size(),
            num_hashes: filter.num_hashes(),
            hash_strategy: strategy_to_id(filter.hash_strategy()),
            hasher_type: filter.hasher_name().to_string(), // ✅ CAPTURE HASHER TYPE
            bits: raw_data,
            num_bits_set,
        }
    }

    /// Convert to StandardBloomFilter.
    #[allow(dead_code)]
    fn to_filter<H: BloomHasher + Default + Clone>(&self) -> Result<StandardBloomFilter<String, H>> {
        // ✅ VALIDATE VERSION
        if self.version != FORMAT_VERSION {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Incompatible serialization version: expected {}, got {}",
                FORMAT_VERSION, self.version
            )));
        }

        // ✅ VALIDATE PARAMETERS
        if self.size == 0 {
            return Err(BloomCraftError::invalid_filter_size(self.size));
        }
        if self.num_hashes == 0 || self.num_hashes > 32 {
            return Err(BloomCraftError::invalid_hash_count(
                self.num_hashes,
                1,
                32,
            ));
        }

        let strategy = id_to_strategy(self.hash_strategy)?;

        // ✅ VALIDATE HASHER TYPE MATCHES
        let expected_hasher = H::default();
        let expected_name = expected_hasher.name();

        if self.hasher_type != expected_name {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Hasher type mismatch: filter was serialized with '{}', \
                 but you're trying to deserialize with '{}'. \
                 These are incompatible - wrong hasher will cause 100% false negatives. \
                 Use the same hasher type for both operations.",
                self.hasher_type, expected_name
            )));
        }

        // Reconstruct bit vector
        let bits = BitVec::from_raw(self.bits.clone(), self.size)?;

        // Reconstruct filter
        StandardBloomFilter::from_parts(bits, self.num_hashes, strategy)
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
        _ => Err(BloomCraftError::invalid_parameters(format!(
            "Unknown hash strategy ID: {}",
            id
        ))),
    }
}

impl<T: std::hash::Hash, H: BloomHasher + Default + Clone> Serialize
    for StandardBloomFilter<T, H>
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serde_repr = StandardBloomFilterSerde::from_filter(self);
        serde_repr.serialize(serializer)
    }
}

impl<'de, T: std::hash::Hash, H: BloomHasher + Default + Clone> Deserialize<'de>
    for StandardBloomFilter<T, H>
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serde_repr = StandardBloomFilterSerde::deserialize(deserializer)?;

        // ✅ VALIDATE VERSION
        if serde_repr.version != FORMAT_VERSION {
            return Err(de::Error::custom(format!(
                "Incompatible serialization version: expected {}, got {}",
                FORMAT_VERSION, serde_repr.version
            )));
        }

        // ✅ VALIDATE PARAMETERS
        if serde_repr.size == 0 {
            return Err(de::Error::custom("Invalid filter size: 0"));
        }
        if serde_repr.num_hashes == 0 || serde_repr.num_hashes > 32 {
            return Err(de::Error::custom(format!(
                "Invalid hash count: {}",
                serde_repr.num_hashes
            )));
        }

        let strategy = id_to_strategy(serde_repr.hash_strategy).map_err(de::Error::custom)?;

        // ✅ VALIDATE HASHER TYPE MATCHES
        let expected_hasher = H::default();
        let expected_name = expected_hasher.name();

        if serde_repr.hasher_type != expected_name {
            return Err(de::Error::custom(format!(
                "Hasher type mismatch: filter was serialized with '{}', \
                 but you're trying to deserialize with '{}'. \
                 These are incompatible - wrong hasher will cause 100% false negatives. \
                 Use the same hasher type for both operations.",
                serde_repr.hasher_type, expected_name
            )));
        }

        // Reconstruct bit vector
        let bits = BitVec::from_raw(serde_repr.bits, serde_repr.size).map_err(de::Error::custom)?;

        // Reconstruct filter with the target type T
        StandardBloomFilter::<T, H>::from_parts(bits, serde_repr.num_hashes, strategy)
            .map_err(de::Error::custom)
    }
}

/// Helper type for serde support documentation.
pub struct StandardFilterSerdeSupport;

impl StandardFilterSerdeSupport {
    /// Estimate serialized size in bytes.
    ///
    /// This is an approximation. Actual size depends on the serialization format.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let estimated_bytes = StandardFilterSerdeSupport::estimate_size(10_000, 0.01);
    /// println!("Estimated size: {} bytes", estimated_bytes);
    /// ```
    pub fn estimate_size(expected_items: usize, fp_rate: f64) -> usize {
        use crate::core::params;

        let size = params::optimal_bit_count(expected_items, fp_rate).unwrap_or(0);
        let _num_hashes = params::optimal_hash_count(size, expected_items).unwrap_or(7);

        // Overhead: version (2) + size (8) + num_hashes (8) + strategy (1) + hasher_type (~20) + length (8)
        let metadata_size = 47;

        // Data: bit vector as u64 words
        let data_size = (size + 63) / 64 * 8;

        metadata_size + data_size
    }

    /// Serialize filter to bytes using bincode.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// let bytes = StandardFilterSerdeSupport::to_bytes(&filter).unwrap();
    /// ```
    pub fn to_bytes<T: std::hash::Hash, H: BloomHasher + Default + Clone>(
        filter: &StandardBloomFilter<T, H>,
    ) -> Result<Vec<u8>> {
        bincode::serialize(filter).map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Deserialize filter from bytes using bincode.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// let bytes = StandardFilterSerdeSupport::to_bytes(&filter).unwrap();
    /// let restored: StandardBloomFilter<&str> = StandardFilterSerdeSupport::from_bytes(&bytes).unwrap();
    /// ```
    pub fn from_bytes<T: std::hash::Hash, H: BloomHasher + Default + Clone>(
        bytes: &[u8],
    ) -> Result<StandardBloomFilter<T, H>> {
        bincode::deserialize(bytes)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Serialize filter to JSON string.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// let json = StandardFilterSerdeSupport::to_json(&filter).unwrap();
    /// ```
    pub fn to_json<T: std::hash::Hash, H: BloomHasher + Default + Clone>(
        filter: &StandardBloomFilter<T, H>,
    ) -> Result<String> {
        serde_json::to_string(filter)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Deserialize filter from JSON string.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
    /// let json = StandardFilterSerdeSupport::to_json(&filter).unwrap();
    /// let restored: StandardBloomFilter<&str> = StandardFilterSerdeSupport::from_json(&json).unwrap();
    /// ```
    pub fn from_json<T: std::hash::Hash, H: BloomHasher + Default + Clone>(
        json: &str,
    ) -> Result<StandardBloomFilter<T, H>> {
        serde_json::from_str(json)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::hasher::StdHasher;

    #[test]
    fn test_serialize_deserialize_bincode() {
        let mut filter = StandardBloomFilter::<&str, StdHasher>::new(1000, 0.01);
        filter.insert(&"hello");
        filter.insert(&"world");

        // Serialize
        let bytes = bincode::serialize(&filter).unwrap();
        assert!(!bytes.is_empty());

        // Deserialize
        let restored: StandardBloomFilter<&str, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert!(restored.contains(&"hello"));
        assert!(restored.contains(&"world"));
        assert!(!restored.contains(&"missing"));
    }

    #[test]
    fn test_serialize_deserialize_json() {
        let mut filter = StandardBloomFilter::<&str, StdHasher>::new(100, 0.01);
        filter.insert(&"test");

        // Serialize
        let json = serde_json::to_string(&filter).unwrap();
        assert!(json.contains("version"));
        assert!(json.contains("bits"));
        assert!(json.contains("hasher_type")); // ✅ NEW: Check hasher is serialized
        assert!(json.contains("StdHasher")); // ✅ NEW: Check correct hasher name

        // Deserialize
        let restored: StandardBloomFilter<&str, StdHasher> =
            serde_json::from_str(&json).unwrap();

        assert!(restored.contains(&"test"));
    }

    /// ⭐ NEW TEST: Verify hasher type validation
    #[test]
    fn test_hasher_type_mismatch_rejected() {
        let mut filter = StandardBloomFilter::<&str, StdHasher>::new(100, 0.01);
        filter.insert(&"test");

        // Serialize with StdHasher
        let mut serde_repr = StandardBloomFilterSerde::from_filter(&filter);

        // Corrupt the hasher type
        serde_repr.hasher_type = "WrongHasher".to_string();

        // Try to deserialize - should fail
        let result = serde_repr.to_filter::<StdHasher>();
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Hasher type mismatch"));
        assert!(err_msg.contains("WrongHasher"));
        assert!(err_msg.contains("StdHasher"));
    }

    /// ⭐ NEW TEST: Verify correct hasher type is accepted
    #[test]
    fn test_correct_hasher_type_accepted() {
        let mut filter = StandardBloomFilter::<&str, StdHasher>::new(100, 0.01);
        filter.insert(&"test");

        let serde_repr = StandardBloomFilterSerde::from_filter(&filter);

        // Should succeed with correct hasher
        let result = serde_repr.to_filter::<StdHasher>();
        assert!(result.is_ok());

        // Note: to_filter returns StandardBloomFilter<String, H>
        let restored = result.unwrap();
        assert!(restored.contains(&"test".to_string()));
    }

    #[test]
    fn test_helper_methods() {
        let mut filter = StandardBloomFilter::<&str, StdHasher>::new(1000, 0.01);
        filter.insert(&"hello");

        // to_bytes / from_bytes
        let bytes = StandardFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: StandardBloomFilter<&str, StdHasher> =
            StandardFilterSerdeSupport::from_bytes(&bytes).unwrap();
        assert!(restored.contains(&"hello"));

        // to_json / from_json
        let json = StandardFilterSerdeSupport::to_json(&filter).unwrap();
        let restored: StandardBloomFilter<&str, StdHasher> =
            StandardFilterSerdeSupport::from_json(&json).unwrap();
        assert!(restored.contains(&"hello"));
    }

    #[test]
    fn test_estimate_size() {
        let estimated = StandardFilterSerdeSupport::estimate_size(10_000, 0.01);
        assert!(estimated > 1000);
        assert!(estimated < 100_000);
    }

    #[test]
    fn test_empty_filter() {
        let filter = StandardBloomFilter::<String, StdHasher>::new(1000, 0.01);

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<String, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert!(restored.is_empty());
    }

    #[test]
    fn test_full_filter() {
        let mut filter = StandardBloomFilter::<i32, StdHasher>::new(100, 0.01);
        for i in 0..100 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<i32, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        for i in 0..100 {
            assert!(restored.contains(&i));
        }
    }

    #[test]
    fn test_different_strategies() {
        // Note: Currently all filters use EnhancedDoubleHashing internally,
        // so the strategy parameter is stored but the actual hashing uses EnhancedDouble.
        // This test verifies serialization/deserialization works correctly.
        for strategy in [
            HashStrategy::Double,
            HashStrategy::EnhancedDouble,
            HashStrategy::Triple,
        ] {
            let mut filter = StandardBloomFilter::<&str, StdHasher>::with_strategy(1000, 7, strategy);
            filter.insert(&"test");

            let bytes = bincode::serialize(&filter).unwrap();
            let restored: StandardBloomFilter<&str, StdHasher> =
                bincode::deserialize(&bytes).unwrap();

            assert!(restored.contains(&"test"));
            // All filters currently use EnhancedDouble internally
            assert_eq!(restored.hash_strategy(), HashStrategy::EnhancedDouble);
        }
    }

    #[test]
    fn test_version_mismatch() {
        let serde_repr = StandardBloomFilterSerde {
            version: 99, // Invalid version
            size: 1000,
            num_hashes: 7,
            hash_strategy: 1,
            hasher_type: "StdHasher".to_string(),
            bits: vec![0; 16],
            num_bits_set: Some(0),
        };

        let result = serde_repr.to_filter::<StdHasher>();
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_hash_strategy() {
        let result = id_to_strategy(99);
        assert!(result.is_err());
    }

    #[test]
    fn test_strategy_round_trip() {
        for strategy in [
            HashStrategy::Double,
            HashStrategy::EnhancedDouble,
            HashStrategy::Triple,
        ] {
            let id = strategy_to_id(strategy);
            let restored = id_to_strategy(id).unwrap();
            assert_eq!(strategy, restored);
        }
    }

    #[test]
    fn test_large_filter_serialization() {
        let mut filter = StandardBloomFilter::<i32, StdHasher>::new(100_000, 0.001);
        for i in 0..10_000 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        println!("Large filter size: {} bytes", bytes.len());

        let restored: StandardBloomFilter<i32, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        for i in 0..10_000 {
            assert!(restored.contains(&i));
        }
    }

    #[test]
    fn test_serialization_preserves_statistics() {
        let mut filter = StandardBloomFilter::<i32, StdHasher>::new(1000, 0.01);
        for i in 0..100 {
            filter.insert(&i);
        }

        let original_len = filter.count_set_bits();
        let original_fp_rate = filter.estimate_fpr();

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<i32, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert_eq!(restored.count_set_bits(), original_len);
        assert!((restored.estimate_fpr() - original_fp_rate).abs() < 0.001);
    }
}
