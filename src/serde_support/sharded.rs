//! Serialization support for `ShardedBloomFilter`.
//!
//! Provides serde `Serialize`/`Deserialize` implementations for concurrent
//! sharded Bloom filters, enabling filters to be persisted and restored.
//!
//! # Format
//!
//! The serialization format includes:
//! - Format version (for compatibility checking)
//! - Filter parameters (expected_items, fpr, shard_count)
//! - Number of hash functions
//! - Hasher type identifier (prevents data corruption)
//! - Serialized data for each shard (BitVec per shard)
//!
//! # Safety
//!
//! Deserialization validates that the hasher type matches. Attempting to
//! deserialize with a different hasher will fail with a clear error message.
//!
//! # Examples
//!
//! ```ignore
//! use bloomcraft::sync::ShardedBloomFilter;
//! use serde_json;
//!
//! let mut filter = ShardedBloomFilter::<String>::new(10_000, 0.01);
//! filter.insert(&"test".to_string());
//!
//! // Serialize
//! let json = serde_json::to_string(&filter).unwrap();
//!
//! // Deserialize
//! let restored: ShardedBloomFilter<String> = serde_json::from_str(&json).unwrap();
//! assert!(restored.contains(&"test".to_string()));
//! ```

#![cfg(feature = "serde")]

use crate::core::BloomFilter;
use crate::error::{BloomCraftError, Result};
use crate::hash::BloomHasher;
use crate::sync::ShardedBloomFilter;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::hash::Hash;

/// Serialization format version for compatibility checking.
const SERIALIZATION_VERSION: u16 = 1;

/// Intermediate serialization format for `ShardedBloomFilter`.
///
/// This struct captures all the state needed to reconstruct a sharded filter.
#[derive(Debug, Serialize, Deserialize)]
struct SerializableShardedBloomFilter {
    /// Format version for future compatibility
    version: u16,
    /// Expected number of items (for documentation/validation)
    expected_items: usize,
    /// Target false positive rate
    target_fpr: f64,
    /// Number of shards
    shard_count: usize,
    /// Number of hash functions (k)
    k: usize,
    /// Hasher type name (for validation)
    hasher_name: String,
    /// Serialized bit vectors for each shard (m bits per shard)
    shard_bits: Vec<Vec<u64>>,
}

impl<T, H> Serialize for ShardedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;

        // Extract shard data with proper error handling
        let shard_bits: Result<Vec<Vec<u64>>> = (0..self.shard_count())
            .map(|shard_idx| self.shard_raw_bits(shard_idx))
            .collect();

        let shard_bits = shard_bits.map_err(|e| {
            S::Error::custom(format!("Failed to extract shard bits: {:?}", e))
        })?;

        let data = SerializableShardedBloomFilter {
            version: SERIALIZATION_VERSION,
            expected_items: self.expected_items_configured(),
            target_fpr: self.target_fpr(),
            shard_count: self.shard_count(),
            k: BloomFilter::hash_count(self),
            hasher_name: self.hasher_name().to_string(),
            shard_bits,
        };

        data.serialize(serializer)
    }
}

impl<'de, T, H> Deserialize<'de> for ShardedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        let data = SerializableShardedBloomFilter::deserialize(deserializer)?;

        // Validate version
        if data.version != SERIALIZATION_VERSION {
            return Err(D::Error::custom(format!(
                "Unsupported serialization version: expected {}, got {}",
                SERIALIZATION_VERSION, data.version
            )));
        }

        // Validate hasher compatibility
        let expected_hasher_name = H::default().name();
        if data.hasher_name != expected_hasher_name {
            return Err(D::Error::custom(format!(
                "Hasher mismatch: filter was serialized with '{}' but deserializing with '{}'",
                data.hasher_name, expected_hasher_name
            )));
        }

        // Validate shard count
        if data.shard_bits.len() != data.shard_count {
            return Err(D::Error::custom(format!(
                "Shard count mismatch: expected {}, got {}",
                data.shard_count,
                data.shard_bits.len()
            )));
        }

        // Reconstruct filter from raw bits
        let filter = ShardedBloomFilter::from_shard_bits(
            data.shard_bits,
            data.k,
            data.expected_items,
            data.target_fpr,
            H::default(),
        )
        .map_err(|e| D::Error::custom(format!("Failed to reconstruct filter: {:?}", e)))?;

        Ok(filter)
    }
}

/// Helper functions for bincode/JSON serialization (standalone API).
pub struct ShardedFilterSerdeSupport;

impl ShardedFilterSerdeSupport {
    /// Serialize filter to bytes using bincode.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_bytes<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        filter: &ShardedBloomFilter<T, H>,
    ) -> Result<Vec<u8>> {
        bincode::serialize(filter).map_err(|e| {
            BloomCraftError::serialization_error(format!("Bincode serialization failed: {}", e))
        })
    }

    /// Deserialize filter from bytes using bincode.
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails or hasher mismatch.
    pub fn from_bytes<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        bytes: &[u8],
    ) -> Result<ShardedBloomFilter<T, H>> {
        bincode::deserialize(bytes).map_err(|e| {
            BloomCraftError::serialization_error(format!("Bincode deserialization failed: {}", e))
        })
    }

    /// Serialize filter to JSON string.
    ///
    /// # Errors
    ///
    /// Returns error if JSON serialization fails.
    pub fn to_json<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        filter: &ShardedBloomFilter<T, H>,
    ) -> Result<String> {
        serde_json::to_string(filter).map_err(|e| {
            BloomCraftError::serialization_error(format!("JSON serialization failed: {}", e))
        })
    }

    /// Deserialize filter from JSON string.
    ///
    /// # Errors
    ///
    /// Returns error if JSON deserialization fails or hasher mismatch.
    pub fn from_json<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        json: &str,
    ) -> Result<ShardedBloomFilter<T, H>> {
        serde_json::from_str(json).map_err(|e| {
            BloomCraftError::serialization_error(format!("JSON deserialization failed: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BloomFilter;

    #[test]
    fn test_sharded_serde_bincode_roundtrip() {
        let mut filter = ShardedBloomFilter::<String>::new(1000, 0.01);
        filter.insert(&"test1".to_string());
        filter.insert(&"test2".to_string());

        // Serialize
        let bytes = ShardedFilterSerdeSupport::to_bytes(&filter).unwrap();

        // Deserialize
        let restored: ShardedBloomFilter<String> =
            ShardedFilterSerdeSupport::from_bytes(&bytes).unwrap();
        
        // Validate
        assert!(restored.contains(&"test1".to_string()));
        assert!(restored.contains(&"test2".to_string()));
        assert!(!restored.contains(&"test3".to_string()));
    }

    #[test]
    fn test_sharded_serde_json_roundtrip() {
        let mut filter = ShardedBloomFilter::<String>::new(1000, 0.01);
        filter.insert(&"hello".to_string());

        // Serialize
        let json = ShardedFilterSerdeSupport::to_json(&filter).unwrap();

        // Deserialize
        let restored: ShardedBloomFilter<String> =
            ShardedFilterSerdeSupport::from_json(&json).unwrap();

        // Validate
        assert!(restored.contains(&"hello".to_string()));
    }

    #[test]
    fn test_sharded_serde_empty_filter() {
        let filter = ShardedBloomFilter::<String>::new(1000, 0.01);
        let bytes = ShardedFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: ShardedBloomFilter<String> =
            ShardedFilterSerdeSupport::from_bytes(&bytes).unwrap();
        assert!(restored.is_empty());
    }

    #[test]
    fn test_sharded_serde_preserves_parameters() {
        let filter = ShardedBloomFilter::<String>::new(5000, 0.001);
        let bytes = ShardedFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: ShardedBloomFilter<String> =
            ShardedFilterSerdeSupport::from_bytes(&bytes).unwrap();
        assert_eq!(restored.expected_items(), 5000);
        assert!((restored.target_fpr() - 0.001).abs() < 1e-9);
    }
}
