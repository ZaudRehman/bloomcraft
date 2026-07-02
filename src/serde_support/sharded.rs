//! Serde integration for [`ShardedBloomFilter`].
//!
//! Implements [`Serialize`] and [`Deserialize`] for [`ShardedBloomFilter`],
//! enabling concurrent sharded filters to be persisted and restored.
//!
//! # Wire Format
//!
//! The serialized representation is a flat struct with the following fields:
//!
//! | Field | Type | Purpose |
//! |-------|------|---------|
//! | `version` | `u16` | Format version; currently always `1` |
//! | `expected_items` | `usize` | Capacity the filter was sized for |
//! | `target_fpr` | `f64` | Target false-positive rate |
//! | `shard_count` | `usize` | Number of shards |
//! | `k` | `usize` | Hash function count |
//! | `hasher_name` | `String` | Canonical hasher name for type-safety checks |
//! | `shard_bits` | `Vec<Vec<u64>>` | One [`BitVec`] per shard, each packed as 64-bit words |
//!
//! # Hasher Safety
//!
//! Deserialization validates that the hasher type matches the one used at
//! serialization time by comparing `H::default().name()` against the stored
//! `hasher_name`. A mismatch returns a clear error.
//!
//! # Examples
//!
//! ## JSON round-trip
//!
//! ```
//! use bloomcraft::sync::ShardedBloomFilter;
//! use bloomcraft::core::SharedBloomFilter;
//!
//! let filter = ShardedBloomFilter::<String>::new(10_000, 0.01);
//! filter.insert(&"test".to_string());
//!
//! let json = serde_json::to_string(&filter).unwrap();
//! let restored: ShardedBloomFilter<String> = serde_json::from_str(&json).unwrap();
//! assert!(restored.contains(&"test".to_string()));
//! ```
//!
//! ## Bincode round-trip
//!
//! ```
//! use bloomcraft::sync::ShardedBloomFilter;
//! use bloomcraft::core::SharedBloomFilter;
//!
//! let filter = ShardedBloomFilter::<String>::new(10_000, 0.01);
//! filter.insert(&"test".to_string());
//!
//! let bytes = bincode::serialize(&filter).unwrap();
//! let restored: ShardedBloomFilter<String> = bincode::deserialize(&bytes).unwrap();
//! assert!(restored.contains(&"test".to_string()));
//! ```

use crate::core::SharedBloomFilter;
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
            k: self.hash_count(),
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

/// Convenience wrapper around [`ShardedBloomFilter`] serialization.
///
/// Provides explicit `to_bytes` / `from_bytes` and `to_json` / `from_json`
/// helpers. The canonical serialization behaviour is defined by the
/// [`Serialize`]/[`Deserialize`] impls above.
pub struct ShardedFilterSerdeSupport;

impl ShardedFilterSerdeSupport {
    /// Serialize a sharded Bloom filter to binary (bincode).
    ///
    /// # Errors
    ///
    /// [`BloomCraftError::SerializationError`] if bincode fails.
    pub fn to_bytes<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        filter: &ShardedBloomFilter<T, H>,
    ) -> Result<Vec<u8>> {
        bincode::serialize(filter).map_err(|e| {
            BloomCraftError::serialization_error(format!("Bincode serialization failed: {}", e))
        })
    }

    /// Deserialize a sharded Bloom filter from binary (bincode).
    ///
    /// The hasher type `H` must match the one used at serialization time;
    /// a mismatch returns [`BloomCraftError::InvalidParameters`].
    ///
    /// # Errors
    ///
    /// [`BloomCraftError::SerializationError`] if bincode fails.
    pub fn from_bytes<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        bytes: &[u8],
    ) -> Result<ShardedBloomFilter<T, H>> {
        bincode::deserialize(bytes).map_err(|e| {
            BloomCraftError::serialization_error(format!("Bincode deserialization failed: {}", e))
        })
    }

    /// Serialize a sharded Bloom filter to JSON.
    ///
    /// # Errors
    ///
    /// [`BloomCraftError::SerializationError`] if JSON serialization fails.
    pub fn to_json<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        filter: &ShardedBloomFilter<T, H>,
    ) -> Result<String> {
        serde_json::to_string(filter).map_err(|e| {
            BloomCraftError::serialization_error(format!("JSON serialization failed: {}", e))
        })
    }

    /// Deserialize a sharded Bloom filter from JSON.
    ///
    /// The hasher type `H` must match the one used at serialization time;
    /// a mismatch returns [`BloomCraftError::InvalidParameters`].
    ///
    /// # Errors
    ///
    /// [`BloomCraftError::SerializationError`] if JSON deserialization fails.
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

    #[test]
    fn test_sharded_serde_bincode_roundtrip() {
        let filter = ShardedBloomFilter::<String>::new(1000, 0.01);
        filter.insert(&"test1".to_string());
        filter.insert(&"test2".to_string());

        let bytes = ShardedFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: ShardedBloomFilter<String> =
            ShardedFilterSerdeSupport::from_bytes(&bytes).unwrap();

        assert!(restored.contains(&"test1".to_string()));
        assert!(restored.contains(&"test2".to_string()));
        assert!(!restored.contains(&"test3".to_string()));
    }

    #[test]
    fn test_sharded_serde_json_roundtrip() {
        let filter = ShardedBloomFilter::<String>::new(1000, 0.01);
        filter.insert(&"hello".to_string());

        let json = ShardedFilterSerdeSupport::to_json(&filter).unwrap();
        let restored: ShardedBloomFilter<String> =
            ShardedFilterSerdeSupport::from_json(&json).unwrap();

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
