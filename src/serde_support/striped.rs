//! Serde integration for [`StripedBloomFilter`].
//!
//! Implements [`Serialize`] and [`Deserialize`] for [`StripedBloomFilter`],
//! enabling concurrent RwLock-partitioned filters to be persisted and restored.
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
//! | `stripe_count` | `usize` | Number of lock stripes |
//! | `k` | `usize` | Hash function count |
//! | `hasher_name` | `String` | Canonical hasher name for type-safety checks |
//! | `bits` | `Vec<u64>` | Full bit vector packed as 64-bit words |
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
//! use bloomcraft::sync::StripedBloomFilter;
//! use bloomcraft::core::SharedBloomFilter;
//!
//! let filter = StripedBloomFilter::<String>::new(10_000, 0.01).unwrap();
//! filter.insert(&"concurrent".to_string());
//!
//! let json = serde_json::to_string(&filter).unwrap();
//! let restored: StripedBloomFilter<String> = serde_json::from_str(&json).unwrap();
//! assert!(restored.contains(&"concurrent".to_string()));
//! ```
//!
//! ## Bincode round-trip
//!
//! ```
//! use bloomcraft::sync::StripedBloomFilter;
//! use bloomcraft::core::SharedBloomFilter;
//!
//! let filter = StripedBloomFilter::<String>::new(10_000, 0.01).unwrap();
//! filter.insert(&"concurrent".to_string());
//!
//! let bytes = bincode::serialize(&filter).unwrap();
//! let restored: StripedBloomFilter<String> = bincode::deserialize(&bytes).unwrap();
//! assert!(restored.contains(&"concurrent".to_string()));
//! ```
use crate::core::SharedBloomFilter;
use crate::error::{BloomCraftError, Result};
use crate::hash::BloomHasher;
use crate::sync::StripedBloomFilter;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::hash::Hash;

/// Serialization format version for compatibility checking.
const SERIALIZATION_VERSION: u16 = 1;

/// Wire representation of a [`StripedBloomFilter`] for serialization.
#[derive(Debug, Serialize, Deserialize)]
struct StripedFilterSerde {
    /// Format version; must equal [`SERIALIZATION_VERSION`].
    version: u16,
    /// Capacity the filter was sized for.
    expected_items: usize,
    /// Target false-positive rate.
    target_fpr: f64,
    /// Number of lock stripes.
    stripe_count: usize,
    /// Number of hash functions (*k*).
    k: usize,
    /// Hasher type name; validated on deserialization.
    hasher_name: String,
    /// Full bit vector packed as 64-bit words.
    bits: Vec<u64>,
}

impl<T, H> Serialize for StripedBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bits = self.raw_bits();

        let data = StripedFilterSerde {
            version: SERIALIZATION_VERSION,
            expected_items: self.expected_items_configured(),
            target_fpr: self.target_fpr(),
            stripe_count: self.stripe_count(),
            k: self.hash_count(),
            hasher_name: self.hasher_name().to_string(),
            bits,
        };

        data.serialize(serializer)
    }
}

impl<'de, T, H> Deserialize<'de> for StripedBloomFilter<T, H>
where
    T: Hash,
    H: BloomHasher + Clone + Default,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;

        let data = StripedFilterSerde::deserialize(deserializer)?;

        if data.version != SERIALIZATION_VERSION {
            return Err(D::Error::custom(format!(
                "Unsupported serialization version: expected {}, got {}",
                SERIALIZATION_VERSION, data.version
            )));
        }

        let expected_hasher_name = H::default().name();
        if data.hasher_name != expected_hasher_name {
            return Err(D::Error::custom(format!(
                "Hasher mismatch: filter was serialized with '{}' but deserializing with '{}'",
                data.hasher_name, expected_hasher_name
            )));
        }

        let filter = StripedBloomFilter::from_raw_bits(
            data.bits,
            data.k,
            data.stripe_count,
            data.expected_items,
            data.target_fpr,
            H::default(),
        )
        .map_err(|e| D::Error::custom(format!("Failed to reconstruct filter: {:?}", e)))?;

        Ok(filter)
    }
}

/// Convenience wrapper around [`StripedBloomFilter`] serialization.
///
/// Provides explicit `to_bytes` / `from_bytes` and `to_json` / `from_json`
/// helpers. The canonical serialization behaviour is defined by the
/// [`Serialize`]/[`Deserialize`] impls above.
pub struct StripedFilterSerdeSupport;

impl StripedFilterSerdeSupport {
    /// Serialize a striped Bloom filter to binary (bincode).
    ///
    /// # Errors
    ///
    /// [`BloomCraftError::SerializationError`] if bincode fails.
    pub fn to_bytes<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        filter: &StripedBloomFilter<T, H>,
    ) -> Result<Vec<u8>> {
        bincode::serialize(filter).map_err(|e| {
            BloomCraftError::serialization_error(format!("Bincode serialization failed: {}", e))
        })
    }

    /// Deserialize a striped Bloom filter from binary (bincode).
    ///
    /// The hasher type `H` must match the one used at serialization time;
    /// a mismatch returns [`BloomCraftError::InvalidParameters`].
    ///
    /// # Errors
    ///
    /// [`BloomCraftError::SerializationError`] if bincode fails.
    pub fn from_bytes<T: Hash, H: BloomHasher + Clone + Default>(
        bytes: &[u8],
    ) -> Result<StripedBloomFilter<T, H>> {
        bincode::deserialize(bytes).map_err(|e| {
            BloomCraftError::serialization_error(format!("Bincode deserialization failed: {}", e))
        })
    }

    /// Serialize a striped Bloom filter to JSON.
    ///
    /// # Errors
    ///
    /// [`BloomCraftError::SerializationError`] if JSON serialization fails.
    pub fn to_json<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        filter: &StripedBloomFilter<T, H>,
    ) -> Result<String> {
        serde_json::to_string(filter).map_err(|e| {
            BloomCraftError::serialization_error(format!("JSON serialization failed: {}", e))
        })
    }

    /// Deserialize a striped Bloom filter from JSON.
    ///
    /// The hasher type `H` must match the one used at serialization time;
    /// a mismatch returns [`BloomCraftError::InvalidParameters`].
    ///
    /// # Errors
    ///
    /// [`BloomCraftError::SerializationError`] if JSON deserialization fails.
    pub fn from_json<T: Hash, H: BloomHasher + Clone + Default>(
        json: &str,
    ) -> Result<StripedBloomFilter<T, H>> {
        serde_json::from_str(json).map_err(|e| {
            BloomCraftError::serialization_error(format!("JSON deserialization failed: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_striped_serde_bincode_roundtrip() {
        let filter = StripedBloomFilter::<String>::new(1000, 0.01).unwrap();
        filter.insert(&"stripe1".to_string());
        filter.insert(&"stripe2".to_string());

        let bytes = StripedFilterSerdeSupport::to_bytes(&filter).unwrap();

        // Deserialize
        let restored: StripedBloomFilter<String> =
            StripedFilterSerdeSupport::from_bytes(&bytes).unwrap();

        // Validate
        assert!(restored.contains(&"stripe1".to_string()));
        assert!(restored.contains(&"stripe2".to_string()));
        assert!(!restored.contains(&"stripe3".to_string()));
    }

    #[test]
    fn test_striped_serde_json_roundtrip() {
        let filter = StripedBloomFilter::<String>::new(1000, 0.01).unwrap();
        filter.insert(&"concurrent".to_string());

        let json = StripedFilterSerdeSupport::to_json(&filter).unwrap();

        // Deserialize
        let restored: StripedBloomFilter<String> =
            StripedFilterSerdeSupport::from_json(&json).unwrap();

        // Validate
        assert!(restored.contains(&"concurrent".to_string()));
    }

    #[test]
    fn test_striped_serde_empty_filter() {
        let filter = StripedBloomFilter::<String>::new(1000, 0.01).unwrap();

        let bytes = StripedFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: StripedBloomFilter<String> =
            StripedFilterSerdeSupport::from_bytes(&bytes).unwrap();

        assert!(restored.is_empty());
    }

    #[test]
    fn test_striped_serde_preserves_parameters() {
        let filter = StripedBloomFilter::<String>::new(5000, 0.001).unwrap();

        let bytes = StripedFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: StripedBloomFilter<String> =
            StripedFilterSerdeSupport::from_bytes(&bytes).unwrap();

        assert_eq!(restored.expected_items(), 5000);
        assert!((restored.target_fpr() - 0.001).abs() < 1e-9);
    }
}
