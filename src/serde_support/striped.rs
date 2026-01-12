//! Serialization support for `StripedBloomFilter`.
//!
//! Provides serde `Serialize`/`Deserialize` implementations for concurrent
//! striped Bloom filters with RwLock-based synchronization.
//!
//! # Format
//!
//! The serialization format includes:
//! - Format version (for compatibility checking)
//! - Filter parameters (expected_items, fpr, stripe_count)
//! - Number of hash functions
//! - Hasher type identifier (prevents data corruption)
//! - Serialized bit vector data (aggregated from all stripes)
//!
//! # Safety
//!
//! Deserialization validates that the hasher type matches. The filter is
//! reconstructed with the same stripe count and lock configuration.
//!
//! # Examples
//!
//! ```ignore
//! use bloomcraft::sync::StripedBloomFilter;
//! use serde_json;
//!
//! let mut filter = StripedBloomFilter::<String>::new(10_000, 0.01);
//! filter.insert(&"concurrent".to_string());
//!
//! // Serialize
//! let json = serde_json::to_string(&filter).unwrap();
//!
//! // Deserialize
//! let restored: StripedBloomFilter<String> = serde_json::from_str(&json).unwrap();
//! assert!(restored.contains(&"concurrent".to_string()));
//! ```

#![cfg(feature = "serde")]
use crate::core::BloomFilter;
use crate::error::{BloomCraftError, Result};
use crate::hash::BloomHasher;
use crate::sync::StripedBloomFilter;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::hash::Hash;

/// Serialization format version for compatibility checking.
const SERIALIZATION_VERSION: u16 = 1;

/// Intermediate serialization format for `StripedBloomFilter`.
#[derive(Debug, Serialize, Deserialize)]
struct StripedFilterSerde {
    /// Format version
    version: u16,
    
    /// Expected number of items
    expected_items: usize,
    
    /// Target false positive rate
    target_fpr: f64,
    
    /// Number of stripes (RwLock segments)
    stripe_count: usize,
    
    /// Number of hash functions (k)
    k: usize,
    
    /// Hasher type name
    hasher_name: String,
    
    /// Serialized bit vector (entire filter, m bits)
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
        use serde::ser::Error;

        // Extract aggregated bit vector (lock all stripes for reading)
        let bits = self
            .raw_bits()
            .map_err(|e| S::Error::custom(format!("Failed to read filter bits: {:?}", e)))?;

        let data = StripedFilterSerde {
            version: SERIALIZATION_VERSION,
            expected_items: self.expected_items_configured(),
            target_fpr: self.target_fpr(),
            stripe_count: self.stripe_count(),
            k: BloomFilter::hash_count(self),
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

        // Reconstruct filter from raw bits
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

/// Helper functions for bincode/JSON serialization (standalone API).
pub struct StripedFilterSerdeSupport;

impl StripedFilterSerdeSupport {
    /// Serialize filter to bytes using bincode.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_bytes<T: Hash + Send + Sync, H: BloomHasher + Clone + Default>(
        filter: &StripedBloomFilter<T, H>,
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
    pub fn from_bytes<T: Hash, H: BloomHasher + Clone + Default>(
        bytes: &[u8],
    ) -> Result<StripedBloomFilter<T, H>> {
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
        filter: &StripedBloomFilter<T, H>,
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
    use crate::core::BloomFilter;

    #[test]
    fn test_striped_serde_bincode_roundtrip() {
        let mut filter = StripedBloomFilter::<String>::new(1000, 0.01);
        filter.insert(&"stripe1".to_string());
        filter.insert(&"stripe2".to_string());

        // Serialize
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
        let mut filter = StripedBloomFilter::<String>::new(1000, 0.01);
        filter.insert(&"concurrent".to_string());

        // Serialize
        let json = StripedFilterSerdeSupport::to_json(&filter).unwrap();

        // Deserialize
        let restored: StripedBloomFilter<String> =
            StripedFilterSerdeSupport::from_json(&json).unwrap();

        // Validate
        assert!(restored.contains(&"concurrent".to_string()));
    }

    #[test]
    fn test_striped_serde_empty_filter() {
        let filter = StripedBloomFilter::<String>::new(1000, 0.01);

        let bytes = StripedFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: StripedBloomFilter<String> =
            StripedFilterSerdeSupport::from_bytes(&bytes).unwrap();

        assert!(restored.is_empty());
    }

    #[test]
    fn test_striped_serde_preserves_parameters() {
        let filter = StripedBloomFilter::<String>::new(5000, 0.001);

        let bytes = StripedFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: StripedBloomFilter<String> =
            StripedFilterSerdeSupport::from_bytes(&bytes).unwrap();

        assert_eq!(restored.expected_items(), 5000);
        assert!((restored.target_fpr() - 0.001).abs() < 1e-9);
    }
}
