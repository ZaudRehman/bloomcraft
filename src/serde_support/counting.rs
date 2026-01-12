//! Serialization for counting Bloom filters.
//!
//! Implements serde Serialize/Deserialize for counting Bloom filters,
//! preserving counter values for filters that support deletion.
//!
//! # Format
//!
//! The serialization format includes:
//! - Format version
//! - Filter parameters (size, hash count, strategy, max_count)
//! - Counter data (4-bit or 8-bit counters)
//!
//! # Examples
//!
//! ## Binary Serialization
//!
//! ```
//! use bloomcraft::filters::CountingBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: CountingBloomFilter<&str> = CountingBloomFilter::new(1000, 0.01);
//! filter.insert(&"hello");
//! filter.insert(&"hello"); // Insert twice
//! filter.insert(&"world");
//!
//! // Serialize
//! let bytes = bincode::serialize(&filter).unwrap();
//!
//! // Deserialize
//! let mut restored: CountingBloomFilter<&str> = bincode::deserialize(&bytes).unwrap();
//! assert!(restored.contains(&"hello"));
//!
//! // Counter state preserved - use delete method
//! restored.delete(&"hello");
//! assert!(restored.contains(&"hello")); // Still contains (count = 1)
//! restored.delete(&"hello");
//! assert!(!restored.contains(&"hello")); // Now removed (count = 0)
//! ```

use crate::hash::{HashStrategy, BloomHasher};
use crate::filters::counting::CountingBloomFilter;
use crate::error::{BloomCraftError, Result};
use serde::{Deserialize, Serialize};

/// Serialization format version for counting filters.
const FORMAT_VERSION: u16 = 1;

/// Serializable representation of a counting Bloom filter.
#[derive(Serialize, Deserialize)]
struct CountingBloomFilterSerde {
    /// Format version
    version: u16,
    /// Filter size (number of counters)
    size: usize,
    /// Number of hash functions
    num_hashes: usize,
    /// Hash strategy identifier
    hash_strategy: u8,
    /// Maximum counter value
    max_count: u8,
    /// Counter bit width (4 or 8)
    counter_bits: u8,
    /// Raw counter data
    /// For 4-bit counters: packed, 2 counters per byte
    /// For 8-bit counters: one counter per byte
    counters: Vec<u8>,
}

impl CountingBloomFilterSerde {
    /// Convert from CountingBloomFilter to serializable format.
    fn from_filter<T: std::hash::Hash, H: BloomHasher + Clone>(filter: &CountingBloomFilter<T, H>) -> Self {
        let raw_counters = filter.raw_counters();
        let counter_bits = filter.counter_bits();

        Self {
            version: FORMAT_VERSION,
            size: filter.size(),
            num_hashes: filter.num_hashes(),
            hash_strategy: strategy_to_id(filter.hash_strategy()),
            max_count: filter.max_count(),
            counter_bits,
            counters: raw_counters,
        }
    }

    /// Convert to CountingBloomFilter.
    #[allow(dead_code)]
    fn to_filter<H: BloomHasher + Default + Clone>(&self) -> Result<CountingBloomFilter<String, H>> {
        // Validate version
        if self.version != FORMAT_VERSION {
            return Err(BloomCraftError::invalid_parameters(
                format!(
                    "Incompatible serialization version: expected {}, got {}",
                    FORMAT_VERSION, self.version
                ),
            ));
        }

        // Validate parameters
        if self.size == 0 {
            return Err(BloomCraftError::invalid_filter_size(self.size));
        }
        if self.num_hashes == 0 || self.num_hashes > 32 {
            return Err(BloomCraftError::invalid_hash_count(self.num_hashes, 1, 32));
        }
        if self.max_count == 0 {
            return Err(BloomCraftError::invalid_parameters(
                "max_count must be > 0".to_string(),
            ));
        }
        if self.counter_bits != 4 && self.counter_bits != 8 {
            return Err(BloomCraftError::invalid_parameters(
                format!("Invalid counter_bits: {}", self.counter_bits),
            ));
        }

        let strategy = id_to_strategy(self.hash_strategy)?;

        // Validate counter data size
        let expected_bytes = self.size;  // 1 counter per byte for 8-bit counters

        if self.counters.len() < expected_bytes {
            return Err(BloomCraftError::invalid_parameters(
                format!(
                    "Counter data size mismatch: expected at least {} bytes, got {}",
                    expected_bytes, self.counters.len()
                ),
            ));
        }

        // Reconstruct filter
        CountingBloomFilter::from_raw(
            self.size,
            self.num_hashes,
            self.max_count,
            strategy,
            &self.counters,
        )
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
        _ => Err(BloomCraftError::invalid_parameters(
            format!("Unknown hash strategy ID: {}", id),
        )),
    }
}

impl<T: std::hash::Hash, H: BloomHasher + Default + Clone> Serialize for CountingBloomFilter<T, H> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let serde_repr = CountingBloomFilterSerde::from_filter(self);
        serde_repr.serialize(serializer)
    }
}

impl<'de, T: std::hash::Hash, H: BloomHasher + Default + Clone> Deserialize<'de> for CountingBloomFilter<T, H> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serde_repr = CountingBloomFilterSerde::deserialize(deserializer)?;
        
        // Validate version
        if serde_repr.version != FORMAT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "Incompatible serialization version: expected {}, got {}",
                FORMAT_VERSION, serde_repr.version
            )));
        }

        // Validate parameters
        if serde_repr.size == 0 {
            return Err(serde::de::Error::custom("Invalid filter size: 0"));
        }
        if serde_repr.num_hashes == 0 || serde_repr.num_hashes > 32 {
            return Err(serde::de::Error::custom(format!(
                "Invalid hash count: {}",
                serde_repr.num_hashes
            )));
        }
        if serde_repr.max_count == 0 {
            return Err(serde::de::Error::custom("max_count must be > 0"));
        }
        if serde_repr.counter_bits != 4 && serde_repr.counter_bits != 8 {
            return Err(serde::de::Error::custom(format!(
                "Invalid counter_bits: {}",
                serde_repr.counter_bits
            )));
        }

        let strategy = id_to_strategy(serde_repr.hash_strategy)
            .map_err(serde::de::Error::custom)?;

        // Reconstruct filter with the target type T
        CountingBloomFilter::<T, H>::from_raw(
            serde_repr.size,
            serde_repr.num_hashes,
            serde_repr.max_count,
            strategy,
            &serde_repr.counters,
        ).map_err(serde::de::Error::custom)
    }
}

/// Helper type for counting filter serde support.
pub struct CountingFilterSerdeSupport;

impl CountingFilterSerdeSupport {
    /// Estimate serialized size in bytes.
    pub fn estimate_size(expected_items: usize, fp_rate: f64, _max_count: u8) -> usize {
        use crate::core::params;

        let size = params::optimal_bit_count(expected_items, fp_rate).unwrap_or(0);
        let _num_hashes = params::optimal_hash_count(size, expected_items).unwrap_or(7);

        // Metadata overhead
        let metadata_size = 32;

        // Counter data (1 byte per counter for 8-bit)
        let data_size = size;

        metadata_size + data_size
    }

    /// Serialize to bytes using bincode.
    pub fn to_bytes<T: std::hash::Hash, H: BloomHasher + Default + Clone>(
        filter: &CountingBloomFilter<T, H>
    ) -> Result<Vec<u8>> {
        bincode::serialize(filter)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Deserialize from bytes using bincode.
    pub fn from_bytes<T: std::hash::Hash, H: BloomHasher + Default + Clone>(bytes: &[u8]) -> Result<CountingBloomFilter<T, H>> {
        bincode::deserialize(bytes)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Serialize to JSON string.
    pub fn to_json<T: std::hash::Hash, H: BloomHasher + Default + Clone>(
        filter: &CountingBloomFilter<T, H>
    ) -> Result<String> {
        serde_json::to_string(filter)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Deserialize from JSON string.
    pub fn from_json<T: std::hash::Hash, H: BloomHasher + Default + Clone>(json: &str) -> Result<CountingBloomFilter<T, H>> {
        serde_json::from_str(json)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::DefaultHasher;

    #[test]
    fn test_serialize_deserialize_bincode() {
        let mut filter = CountingBloomFilter::<&str, DefaultHasher>::new(1000, 0.01);
        filter.insert(&"hello");
        filter.insert(&"hello");  // Insert twice
        filter.insert(&"world");

        // Serialize
        let bytes = bincode::serialize(&filter).unwrap();
        assert!(!bytes.is_empty());

        // Deserialize
        let mut restored: CountingBloomFilter<&str, DefaultHasher> = 
            bincode::deserialize(&bytes).unwrap();

        assert!(restored.contains(&"hello"));
        assert!(restored.contains(&"world"));

        // Counter state preserved
        restored.delete(&"hello");
        assert!(restored.contains(&"hello"));  // Still there (count = 1)
        restored.delete(&"hello");
        assert!(!restored.contains(&"hello"));  // Now gone (count = 0)
    }

    #[test]
    fn test_serialize_deserialize_json() {
        let mut filter = CountingBloomFilter::<&str, DefaultHasher>::new(100, 0.01);
        filter.insert(&"test");

        let json = serde_json::to_string(&filter).unwrap();
        let restored: CountingBloomFilter<&str, DefaultHasher> = 
            serde_json::from_str(&json).unwrap();

        assert!(restored.contains(&"test"));
    }

    #[test]
    fn test_helper_methods() {
        let mut filter = CountingBloomFilter::<&str, DefaultHasher>::new(1000, 0.01);
        filter.insert(&"hello");

        // to_bytes / from_bytes
        let bytes = CountingFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: CountingBloomFilter<&str, DefaultHasher> = CountingFilterSerdeSupport::from_bytes(&bytes).unwrap();
        assert!(restored.contains(&"hello"));

        // to_json / from_json
        let json = CountingFilterSerdeSupport::to_json(&filter).unwrap();
        let restored: CountingBloomFilter<&str, DefaultHasher> = 
            CountingFilterSerdeSupport::from_json(&json).unwrap();
        assert!(restored.contains(&"hello"));
    }

    #[test]
    fn test_estimate_size() {
        let estimated = CountingFilterSerdeSupport::estimate_size(10_000, 0.01, 15);
        assert!(estimated > 1000);
        assert!(estimated < 1_000_000);
    }

    #[test]
    fn test_4bit_counters() {
        // Use with_counter_size to create a filter with 4-bit counters
        let mut filter = CountingBloomFilter::<i32, DefaultHasher>::with_counter_size(1000, 0.01, 4);
        for i in 0..100 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: CountingBloomFilter<i32, DefaultHasher> = 
            bincode::deserialize(&bytes).unwrap();

        for i in 0..100 {
            assert!(restored.contains(&i));
        }
        assert_eq!(restored.counter_bits(), 4);
        assert_eq!(restored.max_count(), 15);
    }

    #[test]
    fn test_8bit_counters() {
        let mut filter = CountingBloomFilter::<i32, DefaultHasher>::with_counter_size(1000, 0.01, 8);
        for i in 0..100 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: CountingBloomFilter<i32, DefaultHasher> = 
            bincode::deserialize(&bytes).unwrap();

        for i in 0..100 {
            assert!(restored.contains(&i));
        }
        assert_eq!(restored.counter_bits(), 8);
        assert_eq!(restored.max_count(), 255);
    }

    #[test]
    fn test_multiple_insertions_preserved() {
        let mut filter = CountingBloomFilter::<&str, DefaultHasher>::new(100, 0.01);

        // Insert "test" 5 times
        for _ in 0..5 {
            filter.insert(&"test");
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let mut restored: CountingBloomFilter<&str, DefaultHasher> = 
            bincode::deserialize(&bytes).unwrap();

        // Should be able to delete 5 times
        for _ in 0..5 {
            assert!(restored.contains(&"test"));
            restored.delete(&"test");
        }
        assert!(!restored.contains(&"test"));
    }

    #[test]
    fn test_empty_filter() {
        let filter = CountingBloomFilter::<String, DefaultHasher>::new(1000, 0.01);

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: CountingBloomFilter<String, DefaultHasher> = 
            bincode::deserialize(&bytes).unwrap();

        assert!(restored.is_empty());
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
            let mut filter = CountingBloomFilter::<&str, DefaultHasher>::with_full_params(
                1000, 7, 15, strategy
            );
            filter.insert(&"test");

            let bytes = bincode::serialize(&filter).unwrap();
            let restored: CountingBloomFilter<&str, DefaultHasher> = 
                bincode::deserialize(&bytes).unwrap();

            assert!(restored.contains(&"test"));
            // All filters currently use EnhancedDouble internally
            assert_eq!(restored.hash_strategy(), HashStrategy::EnhancedDouble);
        }
    }

    #[test]
    fn test_version_mismatch() {
        let serde_repr = CountingBloomFilterSerde {
            version: 99,
            size: 1000,
            num_hashes: 7,
            hash_strategy: 1,
            max_count: 15,
            counter_bits: 4,
            counters: vec![0; 500],
        };

        let result = serde_repr.to_filter::<DefaultHasher>();
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_counter_bits() {
        let serde_repr = CountingBloomFilterSerde {
            version: FORMAT_VERSION,
            size: 1000,
            num_hashes: 7,
            hash_strategy: 1,
            max_count: 15,
            counter_bits: 16,  // Invalid
            counters: vec![0; 1000],
        };

        let result = serde_repr.to_filter::<DefaultHasher>();
        assert!(result.is_err());
    }

    #[test]
    fn test_counter_data_size_mismatch() {
        let serde_repr = CountingBloomFilterSerde {
            version: FORMAT_VERSION,
            size: 1000,
            num_hashes: 7,
            hash_strategy: 1,
            max_count: 15,
            counter_bits: 4,
            counters: vec![0; 100],  // Should be 500
        };

        let result = serde_repr.to_filter::<DefaultHasher>();
        assert!(result.is_err());
    }

    #[test]
    fn test_large_filter_serialization() {
        let mut filter = CountingBloomFilter::<i32, DefaultHasher>::with_counter_size(100_000, 0.001, 8);
        for i in 0..10_000 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        println!("Large counting filter size: {} bytes", bytes.len());

        let restored: CountingBloomFilter<i32, DefaultHasher> = 
            bincode::deserialize(&bytes).unwrap();

        for i in 0..10_000 {
            assert!(restored.contains(&i));
        }
    }

    #[test]
    fn test_deletion_after_deserialization() {
        let mut filter = CountingBloomFilter::<i32, DefaultHasher>::new(1000, 0.01);

        for i in 0..100 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let mut restored: CountingBloomFilter<i32, DefaultHasher> = 
            bincode::deserialize(&bytes).unwrap();

        // Delete half the items
        for i in 0..50 {
            restored.delete(&i);
        }

        // Check state
        for i in 0..50 {
            assert!(!restored.contains(&i));
        }
        for i in 50..100 {
            assert!(restored.contains(&i));
        }
    }
}
