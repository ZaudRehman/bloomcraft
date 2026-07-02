//! Serde integration for [`CountingBloomFilter`].
//!
//! Implements [`Serialize`] and [`Deserialize`] for [`CountingBloomFilter`],
//! preserving counter values so that deletion semantics survive a round-trip.
//!
//! # Wire Format
//!
//! The serialized representation is a flat struct with the following fields:
//!
//! | Field | Type | Purpose |
//! |-------|------|---------|
//! | `version` | `u16` | Format version; currently always `1` |
//! | `size` | `usize` | Total counter-array length *m* |
//! | `num_hashes` | `usize` | Hash function count *k* |
//! | `counter_bits` | `u8` | Bits per counter: `4`, `8`, or `16` |
//! | `expected_items` | `usize` | Capacity the filter was sized for |
//! | `target_fpr` | `f64` | Target false-positive rate |
//! | `counters` | `Vec<u8>` | Counter values, one byte per position (4-bit counters are unpacked to 1 byte each on the wire) |
//!
//! # Hasher Safety
//!
//! Unlike [`StandardBloomFilter`](crate::filters::StandardBloomFilter) serde, this format does **not** store the
//! hasher type name. All counting-filter operations use the hasher that was
//! configured at construction; deserializing with a mismatched hasher will
//! produce a filter whose bit positions are incompatible with the original
//! counter assignments, but the payload itself carries no structural signal
//! to detect this. Callers are responsible for ensuring the hasher type
//! parameter `H` matches the one used at serialization time.
//!
//! # Counter Packing
//!
//! On the wire every counter occupies one byte regardless of its in-memory
//! width. `FourBit` counters are unpacked from two-per-byte to one-per-byte
//! during serialization and re-packed during deserialization. This simplifies
//! the format at the cost of ~2× wire-size for 4-bit counters relative to an
//! optimally packed representation.
//!
//! # Examples
//!
//! ## Binary round-trip
//!
//! ```
//! use bloomcraft::filters::CountingBloomFilter;
//! use bloomcraft::core::BloomFilter;
//!
//! let mut filter: CountingBloomFilter<&str> = CountingBloomFilter::new(1000, 0.01);
//! filter.insert(&"hello");
//! filter.insert(&"hello");
//! filter.insert(&"world");
//!
//! let bytes = bincode::serialize(&filter).unwrap();
//!
//! let mut restored: CountingBloomFilter<&str> = bincode::deserialize(&bytes).unwrap();
//! assert!(restored.contains(&"hello"));
//!
//! restored.delete(&"hello");
//! assert!(restored.contains(&"hello"));
//! restored.delete(&"hello");
//! assert!(!restored.contains(&"hello"));
//! ```

use crate::error::BloomCraftError;
use crate::filters::counting::CountingBloomFilter;
use crate::hash::BloomHasher;
use serde::{Deserialize, Serialize};

/// Serialization format version for counting filters.
const FORMAT_VERSION: u16 = 1;

/// Serializable representation of a counting Bloom filter.
#[derive(Serialize, Deserialize)]
struct CountingBloomFilterSerde {
    /// Format version; must equal [`FORMAT_VERSION`].
    version: u16,
    /// Total number of counter slots (*m*).
    size: usize,
    /// Number of hash functions (*k*).
    num_hashes: usize,
    /// Bits per counter: `4`, `8`, or `16`.
    counter_bits: u8,
    /// Capacity the filter was originally sized for.
    expected_items: usize,
    /// Target false-positive rate.
    target_fpr: f64,
    /// Counter values, one byte per position (4-bit counters unpacked).
    counters: Vec<u8>,
}

impl CountingBloomFilterSerde {
    fn from_filter<T: std::hash::Hash + Send + Sync, H: BloomHasher + Clone>(
        filter: &CountingBloomFilter<T, H>,
    ) -> Self {
        let raw_counters = filter.raw_counters();
        Self {
            version: FORMAT_VERSION,
            size: filter.size(),
            num_hashes: filter.num_hashes(),
            counter_bits: filter.counter_bits(),
            expected_items: filter.expected_items(),
            target_fpr: filter.target_fpr(),
            counters: raw_counters,
        }
    }

    #[allow(dead_code)]
    fn to_filter<T: std::hash::Hash + Send + Sync, H: BloomHasher + Default + Clone>(
        &self,
    ) -> Result<CountingBloomFilter<T, H>, BloomCraftError> {
        if self.version != FORMAT_VERSION {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Incompatible version: expected {}, got {}",
                FORMAT_VERSION, self.version
            )));
        }
        if self.size == 0 {
            return Err(BloomCraftError::invalid_filter_size(self.size));
        }
        if self.num_hashes == 0 || self.num_hashes > 32 {
            return Err(BloomCraftError::invalid_hash_count(self.num_hashes, 1, 32));
        }
        if self.counter_bits != 4 && self.counter_bits != 8 && self.counter_bits != 16 {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Invalid counter_bits: {}",
                self.counter_bits
            )));
        }

        CountingBloomFilter::<T, H>::from_raw(
            self.size,
            self.num_hashes,
            counter_bits_to_size(self.counter_bits),
            &self.counters,
            self.expected_items,
            self.target_fpr,
        )
    }
}

fn counter_bits_to_size(bits: u8) -> crate::filters::CounterSize {
    match bits {
        4 => crate::filters::CounterSize::FourBit,
        8 => crate::filters::CounterSize::EightBit,
        16 => crate::filters::CounterSize::SixteenBit,
        _ => crate::filters::CounterSize::EightBit,
    }
}

impl<T: std::hash::Hash + Send + Sync, H: BloomHasher + Default + Clone> Serialize
    for CountingBloomFilter<T, H>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let serde_repr = CountingBloomFilterSerde::from_filter(self);
        serde_repr.serialize(serializer)
    }
}

impl<'de, T: std::hash::Hash + Send + Sync, H: BloomHasher + Default + Clone> Deserialize<'de>
    for CountingBloomFilter<T, H>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serde_repr = CountingBloomFilterSerde::deserialize(deserializer)?;

        if serde_repr.version != FORMAT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "Incompatible version: expected {}, got {}",
                FORMAT_VERSION, serde_repr.version
            )));
        }
        if serde_repr.size == 0 {
            return Err(serde::de::Error::custom("Invalid filter size: 0"));
        }
        if serde_repr.num_hashes == 0 || serde_repr.num_hashes > 32 {
            return Err(serde::de::Error::custom(format!(
                "Invalid hash count: {}",
                serde_repr.num_hashes
            )));
        }
        if serde_repr.counter_bits != 4
            && serde_repr.counter_bits != 8
            && serde_repr.counter_bits != 16
        {
            return Err(serde::de::Error::custom(format!(
                "Invalid counter_bits: {}",
                serde_repr.counter_bits
            )));
        }

        CountingBloomFilter::<T, H>::from_raw(
            serde_repr.size,
            serde_repr.num_hashes,
            counter_bits_to_size(serde_repr.counter_bits),
            &serde_repr.counters,
            serde_repr.expected_items,
            serde_repr.target_fpr,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Convenience wrapper around [`CountingBloomFilter`] serialization.
///
/// Provides explicit `to_bytes` / `from_bytes` and `to_json` / `from_json`
/// helpers without going through the serde `Serialize`/`Deserialize` derive
/// path. Useful when you need a concrete `Result` type instead of a generic
/// serde error.
pub struct CountingFilterSerdeSupport;

impl CountingFilterSerdeSupport {
    /// Estimate the serialized size (in bytes) for a filter with the given
    /// capacity and false-positive rate. The estimate includes metadata and
    /// the full counter array (one byte per position).
    pub fn estimate_size(expected_items: usize, fp_rate: f64) -> usize {
        use crate::core::params;
        let size = params::optimal_bit_count(expected_items, fp_rate).unwrap_or(0);
        let metadata_size = 32;
        let data_size = size;
        metadata_size + data_size
    }

    /// Serialize a counting Bloom filter to binary (bincode).
    pub fn to_bytes<T: std::hash::Hash + Send + Sync, H: BloomHasher + Default + Clone>(
        filter: &CountingBloomFilter<T, H>,
    ) -> Result<Vec<u8>, BloomCraftError> {
        bincode::serialize(filter).map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Deserialize a counting Bloom filter from binary (bincode).
    ///
    /// # Panics
    ///
    /// The hasher type `H` must match the one used at serialization time.
    /// See [hasher safety](index.html#hasher-safety) for details.
    pub fn from_bytes<T: std::hash::Hash + Send + Sync, H: BloomHasher + Default + Clone>(
        bytes: &[u8],
    ) -> Result<CountingBloomFilter<T, H>, BloomCraftError> {
        bincode::deserialize(bytes).map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Serialize a counting Bloom filter to JSON (serde_json).
    pub fn to_json<T: std::hash::Hash + Send + Sync, H: BloomHasher + Default + Clone>(
        filter: &CountingBloomFilter<T, H>,
    ) -> Result<String, BloomCraftError> {
        serde_json::to_string(filter)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Deserialize a counting Bloom filter from JSON (serde_json).
    ///
    /// # Panics
    ///
    /// The hasher type `H` must match the one used at serialization time.
    /// See [hasher safety](index.html#hasher-safety) for details.
    pub fn from_json<T: std::hash::Hash + Send + Sync, H: BloomHasher + Default + Clone>(
        json: &str,
    ) -> Result<CountingBloomFilter<T, H>, BloomCraftError> {
        serde_json::from_str(json).map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::StdHasher;

    #[test]
    fn test_serialize_deserialize_bincode() {
        let mut filter = CountingBloomFilter::<&str, StdHasher>::new(1000, 0.01);
        filter.insert(&"hello");
        filter.insert(&"hello");
        filter.insert(&"world");

        let bytes = bincode::serialize(&filter).unwrap();
        assert!(!bytes.is_empty());

        let mut restored: CountingBloomFilter<&str, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert!(restored.contains(&"hello"));
        assert!(restored.contains(&"world"));

        restored.delete(&"hello");
        assert!(restored.contains(&"hello"));
        restored.delete(&"hello");
        assert!(!restored.contains(&"hello"));
    }

    #[test]
    fn test_serialize_deserialize_json() {
        let mut filter = CountingBloomFilter::<&str, StdHasher>::new(100, 0.01);
        filter.insert(&"test");

        let json = serde_json::to_string(&filter).unwrap();
        let restored: CountingBloomFilter<&str, StdHasher> = serde_json::from_str(&json).unwrap();

        assert!(restored.contains(&"test"));
    }

    #[test]
    fn test_helper_methods() {
        let mut filter = CountingBloomFilter::<&str, StdHasher>::new(1000, 0.01);
        filter.insert(&"hello");

        let bytes = CountingFilterSerdeSupport::to_bytes(&filter).unwrap();
        let restored: CountingBloomFilter<&str, StdHasher> =
            CountingFilterSerdeSupport::from_bytes(&bytes).unwrap();
        assert!(restored.contains(&"hello"));

        let json = CountingFilterSerdeSupport::to_json(&filter).unwrap();
        let restored: CountingBloomFilter<&str, StdHasher> =
            CountingFilterSerdeSupport::from_json(&json).unwrap();
        assert!(restored.contains(&"hello"));
    }

    #[test]
    fn test_4bit_counters() {
        let mut filter = CountingBloomFilter::<i32, StdHasher>::with_size(
            1000,
            0.01,
            crate::filters::CounterSize::FourBit,
        );
        for i in 0..100 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: CountingBloomFilter<i32, StdHasher> = bincode::deserialize(&bytes).unwrap();

        for i in 0..100 {
            assert!(restored.contains(&i));
        }
    }

    #[test]
    fn test_8bit_counters() {
        let mut filter = CountingBloomFilter::<i32, StdHasher>::with_size(
            1000,
            0.01,
            crate::filters::CounterSize::EightBit,
        );
        for i in 0..100 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: CountingBloomFilter<i32, StdHasher> = bincode::deserialize(&bytes).unwrap();

        for i in 0..100 {
            assert!(restored.contains(&i));
        }
    }

    #[test]
    fn test_multiple_insertions_preserved() {
        let mut filter = CountingBloomFilter::<&str, StdHasher>::new(100, 0.01);

        for _ in 0..5 {
            filter.insert(&"test");
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let mut restored: CountingBloomFilter<&str, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        for _ in 0..5 {
            assert!(restored.contains(&"test"));
            restored.delete(&"test");
        }
        assert!(!restored.contains(&"test"));
    }

    #[test]
    fn test_empty_filter() {
        let filter = CountingBloomFilter::<String, StdHasher>::new(1000, 0.01);

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: CountingBloomFilter<String, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert!(restored.is_empty());
    }

    #[test]
    fn test_version_mismatch() {
        let serde_repr = CountingBloomFilterSerde {
            version: 99,
            size: 1000,
            num_hashes: 7,
            counter_bits: 4,
            expected_items: 1000,
            target_fpr: 0.01,
            counters: vec![0; 500],
        };

        let result = serde_repr.to_filter::<&str, StdHasher>();
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_counter_bits() {
        let serde_repr = CountingBloomFilterSerde {
            version: FORMAT_VERSION,
            size: 1000,
            num_hashes: 7,
            counter_bits: 3,
            expected_items: 1000,
            target_fpr: 0.01,
            counters: vec![0; 1000],
        };

        let result = serde_repr.to_filter::<&str, StdHasher>();
        assert!(result.is_err());
    }

    #[test]
    fn test_large_filter_serialization() {
        let mut filter = CountingBloomFilter::<i32, StdHasher>::with_size(
            100_000,
            0.001,
            crate::filters::CounterSize::EightBit,
        );
        for i in 0..10_000 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: CountingBloomFilter<i32, StdHasher> = bincode::deserialize(&bytes).unwrap();

        for i in 0..10_000 {
            assert!(restored.contains(&i));
        }
    }

    #[test]
    fn test_deletion_after_deserialization() {
        let mut filter = CountingBloomFilter::<i32, StdHasher>::new(1000, 0.01);

        for i in 0..100 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let mut restored: CountingBloomFilter<i32, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        for i in 0..50 {
            restored.delete(&i);
        }

        for i in 0..50 {
            assert!(!restored.contains(&i));
        }
        for i in 50..100 {
            assert!(restored.contains(&i));
        }
    }
}
