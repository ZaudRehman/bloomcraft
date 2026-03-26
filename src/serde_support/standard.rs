//! Serde integration for [`StandardBloomFilter`].
//!
//! Implements [`Serialize`] and [`Deserialize`] for [`StandardBloomFilter`],
//! making it compatible with any serde-backed format — JSON, bincode,
//! MessagePack, CBOR, and so on.
//!
//! # Wire Format
//!
//! The serialized representation is a flat struct with the following fields:
//!
//! | Field          | Type          | Purpose                                      |
//! |----------------|---------------|----------------------------------------------|
//! | `version`      | `u16`         | Format version; currently always `1`         |
//! | `size`         | `usize`       | Total bit-array length *m*                   |
//! | `num_hashes`   | `usize`       | Hash function count *k*                      |
//! | `hash_strategy`| `u8`         | Indexing strategy ID (see below)             |
//! | `hasher_type`  | `String`      | Canonical hasher name for type-safety checks |
//! | `bits`         | `Vec<u64>`    | Raw bit array, packed as 64-bit words        |
//! | `num_bits_set` | `Option<usize>`| Cached popcount; informational only         |
//!
//! The format is intentionally self-describing: every field needed to
//! reconstruct an identical filter is present in the payload.
//!
//! # Hasher Safety
//!
//! A `StandardBloomFilter<T, H>` serialized with hasher `H1` **cannot** be
//! deserialized as `StandardBloomFilter<T, H2>` where `H1 ≠ H2`. Different
//! hashers produce different bit positions for the same item; loading the bit
//! array under the wrong hasher would yield 100% false negatives for all
//! previously inserted items, silently, with no structural signal.
//!
//! Deserialization checks `H::default().name()` against the stored
//! `hasher_type` field and returns a clear [`BloomCraftError::InvalidParameters`]
//! on mismatch rather than producing a corrupt filter.
//!
//! # Hash Strategy Field
//!
//! `hash_strategy` is stored for forward-compatibility. All current filters
//! use [`IndexingStrategy::EnhancedDouble`] internally. On deserialization the
//! field is validated against the known set of IDs — unknown values are
//! rejected — but is not forwarded to [`StandardBloomFilter::from_parts`],
//! which always reconstructs with `EnhancedDouble`. Full strategy round-trip
//! is planned for v0.2.0 when `from_parts` gains strategy support.
//!
//! # Examples
//!
//! ## JSON round-trip
//!
//! ```rust
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1_000, 0.01).unwrap();
//! filter.insert(&"alice");
//! filter.insert(&"bob");
//!
//! let json = serde_json::to_string(&filter).unwrap();
//! let restored: StandardBloomFilter<&str> = serde_json::from_str(&json).unwrap();
//!
//! assert!(restored.contains(&"alice"));
//! assert!(restored.contains(&"bob"));
//! assert!(!restored.contains(&"carol"));
//! ```
//!
//! ## Bincode round-trip
//!
//! ```rust
//! use bloomcraft::filters::StandardBloomFilter;
//!
//! let filter: StandardBloomFilter<i32> = StandardBloomFilter::new(10_000, 0.01).unwrap();
//! for i in 0..1_000_i32 {
//!     filter.insert(&i);
//! }
//!
//! let bytes = bincode::serialize(&filter).unwrap();
//! let restored: StandardBloomFilter<i32> = bincode::deserialize(&bytes).unwrap();
//!
//! for i in 0..1_000_i32 {
//!     assert!(restored.contains(&i));
//! }
//! ```

use crate::core::bitvec::BitVec;
use crate::error::{BloomCraftError, Result};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::hasher::BloomHasher;
use crate::hash::IndexingStrategy;
use serde::de;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Wire-format version tag.
///
/// Bumped on any breaking change to the serialized field layout. Deserialization
/// rejects payloads whose version does not match this constant, preventing
/// silent data corruption when the format evolves.
const FORMAT_VERSION: u16 = 1;

/// Owned, flat representation of a [`StandardBloomFilter`] for serialization.
///
/// This type is an internal detail of the serde implementation. It acts as a
/// data-transfer object: [`StandardBloomFilterSerde::from_filter`] captures a
/// snapshot of a live filter, and [`StandardBloomFilterSerde::into_filter`]
/// validates the snapshot and reconstructs the filter.
///
/// Keeping this separate from `StandardBloomFilter` allows the serde derives to
/// operate on a plain, fully-owned struct without touching the atomic internals
/// of the live type.
#[derive(Serialize, Deserialize)]
struct StandardBloomFilterSerde {
    /// Format version. Checked against [`FORMAT_VERSION`] on deserialization.
    version: u16,
    /// Total bit-array length *m*.
    size: usize,
    /// Number of hash functions *k*.
    num_hashes: usize,
    /// Indexing strategy encoded as a single byte: `0` = Double,
    /// `1` = EnhancedDouble, `2` = Triple. Unknown values are rejected.
    hash_strategy: u8,
    /// Canonical hasher name returned by [`BloomHasher::name`].
    /// Validated against `H::default().name()` on deserialization.
    hasher_type: String,
    /// Raw bit array packed into 64-bit words.
    bits: Vec<u64>,
    /// Cached popcount at serialization time. Stored for diagnostics; not
    /// used during reconstruction — `BitVec::from_raw` is the authoritative
    /// source.
    num_bits_set: Option<usize>,
}

impl StandardBloomFilterSerde {
    /// Snapshot a live filter into the wire representation.
    fn from_filter<T, H>(filter: &StandardBloomFilter<T, H>) -> Self
    where
        T: std::hash::Hash,
        H: BloomHasher + Clone,
    {
        Self {
            version: FORMAT_VERSION,
            size: filter.size(),
            num_hashes: filter.num_hashes(),
            hash_strategy: strategy_to_id(filter.hash_strategy()),
            hasher_type: filter.hasher_name().to_string(),
            bits: filter.raw_bits(),
            num_bits_set: Some(filter.count_set_bits()),
        }
    }

    /// Validate and reconstruct a [`StandardBloomFilter`] from the wire payload.
    ///
    /// Validation order:
    /// 1. Format version must equal [`FORMAT_VERSION`].
    /// 2. `size` must be non-zero.
    /// 3. `num_hashes` must be in `[1, 32]`.
    /// 4. `hash_strategy` must be a known ID.
    /// 5. `hasher_type` must match `H::default().name()`.
    ///
    /// Only after all checks pass is the bit array handed to `BitVec::from_raw`
    /// and then to `from_parts`, preventing partial construction of a corrupt
    /// filter.
    ///
    /// # Errors
    ///
    /// - [`BloomCraftError::InvalidParameters`] — version mismatch, unknown
    ///   strategy ID, or hasher type mismatch.
    /// - [`BloomCraftError::InvalidFilterSize`] — `size == 0`.
    /// - [`BloomCraftError::InvalidHashCount`] — `num_hashes` outside `[1, 32]`.
    /// - Any error propagated from [`BitVec::from_raw`] or
    ///   [`StandardBloomFilter::from_parts`].
    fn into_filter<T, H>(self) -> Result<StandardBloomFilter<T, H>>
    where
        T: std::hash::Hash,
        H: BloomHasher + Default + Clone,
    {
        if self.version != FORMAT_VERSION {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Incompatible serialization version: expected {}, got {}",
                FORMAT_VERSION, self.version
            )));
        }

        if self.size == 0 {
            return Err(BloomCraftError::invalid_filter_size(self.size));
        }

        if self.num_hashes == 0 || self.num_hashes > 32 {
            return Err(BloomCraftError::invalid_hash_count(self.num_hashes, 1, 32));
        }

        // Validate the strategy ID against the known set. The decoded value is
        // not forwarded to from_parts (which always uses EnhancedDouble), but
        // rejecting unknown IDs here prevents future-version payloads from
        // silently deserializing with wrong hash positions.
        id_to_strategy(self.hash_strategy)?;

        let expected = H::default().name();
        if self.hasher_type != expected {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Hasher type mismatch: filter was serialized with '{}', \
                 but the target type uses '{}'. \
                 Using a different hasher produces incorrect bit positions, \
                 causing 100% false negatives for all inserted items. \
                 Deserialize with the same hasher used during serialization.",
                self.hasher_type, expected
            )));
        }

        let bits = BitVec::from_raw(self.bits, self.size)?;
        let filter = StandardBloomFilter::<T, H>::from_parts(bits, self.num_hashes)?;

        let has_bits = self
            .num_bits_set
            .map_or_else(|| filter.count_set_bits() > 0, |n| n > 0);

        if has_bits {
            filter.mark_has_inserts();
        }

        Ok(filter)
    }
}

/// Encode an [`IndexingStrategy`] as its wire-format byte.
///
/// The mapping is stable across versions: values already in circulation must
/// never be reassigned. New strategies are assigned the next available ID.
fn strategy_to_id(strategy: IndexingStrategy) -> u8 {
    match strategy {
        IndexingStrategy::Double => 0,
        IndexingStrategy::EnhancedDouble => 1,
        IndexingStrategy::Triple => 2,
    }
}

/// Decode a wire-format byte into an [`IndexingStrategy`].
///
/// Returns [`BloomCraftError::InvalidParameters`] for any unrecognised ID.
/// Callers should treat an error here as a corrupt or future-version payload.
fn id_to_strategy(id: u8) -> Result<IndexingStrategy> {
    match id {
        0 => Ok(IndexingStrategy::Double),
        1 => Ok(IndexingStrategy::EnhancedDouble),
        2 => Ok(IndexingStrategy::Triple),
        _ => Err(BloomCraftError::invalid_parameters(format!(
            "Unknown hash strategy ID: {}. \
             The payload may be corrupt or produced by a newer version of bloomcraft.",
            id
        ))),
    }
}

impl<T, H> Serialize for StandardBloomFilter<T, H>
where
    T: std::hash::Hash,
    H: BloomHasher + Default + Clone,
{
    /// Serialize the filter to any serde-compatible format.
    ///
    /// The hasher instance is excluded from the payload; it is reconstructed
    /// via [`Default`] on deserialization. Filters backed by a seeded or
    /// non-default hasher instance will lose the seed on round-trip —
    /// use a hasher type that encodes its configuration in its [`Default`]
    /// state if cross-process stability is required.
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        StandardBloomFilterSerde::from_filter(self).serialize(serializer)
    }
}

impl<'de, T, H> Deserialize<'de> for StandardBloomFilter<T, H>
where
    T: std::hash::Hash,
    H: BloomHasher + Default + Clone,
{
    /// Deserialize a filter, validating format version, parameters, and hasher
    /// type before reconstructing the bit array.
    ///
    /// # Errors
    ///
    /// All validation failures from [`StandardBloomFilterSerde::into_filter`]
    /// are forwarded as serde custom errors so they surface naturally in the
    /// deserialization error chain of the caller's chosen format.
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        StandardBloomFilterSerde::deserialize(deserializer)?
            .into_filter::<T, H>()
            .map_err(de::Error::custom)
    }
}

/// Format-level utilities for [`StandardBloomFilter`] serialization.
///
/// Provides convenience wrappers around bincode and JSON and a size estimator
/// for capacity planning. These methods are thin adapters; the canonical
/// serialization behaviour is defined by the [`Serialize`]/[`Deserialize`]
/// impls above.
pub struct StandardFilterSerdeSupport;

impl StandardFilterSerdeSupport {
    /// Estimate the bincode-serialized size in bytes for a filter with the
    /// given parameters.
    ///
    /// This is a conservative lower bound for bincode. JSON and other
    /// human-readable formats are typically 3–4× larger due to field name
    /// overhead and decimal or base64 encoding of the bit array.
    ///
    /// The estimate is useful for pre-allocating send buffers or comparing
    /// serialization costs across parameter choices; it is not suitable as a
    /// hard bound for allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let bytes = StandardFilterSerdeSupport::estimate_size(10_000, 0.01);
    /// assert!(bytes > 1_000 && bytes < 100_000);
    /// ```
    pub fn estimate_size(expected_items: usize, fp_rate: f64) -> usize {
        use crate::core::params;

        let m = params::optimal_bit_count(expected_items, fp_rate).unwrap_or(0);

        // Fixed metadata overhead in bincode encoding:
        //   version(2) + size(8) + num_hashes(8) + hash_strategy(1)
        //   + hasher_type length prefix(8) + ~12 chars name + num_bits_set(9) ≈ 64 bytes.
        let metadata = 64;

        // Bit vector: ⌈m/64⌉ u64 words × 8 bytes each, plus bincode Vec
        // length prefix of 8 bytes.
        let data = ((m + 63) / 64) * 8 + 8;

        metadata + data
    }

    /// Serialize `filter` to bincode bytes.
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::SerializationError`] if bincode fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1_000, 0.01).unwrap();
    /// filter.insert(&"hello");
    ///
    /// let bytes = StandardFilterSerdeSupport::to_bytes(&filter).unwrap();
    /// assert!(!bytes.is_empty());
    /// ```
    #[cfg(feature = "bincode")]
    pub fn to_bytes<T, H>(filter: &StandardBloomFilter<T, H>) -> Result<Vec<u8>>
    where
        T: std::hash::Hash,
        H: BloomHasher + Default + Clone,
    {
        bincode::serialize(filter)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Deserialize a filter from bincode bytes produced by [`to_bytes`].
    ///
    /// The hasher type encoded in the payload must match `H`; a mismatch
    /// returns [`BloomCraftError::InvalidParameters`].
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::SerializationError`] on structural bincode
    /// failures, or [`BloomCraftError::InvalidParameters`] on validation
    /// failures (version, hasher type, strategy ID).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1_000, 0.01).unwrap();
    /// filter.insert(&"hello");
    ///
    /// let bytes = StandardFilterSerdeSupport::to_bytes(&filter).unwrap();
    /// let restored: StandardBloomFilter<&str> =
    ///     StandardFilterSerdeSupport::from_bytes(&bytes).unwrap();
    ///
    /// assert!(restored.contains(&"hello"));
    /// ```
    ///
    /// [`to_bytes`]: StandardFilterSerdeSupport::to_bytes
    #[cfg(feature = "bincode")]
    pub fn from_bytes<T, H>(bytes: &[u8]) -> Result<StandardBloomFilter<T, H>>
    where
        T: std::hash::Hash,
        H: BloomHasher + Default + Clone,
    {
        bincode::deserialize(bytes)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Serialize `filter` to a JSON string.
    ///
    /// The resulting JSON is human-readable and suitable for storage or
    /// transmission where a text format is required. For compact binary
    /// storage, prefer [`to_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::SerializationError`] if JSON serialization
    /// fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1_000, 0.01).unwrap();
    /// filter.insert(&"hello");
    ///
    /// let json = StandardFilterSerdeSupport::to_json(&filter).unwrap();
    /// assert!(json.contains("version"));
    /// ```
    ///
    /// [`to_bytes`]: StandardFilterSerdeSupport::to_bytes
    pub fn to_json<T, H>(filter: &StandardBloomFilter<T, H>) -> Result<String>
    where
        T: std::hash::Hash,
        H: BloomHasher + Default + Clone,
    {
        serde_json::to_string(filter)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }

    /// Deserialize a filter from a JSON string produced by [`to_json`].
    ///
    /// Subject to the same hasher-type validation as [`from_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`BloomCraftError::SerializationError`] on JSON parse failures,
    /// or [`BloomCraftError::InvalidParameters`] on validation failures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bloomcraft::filters::StandardBloomFilter;
    /// use bloomcraft::serde_support::standard::StandardFilterSerdeSupport;
    ///
    /// let filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1_000, 0.01).unwrap();
    /// filter.insert(&"hello");
    ///
    /// let json = StandardFilterSerdeSupport::to_json(&filter).unwrap();
    /// let restored: StandardBloomFilter<&str> =
    ///     StandardFilterSerdeSupport::from_json(&json).unwrap();
    ///
    /// assert!(restored.contains(&"hello"));
    /// ```
    ///
    /// [`to_json`]: StandardFilterSerdeSupport::to_json
    /// [`from_bytes`]: StandardFilterSerdeSupport::from_bytes
    pub fn from_json<T, H>(json: &str) -> Result<StandardBloomFilter<T, H>>
    where
        T: std::hash::Hash,
        H: BloomHasher + Default + Clone,
    {
        serde_json::from_str(json)
            .map_err(|e| BloomCraftError::serialization_error(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::hasher::StdHasher;

    // ── bincode-gated ────────────────────────────────────────────────────────

    #[cfg(feature = "bincode")]
    #[test]
    fn bincode_round_trip() {
        let filter = StandardBloomFilter::<&str, StdHasher>::new(1_000, 0.01).unwrap();
        filter.insert(&"hello");
        filter.insert(&"world");

        let bytes = bincode::serialize(&filter).unwrap();
        assert!(!bytes.is_empty());

        let restored: StandardBloomFilter<&str, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert!(restored.contains(&"hello"));
        assert!(restored.contains(&"world"));
        assert!(!restored.contains(&"missing"));
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn helper_methods_round_trip() {
        // to_bytes / from_bytes are #[cfg(feature = "bincode")]
        let filter = StandardBloomFilter::<&str, StdHasher>::new(1_000, 0.01).unwrap();
        filter.insert(&"hello");

        let bytes = StandardFilterSerdeSupport::to_bytes(&filter).unwrap();
        let from_bytes: StandardBloomFilter<&str, StdHasher> =
            StandardFilterSerdeSupport::from_bytes(&bytes).unwrap();
        assert!(from_bytes.contains(&"hello"));

        let json = StandardFilterSerdeSupport::to_json(&filter).unwrap();
        let from_json: StandardBloomFilter<&str, StdHasher> =
            StandardFilterSerdeSupport::from_json(&json).unwrap();
        assert!(from_json.contains(&"hello"));
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn empty_filter_round_trip() {
        let filter = StandardBloomFilter::<String, StdHasher>::new(1_000, 0.01).unwrap();

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<String, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert!(restored.is_empty());
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn full_filter_round_trip() {
        let filter = StandardBloomFilter::<i32, StdHasher>::new(100, 0.01).unwrap();
        for i in 0..100_i32 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<i32, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        for i in 0..100_i32 {
            assert!(restored.contains(&i));
        }
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn large_filter_round_trip() {
        let filter = StandardBloomFilter::<i32, StdHasher>::new(100_000, 0.001).unwrap();
        for i in 0..10_000_i32 {
            filter.insert(&i);
        }

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<i32, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        for i in 0..10_000_i32 {
            assert!(restored.contains(&i));
        }
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn serialization_preserves_bit_statistics() {
        let filter = StandardBloomFilter::<i32, StdHasher>::new(1_000, 0.01).unwrap();
        for i in 0..100_i32 {
            filter.insert(&i);
        }

        let original_bits = filter.count_set_bits();
        let original_fpr = filter.estimate_fpr();

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<i32, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert_eq!(restored.count_set_bits(), original_bits);
        assert!((restored.estimate_fpr() - original_fpr).abs() < 1e-6);
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn deserialized_non_empty_filter_is_not_reported_as_empty() {
        let filter = StandardBloomFilter::<&str, StdHasher>::new(1_000, 0.01).unwrap();
        filter.insert(&"hello");
        assert!(!filter.is_empty(), "pre-condition: source filter must not be empty");

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<&str, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert!(
            !restored.is_empty(),
            "deserialized non-empty filter must not report is_empty() == true"
        );
        assert!(restored.contains(&"hello"));
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn deserialized_empty_filter_is_still_empty() {
        let filter = StandardBloomFilter::<&str, StdHasher>::new(1_000, 0.01).unwrap();
        assert!(filter.is_empty());

        let bytes = bincode::serialize(&filter).unwrap();
        let restored: StandardBloomFilter<&str, StdHasher> =
            bincode::deserialize(&bytes).unwrap();

        assert!(
            restored.is_empty(),
            "deserialized empty filter must still report is_empty() == true"
        );
    }

    // ── serde-only (no bincode required) ────────────────────────────────────

    #[test]
    fn json_round_trip() {
        let filter = StandardBloomFilter::<&str, StdHasher>::new(100, 0.01).unwrap();
        filter.insert(&"test");

        let json = serde_json::to_string(&filter).unwrap();
        assert!(json.contains("version"));
        assert!(json.contains("bits"));
        assert!(json.contains("hasher_type"));
        assert!(json.contains("StdHasher"));

        let restored: StandardBloomFilter<&str, StdHasher> =
            serde_json::from_str(&json).unwrap();

        assert!(restored.contains(&"test"));
    }

    #[test]
    fn hasher_mismatch_rejected() {
        let filter = StandardBloomFilter::<&str, StdHasher>::new(100, 0.01).unwrap();
        filter.insert(&"test");

        let mut repr = StandardBloomFilterSerde::from_filter(&filter);
        repr.hasher_type = "WrongHasher".to_string();

        let err = repr.into_filter::<&str, StdHasher>().unwrap_err().to_string();
        assert!(err.contains("Hasher type mismatch"));
        assert!(err.contains("WrongHasher"));
        assert!(err.contains("StdHasher"));
    }

    #[test]
    fn correct_hasher_accepted() {
        let filter = StandardBloomFilter::<&str, StdHasher>::new(100, 0.01).unwrap();
        filter.insert(&"test");

        let repr = StandardBloomFilterSerde::from_filter(&filter);
        let restored = repr.into_filter::<&str, StdHasher>().unwrap();
        assert!(restored.contains(&"test"));
    }

    #[test]
    fn estimate_size_plausible() {
        let estimated = StandardFilterSerdeSupport::estimate_size(10_000, 0.01);
        assert!(estimated > 1_000 && estimated < 100_000);
    }

    /// All known strategy IDs must be accepted and yield a working filter.
    ///
    /// The strategy ID is patched directly into the wire repr to exercise the
    /// serde layer in isolation — `with_strategy` does not exist on the public
    /// API.
    #[test]
    fn known_strategy_ids_accepted() {
        let filter = StandardBloomFilter::<&str, StdHasher>::new(1_000, 0.01).unwrap();
        filter.insert(&"test");

        for id in [0u8, 1, 2] {
            let mut repr = StandardBloomFilterSerde::from_filter(&filter);
            repr.hash_strategy = id;

            let restored = repr
                .into_filter::<&str, StdHasher>()
                .unwrap_or_else(|_| panic!("strategy id {} should be accepted", id));

            assert!(
                restored.contains(&"test"),
                "item missing after round-trip with strategy id {}",
                id
            );
        }
    }

    #[test]
    fn unknown_strategy_id_rejected() {
        let filter = StandardBloomFilter::<&str, StdHasher>::new(100, 0.01).unwrap();
        let mut repr = StandardBloomFilterSerde::from_filter(&filter);
        repr.hash_strategy = 99;
        assert!(repr.into_filter::<&str, StdHasher>().is_err());
    }

    #[test]
    fn version_mismatch_rejected() {
        let repr = StandardBloomFilterSerde {
            version: 99,
            size: 1_000,
            num_hashes: 7,
            hash_strategy: 1,
            hasher_type: "StdHasher".to_string(),
            bits: vec![0; 16],
            num_bits_set: Some(0),
        };
        assert!(repr.into_filter::<&str, StdHasher>().is_err());
    }

    #[test]
    fn invalid_strategy_id_returns_error() {
        assert!(id_to_strategy(99).is_err());
    }

    #[test]
    fn strategy_id_encoding_is_stable() {
        let cases = [
            (IndexingStrategy::Double, 0u8),
            (IndexingStrategy::EnhancedDouble, 1),
            (IndexingStrategy::Triple, 2),
        ];
        for (strategy, expected_id) in cases {
            assert_eq!(strategy_to_id(strategy), expected_id);
            assert_eq!(id_to_strategy(expected_id).unwrap(), strategy);
        }
    }
}