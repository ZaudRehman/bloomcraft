# Requirements Document: BloomCraft Correctness Validation

## Introduction

BloomCraft is a production-grade Rust library implementing multiple variants of Bloom filters - probabilistic data structures for approximate set membership queries. This specification defines the correctness requirements and properties that must be validated through comprehensive testing, including property-based testing (PBT).

## Glossary

- **Bloom_Filter**: A space-efficient probabilistic data structure that tests whether an element is a member of a set
- **False_Positive**: When the filter incorrectly reports an element is present when it is not
- **False_Negative**: When the filter incorrectly reports an element is absent when it is present (MUST NEVER OCCUR)
- **BitVec**: Lock-free atomic bit vector used as underlying storage for Bloom filters
- **Hash_Strategy**: Algorithm for generating k hash indices from base hash values
- **Filter_Variant**: Specific implementation of Bloom filter (Standard, Counting, Scalable, etc.)
- **Property_Based_Test**: Test that validates a universal property holds across all valid inputs
- **Cardinality_Estimation**: Approximation of the number of unique items in the filter

## Requirements

### Requirement 1: Core Bloom Filter Correctness

**User Story:** As a developer using BloomCraft, I want the Bloom filter to guarantee no false negatives, so that I can rely on negative query results being definitive.

#### Acceptance Criteria

1. WHEN an item is inserted into a Bloom filter, THEN subsequent contains queries for that item SHALL return true
2. WHEN an item has never been inserted, THEN contains SHALL return false OR true (false positive allowed)
3. WHEN the same item is inserted multiple times, THEN contains SHALL still return true
4. WHEN a filter is cleared, THEN all subsequent contains queries SHALL return false until new items are inserted
5. THE Bloom_Filter SHALL maintain the no-false-negative guarantee across all operations

### Requirement 2: BitVec Atomic Operations

**User Story:** As a concurrent application developer, I want BitVec operations to be thread-safe, so that multiple threads can safely query the filter simultaneously.

#### Acceptance Criteria

1. WHEN multiple threads concurrently set different bits, THEN all bits SHALL be correctly set
2. WHEN a thread sets a bit using Release ordering, THEN threads reading with Acquire ordering SHALL observe the write
3. WHEN bits are set concurrently to the same index, THEN the bit SHALL be set exactly once (idempotent)
4. WHEN counting ones after concurrent writes, THEN the count SHALL equal the number of unique indices set
5. WHEN performing union or intersection operations, THEN the result SHALL correctly reflect the bitwise operation

### Requirement 3: Parameter Calculation Accuracy

**User Story:** As a developer configuring a Bloom filter, I want optimal parameters calculated correctly, so that my filter achieves the target false positive rate.

#### Acceptance Criteria

1. WHEN calculating optimal bit count for n items and ε false positive rate, THEN the result SHALL match the formula m = -n × ln(ε) / (ln 2)²
2. WHEN calculating optimal hash count for m bits and n items, THEN the result SHALL match the formula k = (m/n) × ln 2
3. WHEN calculating expected false positive rate, THEN the result SHALL match the formula (1 - e^(-kn/m))^k
4. WHEN validating parameters, THEN invalid combinations (n=0, ε≤0, ε≥1) SHALL be rejected
5. WHEN parameters result in load factor > 2.0, THEN validation SHALL fail

### Requirement 4: Hash Function Independence

**User Story:** As a Bloom filter implementer, I want hash functions to be statistically independent, so that the filter achieves optimal false positive rates.

#### Acceptance Criteria

1. WHEN generating two hashes from the same input with different seeds, THEN the hashes SHALL be statistically independent
2. WHEN using double hashing strategy, THEN generated indices SHALL be uniformly distributed across the filter
3. WHEN using enhanced double hashing, THEN distribution SHALL be at least as good as standard double hashing
4. WHEN a single bit in input changes, THEN approximately 50% of output bits SHALL change (avalanche property)
5. WHEN generating k indices, THEN no systematic clustering SHALL occur

### Requirement 5: Counting Bloom Filter Deletion

**User Story:** As a developer needing dynamic sets, I want to safely remove items from a counting Bloom filter, so that I can maintain an accurate representation of the current set.

#### Acceptance Criteria

1. WHEN an item is inserted once and removed once, THEN contains SHALL return false
2. WHEN an item is inserted twice and removed once, THEN contains SHALL still return true
3. WHEN attempting to remove an item never inserted, THEN the operation SHALL fail with an error
4. WHEN removing an item, THEN no other items SHALL be affected
5. WHEN a counter would underflow, THEN the operation SHALL fail before modifying state

### Requirement 6: Scalable Bloom Filter Growth

**User Story:** As a developer with unknown dataset size, I want the scalable Bloom filter to grow automatically, so that false positive rate remains bounded.

#### Acceptance Criteria

1. WHEN capacity is exceeded, THEN a new sub-filter SHALL be allocated automatically
2. WHEN querying after growth, THEN all previously inserted items SHALL still return true
3. WHEN inserting beyond initial capacity, THEN false positive rate SHALL remain below 2× target rate
4. WHEN multiple growth events occur, THEN each new sub-filter SHALL have progressively tighter FP rate
5. WHEN querying, THEN all sub-filters SHALL be checked

### Requirement 7: Mergeable Bloom Filter Operations

**User Story:** As a distributed systems developer, I want to merge Bloom filters from different nodes, so that I can aggregate membership information.

#### Acceptance Criteria

1. WHEN two filters with identical parameters are merged via union, THEN the result SHALL contain all items from both filters
2. WHEN filters with different parameters are merged, THEN the operation SHALL fail with an error
3. WHEN performing intersection, THEN the result SHALL only contain items present in both filters
4. WHEN merging filters A and B, THEN union(A, B) SHALL equal union(B, A) (commutative)
5. WHEN merging three filters, THEN union(union(A, B), C) SHALL equal union(A, union(B, C)) (associative)

### Requirement 8: Serialization Round-Trip

**User Story:** As a developer persisting Bloom filters, I want serialization to preserve filter state exactly, so that deserialized filters behave identically.

#### Acceptance Criteria

1. WHEN a filter is serialized then deserialized, THEN all previously inserted items SHALL still return true
2. WHEN a filter is serialized then deserialized, THEN the false positive rate SHALL remain unchanged
3. WHEN using zero-copy serialization, THEN deserialization SHALL be at least 10× faster than standard serde
4. WHEN serializing an empty filter, THEN deserialization SHALL produce an empty filter
5. WHEN serializing a full filter, THEN all bits SHALL be preserved

### Requirement 9: Builder Pattern Validation

**User Story:** As a developer creating Bloom filters, I want the builder to validate parameters at compile-time where possible, so that I catch configuration errors early.

#### Acceptance Criteria

1. WHEN building a filter without setting expected_items, THEN compilation SHALL fail (type-state enforcement)
2. WHEN building a filter without setting false_positive_rate, THEN compilation SHALL fail
3. WHEN building with invalid parameters, THEN build() SHALL return an error
4. WHEN building with valid parameters, THEN the resulting filter SHALL have correct capacity
5. WHEN using builder defaults, THEN reasonable values SHALL be used

### Requirement 10: Concurrent Filter Thread Safety

**User Story:** As a concurrent application developer, I want sharded and striped filters to be safe for concurrent access, so that I can share filters across threads.

#### Acceptance Criteria

1. WHEN multiple threads insert different items concurrently, THEN all items SHALL be correctly inserted
2. WHEN threads query while others insert, THEN no false negatives SHALL occur
3. WHEN using sharded filter, THEN operations SHALL be lock-free
4. WHEN using striped filter, THEN contention SHALL be minimized via striping
5. WHEN concurrent operations complete, THEN the filter SHALL be in a consistent state

### Requirement 11: Cardinality Estimation Accuracy

**User Story:** As a developer analyzing filter contents, I want cardinality estimation to be reasonably accurate, so that I can monitor filter saturation.

#### Acceptance Criteria

1. WHEN a filter contains n unique items, THEN estimate_count SHALL be within 10% of n for 30-70% saturation
2. WHEN a filter is empty, THEN estimate_count SHALL return 0
3. WHEN a filter is nearly full (>90%), THEN estimate_count MAY have higher error (>20%)
4. WHEN the same item is inserted multiple times, THEN estimate_count SHALL not increase
5. WHEN using the formula n = -(m/k) × ln(1 - X/m), THEN the estimate SHALL match this calculation

### Requirement 12: Error Handling Consistency

**User Story:** As a developer integrating BloomCraft, I want consistent error handling, so that I can handle failures predictably.

#### Acceptance Criteria

1. WHEN invalid parameters are provided, THEN a descriptive BloomCraftError SHALL be returned
2. WHEN an operation is unsupported by a variant, THEN UnsupportedOperation error SHALL be returned
3. WHEN filters are incompatible for merging, THEN IncompatibleFilters error SHALL be returned
4. WHEN an index is out of bounds, THEN IndexOutOfBounds error SHALL be returned
5. WHEN a counter would overflow/underflow, THEN CounterOverflow/CounterUnderflow error SHALL be returned

### Requirement 13: Memory Efficiency

**User Story:** As a developer with memory constraints, I want Bloom filters to use optimal space, so that I can maximize capacity within available memory.

#### Acceptance Criteria

1. WHEN a filter is created for n items at ε FP rate, THEN memory usage SHALL be approximately n × (-ln(ε) / (ln 2)²) bits
2. WHEN using BitVec, THEN memory SHALL be ⌈m/64⌉ × 8 bytes plus struct overhead
3. WHEN using counting filter, THEN memory SHALL be approximately 4× standard filter
4. WHEN querying memory_usage(), THEN the returned value SHALL accurately reflect allocated bytes
5. WHEN a filter is cleared, THEN memory SHALL not be deallocated (reusable)

### Requirement 14: Performance Characteristics

**User Story:** As a performance-sensitive developer, I want operations to complete in predictable time, so that I can meet latency requirements.

#### Acceptance Criteria

1. WHEN inserting an item, THEN the operation SHALL complete in O(k) time where k is hash count
2. WHEN querying an item, THEN the operation SHALL complete in O(k) time with early termination possible
3. WHEN counting ones in BitVec, THEN the operation SHALL use CPU POPCNT instruction
4. WHEN using WyHash or XXHash, THEN hashing SHALL be faster than SipHash
5. WHEN using SIMD hasher with batches ≥8, THEN throughput SHALL exceed scalar hashing

### Requirement 15: Mathematical Correctness

**User Story:** As a researcher validating Bloom filter theory, I want implementations to match published formulas exactly, so that empirical results are trustworthy.

#### Acceptance Criteria

1. WHEN calculating optimal parameters, THEN results SHALL match Bloom's 1970 paper formulas
2. WHEN using double hashing, THEN the strategy SHALL match Kirsch & Mitzenmacher 2006
3. WHEN using enhanced double hashing, THEN the strategy SHALL match Dillinger & Manolios 2004
4. WHEN measuring false positive rate empirically, THEN it SHALL be within 15% of theoretical rate
5. WHEN filter is at designed capacity, THEN FP rate SHALL not exceed 2× target rate
