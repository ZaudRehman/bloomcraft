# Design Document: BloomCraft Production-Grade Audit

## Overview

This document details the specific changes required to make BloomCraft production-ready. The changes are organized by category and priority.

## Design Decisions

### D1: Doc Test Type Annotation Strategy

**Decision:** Use explicit type annotations with `&str` for string literal examples.

**Rationale:** 
- `&str` is more ergonomic for examples than `String`
- Avoids the `&"hello"` vs `&String` confusion
- Matches how most users will actually use the library

**Pattern:**
```rust
/// ```
/// use bloomcraft::filters::StandardBloomFilter;
/// 
/// let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
/// filter.insert(&"hello");
/// assert!(filter.contains(&"hello"));
/// ```
```

### D2: Feature-Gated Doc Test Strategy

**Decision:** Use `# #[cfg(feature = "...")]` blocks in doc tests for feature-gated types.

**Pattern:**
```rust
/// ```
/// # #[cfg(feature = "wyhash")]
/// # {
/// use bloomcraft::filters::StandardBloomFilter;
/// use bloomcraft::hash::WyHasher;
/// 
/// let filter: StandardBloomFilter<&str, WyHasher> = 
///     StandardBloomFilter::with_hasher(1000, 0.01, WyHasher::new());
/// # }
/// ```
```

### D3: Unused Field Handling

**Decision:** Keep unused fields with `#[allow(dead_code)]` and documentation explaining future use.

**Rationale:**
- The `hasher` field in `HierarchicalBloomFilter` is stored for future serialization support
- Removing it would break the API when serialization is added

**Pattern:**
```rust
/// Hash function (stored for serialization, actual hashing uses strategy)
#[allow(dead_code)]
hasher: H,
```

### D4: Method Naming Consistency

**Decision:** Use `delete()` consistently for removal operations, not `remove()`.

**Rationale:**
- `delete()` is already implemented in `CountingBloomFilter`
- Matches the semantic of "deleting" an item from a set
- `remove()` in Rust stdlib typically returns the removed item, which doesn't apply here

## Detailed Changes

### Category 1: Doc Test Type Annotations

#### 1.1 Builder Module Doc Tests

**Files:** `src/builder/standard.rs`, `src/builder/counting.rs`, `src/builder/scalable.rs`

**Change Pattern:**
```rust
// BEFORE (fails type inference)
let filter = StandardBloomFilterBuilder::new()
    .expected_items(10_000)
    .false_positive_rate(0.01)
    .build()
    .unwrap();

// AFTER (explicit type)
let filter: StandardBloomFilter<&str> = StandardBloomFilterBuilder::new()
    .expected_items(10_000)
    .false_positive_rate(0.01)
    .build()
    .unwrap();
```

#### 1.2 Filter Module Doc Tests

**Files:** `src/filters/standard.rs`, `src/filters/counting.rs`, `src/filters/partitioned.rs`, `src/filters/scalable.rs`, `src/filters/hierarchical.rs`

**Change Pattern:**
```rust
// BEFORE (type mismatch: String vs &str)
let mut filter: StandardBloomFilter<String> = StandardBloomFilter::new(1000, 0.01);
filter.insert(&"hello");  // ERROR: expected &String, found &&str

// AFTER (consistent types)
let mut filter: StandardBloomFilter<&str> = StandardBloomFilter::new(1000, 0.01);
filter.insert(&"hello");  // OK
```

### Category 2: Import Path Fixes

#### 2.1 WyHasher Import Path

**File:** `src/hash/wyhash.rs`

**Change:**
```rust
// BEFORE (incorrect path)
use bloomcraft::hash::hasher::{BloomHasher, WyHasher};

// AFTER (correct path)
use bloomcraft::hash::{BloomHasher, WyHasher};
```

#### 2.2 XxHasher Import Path

**File:** `src/hash/xxhash.rs`

**Change:**
```rust
// BEFORE (incorrect path)
use bloomcraft::hash::hasher::{BloomHasher, XxHasher};

// AFTER (correct path)
use bloomcraft::hash::{BloomHasher, XxHasher};
```

### Category 3: Method Name Fixes

#### 3.1 CountingBloomFilter remove → delete

**File:** `src/builder/counting.rs` (doc tests)

**Change:**
```rust
// BEFORE (method doesn't exist)
filter.remove(&"hello").unwrap();

// AFTER (correct method name)
filter.delete(&"hello");
```

### Category 4: Code Example Fixes

#### 4.1 BitVec to_raw Example

**File:** `src/core/bitvec.rs`

**Change:**
```rust
// BEFORE (Vec<u64> doesn't support & operator)
let raw = bv.to_raw();
assert_eq!(raw & 1, 1);

// AFTER (index into Vec first)
let raw = bv.to_raw();
assert_eq!(raw[0] & 1, 1);
```

#### 4.2 CountingBloomFilter histogram Example

**File:** `src/filters/counting.rs`

**Change:**
```rust
// BEFORE (Vec doesn't implement Display, syntax error)
println!("Counters with value 0: {}", histogram);
println!("Counters with value 1: {}", histogram);[1]

// AFTER (correct indexing and formatting)
println!("Counters with value 0: {}", histogram[0]);
println!("Counters with value 1: {}", histogram[1]);
```

#### 4.3 Hash Module compare_hashers Example

**File:** `src/hash/mod.rs`

**Change:**
```rust
// BEFORE (type mismatch)
let items = vec![1, 2, 3, 4, 5];
let results = compare_hashers(&items);

// AFTER (correct type)
let items: Vec<&[u8]> = vec![b"item1", b"item2", b"item3"];
let results = compare_hashers(&items);
```

### Category 5: Warning Fixes

#### 5.1 Unused hasher Field

**File:** `src/filters/hierarchical.rs`

**Change:**
```rust
// Add allow attribute with documentation
/// Hash function (stored for future serialization support)
#[allow(dead_code)]
hasher: H,
```

#### 5.2 Unused to_filter Method

**File:** `src/serde_support/standard.rs`

**Options:**
1. Make the method `pub` if it's intended for external use
2. Add `#[allow(dead_code)]` if it's for internal/future use
3. Remove if truly unused

**Recommended:** Add `#[allow(dead_code)]` with comment explaining it's used by deserialization

#### 5.3 Unused Imports

**Files:** `src/serde_support/sharded.rs`, `src/serde_support/striped.rs`

**Change:**
```rust
// BEFORE
use super::*;

// AFTER (remove or use specific imports)
// Remove the line if nothing from super is used
```

## Implementation Order

### Phase 1: Doc Test Fixes (Priority: Critical)

1. Fix `src/filters/standard.rs` doc tests (most examples)
2. Fix `src/builder/standard.rs` doc tests
3. Fix `src/builder/counting.rs` doc tests (including remove → delete)
4. Fix `src/builder/scalable.rs` doc tests
5. Fix `src/filters/counting.rs` doc tests (histogram example)
6. Fix `src/filters/partitioned.rs` doc tests
7. Fix `src/filters/scalable.rs` doc tests
8. Fix `src/filters/hierarchical.rs` doc tests
9. Fix `src/hash/wyhash.rs` doc tests (import paths)
10. Fix `src/hash/xxhash.rs` doc tests (import paths)
11. Fix `src/hash/mod.rs` doc tests (compare_hashers)
12. Fix `src/core/bitvec.rs` doc tests (to_raw)
13. Fix `src/lib.rs` doc tests
14. Fix `src/sync/*.rs` doc tests
15. Fix `src/serde_support/*.rs` doc tests
16. Fix `src/metrics/*.rs` doc tests

### Phase 2: Warning Fixes (Priority: High)

1. Add `#[allow(dead_code)]` to `hasher` field in `HierarchicalBloomFilter`
2. Handle `to_filter` method in `serde_support/standard.rs`
3. Remove unused imports in `serde_support/sharded.rs` and `striped.rs`
4. Fix mutable variable warnings in test files

### Phase 3: Clippy Fixes (Priority: Medium)

1. Run `cargo clippy --all-features`
2. Address any remaining warnings

## Testing Strategy

After each phase:
1. Run `cargo build --all-features` - verify no compilation errors
2. Run `cargo test --all-features` - verify all tests pass
3. Run `cargo test --doc --all-features` - verify doc tests pass
4. Run `cargo clippy --all-features` - verify no warnings

## Success Metrics

- 0 compilation errors
- 0 compilation warnings
- 676+ unit tests passing
- 89 doc tests passing (currently 0)
- 0 clippy warnings
