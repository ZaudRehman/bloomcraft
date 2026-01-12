# Requirements Document: BloomCraft Production-Grade Audit

## Introduction

This specification defines the requirements for auditing and refactoring BloomCraft into a production-grade, standard Rust crate. The goal is to ensure the crate has correct structure, idiomatic APIs, sound trait abstractions, clear documentation, and well-designed tests suitable for public release.

## Current State Summary

- **Unit Tests**: 676 tests pass with `--all-features`
- **Doc Tests**: 89 failures due to type annotation issues and incorrect examples
- **Compilation**: Successful with minor warnings (unused field, unused method)
- **Architecture**: Well-structured with proper module organization

## Glossary

- **Doc_Test**: Executable code examples in documentation comments
- **Type_Annotation**: Explicit type specification required by Rust's type inference
- **API_Coherence**: Consistency of method signatures, naming, and behavior across the crate
- **Idiomatic_Rust**: Code following Rust community conventions and best practices
- **Production_Ready**: Code suitable for use in production systems with stability guarantees

## Requirements

### Requirement 1: Fix Doc Test Type Annotations

**User Story:** As a developer reading BloomCraft documentation, I want code examples to compile and run correctly, so that I can learn from working examples.

#### Acceptance Criteria

1. WHEN a doc test creates a filter via builder pattern, THEN the type annotation SHALL be explicit (e.g., `let filter: StandardBloomFilter<String> = ...`)
2. WHEN a doc test uses string literals with filters, THEN the filter type SHALL match the literal type (use `&str` not `String` when using `&"hello"`)
3. WHEN a doc test calls `build()` or `build_with_metadata()`, THEN the type parameter `T` SHALL be inferable or explicitly specified
4. ALL doc tests SHALL pass when running `cargo test --doc --all-features`
5. WHEN examples use feature-gated types (WyHasher, XxHasher), THEN they SHALL be wrapped in `#[cfg(feature = "...")]` blocks

### Requirement 2: Fix Doc Test Import Paths

**User Story:** As a developer copying examples from documentation, I want import paths to be correct, so that my code compiles without modification.

#### Acceptance Criteria

1. WHEN doc tests import `WyHasher`, THEN the path SHALL be `bloomcraft::hash::WyHasher` (not `bloomcraft::hash::hasher::WyHasher`)
2. WHEN doc tests import `XxHasher`, THEN the path SHALL be `bloomcraft::hash::XxHasher`
3. ALL public types SHALL be importable via documented paths
4. WHEN a type is re-exported in `prelude`, THEN doc tests MAY use `use bloomcraft::prelude::*`
5. WHEN doc tests use internal types, THEN they SHALL use the public re-export path

### Requirement 3: Fix Doc Test Method Calls

**User Story:** As a developer using BloomCraft, I want documented methods to exist and work as shown, so that I can trust the documentation.

#### Acceptance Criteria

1. WHEN doc tests call `filter.remove()`, THEN the method SHALL exist OR the example SHALL use `filter.delete()` instead
2. WHEN doc tests use `histogram[0]` syntax, THEN the example SHALL use correct indexing (not `histogram` without index)
3. WHEN doc tests use `raw & 1`, THEN the example SHALL use `raw[0] & 1` for Vec<u64> types
4. ALL method calls in doc tests SHALL match actual method signatures
5. WHEN a method returns `Result`, THEN doc tests SHALL handle the result appropriately

### Requirement 4: Fix Type Mismatches in Doc Tests

**User Story:** As a developer, I want doc test examples to use consistent types, so that I understand the correct usage patterns.

#### Acceptance Criteria

1. WHEN a filter is typed as `StandardBloomFilter<String>`, THEN insert/contains calls SHALL use `&String` or `&str` consistently
2. WHEN using string literals like `&"hello"`, THEN the filter type SHALL be `StandardBloomFilter<&str>` (not `String`)
3. WHEN batch methods expect `&[T]`, THEN doc tests SHALL pass the correct slice type
4. ALL type annotations in doc tests SHALL match the actual generic parameters
5. WHEN examples show type inference, THEN the inferred type SHALL be documented in comments

### Requirement 5: Remove Unused Code Warnings

**User Story:** As a maintainer, I want the codebase to have no warnings, so that real issues are not hidden by noise.

#### Acceptance Criteria

1. WHEN a field is unused (e.g., `hasher` in `HierarchicalBloomFilter`), THEN it SHALL be either used or removed with `#[allow(dead_code)]` and justification
2. WHEN a method is unused (e.g., `to_filter` in serde_support), THEN it SHALL be either made public, used internally, or removed
3. WHEN imports are unused, THEN they SHALL be removed
4. `cargo clippy --all-features` SHALL produce no warnings
5. `cargo build --all-features` SHALL produce no warnings

### Requirement 6: Ensure API Coherence

**User Story:** As a developer using multiple filter variants, I want consistent APIs across all filter types, so that I can switch between them easily.

#### Acceptance Criteria

1. ALL filter types SHALL implement the `BloomFilter<T>` trait
2. WHEN a filter supports deletion, THEN it SHALL implement `DeletableBloomFilter<T>` with `delete()` method
3. WHEN a filter supports merging, THEN it SHALL implement `MergeableBloomFilter<T>`
4. ALL filter constructors SHALL follow the pattern: `new(expected_items, fpr)` and `with_hasher(...)`
5. ALL filter types SHALL have consistent method naming (e.g., `size()`, `hash_count()`, `is_empty()`)

### Requirement 7: Validate Public API Stability

**User Story:** As a library consumer, I want stable public APIs, so that my code doesn't break on minor version updates.

#### Acceptance Criteria

1. ALL public types SHALL be documented with `///` doc comments
2. ALL public methods SHALL have `# Examples` sections in their documentation
3. ALL public methods returning `Result` SHALL have `# Errors` sections
4. ALL public methods that can panic SHALL have `# Panics` sections
5. NO public API SHALL expose internal implementation details

### Requirement 8: Ensure Thread Safety Documentation

**User Story:** As a concurrent application developer, I want clear documentation about thread safety, so that I can use filters correctly in multi-threaded contexts.

#### Acceptance Criteria

1. ALL filter types SHALL document their `Send` and `Sync` bounds
2. WHEN a method requires `&mut self`, THEN documentation SHALL explain why exclusive access is needed
3. WHEN a method is lock-free, THEN documentation SHALL state this explicitly
4. WHEN atomic ordering is used, THEN the choice SHALL be documented with `// SAFETY:` comments
5. ALL concurrent filter types (Sharded, Striped) SHALL have thread safety examples

### Requirement 9: Validate Error Handling

**User Story:** As a developer handling errors, I want comprehensive error types, so that I can provide meaningful error messages to users.

#### Acceptance Criteria

1. ALL fallible operations SHALL return `Result<T, BloomCraftError>`
2. ALL error variants SHALL have descriptive messages
3. `BloomCraftError` SHALL implement `std::error::Error`
4. ALL error constructors SHALL be documented
5. WHEN an error can be recovered from, THEN documentation SHALL explain recovery strategies

### Requirement 10: Ensure Feature Flag Correctness

**User Story:** As a developer with minimal dependencies, I want feature flags to work correctly, so that I only compile what I need.

#### Acceptance Criteria

1. WHEN `serde` feature is disabled, THEN serialization code SHALL not compile
2. WHEN `wyhash` feature is disabled, THEN `WyHasher` SHALL not be available
3. WHEN `xxhash` feature is disabled, THEN `XxHasher` SHALL not be available
4. `cargo build` (no features) SHALL succeed
5. `cargo build --all-features` SHALL succeed
6. `cargo test` (no features) SHALL pass all applicable tests

### Requirement 11: Validate Mathematical Correctness

**User Story:** As a researcher, I want implementations to match published formulas, so that empirical results are trustworthy.

#### Acceptance Criteria

1. WHEN calculating optimal parameters, THEN results SHALL match Bloom's 1970 paper formulas
2. WHEN using double hashing, THEN the strategy SHALL match Kirsch & Mitzenmacher 2006
3. WHEN measuring false positive rate empirically, THEN it SHALL be within 15% of theoretical rate
4. ALL mathematical formulas SHALL be documented in code comments
5. ALL parameter calculations SHALL have unit tests verifying correctness

### Requirement 12: Ensure Memory Safety

**User Story:** As a systems developer, I want guaranteed memory safety, so that I can use BloomCraft in safety-critical applications.

#### Acceptance Criteria

1. ALL `unsafe` blocks SHALL have `// SAFETY:` comments explaining why they are safe
2. NO undefined behavior SHALL be possible through the public API
3. ALL pointer operations SHALL be bounds-checked or documented as unchecked
4. `cargo miri test` SHALL pass (if applicable)
5. ALL atomic operations SHALL use appropriate memory ordering

## Implementation Priority

### Phase 1: Critical (Doc Test Fixes)
1. Fix type annotations in all doc tests
2. Fix import paths for WyHasher/XxHasher
3. Fix method name mismatches (remove → delete)
4. Fix type mismatches (String vs &str)

### Phase 2: Warnings
1. Address unused field warnings
2. Address unused method warnings
3. Remove unused imports

### Phase 3: Polish
1. Add missing `# Errors` sections
2. Add missing `# Panics` sections
3. Add `// SAFETY:` comments to unsafe blocks
4. Run clippy and fix all warnings

## Success Criteria

1. `cargo build` passes with no warnings
2. `cargo build --all-features` passes with no warnings
3. `cargo test` passes all tests
4. `cargo test --all-features` passes all tests (including doc tests)
5. `cargo clippy --all-features` produces no warnings
6. `cargo doc --all-features --no-deps` produces no warnings

## Files Requiring Changes

| File | Primary Issues |
|------|----------------|
| `src/filters/standard.rs` | Doc test type mismatches (String vs &str) |
| `src/filters/counting.rs` | Doc test histogram example, missing `remove` → `delete` |
| `src/filters/hierarchical.rs` | Doc test type annotations, unused `hasher` field |
| `src/filters/partitioned.rs` | Doc test type annotations |
| `src/filters/scalable.rs` | Doc test type annotations |
| `src/builder/standard.rs` | Doc test type annotations |
| `src/builder/counting.rs` | Doc test type annotations, `remove` → `delete` |
| `src/builder/scalable.rs` | Doc test type annotations |
| `src/hash/wyhash.rs` | Doc test import paths |
| `src/hash/xxhash.rs` | Doc test import paths |
| `src/hash/mod.rs` | Doc test type for `compare_hashers` |
| `src/core/bitvec.rs` | Doc test `raw & 1` → `raw[0] & 1` |
| `src/serde_support/standard.rs` | Unused `to_filter` method |
| `src/lib.rs` | Doc test examples |
| `src/sync/*.rs` | Doc test examples |
| `src/metrics/*.rs` | Doc test examples |
