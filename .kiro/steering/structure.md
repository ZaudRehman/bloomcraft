# BloomCraft Project Structure

## Module Organization

### Core Architecture
```
src/
├── lib.rs              # Public API, re-exports, prelude
├── error.rs            # Centralized error handling with BloomCraftError
├── core/               # Fundamental traits and utilities
│   ├── filter.rs       # BloomFilter trait definition
│   ├── bitvec.rs       # Lock-free atomic bit vector
│   └── params.rs       # Mathematical parameter calculations
├── hash/               # Hash function implementations
│   ├── hasher.rs       # BloomHasher trait
│   ├── strategies.rs   # Double/triple hashing strategies
│   ├── wyhash.rs       # WyHash implementation (feature-gated)
│   ├── xxhash.rs       # XXHash implementation (feature-gated)
│   └── simd.rs         # SIMD-optimized batch hashing
└── filters/            # Bloom filter variants
    ├── standard.rs     # StandardBloomFilter (optimal space)
    ├── counting.rs     # CountingBloomFilter (supports deletion)
    ├── scalable.rs     # ScalableBloomFilter (dynamic growth)
    ├── partitioned.rs  # PartitionedBloomFilter (cache-optimized)
    ├── hierarchical.rs # HierarchicalBloomFilter (multi-level)
    ├── classic_bits.rs # Historical: Burton Bloom Method 2 (1970)
    └── classic_hash.rs # Historical: Burton Bloom Method 1 (1970)
```

### Supporting Modules
```
src/
├── builder/            # Type-safe builder pattern
│   ├── standard.rs     # StandardBloomFilterBuilder
│   ├── counting.rs     # CountingBloomFilterBuilder
│   └── scalable.rs     # ScalableBloomFilterBuilder
├── sync/               # Concurrent implementations
│   ├── sharded.rs      # ShardedBloomFilter (lock-free)
│   ├── striped.rs      # StripedBloomFilter (striped locking)
│   └── atomic_counter.rs # Atomic counter utilities
├── metrics/            # Observability and monitoring
│   ├── collector.rs    # MetricsCollector interface
│   ├── tracker.rs      # FalsePositiveTracker
│   └── histogram.rs    # LatencyHistogram
├── serde_support/      # Serialization (feature-gated)
│   ├── standard.rs     # Standard filter serialization
│   ├── counting.rs     # Counting filter serialization
│   └── zerocopy.rs     # Zero-copy serialization
└── util/               # Internal utilities
    ├── atomic.rs       # Atomic operation helpers
    └── bitops.rs       # Bit manipulation utilities
```

## Design Patterns

### Trait-Based Architecture
- **Core Trait**: `BloomFilter<T>` - Common interface for all variants
- **Extension Traits**: `DeletableBloomFilter`, `MergeableBloomFilter`, `ScalableBloomFilter`
- **Hash Abstraction**: `BloomHasher` trait for pluggable hash functions

### Builder Pattern
- Type-state pattern enforces required parameters at compile time
- Fluent API with method chaining
- Runtime validation for parameter ranges

### Error Handling
- Centralized `BloomCraftError` enum with structured variants
- `Result<T>` type alias throughout crate
- Convenience constructors for common error cases

### Feature Flags
- Optional dependencies behind feature gates
- `serde` feature for serialization support
- `xxhash`/`wyhash` features for alternative hash functions

## Naming Conventions

### Types
- **Filters**: `{Variant}BloomFilter` (e.g., `StandardBloomFilter`)
- **Builders**: `{Variant}BloomFilterBuilder`
- **Errors**: `BloomCraftError` with descriptive variants
- **Traits**: Descriptive names (`BloomFilter`, `BloomHasher`)

### Functions
- **Constructors**: `new()`, `with_*()` for customization
- **Operations**: `insert()`, `contains()`, `delete()` (if supported)
- **Queries**: `len()`, `is_empty()`, `false_positive_rate()`
- **Batch**: `*_batch()` suffix for batch operations

### Modules
- **Lowercase**: All module names in snake_case
- **Descriptive**: Clear purpose (e.g., `serde_support`, `hash`)
- **Hierarchical**: Logical grouping by functionality

## File Organization Rules

### Module Structure
- Each major component gets its own module directory
- `mod.rs` files provide public API and re-exports
- Implementation files named after the type they contain

### Documentation
- Every public item has comprehensive documentation
- Module-level docs explain purpose and usage patterns
- Examples in doc comments for all public APIs

### Testing
- Unit tests in same file as implementation (`#[cfg(test)]`)
- Integration tests in `tests/` directory
- Property-based tests for mathematical correctness
- Benchmark tests in `benches/` directory

## Import Patterns

### Internal Imports
```rust
use crate::core::{BloomFilter, BitVec};
use crate::error::{BloomCraftError, Result};
use crate::hash::BloomHasher;
```

### External Dependencies
```rust
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
```

### Prelude Usage
```rust
// For users
use bloomcraft::prelude::*;

// Internal prelude in core modules
use crate::core::prelude::*;
```

## Code Organization Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Dependency Direction**: Core modules don't depend on higher-level modules
3. **Feature Isolation**: Optional features cleanly separated behind feature gates
4. **API Consistency**: Similar patterns across all filter variants
5. **Performance Focus**: Hot paths optimized, cold paths prioritize clarity