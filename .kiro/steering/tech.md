# BloomCraft Technical Stack

## Build System & Language
- **Language**: Rust 2024 edition (MSRV: 1.70+)
- **Build Tool**: Cargo with optimized release profiles
- **Package Manager**: Cargo with feature flags for optional dependencies

## Core Dependencies
- `thiserror` - Structured error handling
- `num_cpus` - CPU core detection for concurrency
- Optional: `serde` + `bincode` + `bytemuck` (serialization)
- Optional: `xxhash-rust`, `wyhash` (alternative hash functions)

## Development Dependencies
- `criterion` - Performance benchmarking
- `proptest` - Property-based testing
- `rand` - Random data generation for tests
- `serde_json` - JSON serialization testing

## Feature Flags
```toml
default = []
serde = ["dep:serde", "dep:bincode", "dep:bytemuck"]
xxhash = ["dep:xxhash-rust"]
wyhash = ["dep:wyhash"]
all-features = ["serde", "xxhash", "wyhash"]
simd = []
```

## Common Commands

### Building
```bash
# Standard build
cargo build

# Release build with optimizations
cargo build --release

# Build with all features
cargo build --all-features

# Check without building
cargo check
```

### Testing
```bash
# Run all tests
cargo test

# Run tests with all features
cargo test --all-features

# Run specific test module
cargo test core::

# Run property-based tests (may take longer)
cargo test --release -- --ignored
```

### Benchmarking
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench insert

# Compare hash functions
cargo bench hash_functions
```

### Linting & Formatting
```bash
# Format code
cargo fmt

# Run clippy lints
cargo clippy --all-features -- -D warnings

# Check documentation
cargo doc --all-features --no-deps
```

## Release Profile Optimizations
- `opt-level = 3` - Maximum optimization
- `lto = "fat"` - Link-time optimization
- `codegen-units = 1` - Single codegen unit for better optimization
- `strip = true` - Remove debug symbols from release builds

## Platform Support
- **Tier 1**: x86_64-linux, x86_64-windows, aarch64-macos
- **Tier 2**: wasm32-wasi, embedded targets
- **no_std**: Core data structures compatible (requires `alloc`)