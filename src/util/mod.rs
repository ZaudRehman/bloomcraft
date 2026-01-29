//! Internal utility functions and helpers.
//!
//! This module provides low-level utilities used throughout the BloomCraft crate.
//! These are implementation details and not part of the public API.
//!
//! # Modules
//!
//! - [`atomic`] - Atomic operation helpers and lock-free primitives
//! - [`bitops`] - Bit manipulation utilities and optimizations
//! - [`cache_detect`] - CPU cache size detection for performance tuning

#![allow(clippy::pedantic)]

pub mod atomic;
pub mod bitops;
pub mod cache_detect;

// Re-export commonly used items
pub use atomic::AtomicCounter;
pub use bitops::{count_ones, next_power_of_two, is_power_of_two};
pub use cache_detect::{detect_cache_sizes, CacheSizes};
