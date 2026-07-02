//! Cache-padded atomic counters and memory-ordering constants.
//!
//! Thin wrappers around [`AtomicU64`] and [`AtomicUsize`] with two additions
//! that the standard library does not provide out of the box:
//!
//! 1. **Cache-line padding** — [`AtomicCounter`] uses `#[repr(align(64))]` and
//!    internal padding so adjacent instances never share a cache line. This
//!    eliminates false-sharing on multi-socket or high-core-count machines
//!    when counters are packed in a contiguous allocation.
//! 2. **Convenience methods** — `increment`, `add`, `sub`, `reset`,
//!    `fetch_increment`, `compare_exchange` are provided so callers do not
//!    need to spell out `Ordering::Relaxed` for every trivial update.
//!
//! All arithmetic operations use [`Ordering::Relaxed`]. If you need acquire
//! or release semantics, use the `load` / `store` methods directly.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Cache line size for preventing false sharing.
///
/// Most modern CPUs use 64-byte cache lines. We pad structures to this size
/// to prevent false sharing between threads. This constant documents the
/// alignment used by `AtomicCounter` (which uses `#[repr(align(64))]`).
pub const CACHE_LINE_SIZE: usize = 64;

/// Cache-line-padded atomic counter.
///
/// Wraps [`AtomicU64`] with 28 bytes of padding on each side so that adjacent
/// `AtomicCounter` values in a `Vec` or array never share a 64-byte cache
/// line, preventing false-sharing under concurrent writes from different
/// cores.
///
/// # Memory Layout
///
/// ```text
/// | padding | AtomicU64 | padding |
/// |---------|-----------|---------|
/// | 28 B    | 8 B       | 28 B    | = 64 bytes total
/// ```
///
/// All read-modify-write operations use [`Ordering::Relaxed`]. For stronger
/// ordering, use [`load`](Self::load) / [`store`](Self::store) with the
/// desired [`Ordering`].
///
/// # Examples
///
/// ```
/// use bloomcraft::util::AtomicCounter;
///
/// let counter = AtomicCounter::new(0);
/// assert_eq!(counter.get(), 0);
///
/// counter.increment();
/// assert_eq!(counter.get(), 1);
///
/// counter.add(5);
/// assert_eq!(counter.get(), 6);
/// ```
#[repr(align(64))] // Align to cache line
#[derive(Debug)]
pub struct AtomicCounter {
    _pad0: [u8; 28],
    value: AtomicU64,
    _pad1: [u8; 28],
}

impl AtomicCounter {
    /// Create a new atomic counter with initial value.
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial counter value
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(42);
    /// assert_eq!(counter.get(), 42);
    /// ```
    #[must_use]
    pub const fn new(initial: u64) -> Self {
        Self {
            _pad0: [0; 28],
            value: AtomicU64::new(initial),
            _pad1: [0; 28],
        }
    }

    /// Load the current value with [`Ordering::Relaxed`].
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(10);
    /// assert_eq!(counter.get(), 10);
    /// ```
    #[inline]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Get current counter value with specific memory ordering.
    ///
    /// # Arguments
    ///
    /// * `ordering` - Memory ordering to use
    ///
    /// # Returns
    ///
    /// Current counter value
    #[inline]
    pub fn load(&self, ordering: Ordering) -> u64 {
        self.value.load(ordering)
    }

    /// Store a value with [`Ordering::Relaxed`].
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(0);
    /// counter.set(100);
    /// assert_eq!(counter.get(), 100);
    /// ```
    #[inline]
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Set counter value with specific memory ordering.
    ///
    /// # Arguments
    ///
    /// * `value` - New counter value
    /// * `ordering` - Memory ordering to use
    #[inline]
    pub fn store(&self, value: u64, ordering: Ordering) {
        self.value.store(value, ordering);
    }

    /// Atomic increment by 1, returning the previous value.
    ///
    /// Equivalent to `fetch_add(1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(5);
    /// let prev = counter.fetch_increment();
    /// assert_eq!(prev, 5);
    /// assert_eq!(counter.get(), 6);
    /// ```
    #[inline]
    pub fn fetch_increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed)
    }

    /// Increment counter by 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(0);
    /// counter.increment();
    /// assert_eq!(counter.get(), 1);
    /// ```
    #[inline]
    pub fn increment(&self) {
        self.fetch_increment();
    }

    /// Add to counter and return previous value.
    ///
    /// # Arguments
    ///
    /// * `delta` - Amount to add
    ///
    /// # Returns
    ///
    /// Previous counter value
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(10);
    /// let prev = counter.fetch_add(5);
    /// assert_eq!(prev, 10);
    /// assert_eq!(counter.get(), 15);
    /// ```
    #[inline]
    pub fn fetch_add(&self, delta: u64) -> u64 {
        self.value.fetch_add(delta, Ordering::Relaxed)
    }

    /// Add to counter.
    ///
    /// # Arguments
    ///
    /// * `delta` - Amount to add
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(10);
    /// counter.add(5);
    /// assert_eq!(counter.get(), 15);
    /// ```
    #[inline]
    pub fn add(&self, delta: u64) {
        self.fetch_add(delta);
    }

    /// Subtract from counter and return previous value.
    ///
    /// # Arguments
    ///
    /// * `delta` - Amount to subtract
    ///
    /// # Returns
    ///
    /// Previous counter value
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(10);
    /// let prev = counter.fetch_sub(3);
    /// assert_eq!(prev, 10);
    /// assert_eq!(counter.get(), 7);
    /// ```
    #[inline]
    pub fn fetch_sub(&self, delta: u64) -> u64 {
        self.value.fetch_sub(delta, Ordering::Relaxed)
    }

    /// Subtract from counter.
    ///
    /// # Arguments
    ///
    /// * `delta` - Amount to subtract
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(10);
    /// counter.sub(3);
    /// assert_eq!(counter.get(), 7);
    /// ```
    #[inline]
    pub fn sub(&self, delta: u64) {
        self.fetch_sub(delta);
    }

    /// Compare and swap counter value.
    ///
    /// Atomically compares the current value with `current` and if they match,
    /// replaces it with `new`. Returns the previous value.
    ///
    /// # Arguments
    ///
    /// * `current` - Expected current value
    /// * `new` - New value to set if current matches
    ///
    /// # Returns
    ///
    /// Result containing previous value and whether swap succeeded
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(10);
    ///
    /// // Successful swap
    /// let result = counter.compare_exchange(10, 20);
    /// assert!(result.is_ok());
    /// assert_eq!(counter.get(), 20);
    ///
    /// // Failed swap (value is now 20, not 10)
    /// let result = counter.compare_exchange(10, 30);
    /// assert!(result.is_err());
    /// assert_eq!(counter.get(), 20);
    /// ```
    #[inline]
    pub fn compare_exchange(&self, current: u64, new: u64) -> Result<u64, u64> {
        self.value
            .compare_exchange(current, new, Ordering::Relaxed, Ordering::Relaxed)
    }

    /// Reset counter to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::util::AtomicCounter;
    ///
    /// let counter = AtomicCounter::new(100);
    /// counter.reset();
    /// assert_eq!(counter.get(), 0);
    /// ```
    #[inline]
    pub fn reset(&self) {
        self.set(0);
    }
}

impl Default for AtomicCounter {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Clone for AtomicCounter {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

/// Thin wrapper around [`AtomicUsize`] for array-index use.
///
/// Unlike [`AtomicCounter`], this type has **no** cache-line padding — it
/// occupies only the size of `AtomicUsize`. Use this for counters that are
/// embedded in larger structs or that do not sit in a contiguous slice where
/// false-sharing is a concern.
#[derive(Debug)]
pub struct AtomicIndex {
    value: AtomicUsize,
}

impl AtomicIndex {
    /// Create a new atomic index with initial value.
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial index value
    #[must_use]
    pub const fn new(initial: usize) -> Self {
        Self {
            value: AtomicUsize::new(initial),
        }
    }

    /// Get current index value.
    #[inline]
    pub fn get(&self) -> usize {
        self.value.load(Ordering::Relaxed)
    }

    /// Set index value.
    #[inline]
    pub fn set(&self, value: usize) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment index by 1.
    #[inline]
    pub fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::Relaxed)
    }

    /// Add to index.
    #[inline]
    pub fn add(&self, delta: usize) -> usize {
        self.value.fetch_add(delta, Ordering::Relaxed)
    }
}

impl Default for AtomicIndex {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Alias constants for [`Ordering`] variants.
///
/// Provided so callers can write `ordering::RELAXED` instead of
/// `Ordering::Relaxed` when they want to avoid importing the enum.
/// Semantically identical to the [`std::sync::atomic::Ordering`] values.
pub mod ordering {
    use super::*;

    /// [`Ordering::Relaxed`]
    pub const RELAXED: Ordering = Ordering::Relaxed;
    /// [`Ordering::Acquire`]
    pub const ACQUIRE: Ordering = Ordering::Acquire;
    /// [`Ordering::Release`]
    pub const RELEASE: Ordering = Ordering::Release;
    /// [`Ordering::AcqRel`]
    pub const ACQ_REL: Ordering = Ordering::AcqRel;
    /// [`Ordering::SeqCst`]
    pub const SEQ_CST: Ordering = Ordering::SeqCst;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_atomic_counter_new() {
        let counter = AtomicCounter::new(42);
        assert_eq!(counter.get(), 42);
    }

    #[test]
    fn test_atomic_counter_default() {
        let counter = AtomicCounter::default();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_atomic_counter_set_get() {
        let counter = AtomicCounter::new(0);
        counter.set(100);
        assert_eq!(counter.get(), 100);
    }

    #[test]
    fn test_atomic_counter_increment() {
        let counter = AtomicCounter::new(5);
        counter.increment();
        assert_eq!(counter.get(), 6);
        counter.increment();
        assert_eq!(counter.get(), 7);
    }

    #[test]
    fn test_atomic_counter_fetch_increment() {
        let counter = AtomicCounter::new(10);
        let prev = counter.fetch_increment();
        assert_eq!(prev, 10);
        assert_eq!(counter.get(), 11);
    }

    #[test]
    fn test_atomic_counter_add() {
        let counter = AtomicCounter::new(10);
        counter.add(5);
        assert_eq!(counter.get(), 15);
    }

    #[test]
    fn test_atomic_counter_fetch_add() {
        let counter = AtomicCounter::new(10);
        let prev = counter.fetch_add(7);
        assert_eq!(prev, 10);
        assert_eq!(counter.get(), 17);
    }

    #[test]
    fn test_atomic_counter_sub() {
        let counter = AtomicCounter::new(20);
        counter.sub(5);
        assert_eq!(counter.get(), 15);
    }

    #[test]
    fn test_atomic_counter_fetch_sub() {
        let counter = AtomicCounter::new(20);
        let prev = counter.fetch_sub(8);
        assert_eq!(prev, 20);
        assert_eq!(counter.get(), 12);
    }

    #[test]
    fn test_atomic_counter_compare_exchange_success() {
        let counter = AtomicCounter::new(10);
        let result = counter.compare_exchange(10, 20);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 10);
        assert_eq!(counter.get(), 20);
    }

    #[test]
    fn test_atomic_counter_compare_exchange_failure() {
        let counter = AtomicCounter::new(10);
        let result = counter.compare_exchange(5, 20);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), 10);
        assert_eq!(counter.get(), 10);
    }

    #[test]
    fn test_atomic_counter_reset() {
        let counter = AtomicCounter::new(100);
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_atomic_counter_clone() {
        let counter1 = AtomicCounter::new(42);
        let counter2 = counter1.clone();
        assert_eq!(counter2.get(), 42);

        counter1.increment();
        assert_eq!(counter1.get(), 43);
        assert_eq!(counter2.get(), 42); // Clone is independent
    }

    #[test]
    fn test_atomic_counter_concurrent() {
        let counter = Arc::new(AtomicCounter::new(0));
        let mut handles = vec![];

        // Spawn 10 threads, each incrementing 1000 times
        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    counter_clone.increment();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.get(), 10000);
    }

    #[test]
    fn test_atomic_counter_memory_ordering() {
        let counter = AtomicCounter::new(0);

        counter.store(10, Ordering::Release);
        let value = counter.load(Ordering::Acquire);
        assert_eq!(value, 10);
    }

    #[test]
    fn test_atomic_index_new() {
        let index = AtomicIndex::new(5);
        assert_eq!(index.get(), 5);
    }

    #[test]
    fn test_atomic_index_default() {
        let index = AtomicIndex::default();
        assert_eq!(index.get(), 0);
    }

    #[test]
    fn test_atomic_index_set_get() {
        let index = AtomicIndex::new(0);
        index.set(42);
        assert_eq!(index.get(), 42);
    }

    #[test]
    fn test_atomic_index_increment() {
        let index = AtomicIndex::new(10);
        let prev = index.increment();
        assert_eq!(prev, 10);
        assert_eq!(index.get(), 11);
    }

    #[test]
    fn test_atomic_index_add() {
        let index = AtomicIndex::new(10);
        let prev = index.add(5);
        assert_eq!(prev, 10);
        assert_eq!(index.get(), 15);
    }

    #[test]
    fn test_cache_line_size() {
        assert_eq!(CACHE_LINE_SIZE, 64);
    }

    #[test]
    fn test_atomic_counter_size() {
        use std::mem;

        // Should be cache-line aligned
        assert_eq!(mem::size_of::<AtomicCounter>(), 64);
        assert_eq!(mem::align_of::<AtomicCounter>(), 64);
    }

    #[test]
    fn test_ordering_constants() {
        use ordering::*;

        assert!(matches!(RELAXED, Ordering::Relaxed));
        assert!(matches!(ACQUIRE, Ordering::Acquire));
        assert!(matches!(RELEASE, Ordering::Release));
        assert!(matches!(ACQ_REL, Ordering::AcqRel));
        assert!(matches!(SEQ_CST, Ordering::SeqCst));
    }
}
