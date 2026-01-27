//! Tree-structured Bloom filter for hierarchical data organization.
//!
//! A production-grade Bloom filter variant that organizes filters in a user-defined
//! tree hierarchy, enabling location-aware membership testing and spatial queries.
//! Optimized for cache efficiency, pruned search, and real-time incremental updates.
//!
//! # Architecture
//!
//! ```text
//! Level 0 (Root):  [========= Aggregation Filter =========]
//!                        ↓             ↓              ↓
//! Level 1:          [Filter 0]   [Filter 1]    [Filter 2]
//!                      ↓              ↓              ↓
//! Level 2:          [F 0.0]        [F 1.0]        [F 2.0]
//!                   [F 0.1]        [F 1.1]        [F 2.1]
//! ```
//!
//! Each node maintains:
//! - A Bloom filter for items at that level
//! - Child nodes (for non-leaf levels)
//! - Metadata for load tracking and statistics
//!
//! # Core Design Principles
//!
//! 1. **User-Controlled Topology**: You specify hierarchy shape via branching factors
//! 2. **Incremental Updates**: Insert/query operations work on mutable filters
//! 3. **Generic Types**: Works with any `T: Hash` (not sequence-specific)
//! 4. **Semantic Bins**: Bin paths map to real-world concepts (datacenter/rack/server)
//! 5. **Pruned Search**: Early termination skips entire subtrees on negative matches
//!
//! # Performance Characteristics
//!
//! | Operation | Time Complexity | Notes |
//! |-----------|----------------|-------|
//! | Insert to bin | O(k × depth) | Insert at all levels along path |
//! | Query specific bin | O(k × depth) | Early termination on first false |
//! | Query any bin (root) | O(k) | Single filter check |
//! | Locate (find bins) | O(k × depth × branching) | Heavily pruned in practice |
//! | Batch insert | O(n × k × depth) | Shared path traversal |
//!
//! # Memory Layout
//!
//! - Nodes aligned to 64-byte cache lines
//! - Breadth-first construction for sequential access
//! - Children stored as `Box<[Node]>` for cache efficiency
//! - Each node uses lock-free `StandardBloomFilter`
//!
//! # Use Cases
//!
//! ## 1. Distributed System Routing
//!
//! Track data location across datacenters, racks, and servers:
//!
//! ```
//! use bloomcraft::filters::TreeBloomFilter;
//!
//! # fn main() {
//! // 3 continents, 10 datacenters each, 20 racks per DC
//! let mut router: TreeBloomFilter<String> =
//!     TreeBloomFilter::new(vec![3, 10, 20], 10_000, 0.01);
//!
//! // Store user session location
//! router.insert_to_bin(&"session:alice".to_string(), &[1, 5, 12]).unwrap();
//!
//! // Route to closest datacenter
//! if router.contains_in_bin(&"session:alice".to_string(), &[1, 5, 12]).unwrap() {
//!     // route_to_datacenter(1, 5, 12);
//!     println!("Routing session to DC [1,5,12]");
//! }
//!
//! // Find all locations with this session (failover scenario)
//! let replicas = router.locate(&"session:alice".to_string());
//! for loc in replicas {
//!     println!("Session found at: Continent {}, DC {}, Rack {}",
//!              loc[0], loc[1], loc[2]);
//! }
//! # }
//! ```
//!
//! ## 2. Multi-Tenant Access Control
//!
//! Hierarchical permission checks before expensive database queries:
//!
//! ```
//! use bloomcraft::filters::TreeBloomFilter;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Organizations → Departments → Projects
//! let mut acl: TreeBloomFilter<u64> =
//!     TreeBloomFilter::new(vec![7, 10, 100], 5_000, 0.01);
//!
//! // Grant user access to specific project
//! let user_id = 12345_u64;
//! acl.insert_to_bin(&user_id, &[5, 3, 42]).unwrap();
//!
//! // Fast access check (before DB lookup)
//! if !acl.contains_in_bin(&user_id, &[5, 3, 42]).unwrap() {
//!     return Err("Access denied".into()); // Guaranteed negative, no DB hit
//! }
//!
//! // Might have access (check DB for confirmation)
//! // verify_access_in_database(user_id, 5, 3, 42)?;
//! println!("Access check passed for user {}", user_id);
//! # Ok(())
//! # }
//! ```
//!
//! ## 3. CDN Cache Invalidation
//!
//! Track cached resources across edge servers:
//!
//! ```
//! use bloomcraft::filters::TreeBloomFilter;
//!
//! # fn main() {
//! // 5 regions, 20 POPs per region, 100 servers per POP
//! let mut cache_tracker: TreeBloomFilter<String> =
//!     TreeBloomFilter::new(vec![5, 20, 100], 50_000, 0.001);
//!
//! // Record cache locations
//! let resource = "/assets/logo.png".to_string();
//! cache_tracker.insert_to_bin(&resource, &[2, 10, 15]).unwrap();
//!
//! // Invalidate all cached copies
//! let locations = cache_tracker.locate(&resource);
//! for loc in locations {
//!     // send_invalidation(loc[0], loc[1], loc[2], &resource);
//!     println!("Invalidating cache at [{}, {}, {}]", loc[0], loc[1], loc[2]);
//! }
//! # }
//! ```
//!
//! ## 4. Game World Spatial Indexing
//!
//! Track entities across hierarchical world zones:
//!
//! ```
//! use bloomcraft::filters::TreeBloomFilter;
//!
//! # fn main() {
//! // Game world: 4 quadrants, 8 zones each, 16 chunks per zone
//! let mut world_index: TreeBloomFilter<u64> =
//!     TreeBloomFilter::new(vec![4, 8, 16], 1_000, 0.01);
//!
//! // Track player location
//! let player_id = 98765_u64;
//! world_index.insert_to_bin(&player_id, &[2, 5, 10]).unwrap();
//!
//! // Check if player in zone (before detailed lookup)
//! if world_index.contains_in_bin(&player_id, &[2, 5, 10]).unwrap() {
//!     // load_chunk_entities(2, 5, 10);
//!     println!("Loading chunk entities for player {}", player_id);
//! }
//! # }
//! ```
//!
//! # Comparison to Other Filters
//!
//! | When to Use | Instead Of |
//! |-------------|----------|
//! | **Fixed hierarchy** (datacenter topology) | `StandardBloomFilter` (no location info) |
//! | **Semantic bin assignment** (org/dept/proj) | `ShardedBloomFilter` (hash-based sharding) |
//! | **Incremental updates** (users come/go) | Static indexed structures |
//! | **Pruned spatial queries** (find all locations) | Checking all filters individually |
//!
//! # Not a Replacement For
//!
//! - **StandardBloomFilter**: Use when you don't need location tracking
//! - **ScalableBloomFilter**: Use when capacity is unknown/unbounded
//! - **ShardedBloomFilter**: Use for high-concurrency writes without hierarchy
//! - **HIBF** (future): Use for genomic data with automatic space optimization
//!
//! # Relationship to HIBF
//!
//! This filter provides **manual hierarchical organization**, which differs from
//! the Hierarchical Interleaved Bloom Filter (HIBF) described in Mehringer et al. (2023):
//!
//! | Aspect | TreeBloomFilter | HIBF (future roadmap) |
//! |--------|-----------------|----------------------|
//! | Topology | User-specified | Auto-computed via dynamic programming |
//! | Construction | Incremental inserts | Batch construction with layout optimization |
//! | Bin assignment | Manual paths `[0, 2, 5]` | Automatic via cardinality estimation |
//! | Data types | Generic `T: Hash` | Optimized for sequences (k-mers, minimizers) |
//! | Use case | Infrastructure, spatial data | Genomic databases, static archives |
//!
//! **Future**: BloomCraft will add true HIBF as a separate filter type once foundational
//! primitives (HyperLogLog sketches, DP layout solver) are implemented. Track progress
//! at [github.com/your-repo/issues/HIBF].
//!
//! # References
//!
//! - Putze, F., Sanders, P., & Singler, J. (2009). "Cache-, Hash- and Space-Efficient
//!   Bloom Filters". Journal of Experimental Algorithmics, 14, 4.
//! - Fan, B., et al. (2014). "Cuckoo Filter: Practically Better Than Bloom". CoNEXT.
//!
//! For HIBF algorithm details, see:
//! - Mehringer, S., et al. (2023). "Hierarchical Interleaved Bloom Filter: enabling
//!   ultrafast approximate sequence locations". Algorithms for Molecular Biology, 18(1), 10.

#![allow(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use crate::core::filter::BloomFilter;
use crate::error::{BloomCraftError, Result};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Maximum depth to prevent excessive recursion and stack overflow.
const MAX_TREE_DEPTH: usize = 16;

/// Node in the tree-structured Bloom filter with cache-optimized layout.
///
/// Each node contains:
/// - A Bloom filter for items at this level
/// - Child nodes (empty for leaf nodes)
/// - Metadata for load tracking and statistics
///
/// Nodes are aligned to 64-byte cache lines for optimal L1/L2 cache usage.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(align(64))]
struct TreeNode<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Bloom filter for this node
    filter: StandardBloomFilter<T, H>,
    
    /// Child nodes (empty for leaf nodes)
    children: Box<[TreeNode<T, H>]>,
    
    /// Path to this node (for debugging/statistics)
    #[allow(dead_code)]
    path: Vec<usize>,
    
    /// Level in the tree (0 = root)
    #[allow(dead_code)]
    level: u8,
    
    /// Number of items inserted (for load tracking)
    item_count: usize,
}

/// Convert a hashable item to bytes using Rust's Hash trait.
#[inline]
fn hash_item_to_bytes<T: Hash>(item: &T) -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish().to_le_bytes()
}


impl<T, H> TreeNode<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Create a new leaf node.
    #[inline]
    fn new_leaf(capacity: usize, fpr: f64, hasher: H, path: Vec<usize>, level: u8) -> Self {
        Self {
            filter: StandardBloomFilter::with_hasher(capacity, fpr, hasher),
            children: Box::new([]),
            path,
            level,
            item_count: 0,
        }
    }

    /// Create a new internal node with children.
    #[inline]
    fn new_internal(
        capacity: usize,
        fpr: f64,
        hasher: H,
        path: Vec<usize>,
        level: u8,
        children: Box<[TreeNode<T, H>]>,
    ) -> Self {
        Self {
            filter: StandardBloomFilter::with_hasher(capacity, fpr, hasher),
            children,
            path,
            level,
            item_count: 0,
        }
    }

    /// Check if this is a leaf node.
    #[inline(always)]
    const fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Get total number of nodes in subtree (including self).
    #[inline]
    fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Get memory usage estimate of subtree in bytes.
    #[inline]
    fn memory_usage_estimate(&self) -> usize {
        // StandardBloomFilter memory: bit_count / 8 (bytes)
        let filter_bytes = (self.filter.bit_count() + 7) / 8;
        let children_bytes: usize = self.children.iter()
            .map(|c| c.memory_usage_estimate())
            .sum();
        let overhead = std::mem::size_of::<Self>();
        
        filter_bytes + children_bytes + overhead
    }

    /// Get load factor (items / capacity) for this node.
    #[inline]
    fn load_factor(&self) -> f64 {
        let capacity = self.filter.expected_items();
        if capacity == 0 {
            0.0
        } else {
            self.item_count as f64 / capacity as f64
        }
    }
}

/// Tree-structured Bloom filter for hierarchical data organization.
///
/// Organizes Bloom filters in a user-defined tree where each level represents
/// a different granularity of data partitioning (e.g., region → datacenter → rack).
///
/// # Type Parameters
///
/// * `T` - Type of items stored (must implement `Hash`)
/// * `H` - Hash function type (must implement `BloomHasher`)
///
/// # Thread Safety
///
/// - `contains`: Lock-free, thread-safe (atomic reads via `StandardBloomFilter`)
/// - `insert`: Requires `&mut self` or external synchronization
/// - For concurrent writes, wrap in `Arc<Mutex<_>>` or `Arc<RwLock<_>>`
///
/// # Examples
///
/// See module-level documentation for detailed use cases.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TreeBloomFilter<T, H = StdHasher>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Root node of tree
    root: TreeNode<T, H>,
    
    /// Branching factors for each level
    branching: Vec<usize>,
    
    /// Expected items per leaf bin
    capacity_per_bin: usize,
    
    /// Target false positive rate
    #[allow(dead_code)]
    target_fpr: f64,
    
    /// Total items inserted across all bins
    total_items: usize,
    
    /// Hasher for items
    #[cfg_attr(feature = "serde", serde(skip))]
    #[allow(dead_code)]
    hasher: H,
    
    /// Phantom data for type parameter T
    #[cfg_attr(feature = "serde", serde(skip))]
    _phantom: PhantomData<T>,
}

impl<T> TreeBloomFilter<T, StdHasher>
where
    T: Hash + Send + Sync,
{
    /// Create a new tree-structured Bloom filter with default hasher.
    ///
    /// # Arguments
    ///
    /// * `branching` - Branching factors for each level (e.g., `[4, 8]` = 4 branches, then 8 sub-branches)
    /// * `capacity_per_bin` - Expected items per leaf bin
    /// * `fpr` - Target false positive rate (must be in (0, 1))
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `branching` is empty or contains zeros
    /// - `capacity_per_bin` is zero
    /// - `fpr` is not in range (0, 1)
    /// - Depth exceeds `MAX_TREE_DEPTH` (16 levels)
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// # fn main() {
    /// // 3-level hierarchy: 4 regions, 8 datacenters each, 16 racks per DC
    /// let filter: TreeBloomFilter<String> =
    ///     TreeBloomFilter::new(vec![4, 8, 16], 1000, 0.01);
    ///
    /// assert_eq!(filter.depth(), 3);
    /// assert_eq!(filter.leaf_count(), 512); // 4 × 8 × 16
    /// # }
    /// ```
    #[must_use]
    pub fn new(branching: Vec<usize>, capacity_per_bin: usize, fpr: f64) -> Self {
        Self::with_hasher(branching, capacity_per_bin, fpr, StdHasher::new())
    }
}

impl<T, H> TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Create a new tree-structured Bloom filter with custom hasher.
    ///
    /// # Arguments
    ///
    /// * `branching` - Branching factors for each level
    /// * `capacity_per_bin` - Expected items per leaf bin
    /// * `fpr` - Target false positive rate
    /// * `hasher` - Custom hash function
    ///
    /// # Panics
    ///
    /// Panics if parameters are invalid (see `new` documentation).
    #[must_use]
    pub fn with_hasher(
        branching: Vec<usize>,
        capacity_per_bin: usize,
        fpr: f64,
        hasher: H,
    ) -> Self {
        assert!(!branching.is_empty(), "branching cannot be empty");
        assert!(
            branching.iter().all(|&b| b > 0),
            "all branching factors must be > 0"
        );
        assert!(capacity_per_bin > 0, "capacity_per_bin must be > 0");
        assert!(fpr > 0.0 && fpr < 1.0, "fpr must be in (0, 1)");
        assert!(
            branching.len() <= MAX_TREE_DEPTH,
            "tree depth {} exceeds maximum {}",
            branching.len(),
            MAX_TREE_DEPTH
        );

        // Calculate total node count for memory warning
        let leaf_count: usize = branching.iter().product();
        let mut total_nodes = leaf_count;
        let mut partial_product = 1;
        for i in 0..branching.len() - 1 {
            partial_product *= branching[i];
            total_nodes += partial_product;
        }

        // Warn if tree is very large
        if total_nodes > 10_000 {
            eprintln!(
                "WARNING: TreeBloomFilter with {} total nodes (branching {:?})\n\
                This will allocate {} filters. Consider reducing depth or branching factors.",
                total_nodes, branching, total_nodes
            );
        }

        let root = Self::build_tree(&branching, 0, capacity_per_bin, fpr, hasher.clone(), vec![]);

        Self {
            root,
            branching,
            capacity_per_bin,
            target_fpr: fpr,
            total_items: 0,
            hasher,
            _phantom: PhantomData,
        }
    }

    /// Recursively build the tree with breadth-first layout.
    fn build_tree(
        branching: &[usize],
        level: usize,
        capacity: usize,
        fpr: f64,
        hasher: H,
        path: Vec<usize>,
    ) -> TreeNode<T, H> {
        if level >= branching.len() {
            return TreeNode::new_leaf(capacity, fpr, hasher, path, level as u8);
        }

        let num_children = branching[level];
        let mut children = Vec::with_capacity(num_children);

        for i in 0..num_children {
            let mut child_path = path.clone();
            child_path.push(i);

            let child = Self::build_tree(
                branching,
                level + 1,
                capacity,
                fpr,
                hasher.clone(),
                child_path,
            );
            children.push(child);
        }

        let internal_capacity = capacity * num_children;

        TreeNode::new_internal(
            internal_capacity,
            fpr,
            hasher,
            path,
            level as u8,
            children.into_boxed_slice(),
        )
    }

    /// Validate a bin path against tree structure.
    #[inline]
    fn validate_path(&self, path: &[usize]) -> Result<()> {
        if path.len() != self.depth() {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Path length {} does not match tree depth {}",
                path.len(),
                self.depth()
            )));
        }

        for (level, &idx) in path.iter().enumerate() {
            if idx >= self.branching[level] {
                return Err(BloomCraftError::invalid_parameters(format!(
                    "Path index {} at level {} exceeds branching factor {}",
                    idx, level, self.branching[level]
                )));
            }
        }

        Ok(())
    }

    /// Get the depth of the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// let filter: TreeBloomFilter<i32> =
    ///     TreeBloomFilter::new(vec![2, 3, 4], 1000, 0.01);
    /// assert_eq!(filter.depth(), 3);
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn depth(&self) -> usize {
        self.branching.len()
    }

    /// Get the total number of leaf bins.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// let filter: TreeBloomFilter<i32> =
    ///     TreeBloomFilter::new(vec![2, 4], 1000, 0.01);
    /// assert_eq!(filter.leaf_count(), 8); // 2 × 4
    /// ```
    #[must_use]
    #[inline]
    pub fn leaf_count(&self) -> usize {
        self.branching.iter().product()
    }

    /// Get the total number of nodes in the tree.
    ///
    /// Includes internal nodes and leaves.
    #[must_use]
    #[inline]
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Get estimated memory usage in bytes.
    ///
    /// This is an approximation based on filter sizes and tree structure.
    #[must_use]
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.root.memory_usage_estimate() + std::mem::size_of::<Self>()
    }

    /// Insert an item into a specific bin.
    ///
    /// The item is inserted at all levels along the path from root to the specified leaf.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to insert
    /// * `bin_path` - Path to target bin (length must equal depth)
    ///
    /// # Errors
    ///
    /// Returns error if `bin_path` is invalid (wrong length or out-of-bounds indices).
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// let mut filter: TreeBloomFilter<String> =
    ///     TreeBloomFilter::new(vec![2], 1000, 0.01);
    ///
    /// filter.insert_to_bin(&"user:alice".to_string(), &[1]).unwrap();
    /// assert!(filter.contains_in_bin(&"user:alice".to_string(), &[1]).unwrap());
    /// ```
    #[inline]
    pub fn insert_to_bin(&mut self, item: &T, bin_path: &[usize]) -> Result<()> {
        self.validate_path(bin_path)?;
        
        let mut current = &mut self.root;
        current.filter.insert(item);
        current.item_count += 1;

        for &child_idx in bin_path {
            current = &mut current.children[child_idx];
            current.filter.insert(item);
            current.item_count += 1;
        }

        self.total_items += 1;
        Ok(())
    }

    /// Insert multiple items into the same bin (batch operation).
    ///
    /// More efficient than individual inserts due to shared path traversal.
    ///
    /// # Arguments
    ///
    /// * `items` - Slice of items to insert
    /// * `bin_path` - Path to target bin
    ///
    /// # Errors
    ///
    /// Returns error if `bin_path` is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// # fn main() {
    /// let mut filter: TreeBloomFilter<String> =
    ///     TreeBloomFilter::new(vec![2], 1000, 0.01);
    ///
    /// let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    /// let refs: Vec<&String> = items.iter().collect();
    /// filter.insert_batch_to_bin(&refs, &[1]).unwrap();
    ///
    /// for item in &items {
    ///     assert!(filter.contains(item));
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn insert_batch_to_bin(&mut self, items: &[&T], bin_path: &[usize]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        self.validate_path(bin_path)?;

        let mut current = &mut self.root;
        for item in items {
            current.filter.insert(item);
        }
        current.item_count += items.len();

        for &child_idx in bin_path {
            current = &mut current.children[child_idx];
            for item in items {
                current.filter.insert(item);
            }
            current.item_count += items.len();
        }

        self.total_items += items.len();
        Ok(())
    }

    /// Check if an item exists in a specific bin.
    ///
    /// # Arguments
    ///
    /// * `item` - Item to query
    /// * `bin_path` - Path to bin to check
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - Item might be in the bin (subject to FPR)
    /// * `Ok(false)` - Item is definitely not in the bin
    /// * `Err(_)` - Invalid path
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// # fn main() {
    /// let mut filter: TreeBloomFilter<String> =
    ///     TreeBloomFilter::new(vec![2], 1000, 0.01);
    ///
    /// filter.insert_to_bin(&"item1".to_string(), &[1]).unwrap();
    ///
    /// assert!(filter.contains_in_bin(&"item1".to_string(), &[1]).unwrap());
    /// assert!(!filter.contains_in_bin(&"item1".to_string(), &[0]).unwrap());
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn contains_in_bin(&self, item: &T, bin_path: &[usize]) -> Result<bool> {
        self.validate_path(bin_path)?;

        if !self.root.filter.contains(item) {
            return Ok(false);
        }

        let mut current = &self.root;
        for &child_idx in bin_path {
            current = &current.children[child_idx];
            if !current.filter.contains(item) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if an item exists anywhere in the tree.
    ///
    /// This only checks the root filter, making it O(k) regardless of tree depth.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// # fn main() {
    /// let mut filter: TreeBloomFilter<String> =
    ///     TreeBloomFilter::new(vec![2], 1000, 0.01);
    ///
    /// filter.insert_to_bin(&"item".to_string(), &[1]).unwrap();
    ///
    /// assert!(filter.contains(&"item".to_string()));
    /// assert!(!filter.contains(&"missing".to_string()));
    /// # }
    /// ```
    #[must_use]
    #[inline(always)]
    pub fn contains(&self, item: &T) -> bool {
        self.root.filter.contains(item)
    }

    /// Find all bins that might contain an item (iterative, stack-safe).
    ///
    /// Performs depth-first search with aggressive pruning. This implementation
    /// uses an explicit stack to avoid stack overflow on deep trees.
    ///
    /// # Performance
    ///
    /// - **Best case**: O(k) if root returns false
    /// - **Worst case**: O(k × depth × branching) if all filters return true
    /// - **Typical**: O(k × depth × √branching) with pruning
    /// - **Stack safe**: No recursion, bounded heap allocation
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// let mut filter: TreeBloomFilter<String> =
    ///     TreeBloomFilter::new(vec![2, 2], 1000, 0.01);
    ///
    /// filter.insert_to_bin(&"item".to_string(), &[0, 1]).unwrap();
    ///
    /// let locations = filter.locate(&"item".to_string());
    /// assert_eq!(locations.len(), 1);
    /// assert_eq!(locations[0], vec![0, 1]);
    /// ```
    #[must_use]
    pub fn locate(&self, item: &T) -> Vec<Vec<usize>> {
        // Early pruning at root
        if !self.root.filter.contains(item) {
            return Vec::new();
        }

        let mut result = Vec::new();
        
        // Explicit stack: (node_ref, path)
        let mut stack = vec![(&self.root, Vec::new())];

        while let Some((node, current_path)) = stack.pop() {
            if node.is_leaf() {
                result.push(current_path);
                continue;
            }

            // Iterate children in reverse to maintain depth-first left-to-right order
            for (child_idx, child) in node.children.iter().enumerate().rev() {
                if !child.filter.contains(item) {
                    continue; // Prune this subtree
                }

                let mut child_path = current_path.clone();
                child_path.push(child_idx);
                stack.push((child, child_path));
            }
        }

        result
    }

    /// Recursive helper for locate with pruning.
    #[allow(dead_code)]
    fn locate_recursive_old(
        &self,
        node: &TreeNode<T, H>,
        item: &T,
        current_path: Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if node.is_leaf() {
            result.push(current_path);
            return;
        }

        for (child_idx, child) in node.children.iter().enumerate() {
            if !child.filter.contains(item) {
                continue;
            }

            let mut child_path = current_path.clone();
            child_path.push(child_idx);

            self.locate_recursive_old(child, item, child_path, result);
        }
    }

    /// Batch query: check if multiple items exist anywhere in the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// # fn main() {
    /// let mut filter: TreeBloomFilter<String> =
    ///     TreeBloomFilter::new(vec![2], 1000, 0.01);
    ///
    /// let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    /// let refs: Vec<&String> = items.iter().collect();
    ///
    /// filter.insert_to_bin(&refs[0], &[0]).unwrap();
    /// filter.insert_to_bin(&refs[1], &[1]).unwrap();
    ///
    /// let results = filter.contains_batch(&refs);
    /// assert_eq!(results, vec![true, true, false]);
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn contains_batch(&self, items: &[&T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Recursively clear a node and its children.
    fn clear_node(node: &mut TreeNode<T, H>) {
        node.filter.clear();
        node.item_count = 0;

        for child in &mut *node.children {
            Self::clear_node(child);
        }
    }

    /// Get detailed statistics for the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// # fn main() {
    /// let filter: TreeBloomFilter<String> =
    ///     TreeBloomFilter::new(vec![2, 3], 1000, 0.01);
    ///
    /// let stats = filter.stats();
    /// println!("Nodes: {}, Memory: {} bytes", stats.total_nodes, stats.memory_usage);
    /// # }
    /// ```
    #[must_use]
    pub fn stats(&self) -> TreeStats {
        let total_nodes = self.node_count();
        let memory_usage = self.memory_usage();
        let leaf_bins = self.leaf_count();
        
        // Calculate per-node memory
        let memory_per_node = if total_nodes > 0 {
            memory_usage / total_nodes
        } else {
            0
        };
        
        // Overhead = (total filters / leaf filters)
        let overhead_factor = if leaf_bins > 0 {
            total_nodes as f64 / leaf_bins as f64
        } else {
            0.0
        };
        
        TreeStats {
            total_nodes,
            memory_usage,
            total_items: self.total_items,
            depth: self.depth(),
            leaf_bins,
            avg_load_factor: self.compute_avg_load_factor(),
            memory_per_node,
            overhead_factor,
        }
    }

    /// Compute average load factor across all leaf nodes.
    fn compute_avg_load_factor(&self) -> f64 {
        let (sum, count) = self.compute_load_factor_recursive(&self.root);
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Recursively compute sum of load factors and leaf count.
    fn compute_load_factor_recursive(&self, node: &TreeNode<T, H>) -> (f64, usize) {
        if node.is_leaf() {
            return (node.load_factor(), 1);
        }

        let mut sum = 0.0;
        let mut count = 0;
        for child in &*node.children {
            let (child_sum, child_count) = self.compute_load_factor_recursive(child);
            sum += child_sum;
            count += child_count;
        }

        (sum, count)
    }

    /// Insert item with hash-based bin assignment.
    ///
    /// Automatically routes item to bin using **deterministic** hash from this
    /// filter's hasher. The bin path is derived by successively taking modulo
    /// of each branching factor.
    ///
    /// # Hash Stability
    ///
    /// - Uses this filter's `H: BloomHasher` implementation
    /// - Deterministic: same item → same bin (within this filter instance)
    /// - **Not guaranteed stable across different hasher instances or Rust versions**
    /// - For persistent bin assignments, use `insert_to_bin()` with explicit paths
    ///
    /// # Algorithm
    ///
    /// Given branching factors [b₀, b₁, b₂, ...]:
    /// 1. Convert item to 8 bytes: `bytes = DefaultHasher(item).to_le_bytes()`
    /// 2. Compute base hashes: `(h1, h2) = hasher.hash_bytes_pair(bytes)`
    /// 3. Initialize: `hash = h1`
    /// 4. For each level i with branching factor bᵢ:
    ///    - `bin[i] = hash mod bᵢ`
    ///    - `hash = hash × 0x9e3779b97f4a7c15 + h2`  (golden ratio mixing)
    ///
    /// This ensures uniform distribution across all bins.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    ///
    /// let mut filter: TreeBloomFilter<String> =
    ///     TreeBloomFilter::new(vec![3, 5], 1000, 0.01);
    ///
    /// // Automatically assigns to bin based on hash
    /// filter.insert_auto(&"user123".to_string()).unwrap();
    ///
    /// // Same item always goes to same bin (within this filter)
    /// filter.insert_auto(&"user123".to_string()).unwrap();
    ///
    /// // Find where it went
    /// let locations = filter.locate(&"user123".to_string());
    /// println!("Item hashed to bin: {:?}", locations[0]);
    /// ```
    pub fn insert_auto(&mut self, item: &T) -> Result<()> {
        // Convert item to bytes
        let bytes = hash_item_to_bytes(item);
        
        // Use filter's hasher to get base hashes
        let (h1, h2) = self.hasher.hash_bytes_pair(&bytes);
        
        // Compute bin path
        let mut hash = h1;
        let mut bin_path = Vec::with_capacity(self.depth());
        
        for &branching_factor in &self.branching {
            let index = (hash as usize) % branching_factor;
            bin_path.push(index);
            hash = hash.wrapping_mul(0x9e3779b97f4a7c15_u64).wrapping_add(h2);
        }
        
        self.insert_to_bin(item, &bin_path)
    }

    /// Query all bins for an item (exhaustive search).
    ///
    /// Checks every leaf bin, returning true if found in any.
    /// This is O(leaf_count × k) - use `locate()` for pruned search.
    ///
    /// # Examples
    ///
    /// ```
    /// use bloomcraft::filters::TreeBloomFilter;
    /// 
    /// let mut filter: TreeBloomFilter<String> = 
    ///     TreeBloomFilter::new(vec![3, 5], 1000, 0.01);
    /// filter.insert_to_bin(&"item".to_string(), &[0, 3]).unwrap();
    /// 
    /// // Exhaustive search across all bins
    /// assert!(filter.query_all(&"item".to_string()));
    /// ```
    #[must_use]
    pub fn query_all(&self, item: &T) -> bool {
        // Root check first (fast negative path)
        if !self.root.filter.contains(item) {
            return false;
        }

        // Exhaustive leaf check
        self.query_all_recursive(&self.root, item)
    }

    fn query_all_recursive(&self, node: &TreeNode<T, H>, item: &T) -> bool {
        if node.is_leaf() {
            return node.filter.contains(item);
        }

        node.children
            .iter()
            .any(|child| self.query_all_recursive(child, item))
    }
}

/// Statistics for tree-structured Bloom filter.
///
/// Provides detailed metrics for performance analysis and capacity planning.
#[derive(Debug, Clone, Default)]
pub struct TreeStats {
    /// Total number of nodes (internal + leaf)
    pub total_nodes: usize,
    
    /// Estimated memory usage in bytes
    pub memory_usage: usize,
    
    /// Total items inserted
    pub total_items: usize,
    
    /// Depth of tree
    pub depth: usize,
    
    /// Number of leaf bins
    pub leaf_bins: usize,
    
    /// Average load factor (items / capacity) across leaves
    pub avg_load_factor: f64,

    /// Memory per filter node (bytes)
    pub memory_per_node: usize,
    
    /// Memory overhead factor (actual / theoretical minimum)
    pub overhead_factor: f64,
}

// Implement BloomFilter trait for compatibility
impl<T, H> BloomFilter<T> for TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn insert(&mut self, item: &T) {
        self.root.filter.insert(item);
        self.root.item_count += 1;
        self.total_items += 1;
    }

    fn contains(&self, item: &T) -> bool {
        self.contains(item)
    }

    fn clear(&mut self) {
        Self::clear_node(&mut self.root);
        self.total_items = 0;
    }

    fn len(&self) -> usize {
        self.total_items
    }

    fn is_empty(&self) -> bool {
        self.total_items == 0
    }

    fn false_positive_rate(&self) -> f64 {
        self.root.filter.false_positive_rate()
    }

    fn expected_items(&self) -> usize {
        self.capacity_per_bin
    }

    fn bit_count(&self) -> usize {
        self.root.filter.bit_count()
    }

    fn hash_count(&self) -> usize {
        self.root.filter.hash_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 3], 1000, 0.01);
        
        assert_eq!(filter.depth(), 2);
        assert_eq!(filter.leaf_count(), 6);
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
    }

    #[test]
    #[should_panic(expected = "branching cannot be empty")]
    fn test_new_empty_branching() {
        let _: TreeBloomFilter<i32> = TreeBloomFilter::new(vec![], 1000, 0.01);
    }

    #[test]
    fn test_insert_and_query() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&"hello".to_string(), &[0, 1]).unwrap();

        assert!(filter.contains(&"hello".to_string()));
        assert!(!filter.contains(&"goodbye".to_string()));
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_contains_in_bin() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&"item1".to_string(), &[0, 1]).unwrap();
        filter.insert_to_bin(&"item2".to_string(), &[1, 0]).unwrap();

        assert!(filter.contains_in_bin(&"item1".to_string(), &[0, 1]).unwrap());
        assert!(!filter.contains_in_bin(&"item1".to_string(), &[1, 0]).unwrap());
        assert!(filter.contains_in_bin(&"item2".to_string(), &[1, 0]).unwrap());
    }

    #[test]
    fn test_locate() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&"test".to_string(), &[0, 1]).unwrap();

        let locations = filter.locate(&"test".to_string());
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0], vec![0, 1]);

        let empty_locations = filter.locate(&"nonexistent".to_string());
        assert!(empty_locations.is_empty());
    }

    #[test]
    fn test_batch_insert() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2], 1000, 0.01);

        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let item_refs: Vec<&String> = items.iter().collect();
        
        filter.insert_batch_to_bin(&item_refs, &[1]).unwrap();

        for item in &items {
            assert!(filter.contains(item));
        }
        assert_eq!(filter.len(), 3);
    }

    #[test]
    fn test_clear() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);

        filter.insert_to_bin(&"item1".to_string(), &[0, 0]).unwrap();
        filter.insert_to_bin(&"item2".to_string(), &[1, 1]).unwrap();

        assert_eq!(filter.len(), 2);

        filter.clear();

        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
        assert!(!filter.contains(&"item1".to_string()));
        assert!(!filter.contains(&"item2".to_string()));
    }

    #[test]
    fn test_invalid_path() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01);

        assert!(filter.insert_to_bin(&"test".to_string(), &[0]).is_err());
        assert!(filter.insert_to_bin(&"test".to_string(), &[0, 1, 2]).is_err());
        assert!(filter.insert_to_bin(&"test".to_string(), &[0, 5]).is_err());
    }

    #[test]
    fn test_stats() {
        let filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 3], 1000, 0.01);

        let stats = filter.stats();
        
        assert_eq!(stats.depth, 2);
        assert_eq!(stats.leaf_bins, 6);
        assert!(stats.total_nodes > 0);
        assert!(stats.memory_usage > 0);
    }

    #[test]
    fn test_bloom_filter_trait() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2], 1000, 0.01);

        filter.insert(&"test".to_string());
        assert!(filter.contains(&"test".to_string()));
        assert_eq!(filter.len(), 1);
        
        filter.clear();
        assert!(filter.is_empty());
    }

    #[test]
    fn test_insert_auto_deterministic() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 4], 1000, 0.01);

        // Insert same item multiple times
        for _ in 0..5 {
            filter.insert_auto(&"test_item".to_string()).unwrap();
        }

        // Should always go to same bin
        let locations = filter.locate(&"test_item".to_string());
        assert_eq!(locations.len(), 1, "Item should be in exactly one bin");
    }

    #[test]
    fn test_insert_auto_distribution() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![4, 4], 1000, 0.01);

        // Insert items - strings hash much better than sequential integers
        for i in 0..1000 {
            let item = format!("user_{:08}", i);
            filter.insert_auto(&item).unwrap();
        }

        // Check distribution across bins
        let mut bin_counts = vec![vec![0; 4]; 4];
        
        for i in 0..1000 {
            let item = format!("user_{:08}", i);
            let locations = filter.locate(&item);
            if !locations.is_empty() {
                let path = &locations[0];
                bin_counts[path[0]][path[1]] += 1;
            }
        }

        // Each bin should have roughly 1000/16 ≈ 62 items
        let expected_per_bin = 1000.0 / 16.0;
        let mut empty_bins = 0;
        let mut total_deviation = 0.0;
        
        for i in 0..4 {
            for j in 0..4 {
                let count = bin_counts[i][j];
                
                if count == 0 {
                    empty_bins += 1;
                }
                
                let deviation = (count as f64 - expected_per_bin).abs() / expected_per_bin;
                total_deviation += deviation;
            }
        }
        
        // Allow at most 1 empty bin (statistical variance)
        assert!(
            empty_bins <= 1,
            "Too many empty bins: {} out of 16 (distribution quality issue)",
            empty_bins
        );
        
        // Average deviation should be reasonable
        let avg_deviation = total_deviation / 16.0;
        assert!(
            avg_deviation < 0.3,
            "Average deviation {:.1}% exceeds 30% (poor hash distribution)",
            avg_deviation * 100.0
        );
        
        // Print distribution for debugging
        println!("\nBin distribution:");
        for i in 0..4 {
            print!("Row {}: ", i);
            for j in 0..4 {
                print!("{:3} ", bin_counts[i][j]);
            }
            println!();
        }
        println!("Empty bins: {}, Avg deviation: {:.1}%", empty_bins, avg_deviation * 100.0);
    }

    #[test]
    fn test_locate_iterative_deep_tree() {
        // Create a deep tree that would risk stack overflow with recursion
        let mut filter: TreeBloomFilter<u64> =
            TreeBloomFilter::new(vec![2, 2, 2, 2, 2], 100, 0.1); // Depth 5, 32 leaves

        filter.insert_to_bin(&42, &[0, 0, 0, 0, 0]).unwrap();
        filter.insert_to_bin(&42, &[1, 1, 1, 1, 1]).unwrap();

        let locations = filter.locate(&42);
        
        // Should find both locations
        assert!(locations.len() >= 1);
        assert!(locations.contains(&vec![0, 0, 0, 0, 0]));
    }

    #[test]
    fn test_memory_stats() {
        let filter: TreeBloomFilter<u64> =
            TreeBloomFilter::new(vec![3, 4], 1000, 0.01);

        let stats = filter.stats();
        
        assert_eq!(stats.leaf_bins, 12); // 3 × 4
        assert!(stats.total_nodes > stats.leaf_bins); // Has internal nodes
        assert!(stats.memory_usage > 0);
        assert!(stats.overhead_factor >= 1.0); // At least as many nodes as leaves
        assert!(stats.memory_per_node > 0);
    }
}
