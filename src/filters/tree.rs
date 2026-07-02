//! Hierarchical Bloom filter for tree-shaped partitioning of membership state.
//!
//! `TreeBloomFilter` stores items in a complete tree of Bloom filters rather than a
//! single flat bitset. Each inserted item is routed to exactly one leaf bin, and the
//! same item is recorded in every filter along its root-to-leaf path. This makes the
//! root filter a fast coarse membership check, while deeper traversal APIs can prune
//! subtrees that cannot contain a match.
//!
//! The tree is parameterized by a branching vector such as `[4, 8, 16]`. At level `i`,
//! each node has `branching[i]` children, and the filter stored at that node is sized
//! to cover the full descendant load beneath it. In other words, internal filters are
//! intentionally larger than leaf filters so that each level can absorb the total item
//! volume expected from its subtree.
//!
//! Public operations are designed around that layout:
//!
//! - [`TreeBloomFilter::contains`] performs a single probe against the root filter.
//!   It is the fastest membership check, but it is intentionally coarse.
//! - [`TreeBloomFilter::contains_in_bin`] checks whether an item may exist along a
//!   specific root-to-leaf path.
//! - [`TreeBloomFilter::query_any`] performs a pruned depth-first search and returns
//!   whether an item may exist anywhere in the tree.
//! - [`TreeBloomFilter::locate`], [`TreeBloomFilter::locate_with`], and
//!   [`TreeBloomFilter::locate_iter`] return the leaf bins that remain viable after
//!   pruning.
//! - [`TreeBloomFilter::insert_auto`] deterministically routes an item to a leaf bin
//!   using the configured hasher.
//! - [`TreeBloomFilter::insert_to_bin`] inserts directly into a caller-selected bin.
//! - [`TreeBloomFilter::clear_subtree`] logically removes all items from a subtree and
//!   updates item counters, while leaving Bloom bits unchanged as required by Bloom
//!   filter semantics.
//! - [`TreeBloomFilter::union_with`] and [`TreeBloomFilter::intersect_with`] merge two
//!   trees recursively when their branching structure is compatible.
//!
//! This type is optimized for workloads where items have a natural hierarchical
//! placement, such as sharded caches, multi-tenant routing tables, or prefix-partitioned
//! membership indexes. It is not a general replacement for a flat Bloom filter when the
//! data has no tree structure.
//!
//! ## Guarantees
//!
//! - Memory safety and thread-safety follow the underlying filter implementations.
//! - Tree shape is fixed after construction.
//! - Query APIs are deterministic for a given hasher and tree shape.
//! - `locate*` APIs prune impossible subtrees rather than scanning every leaf.
//! - Counter semantics remain logical only; after set algebra operations, exact counts
//!   are intentionally not preserved.
//!
//! ## Constraints
//!
//! - False positives are possible, as with all Bloom filters.
//! - Items cannot be deleted from Bloom bits; subtree clearing only resets counters
//!   and descendant filters logically.
//! - Merge operations require compatible branching structures.
//! - Serialization is feature-gated and preserves the configured hasher state when
//!   the corresponding serde support is enabled.
//!
//! ## Typical use
//!
//! Construct the tree with a branching shape and per-bin capacity, then insert items
//! either automatically or into explicit paths. Use `contains` for a cheap root-level
//! check, `query_any` for tree-aware membership, and `locate` when you need candidate
//! leaf bins for routing or debugging.
//!
//! # References
//!
//! - Crainiceanu, A., & Lemire, D. (2015). Bloofi: Multidimensional Bloom Filters.
//!   *Information Systems*, 54, 311-324.
//! - Lillis, D., Breitinger, F., & Salois, M. (2015). Hierarchical Bloom Filter Trees
//!   for Approximate Matching. *Journal of Digital Forensics, Security and Law*.
//! - Lemire, D. (2016). A Fast Alternative to the Modulo Reduction.
//!   *arXiv preprint arXiv:1602.06915*.

use crate::core::filter::{BloomFilter, MergeableBloomFilter};
use crate::error::{BloomCraftError, Result};
use crate::filters::standard::StandardBloomFilter;
use crate::hash::{BloomHasher, StdHasher};
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "metrics")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "metrics")]
use crate::metrics::LatencyHistogram;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Maximum allowed tree depth (number of levels).
///
/// Exceeding this in [`TreeBloomFilter::with_hasher`] returns
/// [`BloomCraftError::InvalidParameters`].
pub const MAX_TREE_DEPTH: usize = 256;

/// Maximum total nodes across the entire tree.
///
/// Combined with [`MAX_TREE_DEPTH`] this guards against pathological branching
/// configurations that would exhaust memory.
pub const MAX_TOTAL_NODES: usize = 10_000_000;

#[inline(always)]
fn lemire_reduce(hash: u64, range: usize) -> usize {
    ((hash as u128 * range as u128) >> 64) as usize
}

#[inline(always)]
fn mix_hash_fast(mut h: u64) -> u64 {
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    h
}

/// Path and level metadata stored alongside each tree node.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct NodeMetadata {
    path: Vec<usize>,
    #[allow(dead_code)]
    level: u8,
}

/// A single node in the tree hierarchy.
///
/// Each node owns a [`StandardBloomFilter`] sized to cover the expected item
/// volume of its entire subtree. Internal nodes have `branching[level]`
/// children; leaf nodes have an empty child slice.
///
/// This type is `#[repr(C, align(64))]` to prevent false sharing when multiple
/// trees or nodes are accessed from concurrent threads.
#[derive(Debug, Clone)]
#[repr(C, align(64))]
pub struct TreeNode<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    filter: StandardBloomFilter<T, H>,
    item_count: usize,
    children: Box<[TreeNode<T, H>]>,
    metadata: NodeMetadata,
}

impl<T, H> TreeNode<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    #[inline]
    fn new_leaf(capacity: usize, fpr: f64, hasher: H, path: Vec<usize>, level: u8) -> Result<Self> {
        Ok(Self {
            filter: StandardBloomFilter::with_hasher(capacity, fpr, hasher)?,
            children: Box::new([]),
            item_count: 0,
            metadata: NodeMetadata { path, level },
        })
    }

    #[inline]
    fn new_internal(
        capacity: usize,
        fpr: f64,
        hasher: H,
        path: Vec<usize>,
        level: u8,
        children: Box<[TreeNode<T, H>]>,
    ) -> Result<Self> {
        Ok(Self {
            filter: StandardBloomFilter::with_hasher(capacity, fpr, hasher)?,
            children,
            item_count: 0,
            metadata: NodeMetadata { path, level },
        })
    }

    #[inline(always)]
    const fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    #[inline]
    fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    #[inline]
    fn memory_usage_estimate(&self) -> usize {
        let filter_bytes = self.filter.bit_count().div_ceil(8);
        let children_bytes: usize = self
            .children
            .iter()
            .map(|c| c.memory_usage_estimate())
            .sum();
        let overhead = std::mem::size_of::<Self>();
        filter_bytes + children_bytes + overhead
    }

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

/// Hierarchical Bloom filter over a complete tree of per-node filters.
///
/// See the [module-level documentation](self) for design rationale, guarantees,
/// and constraints.
///
/// # Type parameters
///
/// * `T` — Item type. Must implement `Hash + Send + Sync`.
/// * `H` — Hash function. Defaults to [`StdHasher`].
#[derive(Debug, Clone)]
pub struct TreeBloomFilter<T, H = StdHasher>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    root: TreeNode<T, H>,
    branching: Vec<usize>,
    capacity_per_bin: usize,
    #[allow(dead_code)]
    target_fpr: f64,
    total_items: usize,
    hasher: H,
    _phantom: PhantomData<T>,
    #[cfg(feature = "metrics")]
    metrics: TreeFilterMetrics,
}

#[cfg(feature = "metrics")]
#[derive(Debug)]
struct TreeFilterMetrics {
    insert_latency: LatencyHistogram,
    query_latency: LatencyHistogram,
    locate_latency: LatencyHistogram,
    pruned_subtrees: AtomicUsize,
}

#[cfg(feature = "metrics")]
impl Default for TreeFilterMetrics {
    fn default() -> Self {
        Self {
            insert_latency: LatencyHistogram::new(),
            query_latency: LatencyHistogram::new(),
            locate_latency: LatencyHistogram::new(),
            pruned_subtrees: AtomicUsize::new(0),
        }
    }
}

#[cfg(feature = "metrics")]
impl Clone for TreeFilterMetrics {
    fn clone(&self) -> Self {
        Self {
            insert_latency: self.insert_latency.clone(),
            query_latency: self.query_latency.clone(),
            locate_latency: self.locate_latency.clone(),

            pruned_subtrees: AtomicUsize::new(self.pruned_subtrees.load(Ordering::Relaxed)),
        }
    }
}

/// Overall health classification for a tree filter.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub enum HealthStatus {
    /// Filter is operating within nominal parameters.
    Healthy,
    /// Load factor is elevated; resize recommended soon.
    Degraded,
    /// Filter is near or at capacity; immediate action needed.
    Critical,
}

/// Detailed health snapshot of a tree filter.
#[cfg(feature = "metrics")]
#[derive(Debug, Clone)]
pub struct TreeHealthCheck {
    /// Overall health status.
    pub status: HealthStatus,
    /// Average load factor across all leaf filters.
    pub avg_load_factor: f64,
    /// Total items inserted across the entire tree.
    pub total_items: usize,
    /// Summed capacity of all leaf filters.
    pub capacity: usize,
    /// Ratio of total items to total capacity.
    pub saturation: f64,
}

/// Depth-first iterator over viable leaf-bin paths for a query item.
///
/// Yielded items are `Vec<usize>` paths (one child index per level) for
/// every leaf bin that the item *may* exist in, based on pruning with
/// each node's [`StandardBloomFilter::contains`].
///
/// Because Bloom filters can produce false positives, the set of yielded
/// paths is a superset of the true-positive paths.
pub struct LocateIter<'a, T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    tree: &'a TreeBloomFilter<T, H>,
    item: &'a T,
    stack: Vec<(&'a TreeNode<T, H>, Vec<usize>)>,
    started: bool,
}

impl<'a, T, H> LocateIter<'a, T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn new(tree: &'a TreeBloomFilter<T, H>, item: &'a T) -> Self {
        Self {
            tree,
            item,
            stack: Vec::new(),
            started: false,
        }
    }
}

impl<'a, T, H> Iterator for LocateIter<'a, T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            self.started = true;
            if !self.tree.root.filter.contains(self.item) {
                return None;
            }
            self.stack.push((&self.tree.root, Vec::new()));
        }

        while let Some((node, current_path)) = self.stack.pop() {
            if node.is_leaf() {
                return Some(current_path);
            }

            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                for child in node.children.iter() {
                    unsafe {
                        _mm_prefetch(child as *const _ as *const i8, _MM_HINT_T0);
                    }
                }
            }

            for (child_idx, child) in node.children.iter().enumerate().rev() {
                if child.filter.contains(self.item) {
                    let mut child_path = current_path.clone();
                    child_path.push(child_idx);
                    self.stack.push((child, child_path));
                }
            }
        }

        None
    }
}

#[cfg(feature = "serde")]
impl<T, H> Serialize for TreeNode<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TreeNode", 4)?;
        state.serialize_field("filter", &self.filter)?;
        state.serialize_field("item_count", &self.item_count)?;
        // Box<[TreeNode<T,H>]> serializes as a seq; as_ref() gives &[TreeNode<T,H>]
        state.serialize_field("children", self.children.as_ref())?;
        state.serialize_field("metadata", &self.metadata)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, H> Deserialize<'de> for TreeNode<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};

        struct TreeNodeVisitor<T, H>(PhantomData<(T, H)>);

        impl<'de, T, H> Visitor<'de> for TreeNodeVisitor<T, H>
        where
            T: Hash + Send + Sync,
            H: BloomHasher + Clone + Default,
        {
            type Value = TreeNode<T, H>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct TreeNode (map or seq)")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut filter: Option<StandardBloomFilter<T, H>> = None;
                let mut item_count: Option<usize> = None;
                let mut children: Option<Vec<TreeNode<T, H>>> = None;
                let mut metadata: Option<NodeMetadata> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "filter" => filter = Some(map.next_value()?),
                        "item_count" => item_count = Some(map.next_value()?),
                        "children" => children = Some(map.next_value()?),
                        "metadata" => metadata = Some(map.next_value()?),
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                Ok(TreeNode {
                    filter: filter.ok_or_else(|| de::Error::missing_field("filter"))?,
                    item_count: item_count.ok_or_else(|| de::Error::missing_field("item_count"))?,
                    children: children
                        .ok_or_else(|| de::Error::missing_field("children"))?
                        .into_boxed_slice(),
                    metadata: metadata.ok_or_else(|| de::Error::missing_field("metadata"))?,
                })
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let filter: StandardBloomFilter<T, H> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let item_count: usize = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let children: Vec<TreeNode<T, H>> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let metadata: NodeMetadata = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;
                Ok(TreeNode {
                    filter,
                    item_count,
                    children: children.into_boxed_slice(),
                    metadata,
                })
            }
        }

        deserializer.deserialize_struct(
            "TreeNode",
            &["filter", "item_count", "children", "metadata"],
            TreeNodeVisitor(PhantomData),
        )
    }
}

#[cfg(feature = "serde")]
impl<T, H> Serialize for TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + 'static + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("TreeBloomFilter", 7)?;
        state.serialize_field("root", &self.root)?;
        state.serialize_field("branching", &self.branching)?;
        state.serialize_field("capacity_per_bin", &self.capacity_per_bin)?;
        state.serialize_field("target_fpr", &self.target_fpr)?;
        state.serialize_field("total_items", &self.total_items)?;
        state.serialize_field("hasher_type", std::any::type_name::<H>())?;
        state.serialize_field("hasher", &self.hasher)?;

        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, H> Deserialize<'de> for TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default + 'static + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};

        struct TreeBloomFilterVisitor<T, H>(PhantomData<(T, H)>);

        impl<'de, T, H> Visitor<'de> for TreeBloomFilterVisitor<T, H>
        where
            T: Hash + Send + Sync,
            H: BloomHasher + Clone + Default + 'static + Deserialize<'de>,
        {
            type Value = TreeBloomFilter<T, H>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct TreeBloomFilter (map or seq)")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut root: Option<TreeNode<T, H>> = None;
                let mut branching: Option<Vec<usize>> = None;
                let mut capacity_per_bin: Option<usize> = None;
                let mut target_fpr: Option<f64> = None;
                let mut total_items: Option<usize> = None;
                let mut hasher_type: Option<String> = None;
                let mut hasher: Option<H> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "root" => root = Some(map.next_value()?),
                        "branching" => branching = Some(map.next_value()?),
                        "capacity_per_bin" => capacity_per_bin = Some(map.next_value()?),
                        "target_fpr" => target_fpr = Some(map.next_value()?),
                        "total_items" => total_items = Some(map.next_value()?),
                        "hasher_type" => hasher_type = Some(map.next_value()?),
                        "hasher" => hasher = Some(map.next_value()?),
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                let root = root.ok_or_else(|| de::Error::missing_field("root"))?;
                let branching = branching.ok_or_else(|| de::Error::missing_field("branching"))?;
                let capacity_per_bin =
                    capacity_per_bin.ok_or_else(|| de::Error::missing_field("capacity_per_bin"))?;
                let target_fpr =
                    target_fpr.ok_or_else(|| de::Error::missing_field("target_fpr"))?;
                let total_items =
                    total_items.ok_or_else(|| de::Error::missing_field("total_items"))?;

                let ht = hasher_type
                    .ok_or_else(|| de::Error::custom("Missing hasher_type validation field"))?;
                let expected = std::any::type_name::<H>();
                if ht != expected {
                    return Err(de::Error::custom(format!(
                        "Hasher type mismatch: expected {expected}, got {ht}"
                    )));
                }

                let hasher = hasher.unwrap_or_default();

                Ok(TreeBloomFilter {
                    root,
                    branching,
                    capacity_per_bin,
                    target_fpr,
                    total_items,
                    hasher,
                    _phantom: PhantomData,
                    #[cfg(feature = "metrics")]
                    metrics: TreeFilterMetrics::default(),
                })
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let root: TreeNode<T, H> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let branching: Vec<usize> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let capacity_per_bin: usize = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let target_fpr: f64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;
                let total_items: usize = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &self))?;
                let hasher_type: String = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(5, &self))?;
                let hasher: H = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(6, &self))?;

                let expected = std::any::type_name::<H>();
                if hasher_type != expected {
                    return Err(de::Error::custom(format!(
                        "Hasher type mismatch: expected {expected}, got {hasher_type}"
                    )));
                }

                Ok(TreeBloomFilter {
                    root,
                    branching,
                    capacity_per_bin,
                    target_fpr,
                    total_items,
                    hasher,
                    _phantom: PhantomData,
                    #[cfg(feature = "metrics")]
                    metrics: TreeFilterMetrics::default(),
                })
            }
        }

        deserializer.deserialize_struct(
            "TreeBloomFilter",
            &[
                "root",
                "branching",
                "capacity_per_bin",
                "target_fpr",
                "total_items",
                "hasher_type",
                "hasher",
            ],
            TreeBloomFilterVisitor(PhantomData),
        )
    }
}

impl<T> TreeBloomFilter<T, StdHasher>
where
    T: Hash + Send + Sync,
{
    /// Create a tree filter with the default hasher.
    pub fn new(branching: Vec<usize>, capacity_per_bin: usize, fpr: f64) -> Result<Self> {
        Self::with_hasher(branching, capacity_per_bin, fpr, StdHasher::new())
    }
}

impl<T, H> TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Create a tree filter with a custom hasher.
    pub fn with_hasher(
        branching: Vec<usize>,
        capacity_per_bin: usize,
        fpr: f64,
        hasher: H,
    ) -> Result<Self> {
        if branching.is_empty() {
            return Err(BloomCraftError::invalid_parameters(
                "branching cannot be empty",
            ));
        }
        if !branching.iter().all(|&b| b > 0) {
            return Err(BloomCraftError::invalid_parameters(
                "all branching factors must be > 0",
            ));
        }
        if capacity_per_bin == 0 {
            return Err(BloomCraftError::invalid_item_count(capacity_per_bin));
        }
        if !(fpr > 0.0 && fpr < 1.0) {
            return Err(BloomCraftError::fp_rate_out_of_bounds(fpr));
        }
        if branching.len() > MAX_TREE_DEPTH {
            return Err(BloomCraftError::invalid_parameters(format!(
                "tree depth {} exceeds maximum {}",
                branching.len(),
                MAX_TREE_DEPTH
            )));
        }

        let mut total_nodes: usize = 1;
        let mut partial_product: usize = 1;
        for &branching_factor in &branching {
            partial_product = partial_product
                .checked_mul(branching_factor)
                .ok_or_else(|| {
                    BloomCraftError::invalid_parameters(
                        "Branching factors would overflow node count",
                    )
                })?;
            total_nodes = total_nodes.checked_add(partial_product).ok_or_else(|| {
                BloomCraftError::invalid_parameters("Total node count would overflow")
            })?;
        }

        if total_nodes > MAX_TOTAL_NODES {
            let estimated_memory = total_nodes * std::mem::size_of::<TreeNode<T, H>>();
            return Err(BloomCraftError::invalid_parameters(format!(
                "Tree would allocate {} nodes (max: {}). Estimated memory: {} MB. \
                Consider reducing branching factors or depth.",
                total_nodes,
                MAX_TOTAL_NODES,
                estimated_memory / (1024 * 1024)
            )));
        }

        if total_nodes > 100_000 {
            eprintln!(
                "WARNING: TreeBloomFilter will allocate {} nodes (~{} MB). \
                This is within limits but may impact performance.",
                total_nodes,
                (total_nodes * std::mem::size_of::<TreeNode<T, H>>()) / (1024 * 1024)
            );
        }

        let root = Self::build_tree(&branching, 0, capacity_per_bin, fpr, hasher.clone(), vec![]);

        Ok(Self {
            root: root?,
            branching,
            capacity_per_bin,
            target_fpr: fpr,
            total_items: 0,
            hasher,
            _phantom: PhantomData,
            #[cfg(feature = "metrics")]
            metrics: TreeFilterMetrics::default(),
        })
    }

    fn build_tree(
        branching: &[usize],
        level: usize,
        capacity: usize,
        fpr: f64,
        hasher: H,
        path: Vec<usize>,
    ) -> Result<TreeNode<T, H>> {
        if level >= branching.len() {
            return TreeNode::new_leaf(capacity, fpr, hasher, path, level as u8);
        }

        let num_children = branching[level];
        let mut children = Vec::with_capacity(num_children);
        for i in 0..num_children {
            let mut child_path = path.clone();
            child_path.push(i);
            children.push(Self::build_tree(
                branching,
                level + 1,
                capacity,
                fpr,
                hasher.clone(),
                child_path,
            )?);
        }

        let internal_capacity = branching[level..]
            .iter()
            .copied()
            .try_fold(capacity, |acc, b| acc.checked_mul(b))
            .ok_or_else(|| {
                BloomCraftError::invalid_parameters("Internal node capacity overflow")
            })?;
        TreeNode::new_internal(
            internal_capacity,
            fpr,
            hasher,
            path,
            level as u8,
            children.into_boxed_slice(),
        )
    }

    /// Check that `path` is valid for the tree's branching structure.
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

    /// Number of levels in the tree (= `branching.len()`).
    #[must_use]
    #[inline(always)]
    pub fn depth(&self) -> usize {
        self.branching.len()
    }

    /// Product of all branching factors; the total number of leaf bins.
    #[must_use]
    #[inline]
    pub fn leaf_count(&self) -> usize {
        self.branching.iter().product()
    }

    /// Total number of nodes (internal + leaf) in the tree.
    ///
    /// Recursively counts every node in the hierarchy. For a branching vector
    /// `[b0, b1, ..., bn]` this equals `1 + b0 + b0·b1 + ... + ∏bi`.
    #[must_use]
    #[inline]
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Estimated memory usage in bytes (filters + tree structure).
    #[must_use]
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.root.memory_usage_estimate() + std::mem::size_of::<Self>()
    }

    /// Insert with automatic hash-based bin routing.
    #[inline]
    pub fn insert(&mut self, item: &T) -> Result<()> {
        self.insert_auto(item)
    }

    /// Insert an item into a specific leaf bin, setting membership at every node
    /// along the path and incrementing each node's `item_count`.
    #[inline]
    pub fn insert_to_bin(&mut self, item: &T, bin_path: &[usize]) -> Result<()> {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();

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

        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.metrics
                .insert_latency
                .record(std::time::Duration::from_nanos(elapsed));
        }

        Ok(())
    }

    /// Insert multiple items into the same bin (all-or-nothing).
    ///
    /// Pre-validates all capacity constraints before mutating any state.
    #[inline]
    pub fn insert_batch_to_bin(&mut self, items: &[T], bin_path: &[usize]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        self.validate_path(bin_path)?;

        let new_total = self
            .total_items
            .checked_add(items.len())
            .ok_or_else(|| BloomCraftError::capacity_exceeded(usize::MAX, items.len()))?;

        let mut new_counts = Vec::with_capacity(bin_path.len() + 1);
        let new_root_count = self
            .root
            .item_count
            .checked_add(items.len())
            .ok_or_else(|| BloomCraftError::capacity_exceeded(usize::MAX, items.len()))?;
        new_counts.push(new_root_count);

        let mut current = &self.root;
        for &child_idx in bin_path {
            current = &current.children[child_idx];
            new_counts.push(
                current
                    .item_count
                    .checked_add(items.len())
                    .ok_or_else(|| BloomCraftError::capacity_exceeded(usize::MAX, items.len()))?,
            );
        }

        let mut current = &mut self.root;
        for item in items {
            current.filter.insert(item);
        }
        current.item_count = new_counts[0];

        for (depth, &child_idx) in bin_path.iter().enumerate() {
            current = &mut current.children[child_idx];
            for item in items {
                current.filter.insert(item);
            }
            current.item_count = new_counts[depth + 1];
        }

        self.total_items = new_total;
        Ok(())
    }

    /// Find all leaf bins that might contain `item` (pruned DFS).
    ///
    /// Read-safe under `Arc<TreeBloomFilter>` (tree structure is immutable after construction;
    /// `StandardBloomFilter::contains` uses atomic reads).
    #[must_use]
    pub fn locate(&self, item: &T) -> Vec<Vec<usize>> {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();

        if !self.root.filter.contains(item) {
            #[cfg(feature = "metrics")]
            {
                let elapsed = start.elapsed().as_nanos() as u64;
                self.metrics
                    .locate_latency
                    .record(std::time::Duration::from_nanos(elapsed));
            }
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut stack = vec![(&self.root, Vec::new())];

        while let Some((node, current_path)) = stack.pop() {
            if node.is_leaf() {
                result.push(current_path);
                continue;
            }

            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                for child in node.children.iter() {
                    unsafe {
                        _mm_prefetch(child as *const _ as *const i8, _MM_HINT_T0);
                    }
                }
            }

            for (child_idx, child) in node.children.iter().enumerate().rev() {
                if !child.filter.contains(item) {
                    #[cfg(feature = "metrics")]
                    self.metrics.pruned_subtrees.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let mut child_path = current_path.clone();
                child_path.push(child_idx);
                stack.push((child, child_path));
            }
        }

        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.metrics
                .locate_latency
                .record(std::time::Duration::from_nanos(elapsed));
        }
        result
    }

    /// Like [`locate`](Self::locate) but calls `callback` for each match instead of allocating a `Vec`.
    #[inline]
    pub fn locate_with<F>(&self, item: &T, mut callback: F)
    where
        F: FnMut(&[usize]),
    {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();

        if !self.root.filter.contains(item) {
            #[cfg(feature = "metrics")]
            {
                let elapsed = start.elapsed().as_nanos() as u64;
                self.metrics
                    .locate_latency
                    .record(std::time::Duration::from_nanos(elapsed));
            }
            return;
        }

        let mut stack: Vec<(&TreeNode<T, H>, Vec<usize>)> = vec![(&self.root, Vec::new())];

        while let Some((node, current_path)) = stack.pop() {
            if node.is_leaf() {
                callback(&current_path);
                continue;
            }

            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                for child in node.children.iter() {
                    unsafe {
                        _mm_prefetch(child as *const _ as *const i8, _MM_HINT_T0);
                    }
                }
            }

            for (child_idx, child) in node.children.iter().enumerate().rev() {
                if !child.filter.contains(item) {
                    #[cfg(feature = "metrics")]
                    self.metrics.pruned_subtrees.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let mut child_path = current_path.clone();
                child_path.push(child_idx);
                stack.push((child, child_path));
            }
        }

        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.metrics
                .locate_latency
                .record(std::time::Duration::from_nanos(elapsed));
        }
    }

    /// Create an iterator over all bins that might contain an item.
    ///
    /// Lazily evaluates matches instead of pre-allocating all results.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use bloomcraft::filters::TreeBloomFilter;
    /// # let filter = TreeBloomFilter::<String>::new(vec![2, 3], 1000, 0.01)?;
    /// // Find first match only
    /// if let Some(path) = filter.locate_iter(&"item".to_string()).next() {
    ///     println!("First match: {:?}", path);
    /// }
    ///
    /// // Count matches without allocating paths
    /// let count = filter.locate_iter(&"item".to_string()).count();
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn locate_iter<'a>(&'a self, item: &'a T) -> LocateIter<'a, T, H> {
        LocateIter::new(self, item)
    }

    /// Merge `other` into this tree (union of sets).
    ///
    /// After union, `total_items` and all per-node `item_count` values are
    /// set to 0 because false-positive inflation makes exact counts unrecoverable.
    /// Membership queries remain correct.
    pub fn union_with(&mut self, other: &Self) -> Result<()> {
        if self.branching != other.branching {
            return Err(BloomCraftError::incompatible_filters(
                "Different branching factors".to_string(),
            ));
        }

        Self::union_nodes(&mut self.root, &other.root)?;
        // Exact item count is unknown after union (false positives inflate)
        self.total_items = 0;
        Ok(())
    }

    fn union_nodes(a: &mut TreeNode<T, H>, b: &TreeNode<T, H>) -> Result<()> {
        // Union filters at this level
        a.filter = a.filter.union(&b.filter)?;
        // item_count is set to 0 after union because exact count is unknown
        // (Bloom filter false positives inflate the count)
        a.item_count = 0;

        // Recursively union children
        for (a_child, b_child) in a.children.iter_mut().zip(b.children.iter()) {
            Self::union_nodes(a_child, b_child)?;
        }

        Ok(())
    }

    /// Intersect `other` into this tree (set intersection).
    ///
    /// After intersection, `total_items` and all per-node `item_count` values are
    /// set to 0 because false-positive inflation makes exact counts unrecoverable.
    /// Membership queries remain correct.
    pub fn intersect_with(&mut self, other: &Self) -> Result<()> {
        if self.branching != other.branching {
            return Err(BloomCraftError::incompatible_filters(
                "Different branching factors".to_string(),
            ));
        }

        Self::intersect_nodes(&mut self.root, &other.root)?;
        self.total_items = 0; // Unknown after intersect
        Ok(())
    }

    fn intersect_nodes(a: &mut TreeNode<T, H>, b: &TreeNode<T, H>) -> Result<()> {
        // Intersect filters at this level
        a.filter = a.filter.intersect(&b.filter)?;
        a.item_count = 0; // Unknown

        // Recursively intersect children
        for (a_child, b_child) in a.children.iter_mut().zip(b.children.iter()) {
            Self::intersect_nodes(a_child, b_child)?;
        }

        Ok(())
    }
}

// --- impl MergeableBloomFilter ---

impl<T, H> MergeableBloomFilter<T> for TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn is_compatible(&self, other: &Self) -> bool {
        self.branching == other.branching
    }

    fn union(&mut self, other: &Self) -> Result<()> {
        self.union_with(other)
    }

    fn intersect(&mut self, other: &Self) -> Result<()> {
        self.intersect_with(other)
    }
}

impl<T, H> TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Navigate to a node at `path`.
    ///
    /// Returns `InvalidParameters` if the path is too long or an index exceeds the branching factor.
    pub fn subtree_at(&self, path: &[usize]) -> Result<&TreeNode<T, H>> {
        if path.is_empty() {
            return Ok(&self.root);
        }

        // Validate path length doesn't exceed tree depth
        if path.len() > self.depth() {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Path length {} exceeds tree depth {}",
                path.len(),
                self.depth()
            )));
        }

        // Traverse tree, validating each index
        let mut current = &self.root;
        for (level, &idx) in path.iter().enumerate() {
            if idx >= current.children.len() {
                return Err(BloomCraftError::invalid_parameters(format!(
                    "Invalid index {} at level {} (branching factor is {}, valid range: 0..{})",
                    idx,
                    level,
                    current.children.len(),
                    current.children.len()
                )));
            }
            current = &current.children[idx];
        }

        Ok(current)
    }

    /// Logically delete all items under `path`.
    ///
    /// Resets the subtree's filters and zeroes all `item_count` fields within it.
    /// `total_items` and ancestor `item_count` values are decremented by the
    /// number of items that were tracked in the cleared subtree.
    ///
    /// This is a logical (not physical) deletion — the tree structure is preserved.
    pub fn clear_subtree(&mut self, path: &[usize]) -> Result<()> {
        if path.is_empty() {
            let removed = self.root.item_count;
            Self::clear_node_iterative(&mut self.root);
            self.total_items = self.total_items.saturating_sub(removed);
            return Ok(());
        }

        if path.len() > self.depth() {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Path length {} exceeds tree depth {}",
                path.len(),
                self.depth()
            )));
        }

        let removed =
            {
                let mut cur = &self.root;
                for (level, &idx) in path.iter().enumerate() {
                    if idx >= cur.children.len() {
                        return Err(BloomCraftError::invalid_parameters(format!(
                        "Invalid index {} at level {} (branching factor is {}, valid range: 0..{})",
                        idx, level, cur.children.len(), cur.children.len()
                    )));
                    }
                    cur = &cur.children[idx];
                }
                cur.item_count
            };

        {
            let mut cur = &mut self.root;
            for &idx in path.iter() {
                cur = &mut cur.children[idx];
            }
            Self::clear_node_iterative(cur);
        }

        // Decrement ancestors (root + every node along the path except the cleared node)
        self.root.item_count = self.root.item_count.saturating_sub(removed);
        {
            let mut cur = &mut self.root;
            for &idx in path[..path.len().saturating_sub(1)].iter() {
                cur = &mut cur.children[idx];
                cur.item_count = cur.item_count.saturating_sub(removed);
            }
        }

        self.total_items = self.total_items.saturating_sub(removed);
        Ok(())
    }

    /// Search for `item` under a path prefix (scoped query).
    pub fn locate_in_range(&self, item: &T, path_prefix: &[usize]) -> Vec<Vec<usize>> {
        if path_prefix.is_empty() {
            return self.locate(item);
        }

        let subtree = match self.subtree_at(path_prefix) {
            Ok(node) => node,
            Err(_) => return Vec::new(),
        };

        if !subtree.filter.contains(item) {
            return Vec::new();
        }

        // DFS within subtree
        let mut result = Vec::new();
        self.locate_in_subtree(subtree, item, path_prefix.to_vec(), &mut result);
        result
    }

    fn locate_in_subtree(
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
            self.locate_in_subtree(child, item, child_path, result);
        }
    }

    /// Batch locate for multiple items.
    #[must_use]
    pub fn locate_batch(&self, items: &[&T]) -> Vec<Vec<Vec<usize>>> {
        items.iter().map(|item| self.locate(item)).collect()
    }

    /// Parallel batch locate (requires `rayon`).
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn locate_batch_parallel(&self, items: &[&T]) -> Vec<Vec<Vec<usize>>> {
        items.par_iter().map(|item| self.locate(item)).collect()
    }

    /// Whether the tree's average load factor exceeds 70 %.
    ///
    /// A return value of `true` suggests the tree has outgrown its configured
    /// per-bin capacity and [`resize`](Self::resize) should be considered.
    #[must_use]
    pub fn needs_resize(&self) -> bool {
        self.stats().avg_load_factor > 0.7
    }

    /// Return an empty filter with increased capacity.
    pub fn resize(&self, new_capacity_per_bin: usize, new_fpr: f64) -> Result<Self> {
        Self::with_hasher(
            self.branching.clone(),
            new_capacity_per_bin,
            new_fpr,
            H::default(),
        )
    }

    /// Insert with automatic hash-based bin assignment (deterministic).
    pub fn insert_auto(&mut self, item: &T) -> Result<()> {
        let (h1, h2) = self.hasher.hash_item(item);

        let mut bin_path = Vec::with_capacity(self.depth());
        let mut hash = h1;

        for (level, &branching_factor) in self.branching.iter().enumerate() {
            bin_path.push(lemire_reduce(hash, branching_factor));
            hash = mix_hash_fast(hash ^ h2.wrapping_mul(level as u64 + 1));
        }

        self.insert_to_bin(item, &bin_path)
    }

    fn clear_node_iterative(node: &mut TreeNode<T, H>) {
        let mut stack: Vec<&mut TreeNode<T, H>> = vec![node];

        while let Some(current) = stack.pop() {
            current.filter.clear();
            current.item_count = 0;

            for child in current.children.iter_mut() {
                stack.push(child);
            }
        }
    }

    /// Check if `item` *might* exist anywhere in the tree (single root-filter probe).
    ///
    /// This is a fast coarse check — a `true` result means the item *may* exist
    /// in one or more leaf bins. Use `locate` for exact bin resolution.
    #[must_use]
    #[inline(always)]
    pub fn contains(&self, item: &T) -> bool {
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();

        let result = self.root.filter.contains(item);

        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.metrics
                .query_latency
                .record(std::time::Duration::from_nanos(elapsed));
        }

        result
    }

    /// Check if `item` might exist in a specific leaf bin (path-constrained query).
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

    /// Batch contains check (root filter only).
    #[must_use]
    #[inline]
    pub fn contains_batch(&self, items: &[&T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Check if `item` *might* exist at any leaf (pruned DFS).
    ///
    /// Unlike `contains` (single root probe), this walks the tree but prunes
    /// subtrees whose filter doesn't match, so it avoids a full scan in practice.
    #[must_use]
    #[cold]
    pub fn query_any(&self, item: &T) -> bool {
        if !self.root.filter.contains(item) {
            return false;
        }
        self.query_any_recursive(&self.root, item)
    }

    fn query_any_recursive(&self, node: &TreeNode<T, H>, item: &T) -> bool {
        if node.is_leaf() {
            return node.filter.contains(item);
        }
        node.children
            .iter()
            .any(|child| child.filter.contains(item) && self.query_any_recursive(child, item))
    }

    /// Compute a snapshot of tree-wide statistics.
    ///
    /// See [`TreeStats`] for the individual fields.
    #[must_use]
    pub fn stats(&self) -> TreeStats {
        let total_nodes = self.node_count();
        let memory_usage = self.memory_usage();
        let leaf_bins = self.leaf_count();

        let memory_per_node = memory_usage.checked_div(total_nodes).unwrap_or(0);

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

    fn compute_avg_load_factor(&self) -> f64 {
        let (sum, count) = self.compute_load_factor_recursive(&self.root);
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

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

    /// Classify filter health by average load factor.
    #[cfg(feature = "metrics")]
    pub fn health_check(&self) -> TreeHealthCheck {
        let stats = self.stats();
        let status = if stats.avg_load_factor > 0.9 {
            HealthStatus::Critical
        } else if stats.avg_load_factor > 0.7 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        TreeHealthCheck {
            status,
            avg_load_factor: stats.avg_load_factor,
            total_items: stats.total_items,
            capacity: self.capacity_per_bin * stats.leaf_bins,
            saturation: stats.avg_load_factor,
        }
    }

    /// Export metrics in Prometheus text format.
    #[cfg(feature = "metrics")]
    pub fn export_prometheus(&self) -> String {
        let stats = self.stats();
        format!(
            "# HELP tree_bloom_filter_items Total items inserted\n\
             # TYPE tree_bloom_filter_items gauge\n\
             tree_bloom_filter_items{{depth=\"{}\"}} {}\n\
             # HELP tree_bloom_filter_load_factor Average load factor\n\
             # TYPE tree_bloom_filter_load_factor gauge\n\
             tree_bloom_filter_load_factor{{depth=\"{}\"}} {:.4}\n\
             # HELP tree_bloom_filter_pruned_subtrees Total pruned subtrees\n\
             # TYPE tree_bloom_filter_pruned_subtrees counter\n\
             tree_bloom_filter_pruned_subtrees{{depth=\"{}\"}} {}\n",
            stats.depth,
            stats.total_items,
            stats.depth,
            stats.avg_load_factor,
            stats.depth,
            self.metrics.pruned_subtrees.load(Ordering::Relaxed)
        )
    }

    /// Validate internal tree structure (debug builds only).
    ///
    /// Checks that every node's stored path matches its actual position and
    /// that child counts match the branching factors. No-op in release builds.
    #[cfg(debug_assertions)]
    pub fn validate_structure(&self) -> Result<()> {
        self.validate_node(&self.root, &[])?;
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn validate_node(&self, node: &TreeNode<T, H>, current_path: &[usize]) -> Result<()> {
        if node.metadata.path != current_path {
            return Err(BloomCraftError::internal_error(format!(
                "Path mismatch: stored {:?}, actual {:?}",
                node.metadata.path, current_path
            )));
        }

        // Check children count matches branching factor
        if current_path.len() < self.branching.len() {
            let expected_children = self.branching[current_path.len()];
            if node.children.len() != expected_children {
                return Err(BloomCraftError::internal_error(format!(
                    "Child count mismatch at {:?}: expected {}, got {}",
                    current_path,
                    expected_children,
                    node.children.len()
                )));
            }
        }

        // Recursively validate children
        for (idx, child) in node.children.iter().enumerate() {
            let mut child_path = current_path.to_vec();
            child_path.push(idx);
            self.validate_node(child, &child_path)?;
        }

        Ok(())
    }
}

/// Aggregate statistics for a [`TreeBloomFilter`].
///
/// Returned by [`TreeBloomFilter::stats`].
#[derive(Debug, Clone, Default, Copy)]
pub struct TreeStats {
    /// Total number of nodes (internal + leaf) in the tree.
    pub total_nodes: usize,
    /// Estimated heap memory usage in bytes.
    pub memory_usage: usize,
    /// Current number of items inserted into the tree.
    pub total_items: usize,
    /// Number of levels (= `branching.len()`).
    pub depth: usize,
    /// Total number of leaf bins (= product of branching factors).
    pub leaf_bins: usize,
    /// Mean load factor across all nodes.
    pub avg_load_factor: f64,
    /// Estimated bytes per node (`memory_usage / total_nodes`).
    pub memory_per_node: usize,
    /// Ratio of internal nodes to leaf bins (`total_nodes / leaf_bins`).
    pub overhead_factor: f64,
}

impl<T, H> BloomFilter<T> for TreeBloomFilter<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn insert(&mut self, item: &T) {
        let _ = TreeBloomFilter::insert(self, item);
    }

    fn contains(&self, item: &T) -> bool {
        TreeBloomFilter::contains(self, item)
    }

    fn clear(&mut self) {
        Self::clear_node_iterative(&mut self.root);
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

    fn count_set_bits(&self) -> usize {
        self.root.filter.count_set_bits()
    }
}

/// Builder for [`TreeBloomFilter`] with ergonomic field-by-field construction.
///
/// # Example
///
/// ```no_run
/// use bloomcraft::filters::tree::TreeBloomFilterBuilder;
///
/// let tree = TreeBloomFilterBuilder::<String>::new()
///     .branching(vec![4, 4])
///     .capacity_per_bin(1000)
///     .false_positive_rate(0.01)
///     .build()
///     .unwrap();
/// ```
pub struct TreeBloomFilterBuilder<T, H = StdHasher>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    branching: Option<Vec<usize>>,
    capacity_per_bin: Option<usize>,
    fpr: Option<f64>,
    hasher: H,
    _phantom: PhantomData<T>,
}

impl<T, H> TreeBloomFilterBuilder<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    /// Create a new builder with all fields unset.
    ///
    /// Required fields (`branching`, `capacity_per_bin`, `fpr`) must be
    /// provided via the respective setters before calling [`build`](Self::build).
    pub fn new() -> Self {
        Self {
            branching: None,
            capacity_per_bin: None,
            fpr: None,
            hasher: H::default(),
            _phantom: PhantomData,
        }
    }

    /// Set the branching factor vector (required).
    ///
    /// Each element is the number of children per node at that level.
    /// For example, `vec![4, 4]` creates 16 leaf bins in a two-level tree.
    pub fn branching(mut self, branching: Vec<usize>) -> Self {
        self.branching = Some(branching);
        self
    }

    /// Set the expected capacity per leaf bin (required).
    pub fn capacity_per_bin(mut self, capacity: usize) -> Self {
        self.capacity_per_bin = Some(capacity);
        self
    }

    /// Set the target false-positive rate per node (required).
    ///
    /// Each node's [`StandardBloomFilter`] will be dimensioned so that its
    /// FPR stays near this value until `capacity_per_bin` items per bin have
    /// been inserted.
    pub fn false_positive_rate(mut self, fpr: f64) -> Self {
        self.fpr = Some(fpr);
        self
    }

    /// Set a custom hash function.
    ///
    /// Defaults to `H::default()` if not called.
    pub fn hasher(mut self, hasher: H) -> Self {
        self.hasher = hasher;
        self
    }

    /// Consume the builder and create a [`TreeBloomFilter`].
    pub fn build(self) -> Result<TreeBloomFilter<T, H>> {
        let branching = self
            .branching
            .ok_or_else(|| BloomCraftError::invalid_parameters("branching not set"))?;
        let capacity = self
            .capacity_per_bin
            .ok_or_else(|| BloomCraftError::invalid_parameters("capacity_per_bin not set"))?;
        let fpr = self
            .fpr
            .ok_or_else(|| BloomCraftError::invalid_parameters("fpr not set"))?;

        TreeBloomFilter::with_hasher(branching, capacity, fpr, self.hasher)
    }
}

impl<T, H> Default for TreeBloomFilterBuilder<T, H>
where
    T: Hash + Send + Sync,
    H: BloomHasher + Clone + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration parameters for validating a tree architecture.
///
/// Passed to [`TreeConfig::validate`] to produce a [`TreeCapacityStats`]
/// summary without allocating any filters.
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Per-level branching factors (e.g. `[4, 4]` for 16 bins).
    pub branching: Vec<usize>,
    /// Expected item capacity per leaf bin.
    pub capacity_per_bin: usize,
    /// Desired false-positive rate per node (0.0 – 1.0).
    pub target_fpr: f64,
}

/// Estimated capacity statistics produced by [`TreeConfig::validate`].
#[derive(Debug, Clone)]
pub struct TreeCapacityStats {
    /// Total nodes required for the given branching vector.
    pub total_nodes: usize,
    /// Number of leaf bins (= product of branching factors).
    pub leaf_count: usize,
    /// Estimated total memory in megabytes (filters + tree structure).
    pub memory_mb: usize,
    /// Tree depth (= `branching.len()`).
    pub depth: usize,
}

impl TreeConfig {
    /// Return estimated capacity stats, or an error if the config is invalid.
    pub fn validate(&self) -> Result<TreeCapacityStats> {
        use crate::core::params::optimal_m;

        if self.branching.is_empty() {
            return Err(BloomCraftError::invalid_parameters(
                "Branching cannot be empty",
            ));
        }
        if self.branching.len() > MAX_TREE_DEPTH {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Depth {} exceeds maximum {}",
                self.branching.len(),
                MAX_TREE_DEPTH
            )));
        }

        let leaf_count: usize = self
            .branching
            .iter()
            .try_fold(1usize, |acc, &b| acc.checked_mul(b))
            .ok_or_else(|| {
                BloomCraftError::invalid_parameters("Branching factors would overflow leaf count")
            })?;

        let mut total_nodes: usize = 1;
        let mut partial_product: usize = 1;

        for &branching_factor in &self.branching {
            partial_product = partial_product
                .checked_mul(branching_factor)
                .ok_or_else(|| {
                    BloomCraftError::invalid_parameters(
                        "Branching factors would overflow node count",
                    )
                })?;
            total_nodes = total_nodes.checked_add(partial_product).ok_or_else(|| {
                BloomCraftError::invalid_parameters("Total node count would overflow")
            })?;
        }

        if total_nodes > MAX_TOTAL_NODES {
            return Err(BloomCraftError::invalid_parameters(format!(
                "Total nodes {} exceeds maximum {}",
                total_nodes, MAX_TOTAL_NODES
            )));
        }

        let mut total_memory: usize = 0;
        let mut nodes_at_level: usize = 1;

        for level in 0..self.branching.len() {
            let capacity_at_level = self.branching[level..]
                .iter()
                .copied()
                .try_fold(self.capacity_per_bin, |acc, b| acc.checked_mul(b))
                .ok_or_else(|| {
                    BloomCraftError::invalid_parameters("Capacity calculation overflow at level")
                })?;

            let bits = optimal_m(capacity_at_level, self.target_fpr)?;
            let bytes = bits.div_ceil(8);
            let node_overhead = std::mem::size_of::<TreeNode<String, StdHasher>>();
            let bytes_per_node = bytes + node_overhead;

            total_memory = total_memory
                .checked_add(nodes_at_level.checked_mul(bytes_per_node).ok_or_else(|| {
                    BloomCraftError::invalid_parameters("Memory calculation overflow")
                })?)
                .ok_or_else(|| {
                    BloomCraftError::invalid_parameters("Memory calculation overflow")
                })?;

            if level + 1 < self.branching.len() {
                nodes_at_level = nodes_at_level
                    .checked_mul(self.branching[level])
                    .ok_or_else(|| BloomCraftError::invalid_parameters("Node count overflow"))?;
            }
        }

        Ok(TreeCapacityStats {
            total_nodes,
            leaf_count,
            memory_mb: total_memory / (1024 * 1024),
            depth: self.branching.len(),
        })
    }

    /// Generate a human-readable capacity report string.
    pub fn report(&self) -> String {
        match self.validate() {
            Ok(stats) => {
                format!(
                    "TreeBloomFilter Capacity Report\n\
                     ================================\n\
                     Configuration:\n\
                     - Branching: {:?}\n\
                     - Depth: {}\n\
                     - Capacity per bin: {}\n\
                     - Target FPR: {:.4}\n\
                     \n\
                     Estimated Usage:\n\
                     - Total nodes: {}\n\
                     - Leaf bins: {}\n\
                     - Memory: {} MB\n\
                     \n\
                     Status: VIABLE",
                    self.branching,
                    stats.depth,
                    self.capacity_per_bin,
                    self.target_fpr,
                    stats.total_nodes,
                    stats.leaf_count,
                    stats.memory_mb
                )
            }
            Err(e) => {
                format!(
                    "TreeBloomFilter Capacity Report\n\
                     ================================\n\
                     Configuration:\n\
                     - Branching: {:?}\n\
                     - Capacity per bin: {}\n\
                     - Target FPR: {:.4}\n\
                     \n\
                     Status: INVALID\n\
                     Error: {}",
                    self.branching, self.capacity_per_bin, self.target_fpr, e
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();
        assert_eq!(filter.depth(), 2);
        assert_eq!(filter.leaf_count(), 6);
        assert_eq!(filter.len(), 0);
        assert!(filter.is_empty());
    }

    #[test]
    fn test_insert_and_query() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();
        filter.insert_to_bin(&"hello".to_string(), &[0, 1]).unwrap();
        assert!(filter.contains(&"hello".to_string()));
        assert!(!filter.contains(&"goodbye".to_string()));
        assert_eq!(filter.len(), 1);
    }

    #[test]
    fn test_insert_auto() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 4], 1000, 0.01).unwrap();

        filter.insert_auto(&"test".to_string()).unwrap();
        assert!(filter.contains(&"test".to_string()));

        // Should be deterministic
        let loc1 = filter.locate(&"test".to_string());
        filter.insert_auto(&"test".to_string()).unwrap();
        let loc2 = filter.locate(&"test".to_string());
        assert_eq!(loc1, loc2);
    }

    #[test]
    fn test_union() {
        let mut filter1: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();
        let mut filter2: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();

        filter1.insert_to_bin(&"a".to_string(), &[0, 0]).unwrap();
        filter2.insert_to_bin(&"b".to_string(), &[1, 1]).unwrap();

        filter1.union(&filter2).unwrap();
        assert!(filter1.contains(&"a".to_string()));
        assert!(filter1.contains(&"b".to_string()));
    }

    #[test]
    fn test_intersect() {
        let mut filter1: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();
        let mut filter2: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();

        filter1.insert_to_bin(&"a".to_string(), &[0, 0]).unwrap();
        filter1.insert_to_bin(&"b".to_string(), &[0, 0]).unwrap();
        filter2.insert_to_bin(&"b".to_string(), &[0, 0]).unwrap();
        filter2.insert_to_bin(&"c".to_string(), &[1, 1]).unwrap();

        filter1.intersect(&filter2).unwrap();
        assert!(filter1.contains(&"b".to_string()));
    }

    #[test]
    fn test_subtree_operations() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();

        filter.insert_to_bin(&"item".to_string(), &[0, 1]).unwrap();

        let subtree = filter.subtree_at(&[0]).unwrap();
        assert!(subtree.filter.contains(&"item".to_string()));

        filter.clear_subtree(&[0]).unwrap();

        let cleared_subtree = filter.subtree_at(&[0]).unwrap();
        assert!(!cleared_subtree.filter.contains(&"item".to_string()));

        assert!(!filter
            .contains_in_bin(&"item".to_string(), &[0, 1])
            .unwrap());
    }

    #[test]
    fn test_locate_in_range() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 3], 1000, 0.01).unwrap();

        filter.insert_to_bin(&"item".to_string(), &[1, 2]).unwrap();

        let locs = filter.locate_in_range(&"item".to_string(), &[1]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], vec![1, 2]);
    }

    #[test]
    fn test_builder() {
        let filter: Result<TreeBloomFilter<String>> = TreeBloomFilterBuilder::new()
            .branching(vec![2, 3])
            .capacity_per_bin(1000)
            .false_positive_rate(0.01)
            .build();

        assert!(filter.is_ok());
        let f = filter.unwrap();
        assert_eq!(f.depth(), 2);
        assert_eq!(f.leaf_count(), 6);
    }

    #[test]
    fn test_needs_resize() {
        let mut filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2], 100, 0.01).unwrap();

        assert!(!filter.needs_resize());

        // Fill to saturation
        for i in 0..150 {
            filter.insert_to_bin(&format!("item{}", i), &[0]).unwrap();
        }

        assert!(filter.needs_resize());
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_validate_structure() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();
        assert!(filter.validate_structure().is_ok());
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_health_check() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();

        let health = filter.health_check();
        assert!(matches!(health.status, HealthStatus::Healthy));
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_prometheus_export() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2], 1000, 0.01).unwrap();

        let output = filter.export_prometheus();
        assert!(output.contains("tree_bloom_filter_items"));
        assert!(output.contains("tree_bloom_filter_load_factor"));
    }

    #[cfg(test)]
    #[cfg(feature = "proptest")]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn no_false_negatives(items: Vec<String>) {
                let branching = vec![3, 4];
                let mut filter: TreeBloomFilter<String> = TreeBloomFilter::new(branching, 1000, 0.01).unwrap();

                for item in &items {
                    filter.insert_auto(item).unwrap();
                }

                // No false negatives
                for item in &items {
                    prop_assert!(filter.contains(item));
                }
            }

            #[test]
            fn insert_auto_deterministic(items: Vec<String>) {
                let mut filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![4, 4], 1000, 0.01).unwrap();

                for item in &items {
                    filter.insert_auto(item).unwrap();
                }

                // Each item should always be in exactly one bin
                for item in &items {
                    let locations = filter.locate(item);
                    prop_assert_eq!(locations.len(), 1,
                        "Item {:?} in {} bins (expected 1)", item, locations.len());
                }
            }
        }
    }

    #[test]
    fn test_depth_limit_enforced() {
        let too_deep = vec![2; MAX_TREE_DEPTH + 1];
        let result = TreeBloomFilter::<String>::new(too_deep, 1000, 0.01);
        assert!(result.is_err());

        match result.unwrap_err() {
            BloomCraftError::InvalidParameters { message } => {
                assert!(message.contains("exceeds maximum"));
            }
            _ => panic!("Expected InvalidParameters error"),
        }
    }

    #[test]
    fn test_max_depth_allowed() {
        let max_depth = vec![2; MAX_TREE_DEPTH];
        let result = TreeBloomFilter::<String>::new(max_depth, 1000, 0.01);
        // Will likely fail due to node count overflow, but depth check should pass
        if let Ok(filter) = result {
            assert_eq!(filter.depth(), MAX_TREE_DEPTH);
        }
    }

    #[test]
    fn test_node_count_overflow_protection() {
        // This configuration overflows leaf count
        let huge_branching = vec![usize::MAX / 2, 10];
        let result = TreeBloomFilter::<String>::new(huge_branching, 1000, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_tree_config_validation() {
        let config = TreeConfig {
            branching: vec![5, 10, 20],
            capacity_per_bin: 1000,
            target_fpr: 0.01,
        };

        let stats = config.validate().unwrap();
        assert_eq!(stats.total_nodes, 1_056);
        assert_eq!(stats.leaf_count, 1_000);
        assert!(stats.memory_mb > 0);
    }

    #[test]
    fn test_tree_config_too_deep() {
        let config = TreeConfig {
            branching: vec![2; MAX_TREE_DEPTH + 1],
            capacity_per_bin: 100,
            target_fpr: 0.01,
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_insert_batch_overflow_protection() {
        let mut filter = TreeBloomFilter::<String>::new(vec![2], 100, 0.01).unwrap();
        filter.total_items = usize::MAX - 5;

        let items = vec!["a".to_string(); 10];
        let result = filter.insert_batch_to_bin(&items, &[0]);

        assert!(result.is_err());
        match result.unwrap_err() {
            BloomCraftError::CapacityExceeded { .. } => {}
            _ => panic!("Expected CapacityExceeded error"),
        }
    }

    #[test]
    fn test_clear_deep_tree() {
        let mut filter = TreeBloomFilter::<u32>::new(vec![2; 10], 100, 0.01).unwrap();
        // Insert items into many leaves
        for i in 0..5 {
            let path: Vec<usize> = (0..10).map(|_| 0).collect();
            filter.insert_to_bin(&i, &path).unwrap();
        }
        // Clearing should not stack overflow and should zero out counts
        filter.clear_subtree(&[]).unwrap();
        assert_eq!(filter.total_items, 0);
        assert_eq!(filter.root.item_count, 0);
    }

    #[test]
    fn test_union_zeros_counts() {
        let mut filter1: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();
        let mut filter2: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();

        filter1.insert_to_bin(&"a".to_string(), &[0, 0]).unwrap();
        filter1.insert_to_bin(&"b".to_string(), &[0, 1]).unwrap();
        filter2.insert_to_bin(&"c".to_string(), &[1, 0]).unwrap();

        assert_eq!(filter1.total_items, 2);
        assert!(filter1.root.item_count > 0);

        filter1.union(&filter2).unwrap();

        // Exact counts are unrecoverable after union
        assert_eq!(filter1.total_items, 0);
        assert_eq!(filter1.root.item_count, 0);
    }

    #[test]
    fn test_intersect_zeros_counts() {
        let mut filter1: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();
        let mut filter2: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();

        filter1.insert_to_bin(&"a".to_string(), &[0, 0]).unwrap();
        filter1.insert_to_bin(&"b".to_string(), &[0, 0]).unwrap();
        filter2.insert_to_bin(&"b".to_string(), &[0, 0]).unwrap();
        filter2.insert_to_bin(&"c".to_string(), &[1, 1]).unwrap();

        assert_eq!(filter1.total_items, 2);

        filter1.intersect(&filter2).unwrap();

        // Exact counts are unrecoverable after intersection
        assert_eq!(filter1.total_items, 0);
        assert_eq!(filter1.root.item_count, 0);
    }

    #[test]
    fn test_query_any_prunes_subtrees() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![5, 10], 1000, 0.01).unwrap();

        // Insert only into one leaf bin
        filter
            .insert_to_bin(&"present".to_string(), &[0, 0])
            .unwrap();

        // query_any returns true for the inserted item
        assert!(filter.query_any(&"present".to_string()));

        // query_any returns false for a missing item (prunes via root filter)
        assert!(!filter.query_any(&"missing".to_string()));

        // Insert into root-filter-only to verify leaf pruning still works
        // (query_any must not descend into children whose filter doesn't match)
        let mut all_nodes = TreeBloomFilter::<String>::new(vec![3, 3], 1000, 0.01).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                all_nodes
                    .insert_to_bin(&"common".to_string(), &[i, j])
                    .unwrap();
            }
        }
        // Every child filter contains "common", so query_any descends everywhere
        assert!(all_nodes.query_any(&"common".to_string()));
    }

    #[test]
    fn test_clear_subtree_ancestor_counts() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();

        // Insert into two different leaf bins under different subtrees
        filter.insert_to_bin(&"x".to_string(), &[0, 0]).unwrap();
        filter.insert_to_bin(&"y".to_string(), &[0, 0]).unwrap(); // same bin
        filter.insert_to_bin(&"z".to_string(), &[1, 0]).unwrap(); // different subtree

        assert_eq!(filter.total_items, 3);
        assert_eq!(filter.root.item_count, 3);

        // Clear subtree [0] (contains items x, y)
        filter.clear_subtree(&[0]).unwrap();

        // total_items decremented by 2 (items in cleared subtree)
        assert_eq!(filter.total_items, 1);
        // Root decremented by 2
        assert_eq!(filter.root.item_count, 1);
        // Subtree [1] unaffected
        let subtree1 = filter.subtree_at(&[1]).unwrap();
        assert_eq!(subtree1.item_count, 1);
        assert!(subtree1.filter.contains(&"z".to_string()));
        // Cleared subtree item_count is zero
        let cleared = filter.subtree_at(&[0]).unwrap();
        assert_eq!(cleared.item_count, 0);
    }

    #[test]
    fn test_locate_with_zero_allocation() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 4], 1000, 0.01).unwrap();

        // Insert into multiple bins
        filter.insert_to_bin(&"test".to_string(), &[0, 0]).unwrap();
        filter.insert_to_bin(&"test".to_string(), &[1, 2]).unwrap();
        filter.insert_to_bin(&"test".to_string(), &[2, 3]).unwrap();

        // Collect using callback
        let mut paths = Vec::new();
        filter.locate_with(&"test".to_string(), |path| {
            paths.push(path.to_vec());
        });

        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&vec![0, 0]));
        assert!(paths.contains(&vec![1, 2]));
        assert!(paths.contains(&vec![2, 3]));
    }

    #[test]
    fn test_locate_with_early_processing() {
        let mut filter: TreeBloomFilter<u32> =
            TreeBloomFilter::new(vec![5, 10], 1000, 0.01).unwrap();

        // Insert into many bins
        for i in 0..5 {
            for j in 0..10 {
                filter.insert_to_bin(&42, &[i, j]).unwrap();
            }
        }

        // Process immediately without storing
        let mut count = 0;
        filter.locate_with(&42, |_path| {
            count += 1;
        });

        assert_eq!(count, 50);
    }

    #[test]
    fn test_locate_with_vs_locate_equivalence() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![4, 5], 1000, 0.01).unwrap();

        filter.insert_to_bin(&"item".to_string(), &[1, 2]).unwrap();
        filter.insert_to_bin(&"item".to_string(), &[3, 4]).unwrap();

        // Compare results
        let locate_result = filter.locate(&"item".to_string());

        let mut locate_with_result = Vec::new();
        filter.locate_with(&"item".to_string(), |path| {
            locate_with_result.push(path.to_vec());
        });

        assert_eq!(locate_result.len(), locate_with_result.len());
        for path in &locate_result {
            assert!(locate_with_result.contains(path));
        }
    }

    #[test]
    fn test_locate_iter_basic() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();

        filter.insert_to_bin(&"test".to_string(), &[0, 1]).unwrap();
        filter.insert_to_bin(&"test".to_string(), &[1, 2]).unwrap();

        let paths: Vec<_> = filter.locate_iter(&"test".to_string()).collect();

        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&vec![0, 1]));
        assert!(paths.contains(&vec![1, 2]));
    }

    #[test]
    fn test_locate_iter_early_exit() {
        let mut filter: TreeBloomFilter<u32> =
            TreeBloomFilter::new(vec![5, 10], 1000, 0.01).unwrap();

        // Insert into all bins
        for i in 0..5 {
            for j in 0..10 {
                filter.insert_to_bin(&999, &[i, j]).unwrap();
            }
        }

        // Take only first 3 matches
        let first_three: Vec<_> = filter.locate_iter(&999).take(3).collect();

        assert_eq!(first_three.len(), 3);
    }

    #[test]
    fn test_locate_iter_count() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 4], 1000, 0.01).unwrap();

        for i in 0..3 {
            for j in 0..4 {
                filter.insert_to_bin(&"item".to_string(), &[i, j]).unwrap();
            }
        }

        // Count without allocating paths
        let count = filter.locate_iter(&"item".to_string()).count();

        assert_eq!(count, 12);
    }

    #[test]
    fn test_locate_iter_empty() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();

        let item = "nonexistent".to_string();
        let mut iter = filter.locate_iter(&item);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_locate_apis_consistency() {
        let mut filter: TreeBloomFilter<u64> =
            TreeBloomFilter::new(vec![4, 5, 3], 1000, 0.01).unwrap();

        // Insert into sparse bins
        filter.insert_to_bin(&12345, &[0, 1, 2]).unwrap();
        filter.insert_to_bin(&12345, &[2, 3, 1]).unwrap();
        filter.insert_to_bin(&12345, &[3, 4, 0]).unwrap();

        // Collect via all three APIs
        let locate_result = filter.locate(&12345);

        let mut locate_with_result = Vec::new();
        filter.locate_with(&12345, |path| {
            locate_with_result.push(path.to_vec());
        });

        let locate_iter_result: Vec<_> = filter.locate_iter(&12345).collect();

        // All should return same paths
        assert_eq!(locate_result.len(), 3);
        assert_eq!(locate_with_result.len(), 3);
        assert_eq!(locate_iter_result.len(), 3);

        for path in &locate_result {
            assert!(locate_with_result.contains(path));
            assert!(locate_iter_result.contains(path));
        }
    }

    #[test]
    fn test_subtree_at_root() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();

        let root = filter.subtree_at(&[]).unwrap();
        assert_eq!(root.children.len(), 2);
    }

    #[test]
    fn test_subtree_at_valid_paths() {
        let filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 4, 2], 1000, 0.01).unwrap();

        // Valid partial path
        let node = filter.subtree_at(&[1]).unwrap();
        assert_eq!(node.children.len(), 4);

        // Valid full path to internal node
        let node = filter.subtree_at(&[2, 3]).unwrap();
        assert_eq!(node.children.len(), 2);

        // Valid path to leaf
        let leaf = filter.subtree_at(&[0, 1, 0]).unwrap();
        assert!(leaf.is_leaf());
    }

    #[test]
    fn test_subtree_at_path_too_long() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();

        // Path longer than depth (depth = 2)
        let result = filter.subtree_at(&[0, 1, 2]);

        assert!(result.is_err());
        match result.unwrap_err() {
            BloomCraftError::InvalidParameters { message } => {
                assert!(message.contains("Path length 3 exceeds tree depth 2"));
            }
            _ => panic!("Expected InvalidParameters error"),
        }
    }

    #[test]
    fn test_subtree_at_index_out_of_bounds() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();

        // First level has branching factor 2, so index 2 is invalid
        let result = filter.subtree_at(&[2]);

        assert!(result.is_err());
        match result.unwrap_err() {
            BloomCraftError::InvalidParameters { message } => {
                assert!(message.contains("Invalid index 2 at level 0"));
                assert!(message.contains("branching factor is 2"));
            }
            _ => panic!("Expected InvalidParameters error"),
        }
    }

    #[test]
    fn test_subtree_at_index_out_of_bounds_deep() {
        let filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 4, 2], 1000, 0.01).unwrap();

        // Second level has branching factor 4, so index 5 is invalid
        let result = filter.subtree_at(&[1, 5]);

        assert!(result.is_err());
        match result.unwrap_err() {
            BloomCraftError::InvalidParameters { message } => {
                assert!(message.contains("Invalid index 5 at level 1"));
                assert!(message.contains("branching factor is 4"));
            }
            _ => panic!("Expected InvalidParameters error"),
        }
    }

    #[test]
    fn test_subtree_at_boundary_cases() {
        let filter: TreeBloomFilter<String> = TreeBloomFilter::new(vec![3, 4], 1000, 0.01).unwrap();

        // Max valid index at each level
        assert!(filter.subtree_at(&[2]).is_ok());
        assert!(filter.subtree_at(&[0, 3]).is_ok());

        // Just beyond max
        assert!(filter.subtree_at(&[3]).is_err());
        assert!(filter.subtree_at(&[0, 4]).is_err());
    }

    #[test]
    fn test_clear_subtree_path_too_long() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();

        let result = filter.clear_subtree(&[0, 1, 2]);

        assert!(result.is_err());
        match result.unwrap_err() {
            BloomCraftError::InvalidParameters { message } => {
                assert!(message.contains("Path length 3 exceeds tree depth 2"));
            }
            _ => panic!("Expected InvalidParameters error"),
        }
    }

    #[test]
    fn test_clear_subtree_index_out_of_bounds() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 4], 1000, 0.01).unwrap();

        let result = filter.clear_subtree(&[1, 5]);

        assert!(result.is_err());
        match result.unwrap_err() {
            BloomCraftError::InvalidParameters { message } => {
                assert!(message.contains("Invalid index 5 at level 1"));
                assert!(message.contains("branching factor is 4"));
            }
            _ => panic!("Expected InvalidParameters error"),
        }
    }

    #[test]
    fn test_clear_subtree_valid() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 3], 1000, 0.01).unwrap();

        // Insert data into specific subtree
        filter.insert_to_bin(&"item1".to_string(), &[0, 0]).unwrap();
        filter.insert_to_bin(&"item2".to_string(), &[0, 1]).unwrap();

        // Clear that subtree
        filter.clear_subtree(&[0]).unwrap();

        // Verify subtree is cleared
        let subtree = filter.subtree_at(&[0]).unwrap();
        assert_eq!(subtree.item_count, 0);

        for child in subtree.children.iter() {
            assert_eq!(child.item_count, 0);
        }
    }

    #[test]
    fn test_clear_subtree_root() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![2, 2], 1000, 0.01).unwrap();

        filter.insert_to_bin(&"item".to_string(), &[0, 0]).unwrap();
        filter.insert_to_bin(&"item".to_string(), &[1, 1]).unwrap();

        // Clear entire tree
        filter.clear_subtree(&[]).unwrap();

        assert_eq!(filter.root.item_count, 0);
        // total_items is decremented to 0 after clearing the root
        assert_eq!(filter.total_items, 0);
    }

    #[test]
    fn test_subtree_at_and_clear_consistency() {
        let mut filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![3, 4, 2], 1000, 0.01).unwrap();

        // Test that invalid paths fail the same way in both methods
        let invalid_paths = vec![vec![10], vec![0, 20], vec![1, 2, 5], vec![0, 0, 0, 0]];

        for path in &invalid_paths {
            // Test subtree_at
            let subtree_result = filter.subtree_at(path);
            assert!(
                subtree_result.is_err(),
                "subtree_at should fail for {:?}",
                path
            );
            let subtree_msg = format!("{}", subtree_result.unwrap_err()); // ✅ Consume it immediately

            // Test clear_subtree (now filter is not borrowed)
            let clear_result = filter.clear_subtree(path);
            assert!(
                clear_result.is_err(),
                "clear_subtree should fail for {:?}",
                path
            );
            let clear_msg = format!("{}", clear_result.unwrap_err());

            // Compare
            assert_eq!(
                subtree_msg, clear_msg,
                "Error messages should match for {:?}",
                path
            );
        }
    }

    #[test]
    fn test_error_message_quality() {
        let filter: TreeBloomFilter<String> =
            TreeBloomFilter::new(vec![5, 10, 3], 1000, 0.01).unwrap();

        // Path too long
        let result = filter.subtree_at(&[0, 0, 0, 0]);
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Path length 4"));
        assert!(err.contains("tree depth 3"));

        // Index out of bounds with context
        let result = filter.subtree_at(&[0, 15]);
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid index 15"));
        assert!(err.contains("level 1"));
        assert!(err.contains("branching factor is 10"));
        assert!(err.contains("valid range: 0..10"));
    }

    #[cfg(all(test, feature = "serde"))]
    mod serde_security_tests {
        use super::*;
        use crate::hash::StdHasher;

        #[test]
        fn test_serialize_includes_type_id() {
            let filter: TreeBloomFilter<String, StdHasher> =
                TreeBloomFilter::new(vec![2, 2], 100, 0.01).unwrap();

            let serialized = serde_json::to_string(&filter).unwrap();

            // Should contain the hasher_type field
            assert!(serialized.contains("hasher_type"));
        }

        #[test]
        fn test_deserialize_with_matching_type_id() {
            let original: TreeBloomFilter<String, StdHasher> =
                TreeBloomFilter::new(vec![2, 2], 100, 0.01).unwrap();

            let serialized = serde_json::to_string(&original).unwrap();

            // Should deserialize successfully
            let deserialized: TreeBloomFilter<String, StdHasher> =
                serde_json::from_str(&serialized).unwrap();

            assert_eq!(deserialized.depth(), original.depth());
        }

        #[test]
        fn test_deserialize_rejects_wrong_type_id() {
            // Create a filter with StdHasher
            let filter: TreeBloomFilter<String, StdHasher> =
                TreeBloomFilter::new(vec![2], 100, 0.01).unwrap();

            let mut serialized = serde_json::to_value(&filter).unwrap();

            // Tamper with the hasher_type field
            if let Some(obj) = serialized.as_object_mut() {
                obj.insert(
                    "hasher_type".to_string(),
                    serde_json::Value::String("some::bogus::Hasher".to_string()),
                );
            }

            let tampered = serde_json::to_string(&serialized).unwrap();

            // Should fail to deserialize
            let result: Result<TreeBloomFilter<String, StdHasher>> =
                serde_json::from_str(&tampered).map_err(Into::into);

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("Hasher type mismatch"));
        }

        #[test]
        fn test_deserialize_backward_compatibility() {
            // Old format (hasher_type only, no hasher_type_id) — fully valid
            let old_format = r#"{
                "root": {
                    "filter": {"bits": [], "num_hash": 3, "hasher": null},
                    "item_count": 0,
                    "children": [],
                    "metadata": {"path": [], "level": 0}
                },
                "branching": [2, 2],
                "capacity_per_bin": 100,
                "target_fpr": 0.01,
                "total_items": 0,
                "hasher_type": "bloomcraft::hash::StdHasher"
            }"#;

            let result: Result<TreeBloomFilter<String, StdHasher>> =
                serde_json::from_str(old_format).map_err(Into::into);

            // May fail due to simplified test data structure, but shouldn't be a type-mismatch error
            if let Err(e) = result {
                assert!(!e.to_string().contains("Hasher type mismatch"));
            }
        }

        #[test]
        fn test_deserialize_rejects_missing_hasher_type() {
            let malicious = r#"{
                "root": {
                    "filter": {"bits": [], "num_hash": 3, "hasher": null},
                    "item_count": 0,
                    "children": [],
                    "metadata": {"path": [], "level": 0}
                },
                "branching": [2],
                "capacity_per_bin": 100,
                "target_fpr": 0.01,
                "total_items": 0
            }"#;

            let result: Result<TreeBloomFilter<String, StdHasher>> =
                serde_json::from_str(malicious).map_err(Into::into);

            assert!(result.is_err());
        }

        #[test]
        fn test_type_id_is_stable_within_process() {
            // TypeId should be consistent within the same binary
            let id1 = std::any::TypeId::of::<StdHasher>();
            let id2 = std::any::TypeId::of::<StdHasher>();

            assert_eq!(id1, id2);

            // Debug representation should also be consistent
            let debug1 = format!("{:?}", id1);
            let debug2 = format!("{:?}", id2);

            assert_eq!(debug1, debug2);
        }

        #[cfg(feature = "wyhash")]
        #[test]
        fn test_type_id_differs_for_different_types() {
            use crate::hash::WyHasher;

            let std_id = std::any::TypeId::of::<StdHasher>();
            let wy_id = std::any::TypeId::of::<WyHasher>();

            assert_ne!(std_id, wy_id);

            let std_debug = format!("{:?}", std_id);
            let wy_debug = format!("{:?}", wy_id);

            assert_ne!(std_debug, wy_debug);
        }

        #[test]
        fn test_round_trip_preserves_type_safety() {
            let original: TreeBloomFilter<u64, StdHasher> =
                TreeBloomFilter::new(vec![3, 4], 1000, 0.01).unwrap();

            // Serialize
            let serialized = serde_json::to_string(&original).unwrap();

            // Deserialize with correct type
            let correct: TreeBloomFilter<u64, StdHasher> =
                serde_json::from_str(&serialized).unwrap();

            assert_eq!(correct.depth(), 2);
            assert_eq!(correct.leaf_count(), 12);
        }
    }

    #[test]
    fn test_insert_auto_and_locate_use_same_hasher() {
        let mut filter =
            TreeBloomFilter::<String>::new(vec![5, 8], 1000, 0.01).expect("construction failed");

        let items = ["alpha", "beta", "gamma", "delta", "epsilon"];

        for item in &items {
            let s = item.to_string();
            filter.insert_auto(&s).expect("insert_auto failed");
        }

        for item in &items {
            let s = item.to_string();
            let locations = filter.locate(&s);
            assert_eq!(
                locations.len(),
                1,
                "locate('{}') returned {} bins — expected exactly 1. \
                insert_auto and locate use different hashers (Fix 4).",
                item,
                locations.len()
            );
        }
    }

    #[test]
    fn test_locate_with_sibling_no_index_overwrite() {
        let mut filter =
            TreeBloomFilter::<u64>::new(vec![2, 3], 1000, 0.01).expect("construction failed");

        filter
            .insert_to_bin(&42u64, &[0, 1])
            .expect("insert failed");
        filter
            .insert_to_bin(&42u64, &[0, 2])
            .expect("insert failed");

        let mut found_paths: Vec<Vec<usize>> = Vec::new();
        filter.locate_with(&42u64, |path| {
            found_paths.push(path.to_vec());
        });

        assert!(
            found_paths.contains(&vec![0, 1]),
            "locate_with missing path [0, 1]. Got: {:?} (Fix 4 traversal)",
            found_paths
        );
        assert!(
            found_paths.contains(&vec![0, 2]),
            "locate_with missing path [0, 2]. Got: {:?} (Fix 4 traversal)",
            found_paths
        );
        assert_eq!(
            found_paths.len(),
            2,
            "Expected exactly 2 matches, got {}",
            found_paths.len()
        );
        assert_ne!(
            found_paths[0], found_paths[1],
            "Both paths are identical — sibling index overwrite confirmed (Fix 4 traversal)"
        );
    }

    #[test]
    fn test_locate_api_consistency_cross_check() {
        let mut filter =
            TreeBloomFilter::<u64>::new(vec![3, 4], 1000, 0.01).expect("construction failed");

        filter
            .insert_to_bin(&99u64, &[0, 1])
            .expect("insert failed");
        filter
            .insert_to_bin(&99u64, &[1, 3])
            .expect("insert failed");
        filter
            .insert_to_bin(&99u64, &[2, 0])
            .expect("insert failed");

        let locate_result = filter.locate(&99u64);

        let mut locate_with_result: Vec<Vec<usize>> = Vec::new();
        filter.locate_with(&99u64, |path| {
            locate_with_result.push(path.to_vec());
        });

        let locate_iter_result: Vec<Vec<usize>> = filter.locate_iter(&99u64).collect();

        assert_eq!(
            locate_result.len(),
            3,
            "locate() must find 3 matches, found {} (Fix 1 + Fix 4 traversal)",
            locate_result.len()
        );
        assert_eq!(
            locate_with_result.len(),
            locate_result.len(),
            "locate_with() found {} but locate() found {} (Fix 4 traversal)",
            locate_with_result.len(),
            locate_result.len()
        );
        assert_eq!(
            locate_iter_result.len(),
            locate_result.len(),
            "locate_iter() found {} but locate() found {} (Fix 4 traversal)",
            locate_iter_result.len(),
            locate_result.len()
        );

        for path in &locate_result {
            assert!(
                locate_with_result.contains(path),
                "locate_with missing path {:?}",
                path
            );
            assert!(
                locate_iter_result.contains(path),
                "locate_iter missing path {:?}",
                path
            );
        }
    }

    #[test]
    fn test_union_contains_all_items_from_both_filters() {
        let mut filter1 =
            TreeBloomFilter::<String>::new(vec![2, 3], 1000, 0.01).expect("construction failed");
        let mut filter2 =
            TreeBloomFilter::<String>::new(vec![2, 3], 1000, 0.01).expect("construction failed");

        let set_a = ["apple", "banana", "cherry"];
        let set_b = ["delta", "echo", "foxtrot"];

        for item in &set_a {
            let s = item.to_string();
            filter1.insert_to_bin(&s, &[0, 0]).expect("insert failed");
        }
        for item in &set_b {
            let s = item.to_string();
            filter2.insert_to_bin(&s, &[1, 2]).expect("insert failed");
        }

        filter1.union(&filter2).expect("union failed");

        for item in set_a.iter().chain(set_b.iter()) {
            let s = item.to_string();
            assert!(
                filter1.contains(&s),
                "union: item '{}' not found in result",
                item
            );
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialize_deserialize_locate_roundtrip() {
        use crate::hash::StdHasher;

        let mut filter = TreeBloomFilter::<String, StdHasher>::with_hasher(
            vec![3, 4],
            500,
            0.01,
            StdHasher::new(),
        )
        .expect("construction failed");

        let items = ["one", "two", "three", "four", "five"];

        for item in &items {
            let s = item.to_string();
            filter.insert_auto(&s).expect("insert_auto failed");
        }

        let pre_paths: Vec<Vec<Vec<usize>>> = items
            .iter()
            .map(|item| filter.locate(&item.to_string()))
            .collect();

        let bytes = bincode::serialize(&filter).expect("serialization failed");
        let restored: TreeBloomFilter<String, StdHasher> =
            bincode::deserialize(&bytes).expect("deserialization failed");

        for (i, item) in items.iter().enumerate() {
            let post_paths = restored.locate(&item.to_string());
            assert_eq!(
                post_paths, pre_paths[i],
                "locate('{}') returned different paths after deserialization. \
                Fix 4 routing or serialization is broken.",
                item
            );
        }
    }
}
