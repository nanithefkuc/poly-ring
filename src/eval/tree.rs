//! The shared polynomial product-tree topology.
//!
//! Both tree consumers — subproduct-tree multipoint evaluation and the
//! prepared remainder tree — carve the same object: post-order nodes holding
//! a contiguous leaf range's product, with children identified by node id.
//! This module owns that topology once: the node shape, the degree-balanced
//! split, and the recycled node pool. Consumers supply their leaves through a
//! closure that assigns into a pooled buffer, so a warm rebuild over a new
//! point set or modulus list of the same size allocates nothing.

use alloc::vec::Vec;

use fgf::kernel::FieldKernels;

use crate::error::{ConfigError, PolynomialError};
use crate::poly::Polynomial;

/// The closure assigning one leaf's polynomial into a pooled buffer.
type LeafAssigner<'a, F> =
    &'a mut dyn FnMut(usize, &mut Polynomial<F>) -> Result<(), PolynomialError>;

/// Child id marking a leaf node.
pub(super) const LEAF: usize = usize::MAX;

/// One product-tree node: the covered range's product and its child ids.
///
/// Nodes are numbered in construction (post-)order, so the root is the last
/// node and every subtree occupies a contiguous id range. `low`/`high` are
/// the covered leaf range in caller order — traversal reads them from the
/// node, so output order always follows input order.
#[derive(Debug)]
pub(super) struct ProductNode<F: FieldKernels> {
    /// The product of every leaf coefficient range below this node.
    pub(super) polynomial: Polynomial<F>,
    /// Left child id, or [`LEAF`].
    pub(super) left: usize,
    /// Right child id, or [`LEAF`].
    pub(super) right: usize,
    /// First covered leaf index, in caller order.
    pub(super) low: usize,
    /// One past the last covered leaf index.
    pub(super) high: usize,
}

/// Recycled node storage for one product tree.
#[derive(Debug)]
pub(super) struct TreePool<F: FieldKernels> {
    /// The current tree, post-order.
    pub(super) tree: Vec<ProductNode<F>>,
    /// Drained nodes kept for reuse.
    pool: Vec<ProductNode<F>>,
}

impl<F: FieldKernels> TreePool<F> {
    /// Construct an empty pool.
    pub(super) const fn new() -> Self {
        Self {
            tree: Vec::new(),
            pool: Vec::new(),
        }
    }

    /// Build the product tree over `leaf_count` leaves.
    ///
    /// `cumulative(index)` is the total leaf degree strictly below leaf
    /// `index` — a strictly increasing profile, since every leaf in a
    /// product tree has positive degree. Each range `[low, high)` splits at
    /// the index minimizing left/right total-degree imbalance (lowest index
    /// on ties), so a heavy leaf isolates naturally; for uniform degree-one
    /// leaves this is exactly the per-count middle split.
    ///
    /// `leaf(index, destination)` assigns leaf `index`'s polynomial into a
    /// pooled buffer, so a same-size rebuild reuses every allocation.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when a node cannot be reserved or a leaf
    /// or product assignment fails.
    pub(super) fn build(
        &mut self,
        leaf_count: usize,
        cumulative: &dyn Fn(usize) -> usize,
        leaf: LeafAssigner<'_, F>,
    ) -> Result<(), PolynomialError> {
        if leaf_count == 0 {
            self.recycle();
            return Ok(());
        }
        let node_count = leaf_count
            .checked_mul(2)
            .and_then(|doubled| doubled.checked_sub(1))
            .ok_or(ConfigError::GeometryOverflow {
                context: "product tree nodes",
            })?;
        if self.tree.len() == node_count {
            // Same node count: rebuild in place. The split is re-derived from
            // the (possibly changed) degree profile, so the carved topology
            // always matches the profile that produced it.
            let mut cursor = 0_usize;
            self.rebuild_node(0, leaf_count, cumulative, leaf, &mut cursor)?;
            debug_assert_eq!(cursor, node_count);
            return Ok(());
        }
        self.recycle();
        if self.tree.capacity() < node_count {
            self.tree
                .try_reserve(node_count - self.tree.len())
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "product tree",
                    elements: node_count,
                    element_size: core::mem::size_of::<ProductNode<F>>(),
                })?;
        }
        self.build_node(0, leaf_count, cumulative, leaf)?;
        debug_assert_eq!(self.tree.len(), node_count);
        Ok(())
    }

    /// The current tree, post-order.
    pub(super) fn nodes(&self) -> &[ProductNode<F>] {
        &self.tree
    }

    /// Recycle the current tree into the pool, zeroing node buffers.
    pub(super) fn recycle(&mut self) {
        for mut node in self.tree.drain(..) {
            node.polynomial.set_zero();
            self.pool.push(node);
        }
    }

    /// Take a zeroed node from the pool, or a fresh one.
    fn take_node(&mut self) -> ProductNode<F> {
        if let Some(mut node) = self.pool.pop() {
            node.polynomial.set_zero();
            node.left = LEAF;
            node.right = LEAF;
            return node;
        }
        ProductNode {
            polynomial: Polynomial::zero(),
            left: LEAF,
            right: LEAF,
            low: 0,
            high: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_node(
        &mut self,
        low: usize,
        high: usize,
        cumulative: &dyn Fn(usize) -> usize,
        leaf: LeafAssigner<'_, F>,
    ) -> Result<usize, PolynomialError> {
        if high - low == 1 {
            let mut leaf_node = self.take_node();
            leaf(low, &mut leaf_node.polynomial)?;
            leaf_node.low = low;
            leaf_node.high = high;
            let id = self.tree.len();
            self.tree.push(leaf_node);
            return Ok(id);
        }
        let split = balanced_split(cumulative, low, high);
        let left = self.build_node(low, split, cumulative, leaf)?;
        let right = self.build_node(split, high, cumulative, leaf)?;
        let mut node = self.take_node();
        node.low = low;
        node.high = high;
        let product_id = self.tree.len();
        self.tree[left]
            .polynomial
            .multiply_into(&self.tree[right].polynomial, &mut node.polynomial)?;
        node.left = left;
        node.right = right;
        self.tree.push(node);
        debug_assert_eq!(product_id, self.tree.len() - 1);
        Ok(product_id)
    }

    #[allow(clippy::too_many_arguments)]
    fn rebuild_node(
        &mut self,
        low: usize,
        high: usize,
        cumulative: &dyn Fn(usize) -> usize,
        leaf: LeafAssigner<'_, F>,
        cursor: &mut usize,
    ) -> Result<(), PolynomialError> {
        if high - low == 1 {
            let node = &mut self.tree[*cursor];
            leaf(low, &mut node.polynomial)?;
            node.left = LEAF;
            node.right = LEAF;
            node.low = low;
            node.high = high;
            *cursor += 1;
            return Ok(());
        }
        let split = balanced_split(cumulative, low, high);
        self.rebuild_node(low, split, cumulative, leaf, cursor)?;
        // A subtree's root is the last node its recursion filled.
        let left = *cursor - 1;
        self.rebuild_node(split, high, cumulative, leaf, cursor)?;
        let right = *cursor - 1;
        let node = *cursor;
        *cursor += 1;
        let (children, parent) = self.tree.split_at_mut(node);
        children[left]
            .polynomial
            .multiply_into(&children[right].polynomial, &mut parent[0].polynomial)?;
        parent[0].left = left;
        parent[0].right = right;
        parent[0].low = low;
        parent[0].high = high;
        Ok(())
    }
}

/// Split `[low, high)` at the index minimizing left/right total-degree
/// imbalance, lowest index on ties.
///
/// `cumulative` is strictly increasing over the range, so the midpoint
/// crossing can be binary-searched and only the two straddling candidates
/// need weighing. For uniform degree-one leaves this equals
/// `low + (high - low) / 2` exactly, the split the multipoint descent has
/// always used.
fn balanced_split(cumulative: &dyn Fn(usize) -> usize, low: usize, high: usize) -> usize {
    let total = u128::from(cumulative(low) as u64) + u128::from(cumulative(high) as u64);
    let imbalance =
        |index: usize| -> u128 { (2 * u128::from(cumulative(index) as u64)).abs_diff(total) };
    // Smallest split whose doubled prefix reaches the midpoint, or the last
    // split when even that stays below it.
    let mut candidate = high - 1;
    let mut lower = low + 1;
    let mut upper = high - 1;
    while lower <= upper {
        let middle = lower + (upper - lower) / 2;
        if 2 * u128::from(cumulative(middle) as u64) >= total {
            candidate = middle;
            upper = middle - 1;
        } else {
            lower = middle + 1;
        }
    }
    if candidate > low + 1 && imbalance(candidate - 1) <= imbalance(candidate) {
        candidate - 1
    } else {
        candidate
    }
}

#[cfg(test)]
mod tests {
    use super::balanced_split;

    /// Degree-one leaves split exactly at the middle: the shared builder's
    /// degree-balanced rule and the multipoint descent's per-count middle
    /// split carve identical trees, so refactoring multipoint onto the
    /// shared builder is zero behavioral drift.
    #[test]
    fn unit_degree_splits_at_the_middle() {
        for low in 0..64 {
            for high in (low + 2)..64 {
                assert_eq!(
                    balanced_split(&|index| index, low, high),
                    low + (high - low) / 2,
                    "range [{low}, {high})"
                );
            }
        }
    }

    /// A heavy leaf isolates at the first split instead of the light leaves
    /// splitting around it.
    #[test]
    fn heavy_leaf_isolates() {
        // Degrees [64, 1, 1, 1]: prefix [0, 64, 65, 66, 67].
        let prefix = [0, 64, 65, 66, 67];
        assert_eq!(balanced_split(&|index| prefix[index], 0, 4), 1);
        // Degrees [1, 1, 1, 64]: prefix [0, 1, 2, 3, 67].
        let prefix = [0, 1, 2, 3, 67];
        assert_eq!(balanced_split(&|index| prefix[index], 0, 4), 3);
        // Degrees [1, 2, 4, 8]: prefix [0, 1, 3, 7, 15]. Midpoint total 15:
        // split 3 leaves the light prefix against the degree-8 leaf.
        let prefix = [0, 1, 3, 7, 15];
        assert_eq!(balanced_split(&|index| prefix[index], 0, 4), 3);
    }
}
