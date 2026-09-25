//! Subproduct-tree multipoint evaluation and Lagrange interpolation over
//! arbitrary point sets.
//!
//! The subproduct tree of `M(X) = ∏(X − α_i)` turns multipoint evaluation
//! into a remainder descent and interpolation into a divide-and-conquer
//! combination, both `O(M(n) log n)` against the `O(n²)` of per-point Horner
//! and incremental Newton. The leaf is `X − α_i`, not `X + α_i`: only the
//! former reduces to `f(α_i)` outside characteristic two, where the two
//! coincide. Structured (subspace/coset) domains never enter this path: they
//! compose the `butterfly-fft` transform under the `fft` feature instead.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use super::lane::{LaneScratch, evaluate_lane_into};
use super::tree::{ProductNode, TreePool};
use crate::error::{ConfigError, DomainError, EvalError, PolynomialError};
use crate::poly::Polynomial;

/// Point count at or below which per-point Horner wins over the subproduct
/// tree (the tree's setup outweighs its asymptotic advantage). Measured; see
/// `BENCHMARKS.md`.
pub const MULTIPOINT_EVAL_CROSSOVER: usize = 16;

/// Multiply-add step count (points times coefficients) at or below which the
/// lane-parallel Horner route wins over the subproduct-tree descent.
/// Measured; see `BENCHMARKS.md`.
pub const MULTIPOINT_LANE_STEP_CROSSOVER: usize = 67_108_864;

/// Caller-owned reusable storage for subproduct-tree evaluation.
///
/// The tree nodes, the remainder-descent buffers, and their pool are all
/// recycled, so a warmed evaluation over a changed polynomial or point set
/// of the same size performs no heap allocation.
#[derive(Debug)]
pub struct MultipointScratch<F: FieldKernels> {
    tree: TreePool<F>,
    remainders: Vec<Polynomial<F>>,
    quotient: Polynomial<F>,
    lane: LaneScratch<F>,
}

impl<F: FieldKernels> MultipointScratch<F> {
    /// Construct empty reusable multipoint scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tree: TreePool::new(),
            remainders: Vec::new(),
            quotient: Polynomial::zero(),
            lane: LaneScratch::new(),
        }
    }
}

impl<F: FieldKernels> Default for MultipointScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate `polynomial` at every point, dispatching by point count.
///
/// Below [`MULTIPOINT_EVAL_CROSSOVER`] points this is per-point Horner;
/// above it, the lane-parallel Horner route while the multiply-add step
/// count stays at or below [`MULTIPOINT_LANE_STEP_CROSSOVER`], and the
/// subproduct-tree descent beyond that. The routes agree exactly.
///
/// # Errors
///
/// Returns [`PolynomialError`] when a lane, tree, or remainder buffer
/// cannot be reserved.
pub fn evaluate_multipoint<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    points: &[F::Elem],
) -> Result<Vec<F::Elem>, PolynomialError> {
    let mut scratch = MultipointScratch::new();
    let mut values = Vec::new();
    evaluate_multipoint_into(&mut values, polynomial, points, &mut scratch)?;
    Ok(values)
}

/// Write the evaluations at every point into `values`, reusing `scratch`.
///
/// Below [`MULTIPOINT_EVAL_CROSSOVER`] points this is per-point Horner;
/// above it, the lane-parallel Horner route while the multiply-add step
/// count stays at or below [`MULTIPOINT_LANE_STEP_CROSSOVER`], and the
/// subproduct-tree descent beyond that. The routes agree exactly.
///
/// # Errors
///
/// Returns [`PolynomialError`] when a lane, tree, or remainder buffer
/// cannot be reserved.
///
/// # Panics
///
/// The tree-root expectation holds for any point set the tree was just
/// built over.
pub fn evaluate_multipoint_into<F: FieldKernels>(
    values: &mut Vec<F::Elem>,
    polynomial: &Polynomial<F>,
    points: &[F::Elem],
    scratch: &mut MultipointScratch<F>,
) -> Result<(), PolynomialError> {
    values.clear();
    if points.len() < MULTIPOINT_EVAL_CROSSOVER {
        values.extend(
            points
                .iter()
                .copied()
                .map(|point| polynomial.evaluate(point)),
        );
        return Ok(());
    }
    let steps = points.len().saturating_mul(polynomial.coefficient_count());
    if steps <= MULTIPOINT_LANE_STEP_CROSSOVER {
        reserve_values(values, points.len(), "multipoint values")?;
        values.resize(points.len(), F::Elem::ZERO);
        evaluate_lane_into(
            values,
            polynomial.as_packed(),
            polynomial.coefficient_count(),
            points,
            &mut scratch.lane,
        )?;
        return Ok(());
    }
    build_subproduct_tree(points, scratch)?;
    reserve_values(values, points.len(), "multipoint values")?;
    values.resize(points.len(), F::Elem::ZERO);
    let root_id = scratch
        .tree
        .nodes()
        .len()
        .checked_sub(1)
        .expect("nonempty tree");
    // A fixed root slot, swapped out for the descent so its buffer returns
    // intact.
    ensure_remainder_slots(scratch, 1)?;
    let mut root = core::mem::take(&mut scratch.remainders[0]);
    root.assign_from(polynomial);
    descend_evaluate(root_id, 0, points.len(), &root, 0, scratch, values)?;
    scratch.remainders[0] = root;
    Ok(())
}

/// Interpolate the minimal-degree polynomial through the point/value pairs
/// by Lagrange over the subproduct tree.
///
/// Builds `M = ∏(X − α_i)`, evaluates `M'` at every point (one more
/// multipoint pass), and combines `v_i / M'(α_i) · M / (X − α_i)`
/// divide-and-conquer. Agrees with [`super::newton::interpolate_newton`]
/// exactly; the two share no code.
///
/// # Errors
///
/// Returns [`DomainError`] for duplicate points or mismatched lengths, and
/// [`PolynomialError`] when an intermediate buffer cannot be reserved.
///
/// # Panics
///
/// The tree-root expectation holds for any point set the tree was just
/// built over.
pub fn interpolate_lagrange<F: FieldKernels>(
    points: &[F::Elem],
    values: &[F::Elem],
) -> Result<Polynomial<F>, EvalError> {
    if points.len() != values.len() {
        return Err(EvalError::Domain(DomainError::LengthMismatch {
            expected: points.len(),
            found: values.len(),
        }));
    }
    if let Some((first, second)) = super::find_duplicate::<F>(points) {
        return Err(EvalError::Domain(DomainError::DuplicatePoint {
            first,
            second,
        }));
    }
    let mut scratch = MultipointScratch::new();
    build_subproduct_tree(points, &mut scratch)?;
    let root_id = scratch
        .tree
        .nodes()
        .len()
        .checked_sub(1)
        .expect("nonempty tree");
    let master_derivative = scratch.tree.nodes()[root_id]
        .polynomial
        .formal_derivative()?;
    let mut denominators: Vec<F::Elem> = Vec::new();
    reserve_values(&mut denominators, points.len(), "Lagrange denominators")?;
    let mut evaluate_scratch = MultipointScratch::new();
    evaluate_multipoint_into(
        &mut denominators,
        &master_derivative,
        points,
        &mut evaluate_scratch,
    )?;

    // Scaled values c_i = v_i / M'(α_i); `inv(0) == 0` inherits from fgf,
    // and duplicates were rejected above so every denominator is nonzero.
    let mut scaled: Vec<F::Elem> = Vec::with_capacity(values.len());
    for (value, denominator) in values.iter().zip(&denominators) {
        scaled.push(value.mul(denominator.inv()));
    }

    let root_id = scratch
        .tree
        .nodes()
        .len()
        .checked_sub(1)
        .expect("nonempty tree");
    let combined = combine_interpolant(root_id, 0, points.len(), &scaled, scratch.tree.nodes())?;
    Ok(combined)
}

/// Build the subproduct tree of `∏(X − α_i)` over the shared topology.
///
/// The leaves are unit degree, so the shared degree-balanced split carves
/// exactly the per-count middle split this descent has always used, and a
/// point set of `n` points occupies exactly `2n − 1` pooled nodes.
fn build_subproduct_tree<F: FieldKernels>(
    points: &[F::Elem],
    scratch: &mut MultipointScratch<F>,
) -> Result<(), PolynomialError> {
    scratch
        .tree
        .build(points.len(), &|index| index, &mut |index, destination| {
            destination.assign_coefficients(&[points[index].neg(), F::Elem::ONE])
        })
}

/// Descend the remainder tree, writing leaf evaluations into `values`.
///
/// Every recursion depth owns two fixed remainder slots, swapped out for
/// the recursive calls and returned intact, so a warmed descent over the
/// same geometry reuses each slot's buffer exactly.
fn descend_evaluate<F: FieldKernels>(
    node: usize,
    low: usize,
    high: usize,
    value: &Polynomial<F>,
    depth: usize,
    scratch: &mut MultipointScratch<F>,
    values: &mut [F::Elem],
) -> Result<(), PolynomialError> {
    if high - low == 1 {
        values[low] = value.coefficient(0);
        return Ok(());
    }
    let middle = low + (high - low) / 2;
    let (left_child, right_child) = (
        scratch.tree.nodes()[node].left,
        scratch.tree.nodes()[node].right,
    );
    ensure_remainder_slots(scratch, 2 * depth + 3)?;
    let mut left = core::mem::take(&mut scratch.remainders[2 * depth + 1]);
    let mut right = core::mem::take(&mut scratch.remainders[2 * depth + 2]);
    value.div_rem_into(
        &scratch.tree.nodes()[left_child].polynomial,
        &mut scratch.quotient,
        &mut left,
    )?;
    value.div_rem_into(
        &scratch.tree.nodes()[right_child].polynomial,
        &mut scratch.quotient,
        &mut right,
    )?;
    descend_evaluate(left_child, low, middle, &left, depth + 1, scratch, values)?;
    descend_evaluate(
        right_child,
        middle,
        high,
        &right,
        depth + 1,
        scratch,
        values,
    )?;
    scratch.remainders[2 * depth + 1] = left;
    scratch.remainders[2 * depth + 2] = right;
    Ok(())
}

/// Ensure the fixed remainder-slot vector covers `count` slots.
fn ensure_remainder_slots<F: FieldKernels>(
    scratch: &mut MultipointScratch<F>,
    count: usize,
) -> Result<(), PolynomialError> {
    if scratch.remainders.len() < count {
        scratch
            .remainders
            .try_reserve(count - scratch.remainders.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "multipoint remainder slots",
                elements: count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
        while scratch.remainders.len() < count {
            scratch.remainders.push(Polynomial::zero());
        }
    }
    Ok(())
}

/// Divide-and-conquer combination of `Σ c_i · M / (X + α_i)`.
fn combine_interpolant<F: FieldKernels>(
    node: usize,
    low: usize,
    high: usize,
    scaled: &[F::Elem],
    tree: &[ProductNode<F>],
) -> Result<Polynomial<F>, PolynomialError> {
    if high - low == 1 {
        return Polynomial::constant(scaled[low]);
    }
    let middle = low + (high - low) / 2;
    let (left_child, right_child) = (tree[node].left, tree[node].right);
    let left = combine_interpolant(left_child, low, middle, scaled, tree)?;
    let right = combine_interpolant(right_child, middle, high, scaled, tree)?;
    let mut combined = left.multiply(&tree[right_child].polynomial)?;
    combined.add_assign(&right.multiply(&tree[left_child].polynomial)?)?;
    Ok(combined)
}

fn reserve_values<E>(
    values: &mut Vec<E>,
    capacity: usize,
    context: &'static str,
) -> Result<(), PolynomialError> {
    if values.capacity() < capacity {
        values
            .try_reserve(capacity - values.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: capacity,
                element_size: core::mem::size_of::<E>(),
            })?;
    }
    Ok(())
}
