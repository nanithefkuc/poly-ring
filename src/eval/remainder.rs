//! The prepared remainder tree: fast remainders modulo arbitrary moduli.
//!
//! One polynomial (or a batch of lanes) reduced modulo every modulus in a
//! prepared list, by a single descent through the shared product-tree
//! topology. Each node divides by its own monic modulus through the Newton
//! short-division identity — reverse the dividend with an explicit length,
//! multiply by the cached reciprocal of the reversed divisor, reverse the
//! quotient, subtract `q·g` — so a descent costs quasi-linear multiplication
//! in the size of the input, never a long division and never a per-leaf
//! pass over the original polynomial.
//!
//! The construction pins every product geometry: each node's dividend bound
//! is known (`max_coefficients` at the root, the parent's modulus degree
//! below it), so the reciprocals, the broadcast row buffers, and the
//! transform plans are all prepared before the first execution and a warmed
//! `remainders_into` allocates nothing. Execution tracks the live dividend
//! length inside those bounds — a short dividend descends over its own rows,
//! never the structural geometry — and a single-lane descent multiplies the
//! prepared reciprocal and modulus rows in place, with no broadcast copy.

use alloc::vec;
use alloc::vec::Vec;

use fgf::ops;

use super::tree::{LEAF, TreePool};
use crate::error::{ConfigError, PolynomialError, ProductError};
use crate::geometry::checked_product;
use crate::poly::{ConvolutionScratch, Polynomial, PolynomialField, multiply_rows_into};

/// A prepared remainder tree over monic-normalized moduli.
///
/// Construction rejects a zero modulus, normalizes every nonzero modulus to
/// monic, and leaves constant moduli as empty output ranges that never
/// enter the tree. `leaf_offsets` are prefix sums of the moduli's degrees,
/// so output row `leaf_offsets[i] + j` is the degree-`j` coefficient of the
/// remainder modulo modulus `i`, zero-padded.
pub struct RemainderTree<F: PolynomialField> {
    pool: TreePool<F>,
    /// Original modulus index of each positive-degree leaf, in caller order.
    leaf_originals: Vec<usize>,
    /// Prefix sums of the moduli's degrees, length `moduli.len() + 1`.
    leaf_offsets: Vec<usize>,
    /// Per tree node: the dividend length bound (rows), zero for unused.
    bounds: Vec<usize>,
    /// Per tree node: the reciprocal of the reversed monic modulus, as
    /// single-lane packed rows zero-padded to the node's precision; empty
    /// when the copy path always applies.
    reciprocals: Vec<Vec<u8>>,
    /// Per tree node: the monic modulus as single-lane packed rows.
    modulus_rows: Vec<Vec<u8>>,
    max_coefficients: usize,
    total_rows: usize,
    max_depth: usize,
    /// Per node: its recursion depth (root 0).
    depths: Vec<usize>,
    /// Per depth: the largest dividend bound any node at that depth holds.
    depth_bounds: Vec<usize>,
    /// The geometry a compatible scratch must match: the largest reciprocal
    /// precision, modulus row count, and dividend bound over the tree.
    max_precision: usize,
    max_modulus_len: usize,
    max_bound: usize,
}

/// Reusable remainder-descent workspace.
///
/// Holds the per-depth dividend slots, the shared reversal/product/broadcast
/// row buffers, and the prepared convolution engine. All sizes are pinned to
/// the tree's bounds, so a warmed descent allocates nothing. Scratch holds
/// nothing modulus-value-dependent beyond what construction cached: two
/// trees with the same geometry can share one scratch.
pub struct RemainderScratch<F: PolynomialField> {
    max_batch: usize,
    max_coefficients: usize,
    convolution: ConvolutionScratch<F>,
    /// One dividend slot per depth, `bound(depth)` rows of lanes.
    slots: Vec<Vec<u8>>,
    /// Reversed padded dividend, `max_coefficients` rows.
    reversed: Vec<u8>,
    /// Quotient head, `max_precision` rows.
    head: Vec<u8>,
    /// Product `q·g`, `max_coefficients` rows.
    product: Vec<u8>,
    /// Broadcast reciprocal rows, `max_precision` rows.
    reciprocal_broadcast: Vec<u8>,
    /// Broadcast modulus rows, `max_modulus_len` rows.
    modulus_broadcast: Vec<u8>,
    /// The owning tree's geometry, validated before every descent: equal
    /// capacities alone do not make a scratch reusable across trees.
    depth_bounds: Vec<usize>,
    tree_precision: usize,
    tree_modulus_len: usize,
    tree_bound: usize,
    tree_rows: usize,
}

impl<F: PolynomialField> RemainderTree<F> {
    /// Prepare the remainder tree over `moduli`.
    ///
    /// # Errors
    ///
    /// Returns [`ProductError::Polynomial`] with
    /// [`PolynomialError::DivisionByZero`] for a zero modulus, and the
    /// geometry/allocation errors of the products and reciprocals.
    pub fn new(moduli: &[Polynomial<F>], max_coefficients: usize) -> Result<Self, ProductError> {
        if moduli.iter().any(Polynomial::is_zero) {
            return Err(ProductError::Polynomial(PolynomialError::DivisionByZero));
        }
        let mut leaf_offsets = Vec::with_capacity(moduli.len() + 1);
        let mut total_rows = 0_usize;
        leaf_offsets.push(0);
        for modulus in moduli {
            total_rows = total_rows
                .checked_add(modulus.degree().unwrap_or(0))
                .ok_or(ConfigError::GeometryOverflow {
                    context: "remainder tree output rows",
                })?;
            leaf_offsets.push(total_rows);
        }

        // Monic copies of the positive-degree moduli, in caller order;
        // constants contribute empty ranges and stay out of the tree.
        let monic: Vec<Polynomial<F>> = moduli.iter().map(Polynomial::monic).collect();
        let tree_slots: Vec<usize> = (0..moduli.len())
            .filter(|&index| monic[index].degree().is_some_and(|degree| degree > 0))
            .collect();
        let cumulative: Vec<usize> = {
            let mut profile = vec![0_usize; tree_slots.len() + 1];
            for (position, &index) in tree_slots.iter().enumerate() {
                profile[position + 1] = profile[position] + monic[index].coefficient_count() - 1;
            }
            profile
        };

        let mut pool = TreePool::new();
        pool.build(
            tree_slots.len(),
            &|index| cumulative[index],
            &mut |index, destination| {
                destination.assign_packed(monic[tree_slots[index]].as_packed())
            },
        )
        .map_err(ProductError::from)?;

        // Per-node bounds, reciprocals, and modulus rows, by traversal from
        // the root: the root sees the caller's full coefficient count, each
        // child sees the parent's modulus degree (a remainder is shorter
        // than the modulus it was reduced by... than the parent product).
        let node_count = pool.nodes().len();
        let mut bounds = vec![0_usize; node_count];
        let mut reciprocals = vec![Vec::<u8>::new(); node_count];
        let mut modulus_rows = vec![Vec::<u8>::new(); node_count];
        let mut depths = vec![0_usize; node_count];
        let mut max_depth = 0_usize;
        if node_count > 0 {
            let root = node_count - 1;
            Self::prepare_node(
                &pool,
                root,
                max_coefficients,
                0,
                &mut bounds,
                &mut reciprocals,
                &mut modulus_rows,
                &mut depths,
                &mut max_depth,
            )?;
        }
        let (depth_bounds, max_precision, max_modulus_len, max_bound) = Self::summarize_geometry(
            &bounds,
            &depths,
            &reciprocals,
            &modulus_rows,
            max_coefficients,
            max_depth,
        );

        Ok(Self {
            pool,
            leaf_originals: tree_slots,
            leaf_offsets,
            bounds,
            reciprocals,
            modulus_rows,
            max_coefficients,
            total_rows,
            max_depth,
            depths,
            depth_bounds,
            max_precision,
            max_modulus_len,
            max_bound,
        })
    }

    /// Per-depth slot bounds and the shared-buffer maxima a compatible
    /// scratch must match.
    fn summarize_geometry(
        bounds: &[usize],
        depths: &[usize],
        reciprocals: &[Vec<u8>],
        modulus_rows: &[Vec<u8>],
        max_coefficients: usize,
        max_depth: usize,
    ) -> (Vec<usize>, usize, usize, usize) {
        let mut depth_bounds = vec![0_usize; max_depth + 2];
        for node in 0..bounds.len() {
            let depth = depths[node];
            depth_bounds[depth] = depth_bounds[depth].max(bounds[node]);
            // The child slot at `depth + 1` stages this node's remainder:
            // `modulus_degree` rows even when the live dividend is shorter
            // (a leaf's degree can exceed `max_coefficients`).
            let modulus_degree = modulus_rows[node].len() / F::BYTES - 1;
            depth_bounds[depth + 1] = depth_bounds[depth + 1].max(modulus_degree);
        }
        // Slot 0 additionally holds the root dividend.
        depth_bounds[0] = depth_bounds[0].max(max_coefficients);
        for depth in 0..=max_depth + 1 {
            depth_bounds[depth] = depth_bounds[depth].max(depth_bounds[depth.saturating_sub(1)]);
        }
        let max_precision = reciprocals
            .iter()
            .map(|rows| rows.len() / F::BYTES)
            .max()
            .unwrap_or(0);
        let max_modulus_len = modulus_rows
            .iter()
            .map(|rows| rows.len() / F::BYTES)
            .max()
            .unwrap_or(0);
        let max_bound = bounds
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(max_coefficients);
        (depth_bounds, max_precision, max_modulus_len, max_bound)
    }

    /// Prefix sums of the moduli's degrees: leaf `i`'s remainder occupies
    /// output rows `leaf_offsets()[i]..leaf_offsets()[i + 1]`.
    #[must_use]
    pub fn leaf_offsets(&self) -> &[usize] {
        &self.leaf_offsets
    }

    /// Build scratch for batches of up to `batch_capacity` lanes.
    ///
    /// `scratch(0)` is a valid empty-batch capacity.
    ///
    /// # Errors
    ///
    /// Returns [`ProductError::Config`] when a workspace cannot be reserved.
    pub fn scratch(&self, batch_capacity: usize) -> Result<RemainderScratch<F>, ProductError> {
        let lanes = batch_capacity.max(1);
        let reserve = |rows: usize, context: &'static str| -> Result<Vec<u8>, ProductError> {
            let bytes = checked_product(context, rows, lanes)?
                .checked_mul(F::BYTES)
                .ok_or(ConfigError::GeometryOverflow { context })?;
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(bytes)
                .map_err(|_| ConfigError::AllocationFailed {
                    context,
                    elements: bytes,
                    element_size: 1,
                })?;
            buffer.resize(bytes, 0);
            Ok(buffer)
        };
        let max_precision = self.max_precision;
        let max_modulus_len = self.max_modulus_len;
        let max_bound = self.max_bound;
        let max_operand = max_bound.max(max_precision).max(max_modulus_len);
        let mut convolution = ConvolutionScratch::<F>::new(max_operand, max_operand, lanes)?;
        // Prepare every pinned geometry for this lane count (idempotent:
        // the constructor already caches the transform-size ladder).
        for node in 0..self.bounds.len() {
            let bound = self.bounds[node];
            let precision = self.reciprocals[node].len() / F::BYTES;
            if bound > 0 && precision > 0 {
                convolution.prepare_transform(bound + precision - 1, lanes)?;
                convolution.prepare_transform(bound, lanes)?;
            }
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(self.max_depth + 2)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "remainder descent slots",
                elements: self.max_depth + 2,
                element_size: 1,
            })?;
        for depth in 0..=self.max_depth + 1 {
            let bound = self.depth_bounds[depth.min(self.depth_bounds.len() - 1)];
            slots.push(reserve(bound.max(1), "remainder descent slot")?);
        }
        Ok(RemainderScratch {
            max_batch: batch_capacity,
            max_coefficients: self.max_coefficients,
            convolution,
            slots,
            reversed: reserve(max_bound.max(1), "remainder reversal rows")?,
            head: reserve(max_precision.max(1), "remainder quotient head")?,
            product: reserve(max_bound.max(1), "remainder product rows")?,
            reciprocal_broadcast: reserve(max_precision.max(1), "reciprocal broadcast rows")?,
            modulus_broadcast: reserve(max_modulus_len.max(1), "modulus broadcast rows")?,
            depth_bounds: self.depth_bounds.clone(),
            tree_precision: max_precision,
            tree_modulus_len: max_modulus_len,
            tree_bound: max_bound,
            tree_rows: self.total_rows,
        })
    }

    /// Write every modulus's remainder of the batched lanes into `output`.
    ///
    /// `coefficients` holds `coefficient_count` coefficient rows of `batch`
    /// lanes (coefficient-major, lane `l`'s coefficient `d` at
    /// `(d * batch + l) * F::BYTES`); `output` receives `total_rows` rows in
    /// the same layout. Batch zero with empty buffers is a valid empty
    /// geometry. All lengths and capacities are validated before `output`
    /// is touched.
    ///
    /// # Errors
    ///
    /// Returns [`ProductError::Config`] (`BufferLength`, `ScratchTooSmall`,
    /// `GeometryOverflow`) on a geometry violation — before any mutation.
    pub fn remainders_into(
        &self,
        coefficients: &[u8],
        coefficient_count: usize,
        batch: usize,
        scratch: &mut RemainderScratch<F>,
        output: &mut [u8],
    ) -> Result<(), ProductError> {
        let row_bytes = batch
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "remainder lane bytes",
            })?;
        let input_bytes =
            coefficient_count
                .checked_mul(row_bytes)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "remainder input rows",
                })?;
        if coefficients.len() != input_bytes {
            return Err(ProductError::Config(ConfigError::BufferLength {
                context: "remainder coefficients",
                expected: input_bytes,
                actual: coefficients.len(),
            }));
        }
        if coefficient_count > scratch.max_coefficients {
            return Err(ProductError::Config(ConfigError::ScratchTooSmall {
                context: "remainder coefficient capacity",
                required: coefficient_count,
                available: scratch.max_coefficients,
            }));
        }
        if batch > scratch.max_batch {
            return Err(ProductError::Config(ConfigError::ScratchTooSmall {
                context: "remainder lane capacity",
                required: batch,
                available: scratch.max_batch,
            }));
        }
        let output_bytes =
            self.total_rows
                .checked_mul(row_bytes)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "remainder output rows",
                })?;
        if output.len() != output_bytes {
            return Err(ProductError::Config(ConfigError::BufferLength {
                context: "remainder output",
                expected: output_bytes,
                actual: output.len(),
            }));
        }
        if self.total_rows == 0 || batch == 0 {
            return Ok(());
        }

        // Tree-dependent scratch geometry, validated in full before any
        // mutation: equal coefficient/lane capacities do not make a scratch
        // reusable across trees — the descent indexes per-depth slots and
        // shared buffers sized to this tree's bounds.
        if scratch.depth_bounds != self.depth_bounds
            || scratch.tree_precision != self.max_precision
            || scratch.tree_modulus_len != self.max_modulus_len
            || scratch.tree_bound != self.max_bound
            || scratch.tree_rows != self.total_rows
            || scratch.max_coefficients != self.max_coefficients
        {
            return Err(ProductError::Config(ConfigError::ScratchTooSmall {
                context: "remainder scratch geometry",
                required: self.max_bound,
                available: scratch.tree_bound,
            }));
        }

        // Stage the live dividend rows into slot 0; the descent reads no
        // row past `coefficient_count`.
        let root = self.pool.nodes().len() - 1;
        scratch.slots[0][..coefficient_count * row_bytes]
            .copy_from_slice(&coefficients[..coefficient_count * row_bytes]);

        self.descend(root, coefficient_count, batch, scratch, output);
        Ok(())
    }

    /// One node's reduction and recursion. `slot` holds the dividend,
    /// `length` live rows — at most the node's bound; the children's
    /// remainders land in `slot + 1`, and each child descends over the rows
    /// its remainder leaves live.
    fn descend(
        &self,
        node: usize,
        length: usize,
        batch: usize,
        scratch: &mut RemainderScratch<F>,
        output: &mut [u8],
    ) {
        let row_bytes = batch * F::BYTES;
        let modulus_len = self.modulus_rows[node].len() / F::BYTES;
        let modulus_degree = modulus_len - 1;
        let precision = self.reciprocals[node].len() / F::BYTES;
        let depth = self.depths[node];
        let dividend = core::mem::take(&mut scratch.slots[depth]);

        // The child slot receives this node's remainder: `modulus_degree`
        // rows, zero-padded.
        let child_slot = depth + 1;
        let child_bytes = modulus_degree * row_bytes;
        let child_length = if length < modulus_len || precision == 0 {
            // Copy path: the dividend is already shorter than the modulus.
            scratch.slots[child_slot][..child_bytes].fill(0);
            let kept = length.min(modulus_degree) * row_bytes;
            scratch.slots[child_slot][..kept].copy_from_slice(&dividend[..kept]);
            length.min(modulus_degree)
        } else {
            self.reduce_node(node, length, batch, &dividend, scratch);
            modulus_degree
        };
        scratch.slots[depth] = dividend;

        let (left, right, low) = {
            let entry = &self.pool.nodes()[node];
            (entry.left, entry.right, entry.low)
        };
        if left == LEAF {
            // Leaf: `low` indexes the positive-degree moduli in caller
            // order.
            let original = self.leaf_originals[low];
            let start = self.leaf_offsets[original] * row_bytes;
            output[start..start + modulus_degree * row_bytes]
                .copy_from_slice(&scratch.slots[child_slot][..modulus_degree * row_bytes]);
            return;
        }
        self.descend(left, child_length, batch, scratch, output);
        self.descend(right, child_length, batch, scratch, output);
    }

    /// One node's Newton short division: `dividend`'s live `length` rows
    /// reduced modulo the node's monic modulus, into the child slot.
    ///
    /// The live length drives every geometry: rows past `length` are zero
    /// and never enter a product, and the prepared reciprocal is used to
    /// the precision the live length asks for.
    fn reduce_node(
        &self,
        node: usize,
        length: usize,
        batch: usize,
        dividend: &[u8],
        scratch: &mut RemainderScratch<F>,
    ) {
        let row_bytes = batch * F::BYTES;
        let modulus_len = self.modulus_rows[node].len() / F::BYTES;
        let modulus_degree = modulus_len - 1;
        let child_slot = self.depths[node] + 1;
        let child_bytes = modulus_degree * row_bytes;
        let live_precision = length - modulus_len + 1;
        debug_assert!(live_precision <= self.reciprocals[node].len() / F::BYTES);
        // 1. Reverse the live dividend rows.
        let reversed = &mut scratch.reversed[..length * row_bytes];
        for degree in 0..length {
            let source = (length - 1 - degree) * row_bytes;
            reversed[degree * row_bytes..(degree + 1) * row_bytes]
                .copy_from_slice(&dividend[source..source + row_bytes]);
        }
        // 2. Stage the operand images: at one lane the prepared reciprocal
        //    and modulus rows match the lane-row layout and are borrowed in
        //    place; larger batches broadcast them into the scratch row
        //    buffers.
        if batch > 1 {
            let reciprocal = &self.reciprocals[node];
            let broadcast = &mut scratch.reciprocal_broadcast[..live_precision * row_bytes];
            for degree in 0..live_precision {
                for lane in 0..batch {
                    let source = degree * F::BYTES;
                    let target = degree * row_bytes + lane * F::BYTES;
                    broadcast[target..target + F::BYTES]
                        .copy_from_slice(&reciprocal[source..source + F::BYTES]);
                }
            }
            let modulus = &self.modulus_rows[node];
            let modulus_broadcast = &mut scratch.modulus_broadcast[..modulus_len * row_bytes];
            for degree in 0..modulus_len {
                for lane in 0..batch {
                    let source = degree * F::BYTES;
                    let target = degree * row_bytes + lane * F::BYTES;
                    modulus_broadcast[target..target + F::BYTES]
                        .copy_from_slice(&modulus[source..source + F::BYTES]);
                }
            }
        }
        let (reciprocal_rows, modulus_image): (&[u8], &[u8]) = if batch == 1 {
            (
                &self.reciprocals[node][..live_precision * F::BYTES],
                &self.modulus_rows[node],
            )
        } else {
            (
                &scratch.reciprocal_broadcast[..live_precision * row_bytes],
                &scratch.modulus_broadcast[..modulus_len * row_bytes],
            )
        };
        let head = &mut scratch.head[..live_precision * row_bytes];
        multiply_rows_into::<F>(
            head,
            reversed,
            length,
            reciprocal_rows,
            live_precision,
            batch,
            live_precision,
            &mut scratch.convolution,
        )
        .expect("prepared reciprocal product geometry");
        // 3. Reverse the head over its live rows: the quotient.
        for degree in 0..live_precision / 2 {
            let (low, high) = (
                degree * row_bytes,
                (live_precision - 1 - degree) * row_bytes,
            );
            for offset in 0..row_bytes {
                head.swap(low + offset, high + offset);
            }
        }
        // 4. q·g, truncated to the modulus degree: the remainder reads only
        //    these low rows.
        let product = &mut scratch.product[..child_bytes];
        multiply_rows_into::<F>(
            product,
            head,
            live_precision,
            modulus_image,
            modulus_len,
            batch,
            modulus_degree,
            &mut scratch.convolution,
        )
        .expect("prepared quotient product geometry");
        // 5. remainder = dividend − q·g, over the low modulus-degree rows
        //    only and into the child slot: the dividend buffer must survive
        //    intact for the sibling subtree.
        scratch.slots[child_slot][..child_bytes].copy_from_slice(&dividend[..child_bytes]);
        ops::sub_assign::<F>(
            &mut scratch.slots[child_slot][..child_bytes],
            &product[..child_bytes],
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_node(
        pool: &TreePool<F>,
        node: usize,
        bound: usize,
        depth: usize,
        bounds: &mut [usize],
        reciprocals: &mut Vec<Vec<u8>>,
        modulus_rows: &mut Vec<Vec<u8>>,
        depths: &mut [usize],
        max_depth: &mut usize,
    ) -> Result<(), ProductError> {
        depths[node] = depth;
        *max_depth = (*max_depth).max(depth);
        let polynomial = &pool.nodes()[node].polynomial;
        let modulus_len = polynomial.coefficient_count();
        bounds[node] = bound;
        modulus_rows[node] = polynomial.as_packed().to_vec();
        let precision = bound.saturating_sub(modulus_len).saturating_add(1);
        if precision > 0 && bound >= modulus_len {
            // Reverse the monic modulus with an explicit length: its
            // constant term is the (unit) leading coefficient, so the
            // series inverse exists.
            let mut reversed = Vec::new();
            reversed
                .try_reserve_exact(modulus_len * F::BYTES)
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "reversed modulus rows",
                    elements: modulus_len,
                    element_size: F::BYTES,
                })?;
            for coefficient in polynomial.coefficients().rev() {
                let mut encoded = [0_u8; 16];
                F::encode(&mut encoded[..F::BYTES], coefficient);
                reversed.extend_from_slice(&encoded[..F::BYTES]);
            }
            let reversed_poly =
                Polynomial::<F>::from_packed(reversed).ok_or(ConfigError::GeometryOverflow {
                    context: "reversed modulus alignment",
                })?;
            let inverse = reversed_poly
                .inverse_mod_x_power(precision)
                .map_err(ProductError::from)?;
            let mut rows = Vec::new();
            rows.try_reserve_exact(precision * F::BYTES).map_err(|_| {
                ConfigError::AllocationFailed {
                    context: "divisor reciprocal rows",
                    elements: precision,
                    element_size: F::BYTES,
                }
            })?;
            rows.resize(precision * F::BYTES, 0);
            let packed = inverse.as_packed();
            rows[..packed.len()].copy_from_slice(packed);
            reciprocals[node] = rows;
        }
        let (left, right) = (pool.nodes()[node].left, pool.nodes()[node].right);
        if left != LEAF {
            // A remainder is shorter than the modulus that produced it, so
            // each child's dividend bound is this node's modulus degree.
            let child_bound = modulus_len - 1;
            Self::prepare_node(
                pool,
                left,
                child_bound,
                depth + 1,
                bounds,
                reciprocals,
                modulus_rows,
                depths,
                max_depth,
            )?;
            Self::prepare_node(
                pool,
                right,
                child_bound,
                depth + 1,
                bounds,
                reciprocals,
                modulus_rows,
                depths,
                max_depth,
            )?;
        }
        Ok(())
    }
}

impl<F: PolynomialField> core::fmt::Debug for RemainderTree<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RemainderTree")
            .field("moduli", &(self.leaf_offsets.len() - 1))
            .field("total_rows", &self.total_rows)
            .field("max_coefficients", &self.max_coefficients)
            .field("max_depth", &self.max_depth)
            .finish_non_exhaustive()
    }
}

impl<F: PolynomialField> core::fmt::Debug for RemainderScratch<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RemainderScratch")
            .field("max_batch", &self.max_batch)
            .field("max_coefficients", &self.max_coefficients)
            .finish_non_exhaustive()
    }
}
