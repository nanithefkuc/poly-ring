//! Power-series root lifting: Roth–Ruckenstein and Alekhnovich.
//!
//! Both backends find every polynomial `f` of degree at most `max_degree`
//! satisfying `Q(X, f(X)) == 0`, where `Q` is supplied as its low-to-high
//! `Y`-coefficient rows `Q_j(X)` — the representation a bivariate owner
//! (e.g. `gs-engine`'s interpolation layer) already carries. The row-slice
//! algebra lives here; the caller keeps the bivariate type.
//!
//! Roth–Ruckenstein lifts coefficient prefixes iteratively, reusing pooled
//! scratch so a warmed extraction allocates nothing. Alekhnovich solves at
//! half precision first, transforms `Q(X, prefix + X^d Y)`, and finishes
//! only the surviving residual precision; it engages above the measured
//! weighted-size crossover
//! ([`DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER`]) and composes the AFFT batched
//! product through `butterfly-fft` (hence its `fft` feature gate).

use alloc::vec::Vec;
use core::cmp::Ordering;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

#[cfg(feature = "fft")]
use butterfly_fft::core::kernel::ButterflyKernels;

#[cfg(feature = "fft")]
use crate::error::ProductError;
use crate::error::{ConfigError, RootError};
use crate::poly::Polynomial;
#[cfg(feature = "fft")]
use crate::poly::PolynomialProductScratch;
use crate::poly::binomial_odd;
#[cfg(feature = "fft")]
use crate::poly::substitute_y_affine_rows_truncated_into;

use super::equal_degree::{FieldRootScratch, base_field_roots_into, element_key};

/// Caller-provided limits for Roth–Ruckenstein prefix lifting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RothRuckensteinLimits {
    max_work_items: usize,
    max_output_roots: usize,
}

impl RothRuckensteinLimits {
    /// Construct limits on transformed prefix nodes and returned roots.
    #[must_use]
    pub const fn new(max_work_items: usize, max_output_roots: usize) -> Self {
        Self {
            max_work_items,
            max_output_roots,
        }
    }

    /// Maximum number of transformed prefix nodes created during extraction.
    #[must_use]
    pub const fn max_work_items(self) -> usize {
        self.max_work_items
    }

    /// Maximum number of distinct verified output roots.
    #[must_use]
    pub const fn max_output_roots(self) -> usize {
        self.max_output_roots
    }
}

/// Caller-owned reusable storage for Roth–Ruckenstein prefix lifting.
///
/// The frame stack, transformed-node rows, base-field factorization, and
/// candidate polynomials are all recycled through internal pools, so a
/// warmed extraction over a changed input performs no heap allocation.
#[derive(Debug)]
pub struct RothRuckensteinScratch<F: FieldKernels> {
    field_roots: FieldRootScratch<F>,
    prefix: Vec<F::Elem>,
    frames: Vec<Frame<F>>,
    frame_pool: Vec<Frame<F>>,
    row_pool: Vec<Polynomial<F>>,
    sub_powers: Vec<F::Elem>,
    shifted: Vec<Polynomial<F>>,
    constant_coeffs: Vec<F::Elem>,
    constant_y: Polynomial<F>,
    compose_acc: Polynomial<F>,
    compose_product: Polynomial<F>,
    candidate: Polynomial<F>,
    candidate_pool: Vec<Polynomial<F>>,
}

impl<F: FieldKernels> RothRuckensteinScratch<F> {
    /// Construct empty reusable lifting scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            field_roots: FieldRootScratch::new(),
            prefix: Vec::new(),
            frames: Vec::new(),
            frame_pool: Vec::new(),
            row_pool: Vec::new(),
            sub_powers: Vec::new(),
            shifted: Vec::new(),
            constant_coeffs: Vec::new(),
            constant_y: Polynomial::zero(),
            compose_acc: Polynomial::zero(),
            compose_product: Polynomial::zero(),
            candidate: Polynomial::zero(),
            candidate_pool: Vec::new(),
        }
    }

    /// Retained frame-stack and pool capacity available to a subsequent lift.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.frames.capacity()
            + self.frame_pool.capacity()
            + self.row_pool.capacity()
            + self.candidate_pool.capacity()
            + self.field_roots.capacity()
    }

    fn recycle_frames(&mut self) {
        while let Some(mut frame) = self.frames.pop() {
            frame.roots.clear();
            recycle_rows(&mut frame.transformed, &mut self.row_pool);
            self.frame_pool.push(frame);
        }
    }
}

impl<F: FieldKernels> Default for RothRuckensteinScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct Frame<F: FieldKernels> {
    transformed: Vec<Polynomial<F>>,
    roots: Vec<F::Elem>,
    next_root: usize,
    depth: usize,
}

impl<F: FieldKernels> Frame<F> {
    const fn empty() -> Self {
        Self {
            transformed: Vec::new(),
            roots: Vec::new(),
            next_root: 0,
            depth: 0,
        }
    }

    fn next_root(&mut self) -> Option<F::Elem> {
        let root = self.roots.get(self.next_root).copied()?;
        self.next_root += 1;
        Some(root)
    }
}

/// Find every polynomial `f` of degree at most `max_degree` satisfying
/// `Q(X, f(X)) == 0`, where `rows[j]` is the `Y^j` coefficient polynomial.
///
/// The traversal is iterative. One fixed coefficient-prefix buffer is reused
/// by every branch, while the explicit frame stack contains at most one
/// transformed row set per coefficient depth.
///
/// # Errors
///
/// Returns [`RootError`] when supporting arithmetic fails, the rows are
/// empty, or a caller-provided limit is reached.
pub fn roth_ruckenstein_roots<F: FieldKernels>(
    rows: &[Polynomial<F>],
    max_degree: usize,
    limits: RothRuckensteinLimits,
) -> Result<Vec<Polynomial<F>>, RootError> {
    let mut scratch = RothRuckensteinScratch::new();
    let mut output = Vec::new();
    roth_ruckenstein_roots_into(rows, max_degree, limits, &mut scratch, &mut output)?;
    Ok(output)
}

/// Write every bounded-degree polynomial root into reusable `output` storage.
///
/// Existing `output` entries are recycled into an internal pool first. After
/// a warm-up over the same geometry, alternating a changed `rows` input
/// performs no heap allocation.
///
/// # Errors
///
/// Returns [`RootError`] when supporting arithmetic fails, the rows are
/// empty, or a caller-provided limit is reached.
///
/// # Panics
///
/// The internal frame-stack expectations hold for the geometry this
/// function constructs itself.
// A faithful lift of gs-engine's iterative prefix lifting; the explicit
// frame loop reads as one traversal.
#[allow(clippy::too_many_lines)]
pub fn roth_ruckenstein_roots_into<F: FieldKernels>(
    rows: &[Polynomial<F>],
    max_degree: usize,
    limits: RothRuckensteinLimits,
    scratch: &mut RothRuckensteinScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), RootError> {
    scratch.recycle_frames();
    while let Some(mut candidate) = output.pop() {
        candidate.set_zero();
        scratch.candidate_pool.push(candidate);
    }
    if rows.is_empty() {
        return Err(RootError::ZeroBivariatePolynomial);
    }
    let coefficient_count = max_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Roth–Ruckenstein coefficient count",
        })?;
    let y_degree = rows
        .len()
        .checked_sub(1)
        .ok_or(RootError::ZeroBivariatePolynomial)?;
    if y_degree == 0 {
        return Ok(());
    }
    enforce_limit("Roth–Ruckenstein work items", 1, limits.max_work_items)?;

    if scratch.prefix.len() < coefficient_count {
        scratch
            .prefix
            .try_reserve(coefficient_count - scratch.prefix.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "Roth–Ruckenstein coefficient prefix",
                elements: coefficient_count,
                element_size: core::mem::size_of::<F::Elem>(),
            })?;
        scratch.prefix.resize(coefficient_count, F::Elem::ZERO);
    }
    scratch.prefix[..coefficient_count].fill(F::Elem::ZERO);
    let initial_valuation = rows_x_valuation(rows).ok_or(RootError::ZeroBivariatePolynomial)?;
    let mut initial = scratch.frame_pool.pop().unwrap_or_else(Frame::empty);
    rows_divide_by_x_power_into(
        rows,
        initial_valuation,
        &mut scratch.row_pool,
        &mut initial.transformed,
    )?;
    fill_frame_roots(
        &initial.transformed,
        &mut scratch.constant_coeffs,
        &mut scratch.constant_y,
        &mut scratch.field_roots,
        &mut initial.roots,
    )?;
    initial.depth = 0;
    initial.next_root = 0;
    if initial.roots.is_empty() {
        initial.roots.clear();
        recycle_rows(&mut initial.transformed, &mut scratch.row_pool);
        scratch.frame_pool.push(initial);
        return Ok(());
    }
    if scratch.frames.capacity() < coefficient_count {
        let additional = coefficient_count - scratch.frames.capacity();
        scratch
            .frames
            .try_reserve(additional)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "Roth–Ruckenstein frame stack",
                elements: coefficient_count,
                element_size: core::mem::size_of::<Frame<F>>(),
            })?;
    }
    scratch.frames.push(initial);
    let mut work_items = 1_usize;

    loop {
        // The next pending root of the deepest frame, or the frame is
        // exhausted and recycled.
        let (root, depth) = match scratch.frames.last_mut() {
            // The stack empties exactly when the traversal is done.
            None => break,
            #[allow(clippy::single_match_else)]
            Some(frame) => match frame.next_root() {
                Some(root) => (root, frame.depth),
                None => {
                    let mut done = scratch.frames.pop().expect("nonempty frame stack");
                    done.roots.clear();
                    recycle_rows(&mut done.transformed, &mut scratch.row_pool);
                    scratch.frame_pool.push(done);
                    continue;
                }
            },
        };
        scratch.prefix[depth] = root;

        if depth + 1 == coefficient_count {
            scratch
                .candidate
                .assign_coefficients(&scratch.prefix[..coefficient_count])?;
            let is_root = rows_has_root_with(
                rows,
                &scratch.candidate,
                &mut scratch.compose_acc,
                &mut scratch.compose_product,
            )?;
            if is_root && !output.iter().any(|existing| existing == &scratch.candidate) {
                if output.len() >= y_degree {
                    return Err(RootError::FactorizationInvariant {
                        reason: "verified polynomial roots exceed the bivariate Y-degree",
                    });
                }
                enforce_limit(
                    "Roth–Ruckenstein output roots",
                    output.len() + 1,
                    limits.max_output_roots,
                )?;
                let mut buffer = scratch.candidate_pool.pop().unwrap_or_default();
                buffer.assign_from(&scratch.candidate);
                output.push(buffer);
            }
            continue;
        }

        let required_work_items =
            work_items
                .checked_add(1)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "Roth–Ruckenstein work item count",
                })?;
        enforce_limit(
            "Roth–Ruckenstein work items",
            required_work_items,
            limits.max_work_items,
        )?;

        let index = scratch.frames.len() - 1;
        rows_substitute_y_linear_into(
            &scratch.frames[index].transformed,
            root,
            &mut scratch.sub_powers,
            &mut scratch.row_pool,
            &mut scratch.shifted,
        )?;
        let valuation =
            rows_x_valuation(&scratch.shifted).ok_or(RootError::FactorizationInvariant {
                reason: "a nonzero Y substitution produced zero",
            })?;
        let mut child = scratch.frame_pool.pop().unwrap_or_else(Frame::empty);
        rows_divide_by_x_power_into(
            &scratch.shifted,
            valuation,
            &mut scratch.row_pool,
            &mut child.transformed,
        )?;
        fill_frame_roots(
            &child.transformed,
            &mut scratch.constant_coeffs,
            &mut scratch.constant_y,
            &mut scratch.field_roots,
            &mut child.roots,
        )?;
        child.depth = depth + 1;
        child.next_root = 0;
        work_items = required_work_items;
        if child.roots.is_empty() {
            child.roots.clear();
            recycle_rows(&mut child.transformed, &mut scratch.row_pool);
            scratch.frame_pool.push(child);
        } else {
            scratch.frames.push(child);
        }
    }

    output.sort_by(|left, right| compare_polynomials::<F>(left, right));
    output.dedup();
    if output.len() > y_degree {
        return Err(RootError::FactorizationInvariant {
            reason: "deduplicated polynomial roots exceed the bivariate Y-degree",
        });
    }
    for candidate in output.iter() {
        if !rows_has_root_with(
            rows,
            candidate,
            &mut scratch.compose_acc,
            &mut scratch.compose_product,
        )? {
            return Err(RootError::FactorizationInvariant {
                reason: "the final candidate list contains a nonroot",
            });
        }
    }
    Ok(())
}

fn fill_frame_roots<F: FieldKernels>(
    transformed: &[Polynomial<F>],
    coeffs: &mut Vec<F::Elem>,
    constant_y: &mut Polynomial<F>,
    field_roots: &mut FieldRootScratch<F>,
    roots: &mut Vec<F::Elem>,
) -> Result<(), RootError> {
    constant_y_polynomial_into(transformed, coeffs, constant_y)?;
    if base_field_roots_into(constant_y, field_roots, roots)? {
        return Err(RootError::FactorizationInvariant {
            reason: "an X-normalized transformed polynomial has zero constant-X row",
        });
    }
    Ok(())
}

/// Extract the constant-`X` coefficient of each `Y` row into `out`.
fn constant_y_polynomial_into<F: FieldKernels>(
    rows: &[Polynomial<F>],
    coeffs: &mut Vec<F::Elem>,
    out: &mut Polynomial<F>,
) -> Result<(), RootError> {
    coeffs.clear();
    let count = rows.len();
    if coeffs.capacity() < count {
        coeffs.try_reserve(count - coeffs.capacity()).map_err(|_| {
            ConfigError::AllocationFailed {
                context: "Roth–Ruckenstein constant-X polynomial",
                elements: count,
                element_size: core::mem::size_of::<F::Elem>(),
            }
        })?;
    }
    for row in rows {
        coeffs.push(row.coefficient(0));
    }
    out.assign_coefficients(coeffs)?;
    if out.is_zero() {
        Err(RootError::FactorizationInvariant {
            reason: "an X-normalized polynomial yielded a zero constant-X polynomial",
        })
    } else {
        Ok(())
    }
}

/// Order polynomial candidates deterministically: lexicographically by
/// canonical little-endian element key from the low degree up over the
/// shared prefix, then by coefficient count.
pub(super) fn compare_polynomials<F: FieldKernels>(
    left: &Polynomial<F>,
    right: &Polynomial<F>,
) -> Ordering {
    let shared = left.coefficient_count().min(right.coefficient_count());
    for degree in 0..shared {
        let ordering = element_key::<F>(left.coefficient(degree))
            .cmp(&element_key::<F>(right.coefficient(degree)));
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.coefficient_count().cmp(&right.coefficient_count())
}

pub(super) fn enforce_limit(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), RootError> {
    if required > limit {
        Err(RootError::ResourceLimitExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

/// Recycle every row of `rows` into `pool`, leaving it empty.
pub(super) fn recycle_rows<F: FieldKernels>(
    rows: &mut Vec<Polynomial<F>>,
    pool: &mut Vec<Polynomial<F>>,
) {
    while let Some(mut row) = rows.pop() {
        row.set_zero();
        pool.push(row);
    }
}

/// Drop trailing zero rows into `pool` without freeing row buffers.
fn normalize_rows<F: FieldKernels>(rows: &mut Vec<Polynomial<F>>, pool: &mut Vec<Polynomial<F>>) {
    while rows.last().is_some_and(Polynomial::is_zero) {
        let mut row = rows.pop().expect("nonempty rows");
        row.set_zero();
        pool.push(row);
    }
}

/// Ensure `rows` holds exactly `count` zeroed rows, recycling spare rows
/// through `pool` and drawing missing ones from it.
fn prepare_rows<F: FieldKernels>(
    rows: &mut Vec<Polynomial<F>>,
    count: usize,
    pool: &mut Vec<Polynomial<F>>,
) -> Result<(), RootError> {
    recycle_rows(rows, pool);
    if rows.capacity() < count {
        rows.try_reserve(count)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "lifting rows",
                elements: count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
    }
    for _ in 0..count {
        let mut row = pool.pop().unwrap_or_default();
        row.set_zero();
        rows.push(row);
    }
    Ok(())
}

/// Minimum `X` valuation shared by all nonzero rows.
fn rows_x_valuation<F: FieldKernels>(rows: &[Polynomial<F>]) -> Option<usize> {
    rows.iter().filter_map(Polynomial::x_valuation).min()
}

/// Maximum `(1, y_weight)` weighted degree across the rows.
#[cfg(feature = "fft")]
fn rows_weighted_degree<F: FieldKernels>(
    rows: &[Polynomial<F>],
    y_weight: usize,
) -> Result<Option<usize>, RootError> {
    let mut leading = None;
    for (y_degree, row) in rows.iter().enumerate() {
        let Some(x_degree) = row.degree() else {
            continue;
        };
        let weighted = y_degree
            .checked_mul(y_weight)
            .and_then(|weight| weight.checked_add(x_degree))
            .ok_or(ConfigError::GeometryOverflow {
                context: "bivariate weighted degree",
            })?;
        leading = Some(leading.unwrap_or(0).max(weighted));
    }
    Ok(leading)
}

/// Divide every row exactly by `X^power` into pool-backed output rows.
fn rows_divide_by_x_power_into<F: FieldKernels>(
    rows: &[Polynomial<F>],
    power: usize,
    pool: &mut Vec<Polynomial<F>>,
    out: &mut Vec<Polynomial<F>>,
) -> Result<(), RootError> {
    prepare_rows(out, rows.len(), pool)?;
    for (destination, source) in out.iter_mut().zip(rows) {
        source.divide_by_x_power_into(power, destination)?;
    }
    normalize_rows(out, pool);
    Ok(())
}

/// Substitute `Y = constant + X * Z` into the rows, writing pooled output.
fn rows_substitute_y_linear_into<F: FieldKernels>(
    rows: &[Polynomial<F>],
    constant: F::Elem,
    powers: &mut Vec<F::Elem>,
    pool: &mut Vec<Polynomial<F>>,
    out: &mut Vec<Polynomial<F>>,
) -> Result<(), RootError> {
    let Some(y_degree) = rows.len().checked_sub(1) else {
        recycle_rows(out, pool);
        return Ok(());
    };
    let count = y_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "bivariate substitution rows",
        })?;
    if powers.len() < count {
        powers
            .try_reserve(count - powers.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "bivariate substitution powers",
                elements: count,
                element_size: core::mem::size_of::<F::Elem>(),
            })?;
        powers.resize(count, F::Elem::ZERO);
    }
    powers[0] = F::Elem::ONE;
    for exponent in 1..count {
        powers[exponent] = powers[exponent - 1].mul(constant);
    }
    prepare_rows(out, count, pool)?;
    for (source_y, coefficient) in rows.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        for target_y in 0..=source_y {
            if !binomial_odd(source_y, target_y) {
                continue;
            }
            let scale = powers[source_y - target_y];
            if scale.is_zero() {
                continue;
            }
            out[target_y].add_scaled_shifted_assign(scale, coefficient, target_y)?;
        }
    }
    normalize_rows(out, pool);
    Ok(())
}

/// Evaluate `Q(X, candidate(X))` into reusable storage, returning whether
/// the composition is zero.
fn rows_has_root_with<F: FieldKernels>(
    rows: &[Polynomial<F>],
    candidate: &Polynomial<F>,
    accumulator: &mut Polynomial<F>,
    product: &mut Polynomial<F>,
) -> Result<bool, RootError> {
    accumulator.set_zero();
    for coefficient in rows.iter().rev() {
        accumulator.multiply_into(candidate, product)?;
        core::mem::swap(accumulator, product);
        accumulator.add_assign(coefficient)?;
    }
    Ok(accumulator.is_zero())
}

// --- Alekhnovich divide-and-conquer (composes butterfly-fft) -------------

/// Default weighted-coefficient crossover to Alekhnovich divide-and-conquer.
///
/// Roth–Ruckenstein is used at or below this weighted input size (and always
/// on a scalar backend unless overridden). Set explicitly with
/// [`AlekhnovichLimits::with_roth_ruckenstein_crossover`]. See
/// `BENCHMARKS.md`.
#[cfg(feature = "fft")]
pub const DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER: usize = 20_000;

/// Caller-provided bounds for Alekhnovich root extraction.
#[cfg(feature = "fft")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlekhnovichLimits {
    max_work_items: usize,
    max_intermediate_families: usize,
    max_coefficients: usize,
    max_scratch_bytes: usize,
    max_output_roots: usize,
    roth_ruckenstein_crossover: usize,
    backend_adaptive_crossover: bool,
}

#[cfg(feature = "fft")]
impl AlekhnovichLimits {
    /// Construct extraction limits.
    #[must_use]
    pub const fn new(
        max_work_items: usize,
        max_intermediate_families: usize,
        max_coefficients: usize,
        max_scratch_bytes: usize,
        max_output_roots: usize,
    ) -> Self {
        Self {
            max_work_items,
            max_intermediate_families,
            max_coefficients,
            max_scratch_bytes,
            max_output_roots,
            roth_ruckenstein_crossover: DEFAULT_ROTH_RUCKENSTEIN_CROSSOVER,
            backend_adaptive_crossover: true,
        }
    }

    /// Override the weighted-size crossover to Roth–Ruckenstein.
    #[must_use]
    pub const fn with_roth_ruckenstein_crossover(mut self, crossover: usize) -> Self {
        self.roth_ruckenstein_crossover = crossover;
        self.backend_adaptive_crossover = false;
        self
    }

    /// Maximum number of explicit divide-and-conquer nodes.
    #[must_use]
    pub const fn max_work_items(self) -> usize {
        self.max_work_items
    }

    /// Maximum number of affine families materialized during extraction.
    #[must_use]
    pub const fn max_intermediate_families(self) -> usize {
        self.max_intermediate_families
    }

    /// Maximum cumulative coefficient capacity charged to an extraction.
    #[must_use]
    pub const fn max_coefficients(self) -> usize {
        self.max_coefficients
    }

    /// Maximum cumulative temporary storage charged to an extraction.
    #[must_use]
    pub const fn max_scratch_bytes(self) -> usize {
        self.max_scratch_bytes
    }

    /// Maximum number of distinct verified output roots.
    #[must_use]
    pub const fn max_output_roots(self) -> usize {
        self.max_output_roots
    }

    /// Weighted-size crossover at or below which Roth–Ruckenstein is used.
    #[must_use]
    pub const fn roth_ruckenstein_crossover(self) -> usize {
        self.roth_ruckenstein_crossover
    }
}

/// An affine family `prefix(X) + X^tail_degree h(X)` of power-series roots.
#[cfg(feature = "fft")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineRootFamily<F: ButterflyKernels> {
    prefix: Polynomial<F>,
    tail_degree: usize,
}

#[cfg(feature = "fft")]
impl<F: ButterflyKernels> AffineRootFamily<F> {
    fn new(mut prefix: Polynomial<F>, tail_degree: usize) -> Self {
        prefix.truncate(tail_degree);
        Self {
            prefix,
            tail_degree,
        }
    }

    /// Fixed low-degree prefix.
    #[must_use]
    pub fn prefix(&self) -> &Polynomial<F> {
        &self.prefix
    }

    /// First coefficient belonging to the free tail.
    #[must_use]
    pub const fn tail_degree(&self) -> usize {
        self.tail_degree
    }
}

/// Caller-owned reusable stack storage for Alekhnovich extraction.
#[cfg(feature = "fft")]
#[derive(Debug)]
pub struct AlekhnovichScratch<F: ButterflyKernels> {
    frames: Vec<DncFrame<F>>,
    completed: Option<Vec<AffineRootFamily<F>>>,
    products: PolynomialProductScratch<F>,
    transformed: Vec<Polynomial<F>>,
    row_pool: Vec<Polynomial<F>>,
    roth: RothRuckensteinScratch<F>,
    field_roots: FieldRootScratch<F>,
    constant_y: Polynomial<F>,
    constant_y_coeffs: Vec<F::Elem>,
    base_roots: Vec<F::Elem>,
}

#[cfg(feature = "fft")]
impl<F: ButterflyKernels> AlekhnovichScratch<F> {
    /// Construct empty reusable extraction scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            frames: Vec::new(),
            completed: None,
            products: PolynomialProductScratch::new(),
            transformed: Vec::new(),
            row_pool: Vec::new(),
            roth: RothRuckensteinScratch::new(),
            field_roots: FieldRootScratch::new(),
            constant_y: Polynomial::zero(),
            constant_y_coeffs: Vec::new(),
            base_roots: Vec::new(),
        }
    }

    /// Retained explicit-frame capacity available to a subsequent extraction.
    #[must_use]
    pub fn frame_capacity(&self) -> usize {
        self.frames.capacity()
    }

    /// Retained divide-and-conquer frame and Roth–Ruckenstein pool capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.frames.capacity() + self.roth.capacity() + self.field_roots.capacity()
    }

    fn clear(&mut self) {
        self.frames.clear();
        self.completed = None;
    }
}

#[cfg(feature = "fft")]
impl<F: ButterflyKernels> Default for AlekhnovichScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Find every polynomial `f` of degree at most `max_degree` satisfying
/// `Q(X, f(X)) == 0`, where `rows[j]` is the `Y^j` coefficient polynomial.
///
/// The extractor computes reduced affine prefix families with an explicit
/// divide-and-conquer work stack. Each recursive stage first solves at half
/// precision, then transforms `Q(X, prefix + X^d Y)` and solves only the
/// surviving residual precision. Every transformed row is truncated to the
/// precision of its node. Small weighted inputs use the configured
/// Roth–Ruckenstein crossover.
///
/// # Errors
///
/// Returns [`RootError`] when supporting arithmetic fails, the rows are
/// empty, or a caller-provided limit is reached.
#[cfg(feature = "fft")]
pub fn alekhnovich_roots<F: ButterflyKernels>(
    rows: &[Polynomial<F>],
    max_degree: usize,
    limits: AlekhnovichLimits,
    scratch: &mut AlekhnovichScratch<F>,
) -> Result<Vec<Polynomial<F>>, RootError> {
    let mut output = Vec::new();
    alekhnovich_roots_into(rows, max_degree, limits, scratch, &mut output)?;
    Ok(output)
}

/// Extract every bounded-degree root into reusable `output` storage.
///
/// Below the Roth–Ruckenstein crossover the extraction runs entirely on
/// pooled scratch and performs no allocation once warmed; larger inputs use
/// the divide-and-conquer path and replace `output`.
///
/// # Errors
///
/// Returns [`RootError`] when supporting arithmetic fails, the rows are
/// empty, or a caller-provided limit is reached.
#[cfg(feature = "fft")]
pub fn alekhnovich_roots_into<F: ButterflyKernels>(
    rows: &[Polynomial<F>],
    max_degree: usize,
    limits: AlekhnovichLimits,
    scratch: &mut AlekhnovichScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), RootError> {
    scratch.clear();
    let result = alekhnovich_roots_inner(rows, max_degree, limits, scratch, output);
    scratch.clear();
    result
}

#[cfg(feature = "fft")]
// A faithful lift of gs-engine's divide-and-conquer driver; the frame
// state machine is one traversal over its work stack.
#[allow(clippy::too_many_lines)]
fn alekhnovich_roots_inner<F: ButterflyKernels>(
    rows: &[Polynomial<F>],
    max_degree: usize,
    limits: AlekhnovichLimits,
    scratch: &mut AlekhnovichScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), RootError> {
    if rows.is_empty() {
        return Err(RootError::ZeroBivariatePolynomial);
    }
    let y_degree = rows
        .len()
        .checked_sub(1)
        .ok_or(RootError::ZeroBivariatePolynomial)?;
    if y_degree == 0 {
        output.clear();
        return Ok(());
    }
    let initial_valuation = rows_x_valuation(rows).ok_or(RootError::ZeroBivariatePolynomial)?;
    let poly_weighted_degree =
        rows_weighted_degree(rows, max_degree)?.ok_or(RootError::ZeroBivariatePolynomial)?;
    let composition_degree = poly_weighted_degree.checked_sub(initial_valuation).ok_or(
        ConfigError::GeometryOverflow {
            context: "Alekhnovich exact-composition precision",
        },
    )?;
    let precision = composition_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich exact-composition precision",
        })?;
    let weighted_size = precision
        .checked_mul(rows.len())
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich weighted input size",
        })?;

    enforce_limit(
        "Alekhnovich coefficients",
        weighted_size,
        limits.max_coefficients,
    )?;

    let root_backend = crate::cost::select_root(crate::cost::RootCostKey {
        weighted_coefficients: weighted_size,
        y_degree: rows.len(),
        target_precision: max_degree,
        backend: crate::cost::BackendClass::detect::<F>(),
        roth_ruckenstein_crossover: limits.roth_ruckenstein_crossover,
        backend_adaptive: limits.backend_adaptive_crossover,
    });
    if root_backend == crate::cost::RootBackend::RothRuckenstein {
        return roth_ruckenstein_roots_into(
            rows,
            max_degree,
            RothRuckensteinLimits::new(limits.max_work_items, limits.max_output_roots),
            &mut scratch.roth,
            output,
        );
    }

    // The exact-composition frame is only needed by the divide-and-conquer
    // path.
    let initial = rows_divide_by_x_power(rows, initial_valuation)?;
    let initial_bytes = weighted_size
        .checked_mul(F::BYTES)
        .and_then(|bytes| {
            initial
                .len()
                .checked_mul(core::mem::size_of::<Polynomial<F>>())
                .and_then(|row_bytes| bytes.checked_add(row_bytes))
        })
        .and_then(|bytes| bytes.checked_add(core::mem::size_of::<DncFrame<F>>()))
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich initial scratch bytes",
        })?;
    enforce_limit(
        "Alekhnovich scratch bytes",
        initial_bytes,
        limits.max_scratch_bytes,
    )?;

    let mut budget = Budget::new(weighted_size, initial_bytes);
    push_frame(
        scratch,
        DncFrame::new(initial, precision),
        &mut budget,
        limits,
    )?;

    while let Some(mut frame) = scratch.frames.pop() {
        match frame.state {
            FrameState::Enter => {
                if frame.precision == 1 {
                    let constant_y_count = frame.rows.len();
                    let splitter_coefficients = constant_y_count
                        .checked_mul(constant_y_count)
                        .and_then(|count| count.checked_mul(4))
                        .ok_or(ConfigError::GeometryOverflow {
                            context: "Alekhnovich scalar splitter coefficients",
                        })?;
                    let family_bytes = core::mem::size_of::<AffineRootFamily<F>>()
                        .checked_mul(frame.rows.len().saturating_sub(1))
                        .ok_or(ConfigError::GeometryOverflow {
                            context: "Alekhnovich scalar family bytes",
                        })?;
                    budget.charge_materialization::<F>(
                        splitter_coefficients,
                        family_bytes,
                        limits,
                    )?;
                    constant_y_polynomial_into(
                        &frame.rows,
                        &mut scratch.constant_y_coeffs,
                        &mut scratch.constant_y,
                    )?;
                    let all_field = base_field_roots_into(
                        &scratch.constant_y,
                        &mut scratch.field_roots,
                        &mut scratch.base_roots,
                    )?;
                    if all_field {
                        return Err(RootError::FactorizationInvariant {
                            reason: "an X-normalized Alekhnovich node has zero constant-X row",
                        });
                    }
                    let mut families = Vec::new();
                    reserve_exact::<AffineRootFamily<F>>(
                        &mut families,
                        scratch.base_roots.len(),
                        "Alekhnovich scalar root families",
                    )?;
                    for &root in &scratch.base_roots {
                        insert_family(
                            &mut families,
                            AffineRootFamily::new(Polynomial::constant(root)?, 1),
                            &mut budget,
                            limits,
                        )?;
                    }
                    finish_frame(scratch, families);
                } else {
                    let coarse_precision = frame.precision.div_ceil(2);
                    let coarse_coefficients =
                        frame.rows.len().checked_mul(coarse_precision).ok_or(
                            ConfigError::GeometryOverflow {
                                context: "Alekhnovich coarse coefficient bound",
                            },
                        )?;
                    budget.charge_materialization::<F>(
                        coarse_coefficients,
                        core::mem::size_of::<DncFrame<F>>(),
                        limits,
                    )?;
                    let coarse_rows = rows_truncated_x(&frame.rows, coarse_precision);
                    frame.state = FrameState::AwaitCoarse;
                    scratch.frames.push(frame);
                    push_frame(
                        scratch,
                        DncFrame::new(coarse_rows, coarse_precision),
                        &mut budget,
                        limits,
                    )?;
                }
            }
            FrameState::AwaitCoarse => {
                let coarse = take_completed(scratch)?;
                frame.state = FrameState::Refine {
                    coarse,
                    next: 0,
                    refined: Vec::new(),
                };
                scratch.frames.push(frame);
            }
            FrameState::Refine {
                coarse,
                mut next,
                mut refined,
            } => {
                let Some(family) = coarse.get(next).cloned() else {
                    finish_frame(scratch, refined);
                    continue;
                };
                next += 1;
                let transform_bound = frame.rows.len().checked_mul(frame.precision).ok_or(
                    ConfigError::GeometryOverflow {
                        context: "Alekhnovich transformed coefficient bound",
                    },
                )?;
                budget.charge_materialization::<F>(transform_bound, 0, limits)?;
                substitute_y_affine_rows_truncated_into(
                    &frame.rows,
                    family.prefix(),
                    family.tail_degree(),
                    frame.precision,
                    &mut scratch.products,
                    &mut scratch.transformed,
                    &mut scratch.row_pool,
                )
                .map_err(|error| match error {
                    ProductError::Config(error) => RootError::from(error),
                    other => RootError::Product(other),
                })?;
                let Some(valuation) = rows_x_valuation(&scratch.transformed) else {
                    insert_family(&mut refined, family, &mut budget, limits)?;
                    frame.state = FrameState::Refine {
                        coarse,
                        next,
                        refined,
                    };
                    scratch.frames.push(frame);
                    continue;
                };
                let coarse_precision = frame.precision.div_ceil(2);
                if valuation < coarse_precision {
                    return Err(RootError::FactorizationInvariant {
                        reason: "an affine family failed its established coarse precision",
                    });
                }
                if valuation >= frame.precision {
                    insert_family(&mut refined, family, &mut budget, limits)?;
                    frame.state = FrameState::Refine {
                        coarse,
                        next,
                        refined,
                    };
                    scratch.frames.push(frame);
                    continue;
                }
                let residual_precision = frame.precision - valuation;
                budget.charge_materialization::<F>(transform_bound, 0, limits)?;
                let residual = rows_divide_by_x_power(&scratch.transformed, valuation)?;
                frame.state = FrameState::AwaitTail {
                    coarse,
                    next,
                    refined,
                    family,
                };
                scratch.frames.push(frame);
                push_frame(
                    scratch,
                    DncFrame::new(residual, residual_precision),
                    &mut budget,
                    limits,
                )?;
            }
            FrameState::AwaitTail {
                coarse,
                next,
                mut refined,
                family,
            } => {
                let tails = take_completed(scratch)?;
                for tail in tails {
                    let tail_degree = family.tail_degree.checked_add(tail.tail_degree).ok_or(
                        ConfigError::GeometryOverflow {
                            context: "Alekhnovich affine tail degree",
                        },
                    )?;
                    budget.charge_materialization::<F>(tail_degree, 0, limits)?;
                    let mut prefix = family.prefix.clone();
                    prefix.add_assign(&tail.prefix().shifted(family.tail_degree)?)?;
                    insert_family(
                        &mut refined,
                        AffineRootFamily::new(prefix, tail_degree),
                        &mut budget,
                        limits,
                    )?;
                }
                frame.state = FrameState::Refine {
                    coarse,
                    next,
                    refined,
                };
                scratch.frames.push(frame);
            }
        }
    }

    let families = take_completed(scratch)?;
    *output = materialize_candidates(rows, max_degree, y_degree, families, limits)?;
    Ok(())
}

#[cfg(feature = "fft")]
#[derive(Debug)]
struct DncFrame<F: ButterflyKernels> {
    rows: Vec<Polynomial<F>>,
    precision: usize,
    state: FrameState<F>,
}

#[cfg(feature = "fft")]
impl<F: ButterflyKernels> DncFrame<F> {
    fn new(rows: Vec<Polynomial<F>>, precision: usize) -> Self {
        Self {
            rows,
            precision,
            state: FrameState::Enter,
        }
    }
}

#[cfg(feature = "fft")]
#[derive(Debug)]
enum FrameState<F: ButterflyKernels> {
    Enter,
    AwaitCoarse,
    Refine {
        coarse: Vec<AffineRootFamily<F>>,
        next: usize,
        refined: Vec<AffineRootFamily<F>>,
    },
    AwaitTail {
        coarse: Vec<AffineRootFamily<F>>,
        next: usize,
        refined: Vec<AffineRootFamily<F>>,
        family: AffineRootFamily<F>,
    },
}

/// Copy every row truncated to `X^coefficient_count` coefficients.
#[cfg(feature = "fft")]
fn rows_truncated_x<F: FieldKernels>(
    rows: &[Polynomial<F>],
    coefficient_count: usize,
) -> Vec<Polynomial<F>> {
    rows.iter()
        .map(|row| {
            let mut truncated = row.clone();
            truncated.truncate(coefficient_count);
            truncated
        })
        .collect()
}

/// Divide every row exactly by `X^power`, allocating the output.
#[cfg(feature = "fft")]
fn rows_divide_by_x_power<F: FieldKernels>(
    rows: &[Polynomial<F>],
    power: usize,
) -> Result<Vec<Polynomial<F>>, RootError> {
    let mut divided = Vec::new();
    divided
        .try_reserve_exact(rows.len())
        .map_err(|_| ConfigError::AllocationFailed {
            context: "bivariate X-power quotient",
            elements: rows.len(),
            element_size: core::mem::size_of::<Polynomial<F>>(),
        })?;
    for row in rows {
        divided.push(row.divide_by_x_power(power)?);
    }
    Ok(divided)
}

#[cfg(feature = "fft")]
struct Budget {
    work_items: usize,
    families: usize,
    coefficients: usize,
    scratch_bytes: usize,
}

#[cfg(feature = "fft")]
impl Budget {
    const fn new(coefficients: usize, scratch_bytes: usize) -> Self {
        Self {
            work_items: 0,
            families: 0,
            coefficients,
            scratch_bytes,
        }
    }

    fn charge_materialization<F: ButterflyKernels>(
        &mut self,
        coefficients: usize,
        structural_bytes: usize,
        limits: AlekhnovichLimits,
    ) -> Result<(), RootError> {
        let required_coefficients =
            self.coefficients
                .checked_add(coefficients)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "Alekhnovich cumulative coefficients",
                })?;
        enforce_limit(
            "Alekhnovich coefficients",
            required_coefficients,
            limits.max_coefficients,
        )?;
        let coefficient_bytes =
            coefficients
                .checked_mul(F::BYTES)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "Alekhnovich coefficient scratch bytes",
                })?;
        let required_bytes = self
            .scratch_bytes
            .checked_add(coefficient_bytes)
            .and_then(|bytes| bytes.checked_add(structural_bytes))
            .ok_or(ConfigError::GeometryOverflow {
                context: "Alekhnovich cumulative scratch bytes",
            })?;
        enforce_limit(
            "Alekhnovich scratch bytes",
            required_bytes,
            limits.max_scratch_bytes,
        )?;
        self.coefficients = required_coefficients;
        self.scratch_bytes = required_bytes;
        Ok(())
    }
}

#[cfg(feature = "fft")]
fn push_frame<F: ButterflyKernels>(
    scratch: &mut AlekhnovichScratch<F>,
    frame: DncFrame<F>,
    budget: &mut Budget,
    limits: AlekhnovichLimits,
) -> Result<(), RootError> {
    let required = budget
        .work_items
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich work item count",
        })?;
    enforce_limit("Alekhnovich work items", required, limits.max_work_items)?;
    scratch
        .frames
        .try_reserve(1)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "Alekhnovich explicit work stack",
            elements: scratch.frames.len() + 1,
            element_size: core::mem::size_of::<DncFrame<F>>(),
        })?;
    budget.work_items = required;
    scratch.frames.push(frame);
    Ok(())
}

#[cfg(feature = "fft")]
fn finish_frame<F: ButterflyKernels>(
    scratch: &mut AlekhnovichScratch<F>,
    mut families: Vec<AffineRootFamily<F>>,
) {
    families.sort_by(compare_families::<F>);
    scratch.completed = Some(families);
}

#[cfg(feature = "fft")]
fn take_completed<F: ButterflyKernels>(
    scratch: &mut AlekhnovichScratch<F>,
) -> Result<Vec<AffineRootFamily<F>>, RootError> {
    scratch
        .completed
        .take()
        .ok_or(RootError::FactorizationInvariant {
            reason: "an Alekhnovich frame resumed without a child result",
        })
}

#[cfg(feature = "fft")]
fn insert_family<F: ButterflyKernels>(
    families: &mut Vec<AffineRootFamily<F>>,
    family: AffineRootFamily<F>,
    budget: &mut Budget,
    limits: AlekhnovichLimits,
) -> Result<(), RootError> {
    if families
        .iter()
        .any(|existing| family_contains(existing, &family))
    {
        return Ok(());
    }
    families.retain(|existing| !family_contains(&family, existing));
    let required = budget
        .families
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich affine family count",
        })?;
    enforce_limit(
        "Alekhnovich intermediate families",
        required,
        limits.max_intermediate_families,
    )?;
    families
        .try_reserve(1)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "Alekhnovich affine families",
            elements: families.len() + 1,
            element_size: core::mem::size_of::<AffineRootFamily<F>>(),
        })?;
    budget.families = required;
    families.push(family);
    Ok(())
}

#[cfg(feature = "fft")]
fn family_contains<F: ButterflyKernels>(
    outer: &AffineRootFamily<F>,
    inner: &AffineRootFamily<F>,
) -> bool {
    outer.tail_degree <= inner.tail_degree
        && (0..outer.tail_degree)
            .all(|degree| outer.prefix.coefficient(degree) == inner.prefix.coefficient(degree))
}

#[cfg(feature = "fft")]
fn compare_families<F: ButterflyKernels>(
    left: &AffineRootFamily<F>,
    right: &AffineRootFamily<F>,
) -> Ordering {
    left.tail_degree
        .cmp(&right.tail_degree)
        .then_with(|| compare_polynomials::<F>(left.prefix(), right.prefix()))
}

#[cfg(feature = "fft")]
fn materialize_candidates<F: ButterflyKernels>(
    rows: &[Polynomial<F>],
    max_degree: usize,
    y_degree: usize,
    families: Vec<AffineRootFamily<F>>,
    limits: AlekhnovichLimits,
) -> Result<Vec<Polynomial<F>>, RootError> {
    let coefficient_count = max_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "Alekhnovich output coefficient count",
        })?;
    let field_order = usize::try_from(F::ORDER).map_err(|_| ConfigError::GeometryOverflow {
        context: "Alekhnovich field order",
    })?;
    let mut candidates = Vec::new();
    let mut branch = Vec::new();

    for family in families {
        if family
            .prefix()
            .degree()
            .is_some_and(|degree| degree > max_degree)
        {
            continue;
        }
        let free_count = coefficient_count.saturating_sub(family.tail_degree());
        let completion_count = checked_power(field_order, free_count)?;
        let required_outputs = candidates.len().checked_add(completion_count).ok_or(
            ConfigError::GeometryOverflow {
                context: "Alekhnovich output root count",
            },
        )?;
        enforce_limit(
            "Alekhnovich output roots",
            required_outputs,
            limits.max_output_roots,
        )?;
        enforce_limit(
            "Alekhnovich Y-degree root bound",
            required_outputs,
            y_degree,
        )?;

        materialize_family(
            rows,
            &family,
            coefficient_count,
            field_order,
            completion_count,
            &mut branch,
        )?;
        for candidate in branch.drain(..) {
            if !candidates.iter().any(|existing| existing == &candidate) {
                candidates
                    .try_reserve(1)
                    .map_err(|_| ConfigError::AllocationFailed {
                        context: "Alekhnovich output roots",
                        elements: candidates.len() + 1,
                        element_size: core::mem::size_of::<Polynomial<F>>(),
                    })?;
                candidates.push(candidate);
            }
        }
    }

    candidates.sort_by(compare_polynomials::<F>);
    candidates.dedup();
    if candidates.len() > y_degree {
        return Err(RootError::FactorizationInvariant {
            reason: "verified Alekhnovich roots exceed the bivariate Y-degree",
        });
    }
    for candidate in &candidates {
        if !rows_has_root_allocated(rows, candidate)? {
            return Err(RootError::FactorizationInvariant {
                reason: "the final Alekhnovich candidate list contains a nonroot",
            });
        }
    }
    Ok(candidates)
}

/// Complete and verify one affine family's bounded roots into `branch`.
///
/// Each affine root family is an independent branch: expanding its free tail
/// coefficients and checking `Q(X, f(X)) == 0` depends only on the family
/// and the input rows, never on sibling families. This is the unit prepared
/// for optional parallel execution — [`materialize_candidates`] merges
/// branches sequentially so deduplication and the cumulative output/`Y`-degree
/// limits keep their exact failure behavior.
#[cfg(feature = "fft")]
fn materialize_family<F: ButterflyKernels>(
    rows: &[Polynomial<F>],
    family: &AffineRootFamily<F>,
    coefficient_count: usize,
    field_order: usize,
    completion_count: usize,
    branch: &mut Vec<Polynomial<F>>,
) -> Result<(), RootError> {
    branch.clear();
    for ordinal in 0..completion_count {
        let mut candidate = family.prefix().clone();
        let mut digits = ordinal;
        for degree in family.tail_degree()..coefficient_count {
            let key = digits % field_order;
            digits /= field_order;
            candidate.set_coefficient(degree, element_from_key::<F>(key))?;
        }
        if !rows_has_root_allocated(rows, &candidate)? {
            return Err(RootError::FactorizationInvariant {
                reason: "a final Alekhnovich affine family contained a nonroot",
            });
        }
        branch
            .try_reserve(1)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "Alekhnovich family completions",
                elements: branch.len() + 1,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
        branch.push(candidate);
    }
    Ok(())
}

/// Whether `Q(X, candidate(X)) == 0`, allocating the composition scratch.
#[cfg(feature = "fft")]
fn rows_has_root_allocated<F: FieldKernels>(
    rows: &[Polynomial<F>],
    candidate: &Polynomial<F>,
) -> Result<bool, RootError> {
    let mut result = Polynomial::zero();
    for row in rows.iter().rev() {
        result = result.multiply(candidate)?;
        result.add_assign(row)?;
    }
    Ok(result.is_zero())
}

#[cfg(feature = "fft")]
fn checked_power(base: usize, exponent: usize) -> Result<usize, RootError> {
    let mut value = 1_usize;
    for _ in 0..exponent {
        value = value
            .checked_mul(base)
            .ok_or(ConfigError::GeometryOverflow {
                context: "Alekhnovich affine completion count",
            })?;
    }
    Ok(value)
}

#[cfg(feature = "fft")]
fn element_from_key<F: fgf::field::Field>(key: usize) -> F::Elem {
    let bytes = (key as u128).to_le_bytes();
    F::read(&bytes[..F::BYTES])
}

#[cfg(feature = "fft")]
fn reserve_exact<T>(
    values: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), RootError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| ConfigError::AllocationFailed {
            context,
            elements: additional,
            element_size: core::mem::size_of::<T>(),
        })
        .map_err(RootError::from)
}
