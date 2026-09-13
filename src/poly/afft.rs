//! The AFFT tier of the product ladder, composing `butterfly-fft`.
//!
//! AFFT packs every left/right operand across byte-row columns, performs one
//! pair of forward transforms, uses `fgf` elementwise multiplication per
//! point, and inverse-transforms all products as a second packed batch. The
//! transform itself is `butterfly-fft`'s object; this module owns only the
//! packing and the measured decision of when to enter it.

use alloc::vec::Vec;

use butterfly_fft::basis::{
    conversion_scratch_elements, monomial_to_novel_bytes, novel_to_monomial_bytes,
};
use butterfly_fft::core::kernel::ButterflyKernels;
use butterfly_fft::core::transform::TransformPlan;
use fgf::field::Elem as _;
use fgf::ops;

use super::ring::binomial_odd;
use crate::error::{ConfigError, ProductError};
use crate::poly::dense::Polynomial;

/// AFFT product crossover in full-product coefficients, one to three packed products. See `BENCHMARKS.md`.
pub const AFFT_PRODUCT_CROSSOVER: usize = usize::MAX;

/// AFFT product crossover, four to seven packed products. See `BENCHMARKS.md`.
pub const AFFT_BATCH4_CROSSOVER: usize = 65_535;

/// AFFT product crossover, eight to fifteen packed products. See `BENCHMARKS.md`.
pub const AFFT_BATCH8_CROSSOVER: usize = 32_767;

/// AFFT product crossover, sixteen or more packed products. See `BENCHMARKS.md`.
pub const AFFT_BATCH16_CROSSOVER: usize = 8_191;

/// AFFT product crossover, one to three scalar products. See `BENCHMARKS.md`.
pub const SCALAR_AFFT_PRODUCT_CROSSOVER: usize = 511;

/// AFFT product crossover, four to seven scalar products. See `BENCHMARKS.md`.
pub const SCALAR_AFFT_BATCH4_CROSSOVER: usize = 255;

/// AFFT product crossover, eight to fifteen scalar products. See `BENCHMARKS.md`.
pub const SCALAR_AFFT_BATCH8_CROSSOVER: usize = 255;

/// AFFT product crossover, sixteen or more scalar products. See `BENCHMARKS.md`.
pub const SCALAR_AFFT_BATCH16_CROSSOVER: usize = 127;

/// Algorithm selection for polynomial product batches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductStrategy {
    /// Use the measured crossover and fall back when the transform is unavailable.
    Auto,
    /// Always use truncated schoolbook multiplication.
    Schoolbook,
    /// Require the AFFT backend.
    Afft,
}

/// Reusable transform and byte-row storage for polynomial products.
pub struct PolynomialProductScratch<F: ButterflyKernels> {
    plan: Option<TransformPlan<F>>,
    operands: Vec<u8>,
    products: Vec<u8>,
    conversion: Vec<u8>,
    pub(crate) affine_powers: Vec<Polynomial<F>>,
    pub(crate) affine_products: Vec<Polynomial<F>>,
    pub(crate) affine_pairs: Vec<(usize, usize)>,
}

impl<F: ButterflyKernels> PolynomialProductScratch<F> {
    /// Construct empty product scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            plan: None,
            operands: Vec::new(),
            products: Vec::new(),
            conversion: Vec::new(),
            affine_powers: Vec::new(),
            affine_products: Vec::new(),
            affine_pairs: Vec::new(),
        }
    }

    /// Retained operand-row capacity in bytes.
    #[must_use]
    pub fn operand_capacity_bytes(&self) -> usize {
        self.operands.capacity()
    }

    /// Retained product-row capacity in bytes.
    #[must_use]
    pub fn product_capacity_bytes(&self) -> usize {
        self.products.capacity()
    }
}

impl<F: ButterflyKernels> Default for PolynomialProductScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: ButterflyKernels> core::fmt::Debug for PolynomialProductScratch<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("PolynomialProductScratch")
            .field("plan_size", &self.plan.as_ref().map(TransformPlan::size))
            .field("operand_capacity_bytes", &self.operands.capacity())
            .finish_non_exhaustive()
    }
}

/// Multiply independent pairs, truncate every result to `coefficient_count`,
/// and write them to caller-owned output storage.
///
/// AFFT packs every left/right operand across byte-row columns, performs one
/// pair of forward transforms, uses FGF elementwise multiplication per point,
/// and inverse-transforms all products as a second packed batch.
///
/// # Errors
///
/// Returns [`ProductError`] when storage geometry fails, the transform plan
/// cannot be built, or a conversion buffer has inconsistent geometry.
pub fn multiply_batch_truncated<F: ButterflyKernels>(
    pairs: &[(&Polynomial<F>, &Polynomial<F>)],
    coefficient_count: usize,
    strategy: ProductStrategy,
    scratch: &mut PolynomialProductScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), ProductError> {
    multiply_batch_truncated_with(
        pairs.len(),
        |index| pairs[index],
        coefficient_count,
        strategy,
        scratch,
        output,
    )
}

/// Multiply a caller-indexed batch of pairs into `output`.
///
/// # Errors
///
/// Returns [`ProductError`] when storage geometry fails, the transform plan
/// cannot be built, or a conversion buffer has inconsistent geometry.
///
/// # Panics
///
/// The internal plan expectation holds once a matching AFFT plan is
/// prepared in the same call.
// A faithful lift of gs-engine's batched product; splitting it would
// diverge the port from its tested original.
#[allow(clippy::too_many_lines)]
pub fn multiply_batch_truncated_with<'a, F, P>(
    pair_count: usize,
    pair: P,
    coefficient_count: usize,
    strategy: ProductStrategy,
    scratch: &mut PolynomialProductScratch<F>,
    output: &mut Vec<Polynomial<F>>,
) -> Result<(), ProductError>
where
    F: ButterflyKernels,
    P: Copy + Fn(usize) -> (&'a Polynomial<F>, &'a Polynomial<F>),
{
    prepare_output(output, pair_count)?;
    if pair_count == 0 {
        return Ok(());
    }

    let mut max_full_count = 0_usize;
    let mut max_left = 0_usize;
    let mut max_right = 0_usize;
    for index in 0..pair_count {
        let (left, right) = pair(index);
        let full_count = full_product_count(left, right)?;
        max_full_count = max_full_count.max(full_count);
        max_left = max_left.max(left.coefficient_count());
        max_right = max_right.max(right.coefficient_count());
    }
    if max_full_count == 0 || coefficient_count == 0 {
        for polynomial in output {
            polynomial.set_zero();
        }
        return Ok(());
    }

    let use_afft = match strategy {
        ProductStrategy::Schoolbook => false,
        ProductStrategy::Afft => true,
        ProductStrategy::Auto => {
            crate::cost::select_product(crate::cost::ProductCostKey {
                left_coefficients: max_left,
                right_coefficients: max_right,
                output_coefficients: max_full_count,
                batch: pair_count,
                field_order: F::ORDER,
                backend: crate::cost::BackendClass::detect::<F>(),
            }) == crate::cost::ProductBackend::Afft
        }
    };
    let Some(transform_size) = max_full_count.checked_next_power_of_two() else {
        if strategy == ProductStrategy::Afft {
            return Err(ConfigError::GeometryOverflow {
                context: "AFFT polynomial product size",
            }
            .into());
        }
        return schoolbook_batch_with(pair_count, pair, coefficient_count, output);
    };

    if !use_afft {
        return schoolbook_batch_with(pair_count, pair, coefficient_count, output);
    }
    if scratch.plan.as_ref().map(TransformPlan::size) != Some(transform_size) {
        match TransformPlan::<F>::new(transform_size) {
            Ok(plan) => scratch.plan = Some(plan),
            Err(error) if strategy == ProductStrategy::Auto => {
                let _ = error;
                return schoolbook_batch_with(pair_count, pair, coefficient_count, output);
            }
            Err(error) => return Err(error.into()),
        }
    }

    let pair_bytes = pair_count
        .checked_mul(F::BYTES)
        .ok_or(ConfigError::GeometryOverflow {
            context: "AFFT product row bytes",
        })?;
    let operand_row_bytes = pair_bytes
        .checked_mul(2)
        .ok_or(ConfigError::GeometryOverflow {
            context: "AFFT operand row bytes",
        })?;
    let operand_bytes =
        transform_size
            .checked_mul(operand_row_bytes)
            .ok_or(ConfigError::GeometryOverflow {
                context: "AFFT operand bytes",
            })?;
    let product_bytes =
        transform_size
            .checked_mul(pair_bytes)
            .ok_or(ConfigError::GeometryOverflow {
                context: "AFFT product bytes",
            })?;
    let conversion_bytes = conversion_scratch_elements(transform_size)
        .checked_mul(operand_row_bytes)
        .ok_or(ConfigError::GeometryOverflow {
            context: "AFFT conversion bytes",
        })?;
    ensure_len(&mut scratch.operands, operand_bytes, "AFFT operands")?;
    ensure_len(&mut scratch.products, product_bytes, "AFFT products")?;
    ensure_len(
        &mut scratch.conversion,
        conversion_bytes,
        "AFFT conversion scratch",
    )?;
    scratch.operands[..operand_bytes].fill(0);

    for lane in 0..pair_count {
        let (left, right) = pair(lane);
        write_lane::<F>(
            &mut scratch.operands[..operand_bytes],
            operand_row_bytes,
            lane * F::BYTES,
            left,
        );
        write_lane::<F>(
            &mut scratch.operands[..operand_bytes],
            operand_row_bytes,
            pair_bytes + lane * F::BYTES,
            right,
        );
    }

    afft_rows_convolve::<F>(
        scratch
            .plan
            .as_ref()
            .expect("a matching AFFT plan was prepared"),
        &mut scratch.operands[..operand_bytes],
        &mut scratch.products[..product_bytes],
        &mut scratch.conversion[..conversion_bytes],
        pair_bytes,
    )?;

    for (lane, polynomial) in output.iter_mut().enumerate().take(pair_count) {
        let (left, right) = pair(lane);
        let full_count = full_product_count(left, right)?;
        let result_count = full_count.min(coefficient_count);
        let byte_len = result_count
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "AFFT result bytes",
            })?;
        polynomial.set_zero();
        polynomial.resize_coefficients(result_count)?;
        for degree in 0..result_count {
            let source = degree * pair_bytes + lane * F::BYTES;
            let destination = degree * F::BYTES;
            polynomial.coefficients[destination..destination + F::BYTES]
                .copy_from_slice(&scratch.products[source..source + F::BYTES]);
        }
        debug_assert_eq!(polynomial.as_packed().len(), byte_len);
        polynomial.normalize();
    }
    Ok(())
}

/// Run the packed-lane AFFT product pipeline over prepared row buffers.
///
/// `operands` holds the transform's rows of `2 * pair_bytes` bytes — the
/// left-operand lanes in the low half of each row, the right-operand lanes
/// in the high half, monomial order, zero-padded past each operand's
/// coefficients. On return `products` holds the convolution rows in
/// monomial order, one `pair_bytes` row per transform row; coefficient `d`
/// of lane `l` sits at `d * pair_bytes + l * F::BYTES`.
///
/// This is the shared row-buffer worker: the public AFFT batch product and
/// the prepared convolution engine both call it, so there is exactly one
/// conversion/transform/elementwise/inverse implementation to keep correct.
/// `conversion` must hold at least
/// `conversion_scratch_elements(plan.size()) * 2 * pair_bytes` bytes.
pub(crate) fn afft_rows_convolve<F: ButterflyKernels>(
    plan: &TransformPlan<F>,
    operands: &mut [u8],
    products: &mut [u8],
    conversion: &mut [u8],
    pair_bytes: usize,
) -> Result<(), ProductError> {
    let row_count = plan.size();
    let operand_row_bytes = pair_bytes
        .checked_mul(2)
        .ok_or(ConfigError::GeometryOverflow {
            context: "AFFT operand row bytes",
        })?;
    monomial_to_novel_bytes::<F>(operands, operand_row_bytes, plan, conversion)?;
    plan.forward_bytes(operands, operand_row_bytes)?;

    for (operand_row, product_row) in operands
        .chunks_exact(operand_row_bytes)
        .zip(products.chunks_exact_mut(pair_bytes))
        .take(row_count)
    {
        ops::mul_elementwise::<F>(
            product_row,
            &operand_row[..pair_bytes],
            &operand_row[pair_bytes..],
        );
    }
    plan.inverse_bytes(products, pair_bytes)?;
    novel_to_monomial_bytes::<F>(
        products,
        pair_bytes,
        plan,
        &mut conversion[..conversion_scratch_elements(row_count) * pair_bytes],
    )?;
    Ok(())
}

/// Substitute `Y = prefix(X) + X^tail_degree * Z` into the `Y`-coefficient
/// rows of a bivariate polynomial held as a row slice, truncating every
/// output row modulo `X^coefficient_count`.
///
/// This is the affine-prefix transform used by divide-and-conquer root
/// extraction over `rows[j] = Q_j(X)` (the coefficient of `Y^j`). Products
/// are truncated before shifts so coefficients that cannot affect the
/// requested precision are never materialized. Independent coefficient-row
/// products are packed across transform row columns; the measured product
/// crossover retains schoolbook arithmetic for smaller nodes.
///
/// `output` is resized to exactly the needed row count; surplus rows are
/// recycled through `pool` and missing rows are drawn from it.
///
/// # Errors
///
/// Returns [`ProductError`] when storage geometry fails or a batched product
/// cannot be computed.
pub fn substitute_y_affine_rows_truncated_into<F: ButterflyKernels>(
    rows: &[Polynomial<F>],
    prefix: &Polynomial<F>,
    tail_degree: usize,
    coefficient_count: usize,
    scratch: &mut PolynomialProductScratch<F>,
    output: &mut Vec<Polynomial<F>>,
    pool: &mut Vec<Polynomial<F>>,
) -> Result<(), ProductError> {
    let Some(y_degree) = rows.len().checked_sub(1) else {
        recycle_into_pool(output, pool);
        return Ok(());
    };
    if coefficient_count == 0 {
        recycle_into_pool(output, pool);
        return Ok(());
    }
    let power_count = y_degree
        .checked_add(1)
        .ok_or(ConfigError::GeometryOverflow {
            context: "fast affine substitution powers",
        })?;

    let mut prefix_powers = core::mem::take(&mut scratch.affine_powers);
    let mut products = core::mem::take(&mut scratch.affine_products);
    let mut pair_indices = core::mem::take(&mut scratch.affine_pairs);
    let result = (|| {
        if prefix_powers.capacity() < power_count {
            prefix_powers
                .try_reserve_exact(power_count.saturating_sub(prefix_powers.len()))
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "fast affine substitution powers",
                    elements: power_count,
                    element_size: core::mem::size_of::<Polynomial<F>>(),
                })?;
        }
        while prefix_powers.len() < power_count {
            prefix_powers.push(Polynomial::zero());
        }
        prefix_powers.truncate(power_count);
        prefix_powers[0].assign_coefficients(&[F::Elem::ONE])?;
        for exponent in 1..power_count {
            let (previous, current) = prefix_powers.split_at_mut(exponent);
            previous[exponent - 1].multiply_truncated_into(
                prefix,
                coefficient_count,
                &mut current[0],
            )?;
        }

        if pair_indices.capacity() < power_count {
            pair_indices
                .try_reserve_exact(power_count.saturating_sub(pair_indices.len()))
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "fast affine substitution descriptors",
                    elements: power_count,
                    element_size: core::mem::size_of::<(usize, usize)>(),
                })?;
        }
        prepare_rows(output, power_count, pool)?;

        for (target_y, destination) in output.iter_mut().enumerate().take(power_count) {
            let Some(shift) = tail_degree.checked_mul(target_y) else {
                continue;
            };
            if shift >= coefficient_count {
                continue;
            }
            pair_indices.clear();
            for source_y in target_y..rows.len() {
                if binomial_odd(source_y, target_y) {
                    pair_indices.push((source_y, source_y - target_y));
                }
            }
            multiply_batch_truncated_with(
                pair_indices.len(),
                |index| {
                    let (source_y, exponent) = pair_indices[index];
                    (&rows[source_y], &prefix_powers[exponent])
                },
                coefficient_count - shift,
                ProductStrategy::Auto,
                scratch,
                &mut products,
            )?;
            for product in &products {
                if !product.is_zero() {
                    destination.add_scaled_shifted_assign(F::Elem::ONE, product, shift)?;
                }
            }
        }
        drop_trailing_zero_rows(output, pool);
        Ok(())
    })();
    scratch.affine_powers = prefix_powers;
    scratch.affine_products = products;
    scratch.affine_pairs = pair_indices;
    result
}

/// Recycle every row of `output` into `pool`, leaving it empty.
pub(crate) fn recycle_into_pool<F: fgf::kernel::FieldKernels>(
    output: &mut Vec<Polynomial<F>>,
    pool: &mut Vec<Polynomial<F>>,
) {
    while let Some(mut row) = output.pop() {
        row.set_zero();
        pool.push(row);
    }
}

/// Drop trailing zero rows of `output` into `pool`, restoring the normalized
/// row invariant without freeing row buffers.
pub(crate) fn drop_trailing_zero_rows<F: fgf::kernel::FieldKernels>(
    output: &mut Vec<Polynomial<F>>,
    pool: &mut Vec<Polynomial<F>>,
) {
    while output.last().is_some_and(Polynomial::is_zero) {
        let mut row = output.pop().expect("nonempty rows");
        row.set_zero();
        pool.push(row);
    }
}

/// Ensure `output` holds exactly `count` zeroed rows, recycling spare rows
/// through `pool`.
pub(crate) fn prepare_rows<F: fgf::kernel::FieldKernels>(
    output: &mut Vec<Polynomial<F>>,
    count: usize,
    pool: &mut Vec<Polynomial<F>>,
) -> Result<(), ProductError> {
    recycle_into_pool(output, pool);
    if output.capacity() < count {
        output
            .try_reserve(count)
            .map_err(|_| ConfigError::AllocationFailed {
                context: "affine substitution rows",
                elements: count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
    }
    for _ in 0..count {
        let mut row = pool.pop().unwrap_or_default();
        row.set_zero();
        output.push(row);
    }
    Ok(())
}

fn schoolbook_batch_with<'a, F, P>(
    pair_count: usize,
    pair: P,
    coefficient_count: usize,
    output: &mut [Polynomial<F>],
) -> Result<(), ProductError>
where
    F: ButterflyKernels,
    P: Fn(usize) -> (&'a Polynomial<F>, &'a Polynomial<F>),
{
    for (index, polynomial) in output.iter_mut().enumerate().take(pair_count) {
        let (left, right) = pair(index);
        left.multiply_truncated_into(right, coefficient_count, polynomial)?;
    }
    Ok(())
}

fn prepare_output<F: ButterflyKernels>(
    output: &mut Vec<Polynomial<F>>,
    count: usize,
) -> Result<(), ProductError> {
    if output.capacity() < count {
        output
            .try_reserve(count - output.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "polynomial product outputs",
                elements: count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
    }
    while output.len() < count {
        output.push(Polynomial::zero());
    }
    output.truncate(count);
    Ok(())
}

fn full_product_count<F: ButterflyKernels>(
    left: &Polynomial<F>,
    right: &Polynomial<F>,
) -> Result<usize, ConfigError> {
    match (left.coefficient_count(), right.coefficient_count()) {
        (0, _) | (_, 0) => Ok(0),
        (left, right) => left
            .checked_add(right)
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial product coefficients",
            }),
    }
}

fn write_lane<F: ButterflyKernels>(
    rows: &mut [u8],
    row_len: usize,
    lane_offset: usize,
    polynomial: &Polynomial<F>,
) {
    for (degree, coefficient) in polynomial.coefficients().enumerate() {
        let offset = degree * row_len + lane_offset;
        F::write(&mut rows[offset..offset + F::BYTES], coefficient);
    }
}

fn ensure_len(
    values: &mut Vec<u8>,
    required: usize,
    context: &'static str,
) -> Result<(), ProductError> {
    if required > values.len() {
        values
            .try_reserve_exact(required - values.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: required,
                element_size: 1,
            })?;
        values.resize(required, 0);
    }
    Ok(())
}
