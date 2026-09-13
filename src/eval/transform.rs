//! Subspace and coset evaluation/interpolation composing `butterfly-fft`.
//!
//! Fast multipoint evaluation over a size-`2^k` additive subspace *is* the
//! additive FFT, and the transform — its novel/Cantor bases, buffers, and
//! the only SIMD `unsafe` in the L1 tier — is `butterfly-fft`'s object
//! (U2). This module owns only the packing: monomial coefficients are
//! written into transform rows, converted monomial→novel, forwarded
//! (evaluation) or inverted and converted novel→monomial (interpolation).
//! The arbitrary-point paths live in [`super::multipoint`].

use alloc::vec::Vec;

use butterfly_fft::basis::{
    conversion_scratch_elements, inverse_interpolate_bytes, monomial_to_novel_bytes,
};
use butterfly_fft::core::kernel::ButterflyKernels;
use butterfly_fft::core::transform::TransformPlan;
use butterfly_fft::error::TransformLengthError;
use butterfly_fft::shifted::ShiftedPlan;

use crate::error::{ConfigError, PolynomialError};
use crate::poly::Polynomial;

/// Caller-owned reusable byte-row storage for transform composition.
#[derive(Debug)]
pub struct TransformScratch {
    rows: Vec<u8>,
    conversion: Vec<u8>,
}

impl TransformScratch {
    /// Construct empty reusable transform scratch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            rows: Vec::new(),
            conversion: Vec::new(),
        }
    }
}

impl Default for TransformScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate `polynomial` at every point of the plan's subspace.
///
/// # Errors
///
/// Returns [`PolynomialError`] when a row buffer cannot be reserved or the
/// transform reports a length mismatch.
pub fn evaluate_subspace<F: ButterflyKernels>(
    polynomial: &Polynomial<F>,
    plan: &TransformPlan<F>,
    scratch: &mut TransformScratch,
) -> Result<Vec<F::Elem>, PolynomialError> {
    let mut values = Vec::new();
    evaluate_subspace_into(polynomial, plan, scratch, &mut values)?;
    Ok(values)
}

/// Write the evaluations at every subspace point into `values`.
///
/// # Errors
///
/// Returns [`PolynomialError`] when a row buffer cannot be reserved or the
/// transform reports a length mismatch.
pub fn evaluate_subspace_into<F: ButterflyKernels>(
    polynomial: &Polynomial<F>,
    plan: &TransformPlan<F>,
    scratch: &mut TransformScratch,
    values: &mut Vec<F::Elem>,
) -> Result<(), PolynomialError> {
    let size = plan.size();
    values.clear();
    values
        .try_reserve_exact(size)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "subspace evaluations",
            elements: size,
            element_size: core::mem::size_of::<F::Elem>(),
        })?;

    let mut reduced;
    let coefficients = if polynomial.coefficient_count() > size {
        // The vanishing polynomial is zero on the whole subspace, so
        // reducing first leaves every evaluation unchanged.
        let vanishing = Polynomial::<F>::from_coefficients(&plan.vanishing_polynomial())?;
        reduced = polynomial.remainder(&vanishing)?;
        reduced.truncate(size);
        reduced.as_packed()
    } else {
        polynomial.as_packed()
    };

    let row_bytes = size * F::BYTES;
    ensure_len(&mut scratch.rows, row_bytes, "subspace evaluation rows")?;
    ensure_len(
        &mut scratch.conversion,
        conversion_scratch_elements(size) * F::BYTES,
        "subspace evaluation conversion",
    )?;
    scratch.rows[..row_bytes].fill(0);
    scratch.rows[..coefficients.len()].copy_from_slice(coefficients);
    monomial_to_novel_bytes::<F>(
        &mut scratch.rows[..row_bytes],
        F::BYTES,
        plan,
        &mut scratch.conversion,
    )
    .map_err(transform_length_error)?;
    plan.forward_bytes(&mut scratch.rows[..row_bytes], F::BYTES)
        .map_err(transform_length_error)?;
    values.extend(
        (0..row_bytes)
            .step_by(F::BYTES)
            .map(|start| F::read(&scratch.rows[start..start + F::BYTES])),
    );
    Ok(())
}

/// Write the evaluations at every point of the shifted plan's coset into
/// `values`.
///
/// # Errors
///
/// Returns [`PolynomialError`] when a row buffer cannot be reserved or the
/// transform reports a length mismatch.
pub fn evaluate_coset_into<F: ButterflyKernels>(
    polynomial: &Polynomial<F>,
    plan: &ShiftedPlan<F>,
    scratch: &mut TransformScratch,
    values: &mut Vec<F::Elem>,
) -> Result<(), PolynomialError> {
    let size = plan.size();
    values.clear();
    values
        .try_reserve_exact(size)
        .map_err(|_| ConfigError::AllocationFailed {
            context: "coset evaluations",
            elements: size,
            element_size: core::mem::size_of::<F::Elem>(),
        })?;

    let mut reduced;
    let coefficients = if polynomial.coefficient_count() > size {
        let vanishing = Polynomial::<F>::from_coefficients(&plan.plan().vanishing_polynomial())?;
        reduced = polynomial.remainder(&vanishing)?;
        reduced.truncate(size);
        reduced.as_packed()
    } else {
        polynomial.as_packed()
    };

    let row_bytes = size * F::BYTES;
    ensure_len(&mut scratch.rows, row_bytes, "coset evaluation rows")?;
    ensure_len(
        &mut scratch.conversion,
        conversion_scratch_elements(size) * F::BYTES,
        "coset evaluation conversion",
    )?;
    scratch.rows[..row_bytes].fill(0);
    scratch.rows[..coefficients.len()].copy_from_slice(coefficients);
    monomial_to_novel_bytes::<F>(
        &mut scratch.rows[..row_bytes],
        F::BYTES,
        plan.plan(),
        &mut scratch.conversion,
    )
    .map_err(transform_length_error)?;
    plan.forward_bytes(&mut scratch.rows[..row_bytes], F::BYTES)
        .map_err(transform_length_error)?;
    values.extend(
        (0..row_bytes)
            .step_by(F::BYTES)
            .map(|start| F::read(&scratch.rows[start..start + F::BYTES])),
    );
    Ok(())
}

/// Interpolate the degree-below-`plan.size()` polynomial through the plan's
/// points and `values`.
///
/// # Errors
///
/// Returns [`PolynomialError`] on a length mismatch with the plan or when a
/// buffer cannot be reserved.
pub fn interpolate_subspace<F: ButterflyKernels>(
    plan: &TransformPlan<F>,
    values: &[F::Elem],
    scratch: &mut TransformScratch,
) -> Result<Polynomial<F>, PolynomialError> {
    let mut output = Polynomial::zero();
    interpolate_subspace_into(plan, values, scratch, &mut output)?;
    Ok(output)
}

/// Write the inverse-transform interpolant through the plan's points and
/// `values` into `output`.
///
/// # Errors
///
/// Returns [`PolynomialError`] on a length mismatch with the plan or when a
/// buffer cannot be reserved.
pub fn interpolate_subspace_into<F: ButterflyKernels>(
    plan: &TransformPlan<F>,
    values: &[F::Elem],
    scratch: &mut TransformScratch,
    output: &mut Polynomial<F>,
) -> Result<(), PolynomialError> {
    let size = plan.size();
    if values.len() != size {
        return Err(PolynomialError::Config(ConfigError::GeometryOverflow {
            context: "subspace interpolation values",
        }));
    }
    let row_bytes = size * F::BYTES;
    ensure_len(&mut scratch.rows, row_bytes, "subspace interpolation rows")?;
    ensure_len(
        &mut scratch.conversion,
        conversion_scratch_elements(size) * F::BYTES,
        "subspace interpolation conversion",
    )?;
    for (start, &value) in (0..row_bytes).step_by(F::BYTES).zip(values) {
        F::write(&mut scratch.rows[start..start + F::BYTES], value);
    }
    inverse_interpolate_bytes::<F>(
        &mut scratch.rows[..row_bytes],
        F::BYTES,
        plan,
        &mut scratch.conversion,
    )
    .map_err(transform_length_error)?;
    output.assign_packed(&scratch.rows[..row_bytes])
}

fn transform_length_error(error: TransformLengthError) -> PolynomialError {
    // Length mismatches are unreachable after the pre-sized buffers above;
    // surface them as geometry failures carrying both lengths.
    let _ = error;
    PolynomialError::Config(ConfigError::GeometryOverflow {
        context: "butterfly-fft transform geometry",
    })
}

fn ensure_len(
    values: &mut Vec<u8>,
    required: usize,
    context: &'static str,
) -> Result<(), PolynomialError> {
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
