//! Newton-basis interpolation over arbitrary point sets.
//!
//! The basis `N_i(X) = ∏_{j<i}(X + α_j)` with denominators `N_i(α_i)` is
//! built incrementally in `O(n²)`, and an interpolant is materialized by
//! one discrepancy pass in the same cost. This is the small-`n` default
//! (below [`MODULE_INTERPOLATION_CROSSOVER`] points); the Lagrange
//! subproduct-tree path in [`super::multipoint`] takes over above it. The
//! two agree exactly and share no code.

use alloc::vec::Vec;

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::error::{ConfigError, DomainError, EvalError, PolynomialError};
use crate::poly::Polynomial;

/// Point count at or below which incremental Newton interpolation wins over
/// the subproduct-tree Lagrange path. Measured; see `BENCHMARKS.md`.
pub const MODULE_INTERPOLATION_CROSSOVER: usize = 8;

/// A prepared Newton basis and its denominators.
///
/// All parts depend only on the point set, so the basis is built once and
/// reused across every interpolation over the same support.
#[derive(Clone, Debug)]
pub struct NewtonBasis<F: FieldKernels> {
    points: Vec<F::Elem>,
    partials: Vec<Polynomial<F>>,
    denominators: Vec<F::Elem>,
    vanishing: Polynomial<F>,
}

impl<F: FieldKernels> NewtonBasis<F> {
    /// Build the Newton basis for `points`.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::Domain`] for an empty support, a support larger
    /// than the field, or duplicate points, and [`EvalError::Polynomial`]
    /// when a basis buffer cannot be reserved.
    pub fn new(points: &[F::Elem]) -> Result<Self, EvalError> {
        if points.is_empty() {
            return Err(EvalError::Domain(DomainError::Config(
                ConfigError::ZeroParameter {
                    parameter: "interpolation support",
                },
            )));
        }
        if points.len() as u128 > F::ORDER {
            return Err(EvalError::Domain(DomainError::Config(
                ConfigError::FieldCapacityExceeded {
                    points: points.len(),
                    field_order: F::ORDER,
                },
            )));
        }
        let mut partials = Vec::new();
        let mut denominators = Vec::new();
        partials
            .try_reserve_exact(points.len())
            .and_then(|()| denominators.try_reserve_exact(points.len()))
            .map_err(|_| {
                EvalError::Polynomial(PolynomialError::Config(ConfigError::AllocationFailed {
                    context: "Newton basis",
                    elements: points.len(),
                    element_size: core::mem::size_of::<Polynomial<F>>(),
                }))
            })?;
        let mut current = Polynomial::<F>::one()?;
        for (second, &point) in points.iter().enumerate() {
            let denominator = current.evaluate(point);
            if denominator.is_zero() {
                // A zero denominator means this point repeats an earlier
                // one; recover the indices for the domain error.
                let (first, second) = super::find_duplicate::<F>(points)
                    .unwrap_or((second.saturating_sub(1), second));
                return Err(EvalError::Domain(DomainError::DuplicatePoint {
                    first,
                    second,
                }));
            }
            denominators.push(denominator.inv());
            let mut partial = Polynomial::zero();
            partial.assign_packed(current.as_packed())?;
            partials.push(partial);
            current = current.multiply_x_plus(point)?;
        }
        Ok(Self {
            points: points.to_vec(),
            partials,
            denominators,
            vanishing: current,
        })
    }

    /// The support points in basis order.
    #[must_use]
    pub fn points(&self) -> &[F::Elem] {
        &self.points
    }

    /// The vanishing polynomial `∏(X + α_i)` over the support.
    #[must_use]
    pub fn vanishing(&self) -> &Polynomial<F> {
        &self.vanishing
    }

    /// The stored partial products `N_i`, low to high.
    #[must_use]
    pub fn partials(&self) -> &[Polynomial<F>] {
        &self.partials
    }

    /// Number of support points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.partials.len()
    }

    /// Whether the support is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.partials.is_empty()
    }

    /// Write the interpolant through `values` into `output`, reusing its
    /// buffer.
    ///
    /// # Errors
    ///
    /// Returns [`EvalError::Domain`] when `values` does not match the
    /// support length and [`EvalError::Polynomial`] when an intermediate
    /// buffer cannot be reserved.
    pub fn interpolate_into(
        &self,
        values: &[F::Elem],
        output: &mut Polynomial<F>,
    ) -> Result<(), EvalError> {
        if values.len() != self.len() {
            return Err(EvalError::Domain(DomainError::LengthMismatch {
                expected: self.len(),
                found: values.len(),
            }));
        }
        output.set_zero();
        for (((value, point), denominator), partial) in values
            .iter()
            .copied()
            .zip(&self.points)
            .zip(&self.denominators)
            .zip(&self.partials)
        {
            let discrepancy = value.add(output.evaluate(*point));
            let coefficient = discrepancy.mul(*denominator);
            output.add_scaled_assign(coefficient, partial)?;
        }
        Ok(())
    }
}

/// Interpolate the minimal-degree polynomial through the point/value pairs
/// by incremental Newton differences.
///
/// # Errors
///
/// Returns [`EvalError::Domain`] for mismatched lengths, duplicate points,
/// or an over-capacity support, and [`EvalError::Polynomial`] when an
/// intermediate buffer cannot be reserved.
pub fn interpolate_newton<F: FieldKernels>(
    points: &[F::Elem],
    values: &[F::Elem],
) -> Result<Polynomial<F>, EvalError> {
    let mut output = Polynomial::zero();
    interpolate_newton_into(points, values, &mut output)?;
    Ok(output)
}

/// Write the Newton interpolant through the pairs into `output`.
///
/// # Errors
///
/// Returns [`EvalError::Domain`] for mismatched lengths, duplicate points,
/// or an over-capacity support, and [`EvalError::Polynomial`] when an
/// intermediate buffer cannot be reserved.
pub fn interpolate_newton_into<F: FieldKernels>(
    points: &[F::Elem],
    values: &[F::Elem],
    output: &mut Polynomial<F>,
) -> Result<(), EvalError> {
    let basis = NewtonBasis::new(points)?;
    basis.interpolate_into(values, output)
}
