//! Prepared Hasse derivatives of one fixed order.

use alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::ops::Coeff;

use crate::error::HasseError;
use crate::poly::monomial::embed_integer;
use crate::poly::{PolynomialField, binomial_odd};

/// Checked geometry shared by the module.
fn checked_len(context: &'static str, count: usize, unit: usize) -> Result<usize, HasseError> {
    count
        .checked_mul(unit)
        .ok_or(HasseError::GeometryOverflow { context })
}

/// Replace one prime-field lane row's elements by canonical representatives.
fn canonicalize_row<F: Field>(row: &mut [u8]) {
    for slot in row.chunks_exact_mut(F::BYTES) {
        let value = F::decode(slot);
        F::encode(slot, value.add(F::Elem::ZERO));
    }
}

/// The `p`-adic valuation of `n` and its unit part, for `n > 0`.
fn split_prime_power(mut n: u64, prime: u64) -> (u64, u64) {
    let mut valuation = 0_u64;
    while n.is_multiple_of(prime) {
        n /= prime;
        valuation += 1;
    }
    (valuation, n)
}

/// A prepared fixed-order Hasse derivative over whole coefficient vectors.
///
/// The derivative of order `r` maps coefficient `d` of the input to
/// coefficient `d − r` of the output with the factor `C(d, r)` taken in the
/// field's characteristic. Construction precomputes that factor sequence
/// once — parity-tested masks in characteristic two, and an incremental
/// binomial recurrence with tracked `p`-valuation elsewhere — so applying
/// the plan is a single linear pass and a warmed run allocates nothing.
///
/// Output length is `input_coefficients.saturating_sub(order)`, the
/// *coefficient count* difference, not a normalized polynomial degree: every
/// mathematically absent coefficient is written explicitly.
pub struct DerivativePlan<F: PolynomialField> {
    order: usize,
    max_coefficients: usize,
    /// The binomial factors as backend-prepared coefficients, indexed by
    /// output position. Empty in characteristic two, where the parity mask
    /// is the cheaper representation.
    factors: Vec<Coeff<F>>,
}

impl<F: PolynomialField> DerivativePlan<F> {
    /// Prepare the order-`order` derivative for inputs of up to
    /// `max_coefficients` coefficients.
    ///
    /// An order at or above the capacity is legal: it yields a plan whose
    /// output is empty for every in-capacity input.
    ///
    /// # Errors
    ///
    /// Returns [`HasseError::GeometryOverflow`] or
    /// [`HasseError::AllocationFailed`] when the factor table cannot be
    /// represented or reserved.
    pub fn new(order: usize, max_coefficients: usize) -> Result<Self, HasseError> {
        let factors = Self::compute_factors(order, max_coefficients)?;
        Ok(Self {
            order,
            max_coefficients,
            factors,
        })
    }

    /// The order this plan applies.
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }

    /// The coefficient capacity this plan was built for.
    #[must_use]
    pub fn max_coefficients(&self) -> usize {
        self.max_coefficients
    }
    /// The output coefficient count for an input of `input_coefficients`
    /// coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`HasseError::CoefficientCapacityExceeded`] when the input
    /// exceeds the prepared bound — the output length is a capacity fact,
    /// not a pure function.
    pub fn output_coefficients(&self, input_coefficients: usize) -> Result<usize, HasseError> {
        if input_coefficients > self.max_coefficients {
            return Err(HasseError::CoefficientCapacityExceeded {
                maximum: self.max_coefficients,
                actual: input_coefficients,
            });
        }
        Ok(input_coefficients.saturating_sub(self.order))
    }

    /// Apply the derivative to scalar coefficients, low degree first.
    ///
    /// # Errors
    ///
    /// Returns [`HasseError::CoefficientCapacityExceeded`] when the input
    /// exceeds the prepared bound and [`HasseError::LengthMismatch`] when
    /// `output` does not hold exactly
    /// [`Self::output_coefficients`](DerivativePlan::output_coefficients)
    /// elements. `output` is untouched on error.
    pub fn apply_into(
        &self,
        coefficients: &[F::Elem],
        output: &mut [F::Elem],
    ) -> Result<(), HasseError> {
        let count = coefficients.len();
        if count > self.max_coefficients {
            return Err(HasseError::CoefficientCapacityExceeded {
                maximum: self.max_coefficients,
                actual: count,
            });
        }
        let output_count = self.output_coefficients(count)?;
        if output.len() != output_count {
            return Err(HasseError::LengthMismatch {
                argument: "derivative output",
                expected: output_count,
                actual: output.len(),
            });
        }
        if F::CHARACTERISTIC == 2 {
            for (output_degree, destination) in output.iter_mut().enumerate() {
                let source = output_degree + self.order;
                *destination = if binomial_odd(source, self.order) {
                    coefficients[source]
                } else {
                    F::Elem::ZERO
                };
            }
        } else {
            for (output_degree, factor) in self.factors.iter().enumerate().take(output_count) {
                output[output_degree] =
                    coefficients[output_degree + self.order].mul(factor.value());
            }
        }
        Ok(())
    }

    /// Apply the derivative to coefficient-major batched lane rows.
    ///
    /// `coefficients` holds `coefficient_count` rows of `batch` lanes; the
    /// output receives `Self::output_coefficients` rows in the same layout.
    /// Batch zero with empty buffers is a valid empty geometry.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`DerivativePlan::apply_into`] plus
    /// [`HasseError::LengthMismatch`] for a mis-sized input buffer and
    /// [`HasseError::GeometryOverflow`] for unrepresentable byte lengths.
    /// Nothing is mutated on error.
    pub fn apply_batch_into(
        &self,
        coefficients: &[u8],
        coefficient_count: usize,
        batch: usize,
        output: &mut [u8],
    ) -> Result<(), HasseError> {
        let input_bytes = checked_len("derivative input bytes", coefficient_count, batch)
            .and_then(|lanes| checked_len("derivative input bytes", lanes, F::BYTES))?;
        if coefficients.len() != input_bytes {
            return Err(HasseError::LengthMismatch {
                argument: "derivative coefficients",
                expected: input_bytes,
                actual: coefficients.len(),
            });
        }
        if coefficient_count > self.max_coefficients {
            return Err(HasseError::CoefficientCapacityExceeded {
                maximum: self.max_coefficients,
                actual: coefficient_count,
            });
        }
        let output_count = self.output_coefficients(coefficient_count)?;
        let output_bytes = checked_len("derivative output bytes", output_count, batch)
            .and_then(|lanes| checked_len("derivative output bytes", lanes, F::BYTES))?;
        if output.len() != output_bytes {
            return Err(HasseError::LengthMismatch {
                argument: "derivative output",
                expected: output_bytes,
                actual: output.len(),
            });
        }
        let row_bytes = batch * F::BYTES;
        for output_degree in 0..output_count {
            let source = (output_degree + self.order) * row_bytes;
            let destination = output_degree * row_bytes;
            let (source_row, output_row) = (
                &coefficients[source..source + row_bytes],
                &mut output[destination..destination + row_bytes],
            );
            if F::CHARACTERISTIC == 2 {
                if binomial_odd(output_degree + self.order, self.order) {
                    output_row.copy_from_slice(source_row);
                } else {
                    output_row.fill(0);
                }
            } else {
                let factor = &self.factors[output_degree];
                // Copy first, canonicalize the copy, scale in place: the
                // caller's raw prime lanes never reach a packed kernel.
                output_row.copy_from_slice(source_row);
                canonicalize_row::<F>(output_row);
                fgf::ops::mul_assign_with::<F>(output_row, factor);
            }
        }
        Ok(())
    }

    /// The factor `C(j + order, order)` for every output position `j`, as
    /// prepared coefficients.
    fn compute_factors(order: usize, max_coefficients: usize) -> Result<Vec<Coeff<F>>, HasseError> {
        let count = max_coefficients.saturating_sub(order);
        let mut factors = Vec::new();
        factors
            .try_reserve_exact(count)
            .map_err(|_| HasseError::AllocationFailed {
                context: "derivative factor table",
            })?;
        if F::CHARACTERISTIC == 2 || count == 0 {
            // The parity mask carries no per-position factor in
            // characteristic two; nothing to prepare.
            return Ok(factors);
        }
        let prime = F::CHARACTERISTIC;
        // C(r, r) = 1; C(j + r, r) = C(j − 1 + r, r) · (j + r) / j, with the
        // prime powers stripped from the fraction and their valuation
        // tracked: a positive total valuation is a zero factor, a zero
        // valuation leaves the accumulated unit parts to divide exactly.
        let mut numerator_unit = F::Elem::ONE;
        let mut denominator_unit = F::Elem::ONE;
        let mut valuation: i64 = 0;
        factors.push(Coeff::<F>::new(F::Elem::ONE));
        for j in 1..count {
            let (num_val, num_unit) = split_prime_power((j + order) as u64, prime);
            let (den_val, den_unit) = split_prime_power(j as u64, prime);
            valuation += num_val.cast_signed() - den_val.cast_signed();
            numerator_unit = numerator_unit.mul(embed_integer::<F>(num_unit));
            denominator_unit = denominator_unit.mul(embed_integer::<F>(den_unit));
            let value = if valuation > 0 {
                F::Elem::ZERO
            } else {
                numerator_unit.mul(denominator_unit.inv())
            };
            factors.push(Coeff::<F>::new(value));
        }
        Ok(factors)
    }
}

impl<F: PolynomialField> core::fmt::Debug for DerivativePlan<F> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("DerivativePlan")
            .field("order", &self.order)
            .field("max_coefficients", &self.max_coefficients)
            .field("prepared_factors", &self.factors.len())
            .finish_non_exhaustive()
    }
}
