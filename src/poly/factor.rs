//! Square-free, distinct-degree, and equal-degree factorization.

use alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;

use crate::error::{ConfigError, FactorizationError};
use crate::geometry::try_zeroed;

use super::Polynomial;

/// One monic square-free factor and its multiplicity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SquareFreeFactor<F: FieldKernels> {
    /// The monic square-free factor.
    pub factor: Polynomial<F>,
    /// Its positive multiplicity in the input polynomial.
    pub multiplicity: usize,
}

/// The product of all monic irreducible factors of one degree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistinctDegreeFactor<F: FieldKernels> {
    /// The monic product of the irreducible factors.
    pub factor: Polynomial<F>,
    /// Degree of every irreducible factor in `factor`.
    pub factor_degree: usize,
}

/// One monic irreducible factor and its multiplicity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrreducibleFactor<F: FieldKernels> {
    /// The monic irreducible factor.
    pub factor: Polynomial<F>,
    /// Its positive multiplicity in the input polynomial.
    pub multiplicity: usize,
}

impl<F: FieldKernels> Polynomial<F> {
    /// Decompose a nonzero polynomial into monic square-free factors.
    ///
    /// The factors are ordered by increasing multiplicity. A nonzero constant
    /// has no factors. The input's leading coefficient is omitted because it
    /// is a unit in the coefficient field.
    ///
    /// # Errors
    ///
    /// Returns [`FactorizationError::ZeroPolynomial`] for zero, or propagates
    /// checked arithmetic and allocation failures.
    pub fn square_free_factorization(
        &self,
    ) -> Result<Vec<SquareFreeFactor<F>>, FactorizationError> {
        if self.is_zero() {
            return Err(FactorizationError::ZeroPolynomial);
        }
        let mut factors = Vec::new();
        square_free_recursive(&self.monic(), 1, &mut factors)?;
        factors.sort_by_key(|entry| entry.multiplicity);
        Ok(factors)
    }

    /// Group a monic square-free polynomial by irreducible-factor degree.
    ///
    /// Each returned factor is monic and the groups are ordered by increasing
    /// irreducible degree. A nonzero constant has no groups.
    ///
    /// # Errors
    ///
    /// Returns [`FactorizationError::ZeroPolynomial`] for zero and
    /// [`FactorizationError::NotSquareFree`] when the input has a repeated
    /// irreducible factor.
    pub fn distinct_degree_factorization(
        &self,
    ) -> Result<Vec<DistinctDegreeFactor<F>>, FactorizationError> {
        if self.is_zero() {
            return Err(FactorizationError::ZeroPolynomial);
        }
        let mut remaining = self.monic();
        if remaining.degree().is_none_or(|degree| degree == 0) {
            return Ok(Vec::new());
        }
        ensure_square_free(&remaining)?;

        let x = Self::from_coefficients(&[F::Elem::ZERO, F::Elem::ONE])?;
        let mut frobenius = x.clone();
        let mut groups = Vec::new();
        let mut factor_degree = 1_usize;

        while remaining
            .degree()
            .is_some_and(|degree| factor_degree <= degree / 2)
        {
            frobenius = frobenius.pow_mod(F::ORDER, &remaining)?;
            let x_mod = x.remainder(&remaining)?;
            let difference = frobenius.sub(&x_mod)?;
            let factor = remaining.gcd(&difference)?;
            if !factor.is_one() {
                push_group(&mut groups, factor.clone(), factor_degree)?;
                remaining = remaining.divide_exact(&factor)?;
                if remaining.is_one() {
                    break;
                }
                frobenius = frobenius.remainder(&remaining)?;
            }
            factor_degree = factor_degree
                .checked_add(1)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "distinct-degree factor index",
                })?;
        }

        if !remaining.is_one() {
            let degree = remaining.degree().ok_or(FactorizationError::Invariant {
                reason: "a remaining distinct-degree factor was zero",
            })?;
            push_group(&mut groups, remaining, degree)?;
        }
        Ok(groups)
    }

    /// Split a square-free product of irreducibles of one degree.
    ///
    /// The returned factors are monic and sorted by their canonical packed
    /// coefficient representation. The algorithm uses absolute traces in
    /// characteristic two and quadratic characters in odd characteristic.
    /// Candidate polynomials are enumerated deterministically until a proper
    /// divisor is found. The enumeration always terminates, but its worst-case
    /// cost is not polynomially bounded.
    ///
    /// # Errors
    ///
    /// Returns [`FactorizationError::ZeroFactorDegree`] when `factor_degree`
    /// is zero, [`FactorizationError::NotSquareFree`] for repeated factors,
    /// and [`FactorizationError::NotEqualDegree`] when an irreducible factor
    /// has a different degree.
    pub fn equal_degree_factorization(
        &self,
        factor_degree: usize,
    ) -> Result<Vec<Polynomial<F>>, FactorizationError> {
        if factor_degree == 0 {
            return Err(FactorizationError::ZeroFactorDegree);
        }
        if self.is_zero() {
            return Err(FactorizationError::ZeroPolynomial);
        }
        let monic = self.monic();
        if monic.is_one() {
            return Ok(Vec::new());
        }

        let groups = monic.distinct_degree_factorization()?;
        if groups.len() != 1
            || groups[0].factor_degree != factor_degree
            || groups[0].factor != monic
        {
            return Err(FactorizationError::NotEqualDegree { factor_degree });
        }

        let input_degree = monic.degree().ok_or(FactorizationError::Invariant {
            reason: "a validated equal-degree input was zero",
        })?;
        let factor_count = input_degree / factor_degree;
        let mut pending = Vec::new();
        let mut output = Vec::new();
        reserve_polynomials(&mut pending, factor_count, "equal-degree factor stack")?;
        reserve_polynomials(&mut output, factor_count, "equal-degree factors")?;
        pending.push(monic);

        while let Some(factor) = pending.pop() {
            let degree = factor.degree().ok_or(FactorizationError::Invariant {
                reason: "the equal-degree factor stack contained zero",
            })?;
            if degree == factor_degree {
                output.push(factor);
                continue;
            }
            let (left, right) = split_equal_degree(&factor, factor_degree)?;
            pending.push(right);
            pending.push(left);
        }

        output.sort_by(|left, right| left.as_packed().cmp(right.as_packed()));
        Ok(output)
    }

    /// Factor a nonzero polynomial into monic irreducible factors.
    ///
    /// The input's leading coefficient is omitted because it is a unit in the
    /// coefficient field. Factors are sorted by their canonical packed
    /// coefficient representation, then by multiplicity. A nonzero constant
    /// has no factors.
    ///
    /// # Errors
    ///
    /// Returns [`FactorizationError::ZeroPolynomial`] for zero, or propagates
    /// checked arithmetic and allocation failures.
    pub fn factor(&self) -> Result<Vec<IrreducibleFactor<F>>, FactorizationError> {
        let square_free = self.square_free_factorization()?;
        let mut output = Vec::new();
        for component in square_free {
            for group in component.factor.distinct_degree_factorization()? {
                for factor in group
                    .factor
                    .equal_degree_factorization(group.factor_degree)?
                {
                    reserve_one(&mut output, "irreducible factors")?;
                    output.push(IrreducibleFactor {
                        factor,
                        multiplicity: component.multiplicity,
                    });
                }
            }
        }
        output.sort_by(|left, right| {
            left.factor
                .as_packed()
                .cmp(right.factor.as_packed())
                .then_with(|| left.multiplicity.cmp(&right.multiplicity))
        });
        Ok(output)
    }

    /// Whether this polynomial is irreducible over its coefficient field.
    ///
    /// Zero and nonzero constants are not irreducible. The result is
    /// independent of the polynomial's nonzero leading coefficient.
    ///
    /// # Errors
    ///
    /// Returns [`FactorizationError`] when supporting polynomial arithmetic or
    /// storage fails.
    pub fn is_irreducible(&self) -> Result<bool, FactorizationError> {
        let Some(degree) = self.degree() else {
            return Ok(false);
        };
        if degree == 0 {
            return Ok(false);
        }
        let monic = self.monic();
        let derivative = monic.formal_derivative()?;
        if derivative.is_zero() || !monic.gcd(&derivative)?.is_one() {
            return Ok(false);
        }
        let groups = monic.distinct_degree_factorization()?;
        Ok(groups.len() == 1 && groups[0].factor_degree == degree && groups[0].factor == monic)
    }
}

fn square_free_recursive<F: FieldKernels>(
    polynomial: &Polynomial<F>,
    multiplicity_scale: usize,
    output: &mut Vec<SquareFreeFactor<F>>,
) -> Result<(), FactorizationError> {
    if polynomial.degree().is_none_or(|degree| degree == 0) {
        return Ok(());
    }
    let derivative = polynomial.formal_derivative()?;
    if derivative.is_zero() {
        let root = characteristic_root(polynomial)?;
        let scale = multiplicity_scale
            .checked_mul(characteristic_usize::<F>())
            .ok_or(ConfigError::GeometryOverflow {
                context: "square-free factor multiplicity",
            })?;
        return square_free_recursive(&root, scale, output);
    }

    let mut repeated = polynomial.gcd(&derivative)?;
    let mut residual = polynomial.divide_exact(&repeated)?;
    let mut multiplicity = 1_usize;
    while !residual.is_one() {
        let shared = residual.gcd(&repeated)?;
        let factor = residual.divide_exact(&shared)?;
        if !factor.is_one() {
            let full_multiplicity = multiplicity.checked_mul(multiplicity_scale).ok_or(
                ConfigError::GeometryOverflow {
                    context: "square-free factor multiplicity",
                },
            )?;
            push_square_free(output, factor, full_multiplicity)?;
        }
        residual = shared;
        repeated = repeated.divide_exact(&residual)?;
        multiplicity = multiplicity
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "square-free factor multiplicity",
            })?;
    }

    if !repeated.is_one() {
        let root = characteristic_root(&repeated)?;
        let scale = multiplicity_scale
            .checked_mul(characteristic_usize::<F>())
            .ok_or(ConfigError::GeometryOverflow {
                context: "square-free factor multiplicity",
            })?;
        square_free_recursive(&root, scale, output)?;
    }
    Ok(())
}

fn characteristic_root<F: FieldKernels>(
    polynomial: &Polynomial<F>,
) -> Result<Polynomial<F>, FactorizationError> {
    let characteristic = characteristic_usize::<F>();
    let degree = polynomial.degree().ok_or(FactorizationError::Invariant {
        reason: "a characteristic root was requested for zero",
    })?;
    let count = degree / characteristic + 1;
    let mut coefficients = try_zeroed::<F::Elem>("characteristic-root coefficients", count)?;
    let root_exponent = (F::ORDER / u128::from(F::CHARACTERISTIC)) as u64;
    for source_degree in 0..=degree {
        let coefficient = polynomial.coefficient(source_degree);
        if source_degree.is_multiple_of(characteristic) {
            coefficients[source_degree / characteristic] = coefficient.pow(root_exponent);
        } else {
            debug_assert!(coefficient.is_zero());
        }
    }
    Ok(Polynomial::from_coefficients(&coefficients)?)
}

fn ensure_square_free<F: FieldKernels>(
    polynomial: &Polynomial<F>,
) -> Result<(), FactorizationError> {
    let derivative = polynomial.formal_derivative()?;
    if derivative.is_zero() || !polynomial.gcd(&derivative)?.is_one() {
        Err(FactorizationError::NotSquareFree)
    } else {
        Ok(())
    }
}

fn split_equal_degree<F: FieldKernels>(
    factor: &Polynomial<F>,
    irreducible_degree: usize,
) -> Result<(Polynomial<F>, Polynomial<F>), FactorizationError> {
    let factor_degree = factor.degree().ok_or(FactorizationError::Invariant {
        reason: "equal-degree splitting received zero",
    })?;
    let mut digits = try_zeroed::<u128>("factorization candidate digits", factor_degree)?;

    loop {
        for candidate_degree in 1..factor_degree {
            let candidate = candidate_polynomial::<F>(&digits, candidate_degree)?;
            let direct = factor.gcd(&candidate)?;
            if is_proper_factor(&direct, factor_degree) {
                let quotient = factor.divide_exact(&direct)?;
                return Ok((direct, quotient));
            }

            let separator = if F::CHARACTERISTIC == 2 {
                absolute_trace(&candidate, factor, irreducible_degree)?
            } else {
                quadratic_character(&candidate, factor, irreducible_degree)?
                    .sub(&Polynomial::one()?)?
            };
            let divisor = factor.gcd(&separator)?;
            if is_proper_factor(&divisor, factor_degree) {
                let quotient = factor.divide_exact(&divisor)?;
                return Ok((divisor, quotient));
            }
        }
        assert!(
            increment_digits(&mut digits, F::ORDER),
            "the complete candidate set contains an equal-degree separator"
        );
    }
}

fn candidate_polynomial<F: FieldKernels>(
    digits: &[u128],
    degree: usize,
) -> Result<Polynomial<F>, FactorizationError> {
    let mut coefficients = try_zeroed::<F::Elem>("factorization candidate", degree + 1)?;
    for (coefficient, &digit) in coefficients[..degree].iter_mut().zip(digits) {
        *coefficient = digit_element::<F>(digit);
    }
    coefficients[degree] = F::Elem::ONE;
    Ok(Polynomial::from_coefficients(&coefficients)?)
}

fn digit_element<F: FieldKernels>(digit: u128) -> F::Elem {
    if digit == 0 {
        return F::Elem::ZERO;
    }
    let exponent = (digit - 1) as u64;
    F::GENERATOR.pow(exponent)
}

fn increment_digits(digits: &mut [u128], radix: u128) -> bool {
    for digit in digits {
        *digit += 1;
        if *digit != radix {
            return true;
        }
        *digit = 0;
    }
    false
}

fn absolute_trace<F: FieldKernels>(
    candidate: &Polynomial<F>,
    modulus: &Polynomial<F>,
    irreducible_degree: usize,
) -> Result<Polynomial<F>, FactorizationError> {
    let rounds = usize::try_from(F::ORDER.trailing_zeros())
        .ok()
        .and_then(|extension| extension.checked_mul(irreducible_degree))
        .ok_or(ConfigError::GeometryOverflow {
            context: "absolute-trace rounds",
        })?;
    let mut term = candidate.remainder(modulus)?;
    let mut trace = Polynomial::zero();
    for _ in 0..rounds {
        trace.add_assign(&term)?;
        term = term.square_mod(modulus)?;
    }
    Ok(trace)
}

fn quadratic_character<F: FieldKernels>(
    candidate: &Polynomial<F>,
    modulus: &Polynomial<F>,
    irreducible_degree: usize,
) -> Result<Polynomial<F>, FactorizationError> {
    let half_base_order = (F::ORDER - 1) / 2;
    let mut term = candidate.pow_mod(half_base_order, modulus)?;
    let mut character = term.clone();
    for _ in 1..irreducible_degree {
        term = term.pow_mod(F::ORDER, modulus)?;
        character = character.multiply_mod(&term, modulus)?;
    }
    Ok(character)
}

fn is_proper_factor<F: FieldKernels>(factor: &Polynomial<F>, parent_degree: usize) -> bool {
    factor
        .degree()
        .is_some_and(|degree| degree != 0 && degree != parent_degree)
}

fn characteristic_usize<F: Field>() -> usize {
    F::CHARACTERISTIC as usize
}

fn push_square_free<F: FieldKernels>(
    output: &mut Vec<SquareFreeFactor<F>>,
    factor: Polynomial<F>,
    multiplicity: usize,
) -> Result<(), FactorizationError> {
    reserve_one(output, "square-free factors")?;
    output.push(SquareFreeFactor {
        factor,
        multiplicity,
    });
    Ok(())
}

fn push_group<F: FieldKernels>(
    output: &mut Vec<DistinctDegreeFactor<F>>,
    factor: Polynomial<F>,
    factor_degree: usize,
) -> Result<(), FactorizationError> {
    reserve_one(output, "distinct-degree factors")?;
    output.push(DistinctDegreeFactor {
        factor,
        factor_degree,
    });
    Ok(())
}

fn reserve_polynomials<F: FieldKernels>(
    values: &mut Vec<Polynomial<F>>,
    capacity: usize,
    context: &'static str,
) -> Result<(), FactorizationError> {
    if values.capacity() < capacity {
        values
            .try_reserve_exact(capacity - values.capacity())
            .map_err(|_| ConfigError::AllocationFailed {
                context,
                elements: capacity,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })?;
    }
    Ok(())
}

fn reserve_one<T>(values: &mut Vec<T>, context: &'static str) -> Result<(), FactorizationError> {
    values
        .try_reserve(1)
        .map_err(|_| ConfigError::AllocationFailed {
            context,
            elements: values.len().saturating_add(1),
            element_size: core::mem::size_of::<T>(),
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_digits_report_exhaustion() {
        let mut digits = [1_u128];
        assert!(!increment_digits(&mut digits, 2));
        assert_eq!(digits, [0]);
    }
}
