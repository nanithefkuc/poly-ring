//! Euclidean division, exact division, and modular reduction.

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::error::{ConfigError, PolynomialError};
use crate::geometry::try_zeroed;

use super::dense::Polynomial;

impl<F: FieldKernels> Polynomial<F> {
    /// Divide by `divisor`, returning quotient and remainder.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] for a zero divisor and
    /// [`PolynomialError::Config`] when an output buffer cannot be reserved.
    ///
    /// # Panics
    ///
    /// The internal leading-coefficient expectations hold for every nonzero
    /// divisor; the zero divisor is rejected with an error first.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolynomialError> {
        let Some(divisor_degree) = divisor.degree() else {
            return Err(PolynomialError::DivisionByZero);
        };
        let Some(dividend_degree) = self.degree() else {
            return Ok((Self::zero(), Self::zero()));
        };
        if dividend_degree < divisor_degree {
            return Ok((Self::zero(), self.clone()));
        }

        let quotient_count = dividend_degree - divisor_degree + 1;
        let mut quotient = try_zeroed::<F::Elem>("polynomial quotient", quotient_count)?;
        let mut remainder = self.clone();
        let divisor_leading_inverse = divisor
            .leading_coefficient()
            .expect("nonzero divisor has a leading coefficient")
            .inv();

        while let Some(remainder_degree) = remainder.degree() {
            if remainder_degree < divisor_degree {
                break;
            }
            let shift = remainder_degree - divisor_degree;
            let scale = remainder
                .leading_coefficient()
                .expect("nonzero remainder has a leading coefficient")
                .mul(divisor_leading_inverse);
            quotient[shift] = quotient[shift].add(scale);
            // Eliminate with the negated scale: `r -= scale · X^shift · g`.
            // In characteristic two negation is the identity and this is the
            // same XOR it always was.
            remainder.add_scaled_packed_at(scale.neg(), divisor.as_packed(), shift)?;
        }
        Ok((Self::from_coefficients(&quotient)?, remainder))
    }

    /// Divide exactly, returning an error when the remainder is nonzero.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] for a zero divisor,
    /// [`PolynomialError::NonExactDivision`] when the remainder is nonzero,
    /// and [`PolynomialError::Config`] when an output buffer cannot be
    /// reserved.
    pub fn exact_divide(&self, divisor: &Self) -> Result<Self, PolynomialError> {
        let (quotient, remainder) = self.div_rem(divisor)?;
        if remainder.is_zero() {
            Ok(quotient)
        } else {
            Err(PolynomialError::NonExactDivision)
        }
    }

    /// Return a monic copy, leaving zero unchanged.
    #[must_use]
    pub fn monic(&self) -> Self {
        let Some(leading) = self.leading_coefficient() else {
            return Self::zero();
        };
        self.scaled(leading.inv())
    }

    /// Return `self mod modulus`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] for a zero modulus and
    /// [`PolynomialError::Config`] when an output buffer cannot be reserved.
    pub fn remainder(&self, modulus: &Self) -> Result<Self, PolynomialError> {
        self.div_rem(modulus).map(|(_, remainder)| remainder)
    }

    /// Multiply and reduce modulo `modulus`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] for a zero modulus and
    /// [`PolynomialError::Config`] when an output buffer cannot be reserved.
    pub fn multiply_mod(&self, other: &Self, modulus: &Self) -> Result<Self, PolynomialError> {
        self.multiply(other)?.remainder(modulus)
    }

    /// Square and reduce modulo `modulus`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] for a zero modulus and
    /// [`PolynomialError::Config`] when an output buffer cannot be reserved.
    pub fn square_mod(&self, modulus: &Self) -> Result<Self, PolynomialError> {
        self.multiply_mod(self, modulus)
    }

    /// Raise to `exponent` modulo `modulus` by square-and-multiply.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] for a zero modulus and
    /// [`PolynomialError::Config`] when an output buffer cannot be reserved.
    pub fn pow_mod(&self, mut exponent: u128, modulus: &Self) -> Result<Self, PolynomialError> {
        if modulus.is_zero() {
            return Err(PolynomialError::DivisionByZero);
        }
        let mut result = Self::one()?.remainder(modulus)?;
        let mut base = self.remainder(modulus)?;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.multiply_mod(&base, modulus)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.square_mod(modulus)?;
            }
        }
        Ok(result)
    }

    /// Smallest exponent with a nonzero coefficient.
    #[must_use]
    pub fn x_valuation(&self) -> Option<usize> {
        self.coefficients()
            .position(|coefficient| !coefficient.is_zero())
    }

    /// Divide exactly by `X^power`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::NonExactDivision`] when the valuation is
    /// below `power` and [`PolynomialError::Config`] when the quotient buffer
    /// cannot be reserved.
    ///
    /// # Panics
    ///
    /// The coefficient-aligned suffix expectation holds for the validated
    /// valuation this function checks first.
    pub fn divide_by_x_power(&self, power: usize) -> Result<Self, PolynomialError> {
        if self.is_zero() || power == 0 {
            return Ok(self.clone());
        }
        if self.x_valuation().is_none_or(|valuation| valuation < power) {
            return Err(PolynomialError::NonExactDivision);
        }
        let byte_offset = power
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial X-power offset",
            })?;
        let packed = self.coefficients[byte_offset..].to_vec();
        Ok(Self::from_packed(packed).expect("coefficient-aligned suffix"))
    }

    /// Return `scale * X^shift * self`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the shifted buffer cannot be
    /// reserved.
    pub fn scaled_shifted(&self, scale: F::Elem, shift: usize) -> Result<Self, PolynomialError> {
        let mut result = Self::zero();
        result.add_scaled_packed_at(scale, self.as_packed(), shift)?;
        Ok(result)
    }

    /// Write quotient and remainder into reusable output storage.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] for a zero divisor and
    /// [`PolynomialError::Config`] when an output buffer cannot be reserved.
    ///
    /// # Panics
    ///
    /// The internal leading-coefficient expectations hold for every nonzero
    /// divisor; the zero divisor is rejected with an error first.
    pub fn div_rem_into(
        &self,
        divisor: &Self,
        quotient: &mut Self,
        remainder: &mut Self,
    ) -> Result<(), PolynomialError> {
        let Some(divisor_degree) = divisor.degree() else {
            return Err(PolynomialError::DivisionByZero);
        };
        quotient.set_zero();
        let Some(dividend_degree) = self.degree() else {
            remainder.set_zero();
            return Ok(());
        };
        remainder.assign_from(self);
        if dividend_degree < divisor_degree {
            return Ok(());
        }
        quotient.resize_coefficients(dividend_degree - divisor_degree + 1)?;
        let divisor_leading_inverse = divisor
            .leading_coefficient()
            .expect("nonzero divisor has a leading coefficient")
            .inv();
        while let Some(remainder_degree) = remainder.degree() {
            if remainder_degree < divisor_degree {
                break;
            }
            let shift = remainder_degree - divisor_degree;
            let scale = remainder
                .leading_coefficient()
                .expect("nonzero remainder has a leading coefficient")
                .mul(divisor_leading_inverse);
            let start = shift * F::BYTES;
            let updated = F::read(&quotient.coefficients[start..start + F::BYTES]).add(scale);
            F::write(&mut quotient.coefficients[start..start + F::BYTES], updated);
            remainder.add_scaled_shifted_assign(scale.neg(), divisor, shift)?;
        }
        quotient.normalize();
        Ok(())
    }

    /// Divide exactly by `X^power` into reusable output storage.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::NonExactDivision`] when the valuation is
    /// below `power` and [`PolynomialError::Config`] when the quotient buffer
    /// cannot be reserved.
    pub fn divide_by_x_power_into(
        &self,
        power: usize,
        out: &mut Self,
    ) -> Result<(), PolynomialError> {
        if self.is_zero() || power == 0 {
            out.assign_from(self);
            return Ok(());
        }
        if self.x_valuation().is_none_or(|valuation| valuation < power) {
            return Err(PolynomialError::NonExactDivision);
        }
        let byte_offset = power
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial X-power offset",
            })?;
        out.assign_packed(&self.coefficients[byte_offset..])?;
        Ok(())
    }
}
