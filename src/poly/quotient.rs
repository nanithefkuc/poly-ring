//! Prepared arithmetic in polynomial quotient rings.

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::error::PolynomialError;

use super::Polynomial;

/// A reusable plan for arithmetic modulo one positive-degree polynomial.
///
/// Construction normalizes the modulus to monic form. The plan is immutable;
/// mutable intermediate storage belongs to [`ModulusScratch`], so one plan can
/// be shared while each execution owns its workspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModulusPlan<F: FieldKernels> {
    modulus: Polynomial<F>,
}

/// Reusable workspace for [`ModulusPlan`] operations.
///
/// Scratch carries no semantics between calls. Repeating an operation with
/// warmed buffers reuses their retained polynomial storage.
#[derive(Debug)]
pub struct ModulusScratch<F: FieldKernels> {
    quotient: Polynomial<F>,
    product: Polynomial<F>,
    accumulator: Polynomial<F>,
    accumulator_next: Polynomial<F>,
    base: Polynomial<F>,
    base_next: Polynomial<F>,
    remainder: Polynomial<F>,
    r_old: Polynomial<F>,
    r: Polynomial<F>,
    s_old: Polynomial<F>,
    s: Polynomial<F>,
    cofactor_next: Polynomial<F>,
}

impl<F: FieldKernels> ModulusPlan<F> {
    /// Prepare arithmetic modulo `modulus`.
    ///
    /// Multiplying a modulus by a nonzero field element does not change its
    /// ideal, so the stored modulus is monic.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] for the zero polynomial and
    /// [`PolynomialError::ConstantModulus`] for a nonzero constant.
    pub fn new(modulus: &Polynomial<F>) -> Result<Self, PolynomialError> {
        match modulus.degree() {
            None => Err(PolynomialError::DivisionByZero),
            Some(0) => Err(PolynomialError::ConstantModulus),
            Some(_) => Ok(Self {
                modulus: modulus.monic(),
            }),
        }
    }

    /// The monic modulus defining the quotient ring.
    #[must_use]
    pub fn modulus(&self) -> &Polynomial<F> {
        &self.modulus
    }

    /// Construct empty reusable execution workspace.
    #[must_use]
    pub const fn scratch(&self) -> ModulusScratch<F> {
        ModulusScratch::new()
    }

    /// Return the canonical remainder of `value`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when division or output growth fails.
    pub fn reduce(&self, value: &Polynomial<F>) -> Result<Polynomial<F>, PolynomialError> {
        let mut scratch = self.scratch();
        let mut output = Polynomial::zero();
        self.reduce_into(value, &mut scratch, &mut output)?;
        Ok(output)
    }

    /// Overwrite `output` with the canonical remainder of `value`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when division or output growth fails.
    pub fn reduce_into(
        &self,
        value: &Polynomial<F>,
        scratch: &mut ModulusScratch<F>,
        output: &mut Polynomial<F>,
    ) -> Result<(), PolynomialError> {
        value.div_rem_into(&self.modulus, &mut scratch.quotient, output)
    }

    /// Return `left * right` in the quotient ring.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when multiplication, reduction, or output
    /// growth fails.
    pub fn multiply(
        &self,
        left: &Polynomial<F>,
        right: &Polynomial<F>,
    ) -> Result<Polynomial<F>, PolynomialError> {
        let mut scratch = self.scratch();
        let mut output = Polynomial::zero();
        self.multiply_into(left, right, &mut scratch, &mut output)?;
        Ok(output)
    }

    /// Overwrite `output` with `left * right` in the quotient ring.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when multiplication, reduction, or output
    /// growth fails.
    pub fn multiply_into(
        &self,
        left: &Polynomial<F>,
        right: &Polynomial<F>,
        scratch: &mut ModulusScratch<F>,
        output: &mut Polynomial<F>,
    ) -> Result<(), PolynomialError> {
        multiply_reduce(
            &self.modulus,
            output,
            left,
            right,
            &mut scratch.product,
            &mut scratch.quotient,
        )
    }

    /// Return `value^2` in the quotient ring.
    ///
    /// # Errors
    ///
    /// As [`Self::multiply`].
    pub fn square(&self, value: &Polynomial<F>) -> Result<Polynomial<F>, PolynomialError> {
        self.multiply(value, value)
    }

    /// Overwrite `output` with `value^2` in the quotient ring.
    ///
    /// # Errors
    ///
    /// As [`Self::multiply_into`].
    pub fn square_into(
        &self,
        value: &Polynomial<F>,
        scratch: &mut ModulusScratch<F>,
        output: &mut Polynomial<F>,
    ) -> Result<(), PolynomialError> {
        self.multiply_into(value, value, scratch, output)
    }

    /// Return `value^exponent` in the quotient ring.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when multiplication, reduction, or output
    /// growth fails.
    pub fn pow(
        &self,
        value: &Polynomial<F>,
        exponent: u128,
    ) -> Result<Polynomial<F>, PolynomialError> {
        let mut scratch = self.scratch();
        let mut output = Polynomial::zero();
        self.pow_into(value, exponent, &mut scratch, &mut output)?;
        Ok(output)
    }

    /// Overwrite `output` with `value^exponent` in the quotient ring.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when multiplication, reduction, or output
    /// growth fails.
    pub fn pow_into(
        &self,
        value: &Polynomial<F>,
        mut exponent: u128,
        scratch: &mut ModulusScratch<F>,
        output: &mut Polynomial<F>,
    ) -> Result<(), PolynomialError> {
        scratch.accumulator.assign_coefficients(&[F::Elem::ONE])?;
        value.div_rem_into(&self.modulus, &mut scratch.quotient, &mut scratch.base)?;
        while exponent != 0 {
            if exponent & 1 != 0 {
                multiply_reduce(
                    &self.modulus,
                    &mut scratch.accumulator_next,
                    &scratch.accumulator,
                    &scratch.base,
                    &mut scratch.product,
                    &mut scratch.quotient,
                )?;
                core::mem::swap(&mut scratch.accumulator, &mut scratch.accumulator_next);
            }
            exponent >>= 1;
            if exponent != 0 {
                multiply_reduce(
                    &self.modulus,
                    &mut scratch.base_next,
                    &scratch.base,
                    &scratch.base,
                    &mut scratch.product,
                    &mut scratch.quotient,
                )?;
                core::mem::swap(&mut scratch.base, &mut scratch.base_next);
            }
        }
        output.assign_from(&scratch.accumulator);
        Ok(())
    }

    /// Return the multiplicative inverse of `value` in the quotient ring.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::NotInvertibleModulo`] when `value` and the
    /// modulus are not coprime, or propagates arithmetic and storage failures.
    pub fn inverse(&self, value: &Polynomial<F>) -> Result<Polynomial<F>, PolynomialError> {
        let mut scratch = self.scratch();
        let mut output = Polynomial::zero();
        self.inverse_into(value, &mut scratch, &mut output)?;
        Ok(output)
    }

    /// Overwrite `output` with the multiplicative inverse of `value`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::NotInvertibleModulo`] when `value` and the
    /// modulus are not coprime, or propagates arithmetic and storage failures.
    pub fn inverse_into(
        &self,
        value: &Polynomial<F>,
        scratch: &mut ModulusScratch<F>,
        output: &mut Polynomial<F>,
    ) -> Result<(), PolynomialError> {
        scratch.r_old.assign_from(&self.modulus);
        value.div_rem_into(&self.modulus, &mut scratch.quotient, &mut scratch.r)?;
        scratch.s_old.set_zero();
        scratch.s.assign_coefficients(&[F::Elem::ONE])?;

        while !scratch.r.is_zero() {
            scratch.r_old.div_rem_into(
                &scratch.r,
                &mut scratch.quotient,
                &mut scratch.remainder,
            )?;
            scratch
                .quotient
                .multiply_into(&scratch.s, &mut scratch.product)?;
            scratch.cofactor_next.assign_from(&scratch.s_old);
            scratch.cofactor_next.sub_assign(&scratch.product)?;
            core::mem::swap(&mut scratch.r_old, &mut scratch.r);
            core::mem::swap(&mut scratch.r, &mut scratch.remainder);
            core::mem::swap(&mut scratch.s_old, &mut scratch.s);
            core::mem::swap(&mut scratch.s, &mut scratch.cofactor_next);
        }

        if scratch.r_old.degree() != Some(0) {
            return Err(PolynomialError::NotInvertibleModulo);
        }
        let unit = scratch.r_old.coefficient(0);
        scratch.s_old.scale_assign(unit.inv());
        scratch
            .s_old
            .div_rem_into(&self.modulus, &mut scratch.quotient, output)?;
        Ok(())
    }

    /// Return `outer(inner)` in the quotient ring.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when multiplication, reduction, or output
    /// growth fails.
    pub fn compose(
        &self,
        outer: &Polynomial<F>,
        inner: &Polynomial<F>,
    ) -> Result<Polynomial<F>, PolynomialError> {
        let mut scratch = self.scratch();
        let mut output = Polynomial::zero();
        self.compose_into(outer, inner, &mut scratch, &mut output)?;
        Ok(output)
    }

    /// Overwrite `output` with `outer(inner)` in the quotient ring.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError`] when multiplication, reduction, or output
    /// growth fails.
    pub fn compose_into(
        &self,
        outer: &Polynomial<F>,
        inner: &Polynomial<F>,
        scratch: &mut ModulusScratch<F>,
        output: &mut Polynomial<F>,
    ) -> Result<(), PolynomialError> {
        inner.div_rem_into(&self.modulus, &mut scratch.quotient, &mut scratch.base)?;
        scratch.accumulator.set_zero();
        for coefficient in outer.coefficients().rev() {
            multiply_reduce(
                &self.modulus,
                &mut scratch.accumulator_next,
                &scratch.accumulator,
                &scratch.base,
                &mut scratch.product,
                &mut scratch.quotient,
            )?;
            if !coefficient.is_zero() {
                let constant = scratch.accumulator_next.coefficient(0).add(coefficient);
                scratch.accumulator_next.set_coefficient(0, constant)?;
            }
            core::mem::swap(&mut scratch.accumulator, &mut scratch.accumulator_next);
        }
        output.assign_from(&scratch.accumulator);
        Ok(())
    }
}

impl<F: FieldKernels> ModulusScratch<F> {
    /// Construct empty reusable quotient-ring workspace.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            quotient: Polynomial::zero(),
            product: Polynomial::zero(),
            accumulator: Polynomial::zero(),
            accumulator_next: Polynomial::zero(),
            base: Polynomial::zero(),
            base_next: Polynomial::zero(),
            remainder: Polynomial::zero(),
            r_old: Polynomial::zero(),
            r: Polynomial::zero(),
            s_old: Polynomial::zero(),
            s: Polynomial::zero(),
            cofactor_next: Polynomial::zero(),
        }
    }
}

impl<F: FieldKernels> Default for ModulusScratch<F> {
    fn default() -> Self {
        Self::new()
    }
}

fn multiply_reduce<F: FieldKernels>(
    modulus: &Polynomial<F>,
    output: &mut Polynomial<F>,
    left: &Polynomial<F>,
    right: &Polynomial<F>,
    product: &mut Polynomial<F>,
    quotient: &mut Polynomial<F>,
) -> Result<(), PolynomialError> {
    left.multiply_into(right, product)?;
    product.div_rem_into(modulus, quotient, output)
}
