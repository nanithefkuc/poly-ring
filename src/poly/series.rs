//! Truncated power-series arithmetic: Newton inversion and series division.

use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use crate::error::PolynomialError;

use super::dense::Polynomial;

impl<F: FieldKernels> Polynomial<F> {
    /// Return `self^{-1} mod x^t` by Newton doubling.
    ///
    /// The Newton step is `b ← b·(2 − a·b) mod x^{2k}`: if `a·b ≡ 1 (mod
    /// x^k)` then the update is correct modulo `x^{2k}`, so each step doubles
    /// the correct prefix. Seeded from the inverse of the constant term, the
    /// inverse costs `O(M(t))` against the `O(t²)` of the linear
    /// coefficient-at-a-time solve.
    ///
    /// In characteristic two the step collapses to `b ← a·b²`, which is what
    /// this runs there: `2 = 0`, so `b·(2 − a·b) = b·a·b = a·b²`. The two
    /// forms agree, and the cheaper one keeps the binary cost unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::ZeroConstantTerm`] when the constant
    /// coefficient is zero (the series is not invertible modulo `x^t`) and
    /// [`PolynomialError::Config`] when an intermediate buffer cannot be
    /// reserved.
    pub fn inverse_mod_x_power(&self, t: usize) -> Result<Self, PolynomialError> {
        if t == 0 {
            return Ok(Self::zero());
        }
        if self.is_zero() {
            return Err(PolynomialError::ZeroConstantTerm {
                context: "truncated power-series inversion",
            });
        }
        let constant = self.coefficient(0);
        if constant.is_zero() {
            return Err(PolynomialError::ZeroConstantTerm {
                context: "truncated power-series inversion",
            });
        }
        let mut inverse = Self::constant(constant.inv())?;
        let two = Self::constant(F::Elem::ONE.add(F::Elem::ONE))?;
        let mut precision = 1_usize;
        while precision < t {
            let doubled = precision.saturating_mul(2).min(t);
            if F::CHARACTERISTIC == 2 {
                // b ← a·b², the characteristic-two collapse of the general
                // update.
                let squared = inverse.multiply_truncated(&inverse, doubled)?;
                inverse = self.multiply_truncated(&squared, doubled)?;
            } else {
                // b ← b·(2 − a·b), keeping coefficients below the doubled
                // precision throughout.
                let product = self.multiply_truncated(&inverse, doubled)?;
                let correction = two.add_scaled(F::Elem::ONE.neg(), &product)?;
                inverse = inverse.multiply_truncated(&correction, doubled)?;
            }
            precision = doubled;
        }
        inverse.truncate(t);
        Ok(inverse)
    }

    /// Return `self^{-1} mod x^t`, solving one coefficient at a time.
    ///
    /// This is the deliberately naive form kept beside the Newton doubling:
    /// it costs `O(t²)` and shares no code with
    /// [`Self::inverse_mod_x_power`], so the two agree or one is wrong.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::ZeroConstantTerm`] when the constant
    /// coefficient is zero and [`PolynomialError::Config`] when the
    /// coefficient buffer cannot be reserved.
    pub fn inverse_mod_x_power_naive(&self, t: usize) -> Result<Self, PolynomialError> {
        if t == 0 {
            return Ok(Self::zero());
        }
        let constant = self.coefficient(0);
        if constant.is_zero() {
            return Err(PolynomialError::ZeroConstantTerm {
                context: "linear truncated power-series inversion",
            });
        }
        let mut coefficients = crate::geometry::try_zeroed::<F::Elem>("series inverse", t)?;
        coefficients[0] = constant.inv();
        for degree in 1..t {
            // 0 = sum_{j<=degree} a_j · b_{degree−j}, so
            // b_degree = −a_0^{-1} · sum_{j>=1} a_j · b_{degree−j}.
            let mut discrepancy = F::Elem::ZERO;
            for j in 1..=degree {
                discrepancy = discrepancy.add(self.coefficient(j).mul(coefficients[degree - j]));
            }
            coefficients[degree] = discrepancy.neg().mul(coefficients[0]);
        }
        Self::from_coefficients(&coefficients)
    }

    /// Return `self` with coefficients in reverse order.
    ///
    /// `reverse(p)(X) = X^{deg p} · p(1/X)`; the zero polynomial reverses to
    /// itself. This is the standard Padé helper pairing with
    /// [`Self::inverse_mod_x_power`].
    #[must_use]
    pub fn reverse(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let count = self.coefficient_count();
        let mut reversed = self.clone();
        for degree in 0..count {
            let start = degree * F::BYTES;
            let end = start + F::BYTES;
            let destination = count - 1 - degree;
            let destination_start = destination * F::BYTES;
            let destination_end = destination_start + F::BYTES;
            let coefficient = self.coefficients[start..end].to_vec();
            reversed.coefficients[destination_start..destination_end].copy_from_slice(&coefficient);
        }
        reversed
    }
}

/// Return `a * b^{-1} mod x^t`.
///
/// # Errors
///
/// Returns [`PolynomialError::ZeroConstantTerm`] when `b`'s constant
/// coefficient is zero and [`PolynomialError::Config`] when an intermediate
/// buffer cannot be reserved.
pub fn series_divide<F: FieldKernels>(
    a: &Polynomial<F>,
    b: &Polynomial<F>,
    t: usize,
) -> Result<Polynomial<F>, PolynomialError> {
    let inverse = b.inverse_mod_x_power(t)?;
    a.multiply_truncated(&inverse, t)
}
