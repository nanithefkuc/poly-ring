//! Ring operations: addition, scaling, shifts, products, derivatives.

use alloc::vec::Vec;

use fgf::field::{Elem, Field};
use fgf::kernel::{FieldKernels, backend_for};
use fgf::ops;

use crate::error::{ConfigError, PolynomialError};
use crate::geometry::checked_product;

use super::dense::Polynomial;
use super::karatsuba::{KARATSUBA_CROSSOVER, karatsuba_multiply};
use super::monomial::embed_integer;

impl<F: FieldKernels> Polynomial<F> {
    /// Add `other` in place.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the widened buffer cannot be
    /// reserved.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), PolynomialError> {
        self.add_scaled_assign(F::Elem::ONE, other)
    }

    /// Return `self + other`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the widened buffer cannot be
    /// reserved.
    pub fn add(&self, other: &Self) -> Result<Self, PolynomialError> {
        let mut result = self.clone();
        result.add_assign(other)?;
        Ok(result)
    }

    /// Subtract `other` in place.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the widened buffer cannot be
    /// reserved.
    pub fn sub_assign(&mut self, other: &Self) -> Result<(), PolynomialError> {
        self.add_scaled_assign(F::Elem::ONE.neg(), other)
    }

    /// Return `self - other`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the widened buffer cannot be
    /// reserved.
    pub fn sub(&self, other: &Self) -> Result<Self, PolynomialError> {
        let mut result = self.clone();
        result.sub_assign(other)?;
        Ok(result)
    }

    /// Add `scale * other` in place.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the widened buffer cannot be
    /// reserved.
    pub fn add_scaled_assign(
        &mut self,
        scale: F::Elem,
        other: &Self,
    ) -> Result<(), PolynomialError> {
        self.add_scaled_packed_at(scale, other.as_packed(), 0)
    }

    /// Add `scale * X^shift * other` in place.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the widened buffer cannot be
    /// reserved.
    pub fn add_scaled_shifted_assign(
        &mut self,
        scale: F::Elem,
        other: &Self,
        shift: usize,
    ) -> Result<(), PolynomialError> {
        self.add_scaled_packed_at(scale, other.as_packed(), shift)
    }

    /// Return `self + scale * other`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the widened buffer cannot be
    /// reserved.
    pub fn add_scaled(&self, scale: F::Elem, other: &Self) -> Result<Self, PolynomialError> {
        let mut result = self.clone();
        result.add_scaled_assign(scale, other)?;
        Ok(result)
    }

    /// Multiply every coefficient by `scale` in place.
    pub fn scale_assign(&mut self, scale: F::Elem) {
        if self.is_zero() || scale.is_one() {
            return;
        }
        if scale.is_zero() {
            self.coefficients.clear();
            return;
        }
        if use_packed_kernel::<F>(self.coefficients.len()) {
            ops::mul_assign::<F>(&mut self.coefficients, scale);
        } else {
            for start in (0..self.coefficients.len()).step_by(F::BYTES) {
                let coefficient = &mut self.coefficients[start..start + F::BYTES];
                F::encode(coefficient, F::decode(coefficient).mul(scale));
            }
        }
        self.normalize();
    }

    /// Return `scale * self`.
    #[must_use]
    pub fn scaled(&self, scale: F::Elem) -> Self {
        let mut result = self.clone();
        result.scale_assign(scale);
        result
    }

    /// Replace every coefficient by its additive inverse.
    pub fn negate_assign(&mut self) {
        self.scale_assign(F::Elem::ONE.neg());
    }

    /// Return the additive inverse.
    #[must_use]
    pub fn negated(&self) -> Self {
        self.scaled(F::Elem::ONE.neg())
    }

    /// Return `X^amount * self`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the shifted buffer cannot be
    /// reserved.
    pub fn shifted(&self, amount: usize) -> Result<Self, PolynomialError> {
        if self.is_zero() {
            return Ok(Self::zero());
        }
        let mut result = Self::zero();
        result.add_scaled_packed_at(F::Elem::ONE, self.as_packed(), amount)?;
        Ok(result)
    }

    /// Return `(X + constant) * self`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the product buffer cannot be
    /// reserved.
    pub fn multiply_x_plus(&self, constant: F::Elem) -> Result<Self, PolynomialError> {
        let mut result = self.shifted(1)?;
        result.add_scaled_assign(constant, self)?;
        Ok(result)
    }
    /// Return the product, dispatched by operand size.
    ///
    /// Goldilocks, `QuadMersenne31`, and Mersenne31 (embedded) products at
    /// their measured crossovers use an allocating NTT path. Smaller
    /// products and fields without that route use schoolbook or Karatsuba.
    /// Batched products select independently through
    /// [`crate::poly::multiply_batch_truncated_into`].
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the product buffer cannot
    /// be reserved.
    pub fn multiply(&self, other: &Self) -> Result<Self, PolynomialError> {
        if self.is_zero() || other.is_zero() {
            return Ok(Self::zero());
        }
        let output_count = self
            .coefficient_count()
            .checked_add(other.coefficient_count())
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial product coefficients",
            })?;
        if let Some(coefficients) = super::convolution::oneshot_ntt_product::<F>(
            &self.coefficients,
            self.coefficient_count(),
            &other.coefficients,
            other.coefficient_count(),
        ) && let Some(product) = Polynomial::from_packed(coefficients)
        {
            return Ok(product);
        }
        if self.coefficient_count().min(other.coefficient_count()) >= KARATSUBA_CROSSOVER {
            karatsuba_multiply(self, other)
        } else {
            self.multiply_truncated(other, output_count)
        }
    }

    /// Return the product truncated to coefficients below `coefficient_count`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the truncated product buffer
    /// cannot be reserved.
    pub fn multiply_truncated(
        &self,
        other: &Self,
        coefficient_count: usize,
    ) -> Result<Self, PolynomialError> {
        let mut result = Self::zero();
        self.multiply_truncated_into(other, coefficient_count, &mut result)?;
        Ok(result)
    }

    /// Evaluate at one field element with Horner's rule.
    #[must_use]
    pub fn evaluate(&self, point: F::Elem) -> F::Elem {
        self.coefficients()
            .rev()
            .fold(F::Elem::ZERO, |value, coefficient| {
                value.mul(point).add(coefficient)
            })
    }

    /// Evaluate independently at every supplied point.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the output vector cannot be
    /// reserved.
    pub fn evaluate_many(&self, points: &[F::Elem]) -> Result<Vec<F::Elem>, PolynomialError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(points.len())
            .map_err(|_| ConfigError::AllocationFailed {
                context: "polynomial evaluations",
                elements: points.len(),
                element_size: core::mem::size_of::<F::Elem>(),
            })?;
        values.extend(points.iter().copied().map(|point| self.evaluate(point)));
        Ok(values)
    }

    /// Evaluate the Hasse derivative of `order` at `point` without allocating.
    ///
    /// `D^[r]f` is the coefficient of `T^r` in `f(X + T)`, so its degree-`d`
    /// term carries the factor `C(d, r)` reduced into the field — not `d!`
    /// divided by anything, and not an `r`-fold formal derivative.
    ///
    /// Cost is one binomial per surviving term: constant time in
    /// characteristic two (a parity test), and `O(min(r, d − r))` field
    /// operations otherwise. The prepared linear-time form, which shares one
    /// factor sequence across the whole coefficient vector, is the `hasse`
    /// crate's object.
    #[must_use]
    pub fn evaluate_hasse(&self, point: F::Elem, order: usize) -> F::Elem {
        if order >= self.coefficient_count() {
            return F::Elem::ZERO;
        }
        let mut power = F::Elem::ONE;
        let mut value = F::Elem::ZERO;
        for degree in order..self.coefficient_count() {
            if F::CHARACTERISTIC == 2 {
                if binomial_odd(degree, order) {
                    value = value.add(self.coefficient(degree).mul(power));
                }
            } else {
                let factor = binomial::<F>(degree, order);
                if !factor.is_zero() {
                    value = value.add(self.coefficient(degree).mul(factor).mul(power));
                }
            }
            power = power.mul(point);
        }
        value
    }

    /// Return the Hasse derivative of the requested order.
    ///
    /// Coefficient `d − r` of the result is `C(d, r) · a_d`, the binomial
    /// taken in the coefficient field's characteristic.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the derivative buffer cannot
    /// be reserved.
    pub fn hasse_derivative(&self, order: usize) -> Result<Self, PolynomialError> {
        if order >= self.coefficient_count() {
            return Ok(Self::zero());
        }
        let output_count = self.coefficient_count() - order;
        let mut coefficients =
            crate::geometry::try_zeroed::<F::Elem>("Hasse derivative", output_count)?;
        for source_degree in order..self.coefficient_count() {
            if F::CHARACTERISTIC == 2 {
                if binomial_odd(source_degree, order) {
                    coefficients[source_degree - order] = self.coefficient(source_degree);
                }
            } else {
                let factor = binomial::<F>(source_degree, order);
                coefficients[source_degree - order] = self.coefficient(source_degree).mul(factor);
            }
        }
        Self::from_coefficients(&coefficients)
    }

    /// Return the first formal derivative.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the derivative buffer cannot
    /// be reserved.
    pub fn formal_derivative(&self) -> Result<Self, PolynomialError> {
        self.hasse_derivative(1)
    }

    /// Compose with the affine polynomial `constant + linear * X`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when an intermediate product
    /// buffer cannot be reserved.
    pub fn compose_linear(
        &self,
        constant: F::Elem,
        linear: F::Elem,
    ) -> Result<Self, PolynomialError> {
        let affine = Self::from_coefficients(&[constant, linear])?;
        let mut result = Self::zero();
        for coefficient in self.coefficients().rev() {
            result = result.multiply(&affine)?;
            if !coefficient.is_zero() {
                let value = result.coefficient(0).add(coefficient);
                result.set_coefficient(0, value)?;
            }
        }
        Ok(result)
    }

    /// Compose with an arbitrary polynomial, returning `self(inner(X))`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when an intermediate product or
    /// coefficient buffer cannot be reserved.
    pub fn compose(&self, inner: &Self) -> Result<Self, PolynomialError> {
        let mut result = Self::zero();
        for coefficient in self.coefficients().rev() {
            result = result.multiply(inner)?;
            if !coefficient.is_zero() {
                let constant = result.coefficient(0).add(coefficient);
                result.set_coefficient(0, constant)?;
            }
        }
        Ok(result)
    }

    /// Compose with `inner` and reduce modulo `modulus`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::DivisionByZero`] or
    /// [`PolynomialError::ConstantModulus`] for an invalid modulus, or
    /// propagates arithmetic and storage failures.
    pub fn compose_mod(&self, inner: &Self, modulus: &Self) -> Result<Self, PolynomialError> {
        super::quotient::ModulusPlan::new(modulus)?.compose(self, inner)
    }

    /// Return the square `self^2`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the square buffer cannot be
    /// reserved.
    pub fn square(&self) -> Result<Self, PolynomialError> {
        let mut result = Self::zero();
        self.square_into(&mut result)?;
        Ok(result)
    }

    /// Reuse this polynomial's buffer to hold a copy of `source`.
    pub fn assign_from(&mut self, source: &Self) {
        self.coefficients.clone_from(&source.coefficients);
    }

    /// Reset to the zero polynomial while retaining allocated capacity.
    pub fn set_zero(&mut self) {
        self.coefficients.clear();
    }

    /// Overwrite with low-to-high coefficients, reusing existing capacity.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the packed buffer cannot be
    /// reserved.
    pub fn assign_coefficients(&mut self, coefficients: &[F::Elem]) -> Result<(), PolynomialError> {
        self.set_zero();
        let byte_len =
            checked_product("polynomial coefficient bytes", coefficients.len(), F::BYTES)?;
        self.resize_coefficients(coefficients.len())?;
        ops::pack::<F>(&mut self.coefficients[..byte_len], coefficients);
        super::dense::canonicalize::<F>(&mut self.coefficients[..byte_len]);
        self.normalize();
        Ok(())
    }

    /// Write the schoolbook product into reusable output storage.
    ///
    /// Unlike [`Self::multiply`], this method does not select the allocating
    /// Karatsuba or Goldilocks NTT routes.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the product buffer cannot be
    /// reserved.
    pub fn multiply_into(&self, other: &Self, out: &mut Self) -> Result<(), PolynomialError> {
        let output_count = match (self.coefficient_count(), other.coefficient_count()) {
            (0, _) | (_, 0) => {
                out.set_zero();
                return Ok(());
            }
            (left, right) => left
                .checked_add(right)
                .and_then(|sum| sum.checked_sub(1))
                .ok_or(ConfigError::GeometryOverflow {
                    context: "polynomial product coefficients",
                })?,
        };
        self.multiply_truncated_into(other, output_count, out)
    }

    /// Write the square `self^2` into reusable `out`.
    ///
    /// In characteristic two `(sum a_i X^i)^2 = sum a_i^2 X^{2i}`: the cross
    /// terms cancel, so squaring spreads each coefficient to twice its degree
    /// and squares it in the field. That is `O(deg)` rather than the
    /// `O(deg^2)` of a general product, and underlies the modular Frobenius
    /// in base-field factorization. The identity is a characteristic-two
    /// fact — the cross terms are `2·a_i·a_j` — so every other
    /// characteristic takes the ordinary product path instead.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the square buffer cannot be
    /// reserved.
    pub fn square_into(&self, out: &mut Self) -> Result<(), PolynomialError> {
        if F::CHARACTERISTIC != 2 {
            return self.multiply_into(self, out);
        }
        out.set_zero();
        let count = self.coefficient_count();
        if count == 0 {
            return Ok(());
        }
        let output_count = 2 * count - 1;
        out.resize_coefficients(output_count)?;
        for degree in 0..count {
            let coefficient = self.coefficient(degree);
            if coefficient.is_zero() {
                continue;
            }
            let squared = coefficient.mul(coefficient);
            let start = 2 * degree * F::BYTES;
            F::encode(&mut out.coefficients[start..start + F::BYTES], squared);
        }
        out.normalize();
        Ok(())
    }

    /// Write the truncated product into reusable output storage.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the truncated product buffer
    /// cannot be reserved.
    pub fn multiply_truncated_into(
        &self,
        other: &Self,
        coefficient_count: usize,
        out: &mut Self,
    ) -> Result<(), PolynomialError> {
        out.set_zero();
        if self.is_zero() || other.is_zero() || coefficient_count == 0 {
            return Ok(());
        }
        let full_count = self
            .coefficient_count()
            .checked_add(other.coefficient_count())
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(ConfigError::GeometryOverflow {
                context: "polynomial product coefficients",
            })?;
        let output_count = coefficient_count.min(full_count);
        out.resize_coefficients(output_count)?;

        let (source, factors) = if self.coefficient_count() >= other.coefficient_count() {
            (self, other)
        } else {
            (other, self)
        };
        for (shift, scale) in factors.coefficients().enumerate() {
            if shift >= output_count || scale.is_zero() {
                continue;
            }
            let source_count = source.coefficient_count().min(output_count - shift);
            out.add_scaled_packed_at_raw(
                scale,
                &source.as_packed()[..source_count * F::BYTES],
                shift,
            )?;
        }
        out.normalize();
        Ok(())
    }

    /// Add `scale * X^shift * source` and restore the canonical form.
    pub(crate) fn add_scaled_packed_at(
        &mut self,
        scale: F::Elem,
        source: &[u8],
        shift: usize,
    ) -> Result<(), PolynomialError> {
        self.add_scaled_packed_at_raw(scale, source, shift)?;
        self.normalize();
        Ok(())
    }

    #[inline]
    fn add_scaled_packed_at_raw(
        &mut self,
        scale: F::Elem,
        source: &[u8],
        shift: usize,
    ) -> Result<(), PolynomialError> {
        if source.is_empty() || scale.is_zero() {
            return Ok(());
        }
        debug_assert_eq!(source.len() % F::BYTES, 0);
        let source_count = source.len() / F::BYTES;
        let required = shift
            .checked_add(source_count)
            .ok_or(ConfigError::GeometryOverflow {
                context: "shifted polynomial coefficient count",
            })?;
        self.resize_coefficients(required)?;
        let start = shift
            .checked_mul(F::BYTES)
            .ok_or(ConfigError::GeometryOverflow {
                context: "shifted polynomial byte offset",
            })?;
        let destination = &mut self.coefficients[start..start + source.len()];
        if use_packed_kernel::<F>(source.len()) {
            ops::mul_add::<F>(destination, scale, source);
        } else {
            for start in (0..source.len()).step_by(F::BYTES) {
                let (output, input) = (
                    &mut destination[start..start + F::BYTES],
                    &source[start..start + F::BYTES],
                );
                F::encode(output, F::decode(output).add(scale.mul(F::decode(input))));
            }
        }
        Ok(())
    }
}

/// Whether `C(upper, lower)` is odd, by the Lucas/Sierpiński parity rule.
///
/// The binomial coefficient `C(upper, lower)` is odd exactly when every set
/// bit of `lower` is also set in `upper` — the coefficient of `X^lower` in the
/// Hasse derivative of order... equivalently, the terms surviving the
/// characteristic-two binomial expansion.
#[must_use]
pub const fn binomial_odd(upper: usize, lower: usize) -> bool {
    lower <= upper && (upper & lower) == lower
}

/// `C(n, k)` for one base-`p` digit pair, with `k <= n < p`.
///
/// Every factor of the numerator lies in `n − k + 1 ..= n` and every factor
/// of the denominator in `1 ..= k`, so all of them are nonzero residues and
/// the denominator is invertible: this never divides by a multiple of `p`.
/// The symmetry `C(n, k) = C(n, n − k)` bounds the loop by `min(k, n − k)`.
fn digit_binomial<F: Field>(n: u64, k: u64) -> F::Elem {
    let k = k.min(n - k);
    let mut numerator = F::Elem::ONE;
    let mut denominator = F::Elem::ONE;
    let mut top = embed_integer::<F>(n);
    let mut bottom = F::Elem::ONE;
    for _ in 0..k {
        numerator = numerator.mul(top);
        denominator = denominator.mul(bottom);
        top = top.sub(F::Elem::ONE);
        bottom = bottom.add(F::Elem::ONE);
    }
    numerator.mul(denominator.inv())
}

/// The binomial coefficient `C(upper, lower)` as a field element.
///
/// This is the Hasse-derivative factor, and it is a *characteristic* fact:
/// what survives is `C(upper, lower) mod p`, which is why
/// [`Field::CHARACTERISTIC`] and never `Field::ORDER` drives it. In
/// characteristic two the answer is [`binomial_odd`]; otherwise Lucas'
/// theorem factors the coefficient over base-`p` digits, and a lower digit
/// exceeding its upper digit makes the whole coefficient zero.
///
/// ```
/// use fgf::{Gf8B, Mersenne31, field::Elem as _};
/// use poly_ring::binomial;
///
/// // C(4, 2) = 6: even, so zero in characteristic two, six over M31.
/// assert!(binomial::<Gf8B>(4, 2).is_zero());
/// assert_eq!(binomial::<Mersenne31>(4, 2), fgf::mersenne31::Elem::from_raw(6));
/// ```
#[must_use]
pub fn binomial<F: Field>(upper: usize, lower: usize) -> F::Elem {
    if lower > upper {
        return F::Elem::ZERO;
    }
    if F::CHARACTERISTIC == 2 {
        return if binomial_odd(upper, lower) {
            F::Elem::ONE
        } else {
            F::Elem::ZERO
        };
    }
    let modulus = u128::from(F::CHARACTERISTIC);
    let mut upper = upper as u128;
    let mut lower = lower as u128;
    let mut result = F::Elem::ONE;
    while upper != 0 || lower != 0 {
        let upper_digit = (upper % modulus) as u64;
        let lower_digit = (lower % modulus) as u64;
        if lower_digit > upper_digit {
            return F::Elem::ZERO;
        }
        result = result.mul(digit_binomial::<F>(upper_digit, lower_digit));
        upper /= modulus;
        lower /= modulus;
    }
    result
}

/// Dispatch a coefficient-vector width to the packed `fgf` kernels.
///
/// The lane-bytes crossover below which the scalar element loop wins is
/// measured, not guessed; see `BENCHMARKS.md` for the measurement that set it.
#[must_use]
pub(crate) fn use_packed_kernel<F: FieldKernels>(byte_len: usize) -> bool {
    byte_len >= backend_for::<F>().lane_bytes()
}
