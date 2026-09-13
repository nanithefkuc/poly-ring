//! The dense bivariate polynomial `Q(X,Y) = Σ_j Q_j(X)·Y^j`.
//!
//! Row `j` is the coefficient polynomial of `Y^j`; every row is a canonical
//! [`Polynomial`], trailing all-zero `Y` rows are removed, and the zero
//! polynomial has no rows. This is the row-oriented representation the
//! interpolation engines consume; the sparse [`crate::poly::multivariate`]
//! form converts explicitly, never implicitly.

use alloc::vec::Vec;

#[cfg(feature = "fft")]
use butterfly_fft::core::kernel::ButterflyKernels;
use fgf::field::Elem;
use fgf::kernel::FieldKernels;

use super::binomial;
use super::dense::Polynomial;
#[cfg(feature = "fft")]
use crate::error::ProductError;
use crate::error::{ConfigError, PolynomialError};
use crate::geometry::try_zeroed;

/// Leading monomial under a `(1, y_weight)` weighted order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightedTerm {
    /// Exponent of `X`.
    pub x_degree: usize,
    /// Exponent of `Y`.
    pub y_degree: usize,
    /// `x_degree + y_degree * y_weight`.
    pub weighted_degree: usize,
}

/// A bivariate polynomial `Q(X,Y) = sum_j Q_j(X) Y^j`.
#[derive(Clone, Debug)]
pub struct BivariatePolynomial<F: FieldKernels> {
    y_coefficients: Vec<Polynomial<F>>,
}

impl<F: FieldKernels> BivariatePolynomial<F> {
    /// The zero bivariate polynomial.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            y_coefficients: Vec::new(),
        }
    }

    /// Construct from low-to-high `Y` coefficient polynomials.
    #[must_use]
    pub fn from_y_coefficients(mut coefficients: Vec<Polynomial<F>>) -> Self {
        normalize_y_coefficients(&mut coefficients);
        Self {
            y_coefficients: coefficients,
        }
    }

    /// Number of stored `Y` coefficient rows.
    #[must_use]
    pub fn y_coefficient_count(&self) -> usize {
        self.y_coefficients.len()
    }

    /// Degree in `Y` for a nonzero polynomial.
    #[must_use]
    pub fn y_degree(&self) -> Option<usize> {
        self.y_coefficients.len().checked_sub(1)
    }

    /// Stored low-to-high `Y` coefficient rows.
    #[must_use]
    pub fn y_coefficients(&self) -> &[Polynomial<F>] {
        &self.y_coefficients
    }

    /// Coefficient polynomial of `Y^degree`, when stored.
    #[must_use]
    pub fn y_coefficient(&self, degree: usize) -> Option<&Polynomial<F>> {
        self.y_coefficients.get(degree)
    }

    /// Set the coefficient polynomial of `Y^degree`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when the row count cannot be
    /// represented and [`ConfigError::AllocationFailed`] when the row vector
    /// cannot be reserved.
    pub fn set_y_coefficient(
        &mut self,
        degree: usize,
        coefficient: Polynomial<F>,
    ) -> Result<(), ConfigError> {
        let required = degree.checked_add(1).ok_or(ConfigError::GeometryOverflow {
            context: "bivariate Y coefficient count",
        })?;
        if required > self.y_coefficients.len() {
            let additional = required - self.y_coefficients.len();
            self.y_coefficients
                .try_reserve_exact(additional)
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "bivariate Y coefficients",
                    elements: required,
                    element_size: core::mem::size_of::<Polynomial<F>>(),
                })?;
            self.y_coefficients.resize_with(required, Polynomial::zero);
        }
        self.y_coefficients[degree] = coefficient;
        self.normalize();
        Ok(())
    }

    pub(crate) fn prepare_y_rows(&mut self, count: usize) -> Result<(), ConfigError> {
        if count > self.y_coefficients.len() {
            self.y_coefficients
                .try_reserve_exact(count - self.y_coefficients.len())
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "bivariate Y coefficients",
                    elements: count,
                    element_size: core::mem::size_of::<Polynomial<F>>(),
                })?;
            self.y_coefficients.resize_with(count, Polynomial::zero);
        } else {
            self.y_coefficients.truncate(count);
        }
        Ok(())
    }

    /// Whether every coefficient row is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.y_coefficients.is_empty()
    }

    /// Leading term under `(1, y_weight)` weighted degree, breaking ties by
    /// larger `Y`-degree.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::GeometryOverflow`] when a weighted degree
    /// cannot be represented.
    pub fn weighted_leading_term(
        &self,
        y_weight: usize,
    ) -> Result<Option<WeightedTerm>, ConfigError> {
        let mut leading = None;
        for (y_degree, coefficient) in self.y_coefficients.iter().enumerate() {
            let Some(x_degree) = coefficient.degree() else {
                continue;
            };
            let weighted_degree = y_degree
                .checked_mul(y_weight)
                .and_then(|weight| weight.checked_add(x_degree))
                .ok_or(ConfigError::GeometryOverflow {
                    context: "bivariate weighted degree",
                })?;
            let candidate = WeightedTerm {
                x_degree,
                y_degree,
                weighted_degree,
            };
            if leading.is_none_or(|current: WeightedTerm| {
                (
                    candidate.weighted_degree,
                    candidate.y_degree,
                    candidate.x_degree,
                ) > (current.weighted_degree, current.y_degree, current.x_degree)
            }) {
                leading = Some(candidate);
            }
        }
        Ok(leading)
    }

    /// Maximum `(1, y_weight)` weighted degree.
    ///
    /// # Errors
    ///
    /// As [`Self::weighted_leading_term`].
    pub fn weighted_degree(&self, y_weight: usize) -> Result<Option<usize>, ConfigError> {
        Ok(self
            .weighted_leading_term(y_weight)?
            .map(|term| term.weighted_degree))
    }

    /// Evaluate `Q(x,y)`.
    #[must_use]
    pub fn evaluate(&self, x: F::Elem, y: F::Elem) -> F::Elem {
        self.y_coefficients
            .iter()
            .rev()
            .fold(F::Elem::ZERO, |value, coefficient| {
                value.mul(y).add(coefficient.evaluate(x))
            })
    }

    /// Evaluate the bivariate Hasse derivative `Q^[x_order,y_order](x,y)`.
    ///
    /// The `Y` factor is `C(y_degree, y_order) · y^(y_degree - y_order)` —
    /// a characteristic fact taken through [`binomial`], so the value is
    /// correct over binary and prime fields alike, including orders at or
    /// above the characteristic.
    #[must_use]
    pub fn hasse_discrepancy(
        &self,
        x: F::Elem,
        y: F::Elem,
        x_order: usize,
        y_order: usize,
    ) -> F::Elem {
        if y_order >= self.y_coefficients.len() {
            return F::Elem::ZERO;
        }
        let mut y_power = F::Elem::ONE;
        let mut value = F::Elem::ZERO;
        for y_degree in y_order..self.y_coefficients.len() {
            let factor = binomial::<F>(y_degree, y_order);
            if !factor.is_zero() {
                value = value.add(
                    self.y_coefficients[y_degree]
                        .evaluate_hasse(x, x_order)
                        .mul(factor)
                        .mul(y_power),
                );
            }
            y_power = y_power.mul(y);
        }
        value
    }

    /// Row-wise sum with `rhs`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the row vector or a row
    /// buffer cannot be reserved.
    pub fn add(&self, rhs: &Self) -> Result<Self, PolynomialError> {
        let mut result = self.clone();
        result.add_scaled_x_shifted_assign(F::Elem::ONE, rhs, 0)?;
        Ok(result)
    }

    /// Row-wise difference `self - rhs`.
    ///
    /// Subtraction scales by the additive inverse; it is never a XOR.
    ///
    /// # Errors
    ///
    /// As [`Self::add`].
    pub fn sub(&self, rhs: &Self) -> Result<Self, PolynomialError> {
        let mut result = self.clone();
        result.add_scaled_x_shifted_assign(F::Elem::ONE.neg(), rhs, 0)?;
        Ok(result)
    }

    /// Copy with every coefficient scaled by `scale`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when a row buffer cannot be
    /// reserved. A zero scale yields the canonical zero polynomial.
    pub fn scaled(&self, scale: F::Elem) -> Result<Self, PolynomialError> {
        let mut result = self.clone();
        for row in &mut result.y_coefficients {
            row.scale_assign(scale);
        }
        result.normalize();
        Ok(result)
    }

    /// The convolution `self · rhs`, row-wise in `Y`:
    /// row `k` is `Σ_{i+j=k} self_i · rhs_j` through [`Polynomial::multiply`].
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the output row count or any
    /// product cannot be reserved.
    pub fn multiply(&self, rhs: &Self) -> Result<Self, PolynomialError> {
        if self.is_zero() || rhs.is_zero() {
            return Ok(Self::zero());
        }
        let row_count = self
            .y_coefficients
            .len()
            .checked_add(rhs.y_coefficients.len())
            .and_then(|count| count.checked_sub(1))
            .ok_or(ConfigError::GeometryOverflow {
                context: "bivariate product row count",
            })?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(row_count).map_err(|_| {
            PolynomialError::Config(ConfigError::AllocationFailed {
                context: "bivariate product rows",
                elements: row_count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })
        })?;
        rows.resize_with(row_count, Polynomial::zero);
        for (source_y, coefficient) in self.y_coefficients.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            for (other_y, other) in rhs.y_coefficients.iter().enumerate() {
                if other.is_zero() {
                    continue;
                }
                let product = coefficient.multiply(other)?;
                rows[source_y + other_y].add_assign(&product)?;
            }
        }
        Ok(Self::from_y_coefficients(rows))
    }

    /// The bivariate Hasse derivative `D^[x_order,y_order] Q` as a polynomial.
    ///
    /// Row `j >= y_order` contributes `C(j, y_order) · D^[x_order] Q_j(X)`
    /// into output row `j - y_order`, the binomial taken in the field's
    /// characteristic. Derivative orders beyond either support produce the
    /// zero polynomial.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when a row derivative cannot be
    /// reserved.
    pub fn hasse_derivative(
        &self,
        x_order: usize,
        y_order: usize,
    ) -> Result<Self, PolynomialError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(self.y_coefficients.len().saturating_sub(y_order))
            .map_err(|_| {
                PolynomialError::Config(ConfigError::AllocationFailed {
                    context: "bivariate derivative rows",
                    elements: self.y_coefficients.len(),
                    element_size: core::mem::size_of::<Polynomial<F>>(),
                })
            })?;
        for (y_degree, coefficient) in self.y_coefficients.iter().enumerate().skip(y_order) {
            let factor = binomial::<F>(y_degree, y_order);
            let derivative = coefficient.hasse_derivative(x_order)?;
            rows.push(if factor.is_one() {
                derivative
            } else {
                derivative.scaled(factor)
            });
        }
        Ok(Self::from_y_coefficients(rows))
    }

    /// Multiply every `Y` coefficient row by `X + constant`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when a row product cannot be
    /// reserved.
    pub fn multiply_x_plus(&self, constant: F::Elem) -> Result<Self, PolynomialError> {
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(self.y_coefficients.len())
            .map_err(|_| {
                PolynomialError::Config(ConfigError::AllocationFailed {
                    context: "bivariate linear product",
                    elements: self.y_coefficients.len(),
                    element_size: core::mem::size_of::<Polynomial<F>>(),
                })
            })?;
        for coefficient in &self.y_coefficients {
            coefficients.push(coefficient.multiply_x_plus(constant)?);
        }
        Ok(Self::from_y_coefficients(coefficients))
    }

    /// Substitute `Y = constant + X*Z`, returning a polynomial in `(X,Z)`.
    ///
    /// The `Y` expansion runs through [`binomial`], so the substitution is
    /// exact over every characteristic.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] for geometry overflow or a failed
    /// reservation.
    pub fn substitute_y_linear(&self, constant: F::Elem) -> Result<Self, PolynomialError> {
        let Some(y_degree) = self.y_degree() else {
            return Ok(Self::zero());
        };
        let count = y_degree
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "bivariate substitution rows",
            })?;
        let mut powers = try_zeroed::<F::Elem>("bivariate substitution powers", count)?;
        powers[0] = F::Elem::ONE;
        for exponent in 1..count {
            powers[exponent] = powers[exponent - 1].mul(constant);
        }
        let mut output = Vec::new();
        output.try_reserve_exact(count).map_err(|_| {
            PolynomialError::Config(ConfigError::AllocationFailed {
                context: "bivariate substitution rows",
                elements: count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })
        })?;
        output.resize_with(count, Polynomial::zero);

        for (source_y, coefficient) in self.y_coefficients.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            for target_y in 0..=source_y {
                let factor = binomial::<F>(source_y, target_y);
                if factor.is_zero() {
                    continue;
                }
                let scale = powers[source_y - target_y].mul(factor);
                if scale.is_zero() {
                    continue;
                }
                let term = coefficient.scaled_shifted(scale, target_y)?;
                output[target_y].add_assign(&term)?;
            }
        }
        Ok(Self::from_y_coefficients(output))
    }

    /// Substitute `Y = prefix(X) + X^tail_degree*Z`, truncating every output
    /// row modulo `X^coefficient_count`.
    ///
    /// This is the affine-prefix transform used by divide-and-conquer root
    /// extraction. Products are truncated before shifts so coefficients that
    /// cannot affect the requested precision are never materialized. The
    /// `Y` expansion runs through [`binomial`], exact over every
    /// characteristic.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] for geometry overflow or a failed
    /// reservation.
    pub fn substitute_y_affine_truncated(
        &self,
        prefix: &Polynomial<F>,
        tail_degree: usize,
        coefficient_count: usize,
    ) -> Result<Self, PolynomialError> {
        let Some(y_degree) = self.y_degree() else {
            return Ok(Self::zero());
        };
        if coefficient_count == 0 {
            return Ok(Self::zero());
        }
        let power_count = y_degree
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "affine bivariate substitution powers",
            })?;
        let mut prefix_powers = Vec::new();
        prefix_powers.try_reserve_exact(power_count).map_err(|_| {
            PolynomialError::Config(ConfigError::AllocationFailed {
                context: "affine bivariate substitution powers",
                elements: power_count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })
        })?;
        prefix_powers.push(Polynomial::one()?);
        for exponent in 1..power_count {
            prefix_powers
                .push(prefix_powers[exponent - 1].multiply_truncated(prefix, coefficient_count)?);
        }

        let mut output = Vec::new();
        output.try_reserve_exact(power_count).map_err(|_| {
            PolynomialError::Config(ConfigError::AllocationFailed {
                context: "affine bivariate substitution rows",
                elements: power_count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })
        })?;
        output.resize_with(power_count, Polynomial::zero);

        for (source_y, coefficient) in self.y_coefficients.iter().enumerate() {
            for target_y in 0..=source_y {
                let factor = binomial::<F>(source_y, target_y);
                if factor.is_zero() {
                    continue;
                }
                let Some(shift) = tail_degree.checked_mul(target_y) else {
                    continue;
                };
                if shift >= coefficient_count {
                    continue;
                }
                let product = coefficient
                    .multiply_truncated(
                        &prefix_powers[source_y - target_y],
                        coefficient_count - shift,
                    )?
                    .scaled(factor);
                if product.is_zero() {
                    continue;
                }
                output[target_y].add_assign(&product.shifted(shift)?)?;
            }
        }
        Ok(Self::from_y_coefficients(output))
    }

    /// Batched-product variant of [`Self::substitute_y_affine_truncated`].
    ///
    /// Independent coefficient-row products are packed across transform row
    /// columns; the measured product crossover retains schoolbook arithmetic
    /// for smaller nodes. The substitution itself runs through the ring's
    /// batched-product engine.
    ///
    /// # Errors
    ///
    /// Returns [`ProductError`] for geometry overflow, a failed reservation,
    /// or a failed batched product.
    #[cfg(feature = "fft")]
    pub fn substitute_y_affine_truncated_fast(
        &self,
        prefix: &Polynomial<F>,
        tail_degree: usize,
        coefficient_count: usize,
        scratch: &mut super::afft::PolynomialProductScratch<F>,
    ) -> Result<Self, ProductError>
    where
        F: ButterflyKernels,
    {
        use super::afft::{ProductStrategy, multiply_batch_truncated};

        let Some(y_degree) = self.y_degree() else {
            return Ok(Self::zero());
        };
        if coefficient_count == 0 {
            return Ok(Self::zero());
        }
        let power_count = y_degree
            .checked_add(1)
            .ok_or(ConfigError::GeometryOverflow {
                context: "fast affine substitution powers",
            })?;
        let mut prefix_powers = Vec::new();
        prefix_powers.try_reserve_exact(power_count).map_err(|_| {
            ProductError::Config(ConfigError::AllocationFailed {
                context: "fast affine substitution powers",
                elements: power_count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })
        })?;
        prefix_powers.push(Polynomial::one()?);
        for exponent in 1..power_count {
            prefix_powers
                .push(prefix_powers[exponent - 1].multiply_truncated(prefix, coefficient_count)?);
        }

        let pair_capacity =
            power_count
                .checked_mul(power_count)
                .ok_or(ConfigError::GeometryOverflow {
                    context: "fast affine substitution product count",
                })?;
        let mut pairs = Vec::new();
        let mut metadata = Vec::new();
        pairs.try_reserve_exact(pair_capacity).map_err(|_| {
            ProductError::Config(ConfigError::AllocationFailed {
                context: "fast affine substitution products",
                elements: pair_capacity,
                element_size: core::mem::size_of::<(&Polynomial<F>, &Polynomial<F>)>(),
            })
        })?;
        metadata.try_reserve_exact(pair_capacity).map_err(|_| {
            ProductError::Config(ConfigError::AllocationFailed {
                context: "fast affine substitution metadata",
                elements: pair_capacity,
                element_size: core::mem::size_of::<(usize, usize, F::Elem)>(),
            })
        })?;
        for (source_y, coefficient) in self.y_coefficients.iter().enumerate() {
            for target_y in 0..=source_y {
                let factor = binomial::<F>(source_y, target_y);
                if factor.is_zero() {
                    continue;
                }
                let Some(shift) = tail_degree.checked_mul(target_y) else {
                    continue;
                };
                if shift >= coefficient_count {
                    continue;
                }
                pairs.push((coefficient, &prefix_powers[source_y - target_y]));
                metadata.push((target_y, shift, factor));
            }
        }

        let mut products = Vec::new();
        multiply_batch_truncated(
            &pairs,
            coefficient_count,
            ProductStrategy::Auto,
            scratch,
            &mut products,
        )?;
        let mut output = Vec::new();
        output.try_reserve_exact(power_count).map_err(|_| {
            ProductError::Config(ConfigError::AllocationFailed {
                context: "fast affine substitution rows",
                elements: power_count,
                element_size: core::mem::size_of::<Polynomial<F>>(),
            })
        })?;
        output.resize_with(power_count, Polynomial::zero);
        for (mut product, (target_y, shift, factor)) in products.into_iter().zip(metadata) {
            product.scale_assign(factor);
            product.truncate(coefficient_count - shift);
            if !product.is_zero() {
                output[target_y].add_assign(&product.shifted(shift)?)?;
            }
        }
        Ok(Self::from_y_coefficients(output))
    }

    /// Return a copy with every `X` coefficient row reduced modulo
    /// `X^coefficient_count`.
    #[must_use]
    pub fn truncated_x(&self, coefficient_count: usize) -> Self {
        let mut coefficients = self.y_coefficients.clone();
        for coefficient in &mut coefficients {
            coefficient.truncate(coefficient_count);
        }
        Self::from_y_coefficients(coefficients)
    }

    /// Minimum `X` valuation shared by all nonzero coefficient rows.
    #[must_use]
    pub fn x_valuation(&self) -> Option<usize> {
        self.y_coefficients
            .iter()
            .filter_map(Polynomial::x_valuation)
            .min()
    }

    /// Divide every coefficient row exactly by `X^power`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when a row quotient cannot be
    /// reserved, propagating a non-exact division if a row has a lower
    /// `X` valuation than `power`.
    pub fn divide_by_x_power(&self, power: usize) -> Result<Self, PolynomialError> {
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(self.y_coefficients.len())
            .map_err(|_| {
                PolynomialError::Config(ConfigError::AllocationFailed {
                    context: "bivariate X-power quotient",
                    elements: self.y_coefficients.len(),
                    element_size: core::mem::size_of::<Polynomial<F>>(),
                })
            })?;
        for coefficient in &self.y_coefficients {
            coefficients.push(coefficient.divide_by_x_power(power)?);
        }
        Ok(Self::from_y_coefficients(coefficients))
    }

    /// Compose `Y` with `candidate`, returning `Q(X,candidate(X))`.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when a product cannot be reserved.
    pub fn compose_y(&self, candidate: &Polynomial<F>) -> Result<Polynomial<F>, PolynomialError> {
        let mut result = Polynomial::zero();
        for coefficient in self.y_coefficients.iter().rev() {
            result = result.multiply(candidate)?;
            result.add_assign(coefficient)?;
        }
        Ok(result)
    }

    /// Add `scale · other` with every row of `other` shifted `x_shift`
    /// places in `X`, in place.
    ///
    /// The row vector grows to `max(self, other)` rows before any row is
    /// touched, so the update is a single checked bulk operation. This is
    /// the accumulation primitive behind module construction and
    /// interpolation; normalization runs once on success.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the widened row vector or a
    /// shifted row cannot be reserved.
    pub fn add_scaled_x_shifted_assign(
        &mut self,
        scale: F::Elem,
        other: &Self,
        x_shift: usize,
    ) -> Result<(), PolynomialError> {
        self.prepare_y_rows(self.y_coefficient_count().max(other.y_coefficient_count()))?;
        for (target, source) in self.y_coefficients.iter_mut().zip(&other.y_coefficients) {
            target.add_scaled_shifted_assign(scale, source, x_shift)?;
        }
        self.normalize();
        Ok(())
    }

    /// Overwrite every row from packed coefficient slices, ascending `Y`.
    ///
    /// Each item is one ascending-`X` packed row in `fgf`'s little-endian
    /// element layout; items are ascending `Y`. Every byte length is
    /// validated as a whole number of `F::BYTES` elements before any
    /// mutation, so a partial trailing element rejects the whole call
    /// without touching the polynomial. Existing row buffers are reused
    /// through [`Polynomial::assign_packed`], rows beyond the supplied count
    /// are dropped, and the outer vector is normalized once at the end. If a
    /// row reservation fails after earlier rows were written, the polynomial
    /// is normalized before the error returns: the replacement may be
    /// partial but the representation stays canonical. Empty input resets to
    /// zero.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] wrapping
    /// [`ConfigError::BufferLength`] for a partial field element (context
    /// `"bivariate packed row"`), [`ConfigError::GeometryOverflow`] when the
    /// expected row length cannot be represented, and
    /// [`ConfigError::AllocationFailed`] when the row vector or a row buffer
    /// cannot be reserved.
    pub fn assign_y_coefficients_packed<'a, I>(&mut self, rows: I) -> Result<(), PolynomialError>
    where
        I: ExactSizeIterator<Item = &'a [u8]> + Clone,
    {
        for row in rows.clone() {
            let Some(expected) = row.len().checked_next_multiple_of(F::BYTES) else {
                return Err(ConfigError::GeometryOverflow {
                    context: "bivariate packed row",
                }
                .into());
            };
            if row.len() != expected {
                return Err(PolynomialError::Config(ConfigError::BufferLength {
                    context: "bivariate packed row",
                    expected,
                    actual: row.len(),
                }));
            }
        }
        let count = rows.len();
        self.prepare_y_rows(count)?;
        for (slot, row) in self.y_coefficients.iter_mut().zip(rows) {
            if let Err(error) = slot.assign_packed(row) {
                self.normalize();
                return Err(error);
            }
        }
        self.normalize();
        Ok(())
    }

    /// Whether `Q(X, candidate(X))` is the zero polynomial, i.e. whether
    /// `Y - candidate(X)` divides this polynomial.
    ///
    /// Over characteristic two `Y - candidate` and `Y + candidate` are the
    /// same factor; over other characteristics the sign matters, and this
    /// check is the root test proper.
    ///
    /// # Errors
    ///
    /// As [`Self::compose_y`].
    pub fn has_root(&self, candidate: &Polynomial<F>) -> Result<bool, PolynomialError> {
        Ok(self.compose_y(candidate)?.is_zero())
    }

    pub(crate) fn normalize(&mut self) {
        normalize_y_coefficients(&mut self.y_coefficients);
    }
}

impl<F: FieldKernels> PartialEq for BivariatePolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        self.y_coefficients == other.y_coefficients
    }
}

impl<F: FieldKernels> Eq for BivariatePolynomial<F> {}

impl<F: FieldKernels> Default for BivariatePolynomial<F> {
    fn default() -> Self {
        Self::zero()
    }
}

fn normalize_y_coefficients<F: FieldKernels>(coefficients: &mut Vec<Polynomial<F>>) {
    while coefficients.last().is_some_and(Polynomial::is_zero) {
        coefficients.pop();
    }
}
