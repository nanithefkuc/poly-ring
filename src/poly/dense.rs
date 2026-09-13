//! The canonical dense monomial-basis polynomial type.

use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use fgf::field::{Elem, Field};
use fgf::kernel::FieldKernels;
use fgf::ops;

use crate::error::{ConfigError, PolynomialError};
use crate::geometry::{checked_product, try_zeroed};

/// A normalized dense monomial-basis polynomial.
///
/// Coefficients use the field's packed little-endian representation, so wide
/// fixed-scalar operations can execute directly through `fgf` without unsafe
/// casts or representation copies. Zero is represented by an empty buffer;
/// every nonzero value ends in a nonzero coefficient.
///
/// # Invariants
///
/// - The buffer length is a multiple of `F::BYTES` (one packed element per
///   coefficient, low degree first).
/// - No trailing zero coefficient survives any constructor or mutating
///   operation; the zero polynomial is the empty buffer and [`Self::degree`]
///   of it is `None`, so a nonzero constant stays distinguishable from zero.
pub struct Polynomial<F: FieldKernels> {
    pub(crate) coefficients: Vec<u8>,
    field: PhantomData<F>,
}

impl<F: FieldKernels> Clone for Polynomial<F> {
    fn clone(&self) -> Self {
        Self {
            coefficients: self.coefficients.clone(),
            field: PhantomData,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.coefficients.clone_from(&source.coefficients);
    }
}

impl<F: FieldKernels> Polynomial<F> {
    /// The zero polynomial.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            coefficients: Vec::new(),
            field: PhantomData,
        }
    }

    /// Construct a polynomial from low-to-high monomial coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the packed coefficient buffer
    /// length would overflow `usize` or fail to reserve.
    pub fn from_coefficients(coefficients: &[F::Elem]) -> Result<Self, PolynomialError> {
        let byte_len =
            checked_product("polynomial coefficient bytes", coefficients.len(), F::BYTES)?;
        let mut packed = try_zeroed::<u8>("polynomial coefficients", byte_len)?;
        ops::pack::<F>(&mut packed, coefficients);
        canonicalize::<F>(&mut packed);
        let mut polynomial = Self {
            coefficients: packed,
            field: PhantomData,
        };
        polynomial.normalize();
        Ok(polynomial)
    }

    /// Construct a constant polynomial.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when construction of the packed
    /// constant fails.
    pub fn constant(value: F::Elem) -> Result<Self, PolynomialError> {
        Self::from_coefficients(&[value])
    }

    /// The multiplicative identity polynomial.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when construction fails.
    pub fn one() -> Result<Self, PolynomialError> {
        Self::constant(F::Elem::ONE)
    }

    /// Construct from packed field elements, or return `None` for a partial
    /// trailing element.
    #[must_use]
    pub fn from_packed(mut coefficients: Vec<u8>) -> Option<Self> {
        if !coefficients.len().is_multiple_of(F::BYTES) {
            return None;
        }
        canonicalize::<F>(&mut coefficients);
        normalize_bytes::<F>(&mut coefficients);
        Some(Self {
            coefficients,
            field: PhantomData,
        })
    }

    /// Overwrite the stored coefficients with `coefficients` (packed,
    /// coefficient-aligned), restoring the canonical representation.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the resized buffer cannot be
    /// reserved.
    pub fn assign_packed(&mut self, coefficients: &[u8]) -> Result<(), PolynomialError> {
        debug_assert!(coefficients.len().is_multiple_of(F::BYTES));
        let coefficient_count = coefficients.len() / F::BYTES;
        self.resize_coefficients(coefficient_count)?;
        self.coefficients[..coefficients.len()].copy_from_slice(coefficients);
        self.coefficients.truncate(coefficients.len());
        canonicalize::<F>(&mut self.coefficients);
        self.normalize();
        Ok(())
    }

    /// Packed low-to-high coefficient bytes.
    #[must_use]
    pub fn as_packed(&self) -> &[u8] {
        &self.coefficients
    }

    /// Number of stored coefficients, zero for the zero polynomial.
    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        self.coefficients.len() / F::BYTES
    }

    /// Degree of a nonzero polynomial.
    ///
    /// The zero polynomial has no degree; callers distinguish it with
    /// [`Self::is_zero`] rather than reading a sentinel.
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        self.coefficient_count().checked_sub(1)
    }

    /// Whether this is the zero polynomial.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Coefficient of `X^degree`, returning zero beyond the stored degree.
    #[must_use]
    pub fn coefficient(&self, degree: usize) -> F::Elem {
        let Some((start, end)) = degree
            .checked_mul(F::BYTES)
            .and_then(|start| start.checked_add(F::BYTES).map(|end| (start, end)))
        else {
            return F::Elem::ZERO;
        };
        let Some(bytes) = self.coefficients.get(start..end) else {
            return F::Elem::ZERO;
        };
        F::read(bytes)
    }

    /// Stored coefficients in low-to-high order.
    #[must_use]
    pub fn coefficients(
        &self,
    ) -> impl DoubleEndedIterator<Item = F::Elem> + ExactSizeIterator + '_ {
        (0..self.coefficients.len())
            .step_by(F::BYTES)
            .map(|start| F::read(&self.coefficients[start..start + F::BYTES]))
    }

    /// Leading coefficient of a nonzero polynomial.
    #[must_use]
    pub fn leading_coefficient(&self) -> Option<F::Elem> {
        self.coefficients().next_back()
    }

    /// Set one coefficient and restore the canonical representation.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the required coefficient
    /// count overflows or cannot be reserved.
    pub fn set_coefficient(
        &mut self,
        degree: usize,
        value: F::Elem,
    ) -> Result<(), PolynomialError> {
        let required = degree.checked_add(1).ok_or(ConfigError::GeometryOverflow {
            context: "polynomial coefficient count",
        })?;
        self.resize_coefficients(required)?;
        let start = degree * F::BYTES;
        F::write(&mut self.coefficients[start..start + F::BYTES], value);
        canonicalize::<F>(&mut self.coefficients[start..start + F::BYTES]);
        self.normalize();
        Ok(())
    }

    /// Discard coefficients at degrees `>= coefficient_count`.
    pub fn truncate(&mut self, coefficient_count: usize) {
        let byte_len = coefficient_count.saturating_mul(F::BYTES);
        if byte_len < self.coefficients.len() {
            self.coefficients.truncate(byte_len);
            self.normalize();
        }
    }

    /// Ensure the buffer holds at least `coefficient_count` zero coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`PolynomialError::Config`] when the resized buffer length
    /// overflows or cannot be reserved.
    pub fn resize_coefficients(&mut self, coefficient_count: usize) -> Result<(), PolynomialError> {
        let byte_len =
            checked_product("polynomial coefficient bytes", coefficient_count, F::BYTES)?;
        if byte_len > self.coefficients.len() {
            let additional = byte_len - self.coefficients.len();
            self.coefficients
                .try_reserve_exact(additional)
                .map_err(|_| ConfigError::AllocationFailed {
                    context: "polynomial coefficients",
                    elements: byte_len,
                    element_size: 1,
                })?;
            self.coefficients.resize(byte_len, 0);
        }
        Ok(())
    }

    /// Heap capacity retained by the coefficient buffer, in bytes.
    #[must_use]
    pub fn retained_capacity_bytes(&self) -> usize {
        self.coefficients.capacity()
    }

    /// Trim trailing zero coefficients, restoring the canonical form.
    pub fn normalize(&mut self) {
        normalize_bytes::<F>(&mut self.coefficients);
    }
}

impl<F: FieldKernels> Default for Polynomial<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: FieldKernels> PartialEq for Polynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
    }
}

impl<F: FieldKernels> Eq for Polynomial<F> {}

impl<F: FieldKernels> fmt::Debug for Polynomial<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Polynomial")
            .field(&self.coefficients().collect::<Vec<_>>())
            .finish()
    }
}

fn normalize_bytes<F: Field>(coefficients: &mut Vec<u8>) {
    while !coefficients.is_empty()
        && coefficients[coefficients.len() - F::BYTES..]
            .iter()
            .all(|&byte| byte == 0)
    {
        coefficients.truncate(coefficients.len() - F::BYTES);
    }
}

/// Replace every stored lane by its canonical representative.
///
/// `fgf::ops::pack` and a packed byte copy both preserve whatever raw
/// representation the caller handed in, while `fgf`'s packed prime-field
/// kernels are defined on canonical lanes: a noncanonical lane computes an
/// unspecified field value. Every ingress point runs this once, so the
/// arithmetic below it never has to. Characteristic two has exactly one
/// representation per value, so the pass is skipped there entirely — that is
/// also what keeps binary results and binary cost unchanged.
pub(crate) fn canonicalize<F: Field>(coefficients: &mut [u8]) {
    if F::CHARACTERISTIC == 2 {
        return;
    }
    for start in (0..coefficients.len()).step_by(F::BYTES) {
        let slot = &mut coefficients[start..start + F::BYTES];
        F::write(slot, F::read(slot).add(F::Elem::ZERO));
    }
}
